"""Event lineup store: the lineup anchors of every match of an event, so a
later match's lineup card can borrow rows for teams it did not read twice.

Spec: docs/superpowers/specs/2026-11-10-event-lineup-design.md section 1.

Rows: table ``event_anchors`` in frc_scorer.db (created by
web/database.init_db and by ``replace`` / ``clear`` here). Tiles: JPEG files
at ANCHOR_ROOT/<namespace>/<event_key>/<id>.jpg (git-ignored); ``tile_path``
holds that path relative to the root. Ids are AUTOINCREMENT and never
reused, so a re-run's new files can never collide with the files it
replaces.

This module is the SQL + file layer and the sync access path (sqlite3, for
scripts). The app's async path is web/database.fetch_event_anchors /
replace_event_anchors, built on the same SQL and file helpers. The
window-label stage imports only ``Anchor`` and the kind names."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from web.field.match_order import match_sort_key

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "frc_scorer.db"                 # the same file as web.database.DB_PATH
ANCHOR_ROOT = REPO / "data" / "event_anchors"
LIVE = "live"
KIND_DIGITS, KIND_ELIM = "digits", "elimination"
KINDS = (KIND_DIGITS, KIND_ELIM)
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")

DDL = """
CREATE TABLE IF NOT EXISTS event_anchors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    event_key TEXT NOT NULL,
    team TEXT NOT NULL,
    match_key TEXT NOT NULL,
    match_number INTEGER,
    replay_id TEXT,
    frame_num INTEGER,
    alliance_color TEXT,
    kind TEXT NOT NULL,
    p REAL,
    tile_path TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_event_anchors_team ON event_anchors(namespace, event_key, team);
"""

SELECT_SQL = ("SELECT id, team, match_key, match_number, replay_id, frame_num, alliance_color, "
              "kind, p, tile_path FROM event_anchors WHERE namespace = ? AND event_key = ? "
              "AND team IN ({teams}) AND match_key != ?{cutoff} ORDER BY id")
OLD_PATHS_SQL = ("SELECT tile_path FROM event_anchors "
                 "WHERE namespace = ? AND event_key = ? AND match_key = ?")
DELETE_SQL = "DELETE FROM event_anchors WHERE namespace = ? AND event_key = ? AND match_key = ?"
INSERT_SQL = ("INSERT INTO event_anchors (namespace, event_key, team, match_key, match_number, "
              "replay_id, frame_num, alliance_color, kind, p, tile_path) "
              "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '')")
SET_PATH_SQL = "UPDATE event_anchors SET tile_path = ? WHERE id = ?"


@dataclass(frozen=True)
class Anchor:
    """One lineup tile of one team. The stage fills the first six fields;
    the store stamps the match fields and the id."""
    team: str
    kind: str                         # KIND_DIGITS | KIND_ELIM
    p: float                          # digit p (elimination: the smaller partner p)
    frame_num: int                    # source frame of the tile
    alliance_color: Optional[str]     # the team's alliance in the source match
    tile: bytes                       # JPEG
    match_key: Optional[str] = None
    match_number: Optional[int] = None
    replay_id: Optional[str] = None
    id: Optional[int] = None


def schedule_rank(match_key) -> Optional[int]:
    """Schedule position of a TBA match key: qmN -> N; playoffs after every
    qual (level * 1000 + set * 10 + match). None for an unrecognised key."""
    level, a, b = match_sort_key(str(match_key or ""))
    if level == 99:
        return None
    return b if level == 0 else level * 1000 + a * 10 + b


def check_name(name, what: str) -> None:
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"bad {what} {name!r}")


def validate(namespace: str, event_key: str, anchors: Iterable) -> None:
    check_name(namespace, "namespace")
    check_name(event_key, "event_key")
    for a in anchors:
        if a.kind not in KINDS:
            raise ValueError(f"bad anchor kind {a.kind!r}")


def select_query(namespace, event_key, teams, exclude_match_key, max_match_number=None):
    teams = [str(t) for t in teams]
    sql = SELECT_SQL.format(teams=",".join("?" * len(teams)),
                            cutoff="" if max_match_number is None else " AND match_number <= ?")
    args = [namespace, event_key, *teams, exclude_match_key or ""]
    if max_match_number is not None:
        args.append(int(max_match_number))
    return sql, args


def insert_args(namespace, event_key, match_key, match_number, replay_id, a: Anchor) -> tuple:
    return (namespace, event_key, str(a.team), match_key, match_number, replay_id,
            int(a.frame_num), a.alliance_color, a.kind, float(a.p))


def tile_relpath(namespace: str, event_key: str, anchor_id: int) -> str:
    return f"{namespace}/{event_key}/{int(anchor_id)}.jpg"


def write_tile(root, rel: str, data: bytes) -> None:
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def remove_files(paths: Iterable) -> None:
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass


def rows_to_anchors(rows, root) -> dict:
    """team -> [Anchor] (in id order) with the tile bytes read from disk; a
    row whose file is gone is skipped."""
    out: dict = {}
    for aid, team, mk, mn, rid, fn, colour, kind, p, rel in rows:
        try:
            tile = (Path(root) / rel).read_bytes()
        except OSError:
            continue
        out.setdefault(str(team), []).append(Anchor(
            team=str(team), kind=kind, p=float(p), frame_num=int(fn), alliance_color=colour,
            tile=tile, match_key=mk, match_number=mn, replay_id=rid, id=aid))
    return out


def fetch(namespace: str, event_key: str, teams, exclude_match_key: Optional[str],
          max_match_number: Optional[int] = None, *, db_path=None, root=None) -> dict:
    """team -> [Anchor] for ``teams`` from every other match of the event in
    ``namespace`` (only match_number <= ``max_match_number`` when given).
    Read-only; {} when the DB or the table does not exist yet."""
    teams = [str(t) for t in teams]
    db = Path(db_path or DB_PATH)
    if not teams or not db.exists():
        return {}
    sql, args = select_query(namespace, event_key, teams, exclude_match_key, max_match_number)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30.0)
    try:
        rows = con.execute(sql, args).fetchall()
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            return {}
        raise
    finally:
        con.close()
    return rows_to_anchors(rows, root or ANCHOR_ROOT)


def replace(namespace: str, event_key: str, match_key: str, anchors, *, replay_id=None,
            match_number: Optional[int] = None, db_path=None, root=None) -> int:
    """Replace the anchors of (namespace, event_key, match_key): delete its
    rows, insert ``anchors``, write their tiles, commit, then delete the old
    tiles. On any failure before the commit nothing changes (new tiles are
    removed). Returns the number written."""
    anchors = list(anchors)
    validate(namespace, event_key, anchors)
    root = Path(root or ANCHOR_ROOT)
    mn = schedule_rank(match_key) if match_number is None else int(match_number)
    con = sqlite3.connect(str(db_path or DB_PATH), timeout=30.0)
    new_files: list = []
    try:
        con.executescript(DDL)
        con.execute("BEGIN IMMEDIATE")
        old = [r[0] for r in con.execute(OLD_PATHS_SQL, (namespace, event_key, match_key))]
        con.execute(DELETE_SQL, (namespace, event_key, match_key))
        for a in anchors:
            cur = con.execute(INSERT_SQL, insert_args(namespace, event_key, match_key, mn,
                                                      replay_id, a))
            rel = tile_relpath(namespace, event_key, cur.lastrowid)
            con.execute(SET_PATH_SQL, (rel, cur.lastrowid))
            write_tile(root, rel, a.tile)
            new_files.append(root / rel)
        con.commit()
    except BaseException:
        con.rollback()
        remove_files(new_files)
        raise
    finally:
        con.close()
    remove_files(root / r for r in old)
    return len(anchors)


def clear(namespace: str, event_key: str, *, db_path=None, root=None) -> int:
    """Delete every anchor of one event in one namespace. Returns the count."""
    check_name(namespace, "namespace")
    check_name(event_key, "event_key")
    root = Path(root or ANCHOR_ROOT)
    con = sqlite3.connect(str(db_path or DB_PATH), timeout=30.0)
    try:
        con.executescript(DDL)
        old = [r[0] for r in con.execute(
            "SELECT tile_path FROM event_anchors WHERE namespace = ? AND event_key = ?",
            (namespace, event_key))]
        con.execute("DELETE FROM event_anchors WHERE namespace = ? AND event_key = ?",
                    (namespace, event_key))
        con.commit()
    finally:
        con.close()
    remove_files(root / r for r in old)
    return len(old)
