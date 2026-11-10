"""
SQLite database for persisting match results and per-robot scoring data.
"""

import asyncio

import aiosqlite
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from web.embedder import ACTIVE_MODEL
from web.field import event_anchors as _event_anchors

DB_PATH = Path(__file__).parent.parent / "frc_scorer.db"

# Every row written before the embed_model column shipped was embedded with
# dinov2-small; NULL embed_model (pre-migration rows, or a row some other path
# forgot to stamp) is treated as that model rather than as "unknown". The
# LITERAL is intentional and must NOT track embedder.DEFAULT_MODEL: when the
# default flipped to DINOv3 (2026-07-18), an alias would have silently
# re-attributed every legacy dinov2 row to the new model — exactly the
# cross-embedding-space contamination this tag exists to prevent.
_LEGACY_EMBED_MODEL = "facebook/dinov2-small"

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_key TEXT UNIQUE NOT NULL,
    event_name TEXT NOT NULL,
    year INTEGER NOT NULL,
    location TEXT,
    start_date TEXT,
    end_date TEXT,
    ocr_regions TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id),
    tba_match_key TEXT UNIQUE NOT NULL,
    comp_level TEXT NOT NULL,
    match_number INTEGER NOT NULL,
    set_number INTEGER DEFAULT 1,
    video_source TEXT,
    tba_video_key TEXT,
    blue_predicted_score INTEGER DEFAULT 0,
    red_predicted_score INTEGER DEFAULT 0,
    tba_blue_score INTEGER,
    tba_red_score INTEGER,
    tba_blue_hub_total INTEGER,
    tba_red_hub_total INTEGER,
    tba_blue_hub_auto INTEGER,
    tba_red_hub_auto INTEGER,
    tba_blue_hub_teleop INTEGER,
    tba_red_hub_teleop INTEGER,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS match_robots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(id),
    alliance TEXT NOT NULL CHECK(alliance IN ('blue', 'red')),
    position INTEGER NOT NULL CHECK(position BETWEEN 1 AND 3),
    team_number TEXT NOT NULL,
    predicted_fuel_count INTEGER DEFAULT 0,
    corrected_fuel_count INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(match_id, alliance, position)
);

CREATE TABLE IF NOT EXISTS scoring_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(id),
    match_robot_id INTEGER REFERENCES match_robots(id),
    frame_num INTEGER,
    timestamp REAL,
    alliance TEXT NOT NULL,
    confidence TEXT,
    distance_to_hub REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS replays (
    id TEXT PRIMARY KEY,
    created_at TEXT,
    label TEXT,
    event_key TEXT,
    match_key TEXT,
    comp_level TEXT,
    blue_score INTEGER DEFAULT 0,
    red_score INTEGER DEFAULT 0,
    blue_teams TEXT,
    red_teams TEXT,
    video_path TEXT,
    meta_path TEXT,
    source_video_path TEXT,
    thumb_path TEXT,
    fps REAL,
    frame_count INTEGER,
    duration_sec REAL,
    width INTEGER,
    height INTEGER,
    codec TEXT,
    status TEXT,
    processing_sec REAL,
    timings TEXT
);

CREATE TABLE IF NOT EXISTS appearance_gallery (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_key TEXT NOT NULL,
    source_match_key TEXT,
    team TEXT NOT NULL,
    alliance_color TEXT NOT NULL,
    embedding BLOB NOT NULL,
    legibility REAL,
    image BLOB,
    verified INTEGER DEFAULT 0,
    job_id TEXT,
    embed_model TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_gallery_event ON appearance_gallery(event_key, alliance_color);

CREATE TABLE IF NOT EXISTS match_robot_identity (
    job_id TEXT NOT NULL,
    identity_id INTEGER NOT NULL,
    slot INTEGER,
    slot_label TEXT,
    alliance TEXT,
    team_number TEXT,
    confirmed INTEGER DEFAULT 0,
    source TEXT,
    confidence REAL,
    created_at TEXT DEFAULT (datetime('now')),
    label_share REAL,
    runner_up_team TEXT,
    runner_up_share REAL,
    PRIMARY KEY (job_id, identity_id)
);
CREATE TABLE IF NOT EXISTS match_robot_binding (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    identity_id INTEGER NOT NULL,
    slot INTEGER NOT NULL,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);
-- identity_swap_resolve / identity_segment_resolve: per-slot team segments
-- (slots change robots mid-match). Frames are source-video frame numbers,
-- both ends inclusive; team_number NULL = unassigned/contested; score =
-- summed read weight inside the segment naming its team (swap) or the read
-- support behind its label (segments); source = read | stitched | none
-- (segment resolve only, NULL for swap). Written only when a knob is on.
CREATE TABLE IF NOT EXISTS match_robot_identity_segment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    slot INTEGER NOT NULL,
    slot_label TEXT,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    team_number TEXT,
    confirmed INTEGER DEFAULT 0,
    score REAL,
    source TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_identity_segment_job
    ON match_robot_identity_segment(job_id, slot, start_frame);
CREATE TABLE IF NOT EXISTS match_robot_crop (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    identity_id INTEGER NOT NULL,
    legibility REAL,
    embedding BLOB,
    image BLOB,
    embed_model TEXT
);
-- Manual slot corrections (POST /api/replays/{id}/correct), in order (id).
-- The HUD renderer relabels the video from them: old = the slot's label just
-- before the correction (NULL = unlabelled / AMBIGUOUS), new = the team typed.
CREATE TABLE IF NOT EXISTS replay_slot_corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    replay_id TEXT NOT NULL,
    slot INTEGER NOT NULL,
    old_team TEXT,
    new_team TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_replay_slot_corrections
    ON replay_slot_corrections(replay_id, slot, id);
"""


async def _table_columns(db, table: str) -> set[str]:
    cursor = await db.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in await cursor.fetchall()}


async def _migrate_identity_tables(db) -> None:
    identity_columns = await _table_columns(db, "match_robot_identity")
    if identity_columns and "identity_id" not in identity_columns:
        await db.execute("ALTER TABLE match_robot_identity RENAME TO match_robot_identity_legacy")
        await db.execute("""
            CREATE TABLE match_robot_identity (
                job_id TEXT NOT NULL,
                identity_id INTEGER NOT NULL,
                slot INTEGER,
                slot_label TEXT,
                alliance TEXT,
                team_number TEXT,
                confirmed INTEGER DEFAULT 0,
                source TEXT,
                confidence REAL,
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (job_id, identity_id)
            )
        """)
        await db.execute("""
            INSERT INTO match_robot_identity
                (job_id, identity_id, slot, slot_label, alliance, team_number,
                 confirmed, source, confidence, created_at)
            SELECT job_id, slot, slot, slot_label, alliance, team_number,
                   confirmed, source, confidence, created_at
            FROM match_robot_identity_legacy
        """)
        await db.execute("DROP TABLE match_robot_identity_legacy")

    crop_columns = await _table_columns(db, "match_robot_crop")
    if crop_columns and "identity_id" not in crop_columns:
        await db.execute("DROP INDEX IF EXISTS idx_match_crop")
        await db.execute("ALTER TABLE match_robot_crop RENAME TO match_robot_crop_legacy")
        await db.execute("""
            CREATE TABLE match_robot_crop (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                identity_id INTEGER NOT NULL,
                legibility REAL,
                embedding BLOB,
                image BLOB
            )
        """)
        await db.execute("""
            INSERT INTO match_robot_crop
                (id, job_id, identity_id, legibility, embedding, image)
            SELECT id, job_id, slot, legibility, embedding, image
            FROM match_robot_crop_legacy
        """)
        await db.execute("DROP TABLE match_robot_crop_legacy")

    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_match_crop "
        "ON match_robot_crop(job_id, identity_id)"
    )


_IDENTITY_LABEL_CONF_COLUMNS = (("label_share", "REAL"), ("runner_up_team", "TEXT"),
                                ("runner_up_share", "REAL"))


async def _migrate_identity_label_conf(db) -> None:
    """Add match_robot_identity.label_share / runner_up_team / runner_up_share
    to a pre-existing table. Idempotent: only missing columns are added."""
    have = await _table_columns(db, "match_robot_identity")
    for col, typ in _IDENTITY_LABEL_CONF_COLUMNS:
        if col not in have:
            await db.execute(f"ALTER TABLE match_robot_identity ADD COLUMN {col} {typ}")


async def init_db():
    """Initialize the database and create tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        # Event lineup anchor store (web/field/event_anchors.py).
        await db.executescript(_event_anchors.DDL)
        # Lightweight migrations: add columns to a pre-existing replays table.
        for col_def in ("processing_sec REAL", "timings TEXT"):
            try:
                await db.execute(f"ALTER TABLE replays ADD COLUMN {col_def}")
            except Exception:
                pass  # column already exists
        # Lightweight migrations: add columns to a pre-existing appearance_gallery table.
        for col_def in (
            "image BLOB",
            "verified INTEGER DEFAULT 0",
            "job_id TEXT",
            "source_match_key TEXT",
            "embed_model TEXT",
        ):
            try:
                await db.execute(f"ALTER TABLE appearance_gallery ADD COLUMN {col_def}")
            except Exception:
                pass  # column already exists
        await _migrate_identity_tables(db)
        # match_robot_crop may have just been rebuilt (above) by an older-schema
        # rename/recreate that predates embed_model; add it post-hoc too.
        for col_def in ("embed_model TEXT",):
            try:
                await db.execute(f"ALTER TABLE match_robot_crop ADD COLUMN {col_def}")
            except Exception:
                pass  # column already exists
        # Whole-match label confidence (finalize resolve): the label team's
        # share of the slot's team score + the runner-up team and its share.
        # NULL on rows written before the columns existed (or without the
        # global resolve). ``confidence`` stays the best crop's quality.
        await _migrate_identity_label_conf(db)
        # identity_segment_resolve: per-segment label source on a table that
        # predates it.
        if "source" not in await _table_columns(db, "match_robot_identity_segment"):
            await db.execute(
                "ALTER TABLE match_robot_identity_segment ADD COLUMN source TEXT")
        # Every embedding written before this column existed came from dinov2-small
        # (the only backend that ever ran) — pin that fact so a future default-model
        # change can't reinterpret old rows as "unknown model".
        await db.execute(
            "UPDATE appearance_gallery SET embed_model = ? WHERE embed_model IS NULL",
            (_LEGACY_EMBED_MODEL,))
        await db.execute(
            "UPDATE match_robot_crop SET embed_model = ? WHERE embed_model IS NULL",
            (_LEGACY_EMBED_MODEL,))
        await db.commit()


async def get_db():
    """Get a database connection."""
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    return db


# --- Events ---

async def upsert_event(event_key: str, event_name: str, year: int,
                       location: str = None, start_date: str = None,
                       end_date: str = None) -> int:
    """Insert or update an event. Returns the event id."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("""
            INSERT INTO events (event_key, event_name, year, location, start_date, end_date)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_key) DO UPDATE SET
                event_name = excluded.event_name,
                location = excluded.location,
                start_date = excluded.start_date,
                end_date = excluded.end_date
        """, (event_key, event_name, year, location, start_date, end_date))
        await db.commit()

        cursor = await db.execute("SELECT id FROM events WHERE event_key = ?", (event_key,))
        row = await cursor.fetchone()
        return row[0]


async def get_event_by_key(event_key: str) -> Optional[Dict]:
    """Get an event by its TBA key."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM events WHERE event_key = ?", (event_key,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def get_all_events() -> List[Dict]:
    """Get all events."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM events ORDER BY year DESC, event_name")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


# --- Matches ---

async def upsert_match(event_id: int, tba_match_key: str, comp_level: str,
                       match_number: int, set_number: int = 1,
                       tba_blue_score: int = None, tba_red_score: int = None,
                       tba_blue_hub_total: int = None, tba_red_hub_total: int = None,
                       tba_blue_hub_auto: int = None, tba_red_hub_auto: int = None,
                       tba_blue_hub_teleop: int = None, tba_red_hub_teleop: int = None,
                       tba_video_key: str = None) -> int:
    """Insert or update a match. Returns the match id."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("""
            INSERT INTO matches (event_id, tba_match_key, comp_level, match_number, set_number,
                               tba_blue_score, tba_red_score,
                               tba_blue_hub_total, tba_red_hub_total,
                               tba_blue_hub_auto, tba_red_hub_auto,
                               tba_blue_hub_teleop, tba_red_hub_teleop,
                               tba_video_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tba_match_key) DO UPDATE SET
                tba_blue_score = excluded.tba_blue_score,
                tba_red_score = excluded.tba_red_score,
                tba_blue_hub_total = excluded.tba_blue_hub_total,
                tba_red_hub_total = excluded.tba_red_hub_total,
                tba_blue_hub_auto = excluded.tba_blue_hub_auto,
                tba_red_hub_auto = excluded.tba_red_hub_auto,
                tba_blue_hub_teleop = excluded.tba_blue_hub_teleop,
                tba_red_hub_teleop = excluded.tba_red_hub_teleop,
                tba_video_key = COALESCE(excluded.tba_video_key, tba_video_key),
                updated_at = datetime('now')
        """, (event_id, tba_match_key, comp_level, match_number, set_number,
              tba_blue_score, tba_red_score,
              tba_blue_hub_total, tba_red_hub_total,
              tba_blue_hub_auto, tba_red_hub_auto,
              tba_blue_hub_teleop, tba_red_hub_teleop,
              tba_video_key))
        await db.commit()

        cursor = await db.execute("SELECT id FROM matches WHERE tba_match_key = ?", (tba_match_key,))
        row = await cursor.fetchone()
        return row[0]


async def update_match_predictions(match_id: int, blue_predicted: int, red_predicted: int,
                                   video_source: str = None):
    """Update match with prediction scores and video source."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE matches SET
                blue_predicted_score = ?,
                red_predicted_score = ?,
                video_source = COALESCE(?, video_source),
                updated_at = datetime('now')
            WHERE id = ?
        """, (blue_predicted, red_predicted, video_source, match_id))
        await db.commit()


async def update_match_notes(match_id: int, notes: str):
    """Update match notes."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE matches SET notes = ?, updated_at = datetime('now') WHERE id = ?
        """, (notes, match_id))
        await db.commit()


async def get_match(match_id: int) -> Optional[Dict]:
    """Get a match by id with its robots."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM matches WHERE id = ?", (match_id,))
        match = await cursor.fetchone()
        if not match:
            return None

        match_dict = dict(match)

        cursor = await db.execute(
            "SELECT * FROM match_robots WHERE match_id = ? ORDER BY alliance, position",
            (match_id,))
        robots = await cursor.fetchall()
        match_dict['robots'] = [dict(r) for r in robots]

        return match_dict


async def get_match_by_key(tba_match_key: str) -> Optional[Dict]:
    """Get a match by its TBA key (includes event_name from the events table)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT m.*, e.event_name FROM matches m "
            "JOIN events e ON e.id = m.event_id "
            "WHERE m.tba_match_key = ?", (tba_match_key,))
        match = await cursor.fetchone()
        if not match:
            return None

        match_dict = dict(match)

        cursor = await db.execute(
            "SELECT * FROM match_robots WHERE match_id = ? ORDER BY alliance, position",
            (match_dict['id'],))
        robots = await cursor.fetchall()
        match_dict['robots'] = [dict(r) for r in robots]

        return match_dict


async def get_event_matches(event_id: int) -> List[Dict]:
    """Get all matches for an event with robots."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT * FROM matches WHERE event_id = ?
            ORDER BY
                CASE comp_level
                    WHEN 'qm' THEN 1
                    WHEN 'sf' THEN 2
                    WHEN 'f' THEN 3
                    ELSE 4
                END,
                match_number, set_number
        """, (event_id,))
        matches = await cursor.fetchall()

        result = []
        for match in matches:
            match_dict = dict(match)
            cursor = await db.execute(
                "SELECT * FROM match_robots WHERE match_id = ? ORDER BY alliance, position",
                (match_dict['id'],))
            robots = await cursor.fetchall()
            match_dict['robots'] = [dict(r) for r in robots]
            result.append(match_dict)

        return result


# --- Match Robots ---

async def upsert_match_robot(match_id: int, alliance: str, position: int,
                             team_number: str) -> int:
    """Insert or update a match robot. Returns the match_robot id."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("""
            INSERT INTO match_robots (match_id, alliance, position, team_number)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(match_id, alliance, position) DO UPDATE SET
                team_number = excluded.team_number,
                updated_at = datetime('now')
        """, (match_id, alliance, position, team_number))
        await db.commit()

        cursor = await db.execute(
            "SELECT id FROM match_robots WHERE match_id = ? AND alliance = ? AND position = ?",
            (match_id, alliance, position))
        row = await cursor.fetchone()
        return row[0]


async def update_robot_scores(match_robot_id: int, predicted_fuel_count: int = None,
                              corrected_fuel_count: int = None, team_number: str = None):
    """Update a robot's predicted or corrected fuel count and/or team number."""
    async with aiosqlite.connect(DB_PATH) as db:
        updates = []
        params = []

        if predicted_fuel_count is not None:
            updates.append("predicted_fuel_count = ?")
            params.append(predicted_fuel_count)
        if corrected_fuel_count is not None:
            updates.append("corrected_fuel_count = ?")
            params.append(corrected_fuel_count)
        if team_number is not None:
            updates.append("team_number = ?")
            params.append(team_number)

        if not updates:
            return

        updates.append("updated_at = datetime('now')")
        params.append(match_robot_id)

        await db.execute(
            f"UPDATE match_robots SET {', '.join(updates)} WHERE id = ?",
            params)
        await db.commit()


# --- Scoring Events ---

async def save_scoring_events(match_id: int, events: List[Dict]):
    """Save scoring events for a match (replaces existing)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM scoring_events WHERE match_id = ?", (match_id,))
        for event in events:
            await db.execute("""
                INSERT INTO scoring_events (match_id, match_robot_id, frame_num, timestamp,
                                           alliance, confidence, distance_to_hub)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (match_id, event.get('match_robot_id'), event.get('frame_num'),
                  event.get('timestamp'), event['alliance'],
                  event.get('confidence'), event.get('distance_to_hub')))
        await db.commit()


# --- History / Stats ---

async def save_ocr_regions(event_key: str, regions: dict) -> None:
    """Save OCR regions for an event."""
    import json
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE events SET ocr_regions = ? WHERE event_key = ?",
            (json.dumps({k: list(v) for k, v in regions.items()}), event_key)
        )
        await db.commit()

async def get_ocr_regions(event_key: str) -> dict:
    """Get saved OCR regions for an event. Returns None if not set."""
    import json
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT ocr_regions FROM events WHERE event_key = ?", (event_key,)
        )
        row = await cursor.fetchone()
        if row and row[0]:
            data = json.loads(row[0])
            return {k: tuple(v) for k, v in data.items()}
    return None

async def save_hub_positions(event_key: str, positions: dict) -> None:
    """Save hub positions for an event. positions = {blue: [x, y], red: [x, y]}"""
    import json
    async with aiosqlite.connect(DB_PATH) as db:
        # Ensure column exists
        try:
            await db.execute("ALTER TABLE events ADD COLUMN hub_positions TEXT")
        except Exception:
            pass  # Column already exists
        await db.execute(
            "UPDATE events SET hub_positions = ? WHERE event_key = ?",
            (json.dumps(positions), event_key)
        )
        await db.commit()

async def get_hub_positions(event_key: str) -> dict:
    """Get saved hub positions for an event. Returns None if not set."""
    import json
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            cursor = await db.execute(
                "SELECT hub_positions FROM events WHERE event_key = ?", (event_key,)
            )
            row = await cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
        except Exception:
            pass  # Column doesn't exist yet
    return None

async def get_scored_matches(event_id: int = None) -> List[Dict]:
    """Get matches that have been scored (have video_source or predictions)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        query = """
            SELECT m.*, e.event_key, e.event_name
            FROM matches m
            JOIN events e ON m.event_id = e.id
            WHERE m.video_source IS NOT NULL
        """
        params = []
        if event_id:
            query += " AND m.event_id = ?"
            params.append(event_id)
        query += " ORDER BY m.created_at DESC"

        cursor = await db.execute(query, params)
        matches = await cursor.fetchall()

        result = []
        for match in matches:
            match_dict = dict(match)
            cursor = await db.execute(
                "SELECT * FROM match_robots WHERE match_id = ? ORDER BY alliance, position",
                (match_dict['id'],))
            robots = await cursor.fetchall()
            match_dict['robots'] = [dict(r) for r in robots]
            result.append(match_dict)

        return result


# --- Replays ---

_REPLAY_COLUMNS = [
    "id", "created_at", "label", "event_key", "match_key", "comp_level",
    "blue_score", "red_score", "blue_teams", "red_teams",
    "video_path", "meta_path", "source_video_path", "thumb_path",
    "fps", "frame_count", "duration_sec", "width", "height", "codec", "status",
    "processing_sec", "timings",
]


async def insert_replay(row: Dict[str, Any]) -> None:
    """Insert or replace a replay row (keyed by id)."""
    cols = ", ".join(_REPLAY_COLUMNS)
    placeholders = ", ".join("?" for _ in _REPLAY_COLUMNS)
    values = [row.get(c) for c in _REPLAY_COLUMNS]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"INSERT OR REPLACE INTO replays ({cols}) VALUES ({placeholders})", values)
        await db.commit()


async def list_replays() -> List[Dict]:
    """All replays, newest first, enriched with TBA match info when the
    replay's match_key matches an imported match: event_name, comp_level
    (as tba_comp_level), match_number, set_number, and the official
    alliance rosters (tba_blue_teams / tba_red_teams)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT r.*, e.event_name,
                   m.comp_level AS tba_comp_level,
                   m.match_number, m.set_number,
                   m.id AS _match_id
            FROM replays r
            LEFT JOIN matches m ON m.tba_match_key = r.match_key
            LEFT JOIN events e ON e.id = m.event_id
            ORDER BY r.created_at DESC, r.id DESC
        """)
        rows = [dict(r) for r in await cursor.fetchall()]

        match_ids = [r["_match_id"] for r in rows if r["_match_id"] is not None]
        rosters: Dict[int, Dict[str, List[str]]] = {}
        if match_ids:
            placeholders = ",".join("?" for _ in set(match_ids))
            cursor = await db.execute(
                f"SELECT match_id, alliance, team_number FROM match_robots "
                f"WHERE match_id IN ({placeholders}) ORDER BY alliance, position",
                list(set(match_ids)))
            for row in await cursor.fetchall():
                roster = rosters.setdefault(
                    row["match_id"], {"blue": [], "red": []})
                roster[row["alliance"]].append(row["team_number"])

        for r in rows:
            roster = rosters.get(r.pop("_match_id"), {})
            r["tba_blue_teams"] = roster.get("blue", [])
            r["tba_red_teams"] = roster.get("red", [])
        return rows


async def get_replay(replay_id: str) -> Optional[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM replays WHERE id = ?", (replay_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


def get_replay_sync(replay_id: str, db_path=None) -> Optional[Dict]:
    """get_replay for the render subprocess and scripts (plain sqlite3)."""
    import sqlite3
    con = sqlite3.connect(str(db_path or DB_PATH), timeout=30)
    try:
        con.row_factory = sqlite3.Row
        r = con.execute("SELECT * FROM replays WHERE id = ?", (replay_id,)).fetchone()
        return dict(r) if r else None
    finally:
        con.close()


async def delete_replay(replay_id: str) -> None:
    """Delete a replay row and its recorded slot corrections."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM replays WHERE id = ?", (replay_id,))
        await db.execute("DELETE FROM replay_slot_corrections WHERE replay_id = ?",
                         (replay_id,))
        await db.commit()


async def insert_slot_correction(replay_id: str, slot: int, old_team, new_team) -> None:
    """Append one manual slot correction (HUD post-render spec section 2)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO replay_slot_corrections (replay_id, slot, old_team, new_team) "
            "VALUES (?, ?, ?, ?)", (replay_id, int(slot), old_team, new_team))
        await db.commit()


async def get_slot_corrections(replay_id: str) -> List[Dict]:
    """A replay's slot corrections, oldest first."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM replay_slot_corrections WHERE replay_id = ? ORDER BY id",
            (replay_id,))
        return [dict(r) for r in await cur.fetchall()]


def get_slot_corrections_sync(replay_id: str, db_path=None) -> List[Dict]:
    """get_slot_corrections for the render subprocess and scripts (plain
    sqlite3). A DB without the table (older file) has no corrections."""
    import sqlite3
    con = sqlite3.connect(str(db_path or DB_PATH), timeout=30)
    try:
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT * FROM replay_slot_corrections WHERE replay_id = ? ORDER BY id",
                (replay_id,)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(r) for r in rows]
    finally:
        con.close()


async def update_replay_teams(replay_id: str, key: str, value: str) -> None:
    """Update one alliance's team list (JSON string) on a replay row."""
    # Real guard (not assert: assert is stripped under python -O) — the column name
    # is interpolated into the SQL, so it must never be attacker-influenced.
    if key not in ("blue_teams", "red_teams"):
        raise ValueError(f"invalid team column: {key!r}")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"UPDATE replays SET {key} = ? WHERE id = ?", (value, replay_id))
        await db.commit()


# --- Appearance gallery ---

async def insert_gallery_exemplar(row: Dict[str, Any]) -> None:
    """Insert one appearance-gallery exemplar."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO appearance_gallery "
            "(event_key, source_match_key, team, alliance_color, embedding, legibility, "
            "image, verified, job_id, embed_model) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (row["event_key"], row.get("source_match_key"),
             row["team"], row["alliance_color"],
             row["embedding"], row.get("legibility"), row.get("image"),
             int(row.get("verified", 0)), row.get("job_id"), row.get("embed_model")))
        await db.commit()


async def insert_gallery_exemplars(rows: List[Dict[str, Any]]) -> None:
    """Insert many appearance-gallery exemplars in one transaction.

    Same SQL as insert_gallery_exemplar; batching avoids one connection +
    fsync'd commit per row at job finalize."""
    if not rows:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT INTO appearance_gallery "
            "(event_key, source_match_key, team, alliance_color, embedding, legibility, "
            "image, verified, job_id, embed_model) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(row["event_key"], row.get("source_match_key"),
              row["team"], row["alliance_color"],
              row["embedding"], row.get("legibility"), row.get("image"),
              int(row.get("verified", 0)), row.get("job_id"), row.get("embed_model"))
             for row in rows])
        await db.commit()


async def get_gallery_exemplars(
    event_key: str, teams: Optional[List[str]] = None
) -> List[Dict]:
    """Gallery exemplars for an event, optionally limited to specific teams.

    Scoped to the active embed_model (NULL rows count as dinov2-small, the
    only backend that ever ran pre-migration) — a row from a different model
    is a different vector space and must not surface as if it matches.
    """
    team_filter = list(dict.fromkeys(
        str(team) for team in (teams or [])
        if team is not None and str(team)
    ))
    where = "event_key = ? AND COALESCE(embed_model, ?) = ?"
    params: List[Any] = [event_key, _LEGACY_EMBED_MODEL, ACTIVE_MODEL]
    if team_filter:
        where += f" AND team IN ({','.join('?' for _ in team_filter)})"
        params.extend(team_filter)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, source_match_key, team, alliance_color, embedding, legibility, "
            "image, verified, job_id, embed_model "
            f"FROM appearance_gallery WHERE {where} ORDER BY id", params)
        return [dict(r) for r in await cursor.fetchall()]


async def get_previous_gallery_exemplars(
    event_key: str,
    current_match_key: str,
    teams: Optional[List[str]] = None,
    verified_only: bool = False,
) -> List[Dict]:
    """Return exemplars sourced only from matches before ``current_match_key``.

    Rows without match provenance are intentionally excluded: they cannot be
    proven to come from an earlier match.

    ``verified_only`` restricts to human-verified rows: unverified deposits
    are candidates awaiting review in the gallery UI, not identity evidence
    (one mislabeled deposit otherwise votes for the wrong team in every later
    match, and DINOv2 embeddings can't separate same-alliance robots).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT m.comp_level, m.match_number, m.set_number
            FROM matches m
            JOIN events e ON e.id = m.event_id
            WHERE e.event_key = ? AND m.tba_match_key = ?
            """,
            (event_key, current_match_key),
        )
        current = await cursor.fetchone()
        if current is None:
            return []

        level_order = {"qm": 1, "sf": 2, "f": 3}

        def match_rank(row) -> tuple[int, int, int]:
            level = row["comp_level"]
            match_number = int(row["match_number"])
            set_number = int(row["set_number"] or 1)
            if level == "qm":
                return (level_order[level], match_number, set_number)
            return (level_order.get(level, 4), set_number, match_number)

        current_rank = match_rank(current)
        cursor = await db.execute(
            """
            SELECT m.tba_match_key, m.comp_level, m.match_number, m.set_number
            FROM matches m
            JOIN events e ON e.id = m.event_id
            WHERE e.event_key = ?
            """,
            (event_key,),
        )
        previous_keys = [
            row["tba_match_key"]
            for row in await cursor.fetchall()
            if match_rank(row) < current_rank
        ]
        if not previous_keys:
            return []

        team_filter = list(dict.fromkeys(
            str(team) for team in (teams or [])
            if team is not None and str(team)
        ))
        where = (
            f"event_key = ? AND source_match_key IN "
            f"({','.join('?' for _ in previous_keys)})"
        )
        params: List[Any] = [event_key, *previous_keys]
        if team_filter:
            where += f" AND team IN ({','.join('?' for _ in team_filter)})"
            params.extend(team_filter)
        if verified_only:
            where += " AND verified = 1"
        # A stored embedding is only comparable to a freshly computed one if
        # both came from the same backbone. Rows from a different (or absent
        # -> pre-migration dinov2-small) embed_model are excluded rather than
        # cosine-compared against the current model's vectors: that would be
        # a silent garbage comparison, not a real re-ID signal. The team just
        # has no exemplars yet for the new model, which is fine — cold start.
        where += " AND COALESCE(embed_model, ?) = ?"
        params.extend([_LEGACY_EMBED_MODEL, ACTIVE_MODEL])

        cursor = await db.execute(
            "SELECT id, source_match_key, team, alliance_color, embedding, legibility, "
            "image, verified, job_id, embed_model "
            f"FROM appearance_gallery WHERE {where} ORDER BY id",
            params,
        )
        return [dict(r) for r in await cursor.fetchall()]


async def update_gallery_exemplar(gid, team=None, alliance_color=None, verified=None):
    """Relabel team/alliance or toggle the verified flag on a gallery exemplar."""
    sets, vals = [], []
    if team is not None:
        sets.append("team = ?"); vals.append(team)
    if alliance_color is not None:
        sets.append("alliance_color = ?"); vals.append(alliance_color)
    if verified is not None:
        sets.append("verified = ?"); vals.append(int(verified))
    if not sets:
        return
    vals.append(gid)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"UPDATE appearance_gallery SET {', '.join(sets)} WHERE id = ?", vals)
        await db.commit()


async def delete_gallery_exemplar(gid):
    """Delete a gallery exemplar by id."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM appearance_gallery WHERE id = ?", (gid,))
        await db.commit()


async def clear_gallery_exemplars(event_key: str) -> int:
    """Delete every gallery exemplar for an event. Returns the count removed.

    The gallery is a regenerable per-event cache (rebuilt from match processing),
    so a full clear is a routine troubleshooting reset, not data loss.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "DELETE FROM appearance_gallery WHERE event_key = ?", (event_key,))
        await db.commit()
        return cur.rowcount


# --- Match robot identities & crops ---

async def upsert_match_robot_identity(row: Dict[str, Any]) -> None:
    """Insert or replace the identity for one robot in a job."""
    identity_id = row.get("identity_id")
    if identity_id is None:
        identity_id = row.get("slot")
    if identity_id is None:
        raise KeyError("Either 'identity_id' or 'slot' must be provided in row")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO match_robot_identity "
            "(job_id, identity_id, slot, slot_label, alliance, team_number, confirmed, source, "
            "confidence, label_share, runner_up_team, runner_up_share) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (row["job_id"], identity_id, row.get("slot"), row.get("slot_label"), row.get("alliance"),
             row.get("team_number"), int(row.get("confirmed", 0)), row.get("source"),
             row.get("confidence"), row.get("label_share"), row.get("runner_up_team"),
             row.get("runner_up_share")))
        await db.commit()


async def insert_match_robot_crop(row: Dict[str, Any]) -> None:
    """Insert one captured bumper crop for a robot identity."""
    identity_id = row.get("identity_id")
    if identity_id is None:
        identity_id = row.get("slot")
    if identity_id is None:
        raise KeyError("Either 'identity_id' or 'slot' must be provided in row")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO match_robot_crop (job_id, identity_id, legibility, embedding, image, embed_model) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (row["job_id"], identity_id, row.get("legibility"), row.get("embedding"),
             row.get("image"), row.get("embed_model")))
        await db.commit()


def _identity_key(row: Dict[str, Any]):
    identity_id = row.get("identity_id")
    if identity_id is None:
        identity_id = row.get("slot")
    if identity_id is None:
        raise KeyError("Either 'identity_id' or 'slot' must be provided in row")
    return identity_id


async def upsert_match_robot_identities(rows: List[Dict[str, Any]]) -> None:
    """Batch variant of upsert_match_robot_identity: one transaction for all rows."""
    if not rows:
        return
    params = [
        (row["job_id"], _identity_key(row), row.get("slot"), row.get("slot_label"),
         row.get("alliance"), row.get("team_number"), int(row.get("confirmed", 0)),
         row.get("source"), row.get("confidence"), row.get("label_share"),
         row.get("runner_up_team"), row.get("runner_up_share"))
        for row in rows
    ]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT OR REPLACE INTO match_robot_identity "
            "(job_id, identity_id, slot, slot_label, alliance, team_number, confirmed, source, "
            "confidence, label_share, runner_up_team, runner_up_share) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", params)
        await db.commit()


async def insert_match_robot_crops(rows: List[Dict[str, Any]]) -> None:
    """Batch variant of insert_match_robot_crop: one transaction for all rows."""
    if not rows:
        return
    params = [
        (row["job_id"], _identity_key(row), row.get("legibility"),
         row.get("embedding"), row.get("image"), row.get("embed_model"))
        for row in rows
    ]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT INTO match_robot_crop (job_id, identity_id, legibility, embedding, image, embed_model) "
            "VALUES (?, ?, ?, ?, ?, ?)", params)
        await db.commit()


async def get_match_identities(job_id) -> List[Dict]:
    """All robot identities for a job, ordered by identity_id."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM match_robot_identity WHERE job_id = ? ORDER BY identity_id", (job_id,))
        return [dict(r) for r in await cur.fetchall()]


async def get_match_crops(job_id, identity_id) -> List[Dict]:
    """All captured crops for a robot identity, most legible first."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM match_robot_crop WHERE job_id = ? AND identity_id = ? ORDER BY legibility DESC",
            (job_id, identity_id))
        return [dict(r) for r in await cur.fetchall()]


async def replace_match_identity_segments(job_id: str, rows: List[Dict[str, Any]]) -> None:
    """Replace a job's swap-aware identity segments (one transaction).

    ``rows``: ``{slot, slot_label, start_frame, end_frame, team_number,
    confirmed, score[, source]}``. Any previous segments for the job are
    deleted first, so a re-run never accumulates duplicates."""
    params = [
        (job_id, int(r["slot"]), r.get("slot_label"), int(r["start_frame"]),
         int(r["end_frame"]), r.get("team_number"), int(bool(r.get("confirmed"))),
         r.get("score"), r.get("source"))
        for r in rows
    ]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM match_robot_identity_segment WHERE job_id = ?", (job_id,))
        if params:
            await db.executemany(
                "INSERT INTO match_robot_identity_segment "
                "(job_id, slot, slot_label, start_frame, end_frame, team_number, confirmed, "
                "score, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", params)
        await db.commit()


async def get_match_identity_segments(job_id) -> List[Dict]:
    """A job's identity segments ordered by slot then start frame (empty
    unless the job ran with identity_swap_resolve / identity_segment_resolve)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT slot, slot_label, start_frame, end_frame, team_number, confirmed, score, "
            "source FROM match_robot_identity_segment WHERE job_id = ? "
            "ORDER BY slot, start_frame",
            (job_id,))
        return [dict(r) for r in await cur.fetchall()]


async def insert_match_robot_binding(row: Dict[str, Any]) -> None:
    """Insert a robot slot binding event."""
    identity_id = row.get("identity_id")
    if identity_id is None:
        identity_id = row.get("slot")
    if identity_id is None:
        raise KeyError("Either 'identity_id' or 'slot' must be provided in row")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO match_robot_binding (job_id, identity_id, slot, start_frame, end_frame) "
            "VALUES (?, ?, ?, ?, ?)",
            (row["job_id"], identity_id, row["slot"], row["start_frame"], row.get("end_frame")))
        await db.commit()


async def get_match_bindings(job_id) -> List[Dict]:
    """All binding events for a job."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM match_robot_binding WHERE job_id = ? ORDER BY id", (job_id,))
        return [dict(r) for r in await cur.fetchall()]


# --- Event lineup anchor store (web/field/event_anchors.py; spec 2026-11-10 section 1) ---

async def fetch_event_anchors(namespace: str, event_key: str, teams,
                              exclude_match_key: Optional[str],
                              max_match_number: Optional[int] = None, *,
                              root=None) -> Dict[str, list]:
    """Async twin of event_anchors.fetch: team -> [Anchor] for ``teams`` from
    every other match of the event in ``namespace`` (only match_number <=
    ``max_match_number`` when given). Tile files are read off the event loop."""
    ea = _event_anchors
    teams = [str(t) for t in teams]
    if not teams:
        return {}
    sql, args = ea.select_query(namespace, event_key, teams, exclude_match_key,
                                max_match_number)
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(sql, args)
        rows = await cur.fetchall()
    return await asyncio.to_thread(ea.rows_to_anchors, rows, root or ea.ANCHOR_ROOT)


async def replace_event_anchors(namespace: str, event_key: str, match_key: str, anchors, *,
                                replay_id: Optional[str] = None,
                                match_number: Optional[int] = None, root=None) -> int:
    """Async twin of event_anchors.replace (same transaction and file order:
    delete the match's rows, insert, write tiles, commit, then delete the old
    tiles; a failure before the commit changes nothing). Returns the count."""
    ea = _event_anchors
    anchors = list(anchors)
    ea.validate(namespace, event_key, anchors)
    root = Path(root or ea.ANCHOR_ROOT)
    mn = ea.schedule_rank(match_key) if match_number is None else int(match_number)
    new_files: list = []
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute("BEGIN IMMEDIATE")
            cur = await db.execute(ea.OLD_PATHS_SQL, (namespace, event_key, match_key))
            old = [r[0] for r in await cur.fetchall()]
            await db.execute(ea.DELETE_SQL, (namespace, event_key, match_key))
            for a in anchors:
                cur = await db.execute(ea.INSERT_SQL, ea.insert_args(
                    namespace, event_key, match_key, mn, replay_id, a))
                rel = ea.tile_relpath(namespace, event_key, cur.lastrowid)
                await db.execute(ea.SET_PATH_SQL, (rel, cur.lastrowid))
                await asyncio.to_thread(ea.write_tile, root, rel, a.tile)
                new_files.append(root / rel)
            await db.commit()
        except BaseException:
            await db.rollback()
            await asyncio.to_thread(ea.remove_files, new_files)
            raise
    await asyncio.to_thread(ea.remove_files, [root / r for r in old])
    return len(anchors)
