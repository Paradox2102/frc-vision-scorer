"""Load an event's finished replays into ``MatchData`` for the scouting metrics.

One replay per match: several exist when a match was A/B'd. The newest
``complete`` replay whose summary carries ``window_labels`` (the current
identity method) wins; with none, the newest complete one is used and flagged
``legacy_labels``.

Identity: a robot sample belongs to the frame's ``team_number``. Unlabelled
samples are counted in ``tracked_s`` but not attributed. A label naming a team
that is not in the TBA lineup is dropped (``off_roster``). Two slots carrying
the same team in one frame keep one sample (the confirmed one, else the first).

Synchronous sqlite3: called from a thread by the API.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Optional

from web.field.season import season_for_event
from web.scouting.clock import MatchClock
from web.scouting.geometry import to_alliance
from web.scouting.model import MatchData, Sample, TeamMatch

REPO = Path(__file__).resolve().parents[2]
REPLAY_DIR = REPO / "web" / "replays"
_KEY_RE = re.compile(r"^(?P<event>[^_]+)_(?P<level>qm|ef|qf|sf|f)(?:(?P<set>\d+)m)?(?P<num>\d+)$")


def event_season_year(event_key: str) -> int:
    """The season year an event is played on (honours the per-event field-season
    override, POST /api/event/field-season), not just the key's prefix."""
    return int(season_for_event(event_key).year)


def default_db_path() -> Path:
    from web import database
    return Path(database.DB_PATH)


def parse_match_key(match_key: str) -> tuple:
    """'2026cur_qm8' -> ('qm', 1, 8); '2026cur_sf3m1' -> ('sf', 3, 1)."""
    m = _KEY_RE.match(match_key or "")
    if not m:
        return "unknown", 1, 0
    level = m.group("level")
    if m.group("set"):
        return level, int(m.group("set")), int(m.group("num"))
    return level, 1, int(m.group("num"))


def _meta_path(row: dict) -> Optional[Path]:
    for cand in (row.get("meta_path"), REPLAY_DIR / str(row["id"]) / "meta.json"):
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def _read_meta(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def candidate_replays(conn, event_key: str) -> dict:
    """{match_key: [replay rows newest first]} for complete replays."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, match_key, created_at, meta_path FROM replays "
        "WHERE event_key = ? AND status = 'complete' AND match_key IS NOT NULL "
        "AND match_key != '' ORDER BY created_at DESC", (event_key,)).fetchall()
    out: dict = defaultdict(list)
    for r in rows:
        out[r["match_key"]].append(dict(r))
    return out


def pick_replay(rows: list) -> tuple:
    """(row, meta, legacy) for the best replay of one match, or (None, None, None)."""
    fallback = None
    for row in rows:
        path = _meta_path(row)
        meta = _read_meta(path) if path else None
        if not meta:
            continue
        if (meta.get("summary") or {}).get("window_labels"):
            return row, meta, False
        if fallback is None:
            fallback = (row, meta, True)
    return fallback or (None, None, None)


def match_lineup(conn, match_key: str) -> dict:
    """TBA lineup + hub auto totals from the matches tables (empty if unknown)."""
    conn.row_factory = sqlite3.Row
    m = conn.execute("SELECT id, comp_level, match_number, set_number, tba_red_hub_auto, "
                     "tba_blue_hub_auto FROM matches WHERE tba_match_key = ?",
                     (match_key,)).fetchone()
    if not m:
        return {}
    alliances = {"red": [None, None, None], "blue": [None, None, None]}
    for r in conn.execute("SELECT alliance, position, team_number FROM match_robots "
                          "WHERE match_id = ?", (m["id"],)):
        try:
            alliances[r["alliance"]][int(r["position"]) - 1] = int(r["team_number"])
        except (KeyError, ValueError, TypeError, IndexError):
            pass
    return {"comp_level": m["comp_level"], "match_number": m["match_number"],
            "set_number": m["set_number"] or 1, "alliances": alliances,
            "hub_auto": {"red": m["tba_red_hub_auto"], "blue": m["tba_blue_hub_auto"]}}


def _team_int(v) -> Optional[int]:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def build_match(event_key: str, match_key: str, replay: dict, meta: dict, legacy: bool,
                lineup: dict, year: int) -> tuple:
    """(MatchData, None) or (None, skip_reason)."""
    summary = meta.get("summary") or {}
    clock = MatchClock(summary.get("periods") or [])
    if not clock.ok:
        return None, "no match clock (summary.periods)"
    frames = meta.get("frames") or []
    if len(frames) < 2:
        return None, "no frames"
    fps = clock.frames_per_s()
    steps = [b["frame_num"] - a["frame_num"] for a, b in zip(frames, frames[1:])
             if b["frame_num"] > a["frame_num"]]
    dt = (median(steps) / fps) if steps and fps > 0 else 0.2

    level, set_no, num = parse_match_key(match_key)
    alliances = lineup.get("alliances") or {"red": [None] * 3, "blue": [None] * 3}
    roster = {t: a for a in ("red", "blue") for t in alliances.get(a, []) if t is not None}
    md = MatchData(
        event_key=event_key, match_key=match_key,
        comp_level=lineup.get("comp_level") or level,
        match_number=int(lineup.get("match_number") or num),
        set_number=int(lineup.get("set_number") or set_no), year=year,
        replay_id=str(replay["id"]), replay_created_at=replay.get("created_at"),
        legacy_labels=bool(legacy), alliances=alliances,
        hub_auto=lineup.get("hub_auto") or {"red": None, "blue": None},
        frame_period_s=float(dt),
        labelled_s={"auto": 0.0, "teleop": 0.0}, tracked_s={"auto": 0.0, "teleop": 0.0})

    def team_match(team: int, alliance: str) -> TeamMatch:
        tm = md.teams.get(team)
        if tm is None:
            _, station = md.station_of(team)
            tm = md.teams[team] = TeamMatch(team=team, alliance=alliance, station=station)
        return tm

    def sample(fn, at, xy, alliance, confirmed=False, slot=None) -> Sample:
        period, t, remaining = at
        ax, ay = to_alliance(xy[0], xy[1], alliance, year)
        return Sample(frame_num=int(fn), t_match=t, period=period, remaining=remaining,
                      x=float(xy[0]), y=float(xy[1]), ax=ax, ay=ay, dt=dt,
                      confirmed=bool(confirmed), slot=slot)

    slot_votes: dict = defaultdict(Counter)   # auto-period labels per slot
    slot_auto_xy: dict = defaultdict(list)    # slot -> [(frame_num, xy)] in auto
    off_roster = dups = 0
    for fr in frames:
        at = clock.at(int(fr.get("frame_num", -1)))
        if at is None:
            continue
        period = at[0]
        seen: dict = {}
        for rb in fr.get("robots") or []:
            xy = rb.get("field_pos")
            if not xy or len(xy) < 2:
                continue
            md.tracked_s[period] += dt
            if period == "auto" and rb.get("slot"):
                slot_auto_xy[rb["slot"]].append((int(fr["frame_num"]), xy))
            team = _team_int(rb.get("team_number"))
            if team is None:
                continue
            if roster and team not in roster:
                off_roster += 1
                continue
            alliance = roster.get(team) or rb.get("alliance") or "blue"
            if period == "auto" and rb.get("slot"):
                slot_votes[rb["slot"]][team] += 1
            prev = seen.get(team)
            if prev is not None:
                dups += 1
                if prev.get("confirmed") or not rb.get("confirmed"):
                    continue
            seen[team] = rb
        for team, rb in seen.items():
            alliance = roster.get(team) or rb.get("alliance") or "blue"
            team_match(team, alliance).samples.append(
                sample(fr["frame_num"], at, rb["field_pos"], alliance,
                       rb.get("confirmed"), rb.get("slot")))
            md.labelled_s[period] += dt

    _attach_auto_paths(md, summary.get("autonomous_paths") or {}, slot_votes, slot_auto_xy,
                       clock, roster, team_match, sample)
    md.off_roster_samples, md.duplicate_label_samples = off_roster, dups
    return md, None


def _attach_auto_paths(md, paths, slot_votes, slot_auto_xy, clock, roster, team_match,
                       sample) -> None:
    # Each slot's path goes to its majority auto label; a team claimed by two
    # slots keeps the slot with more votes. The pipeline's path starts at the
    # clock's auto start_frame, which is ~CLOCK_LEAD_S late (clock.py); the
    # slot's frames before that (the processing lead) are prepended.
    claims: dict = {}
    for slot, votes in slot_votes.items():
        if slot not in paths or not votes:
            continue
        total = sum(votes.values())
        (team, n), *rest = votes.most_common()
        runner = rest[0][1] / total if rest else 0.0
        if team not in claims or n > claims[team][1]:
            claims[team] = (slot, n, n / total, runner)
    for team, (slot, _n, share, runner) in claims.items():
        alliance = roster.get(team) or ("red" if str(slot).startswith("R") else "blue")
        tm = team_match(team, alliance)
        first = min((int(p.get("frame_num", 1 << 30)) for p in paths[slot]), default=1 << 30)
        pts = []
        for fn, xy in slot_auto_xy.get(slot, []):
            at = clock.at(fn)
            if fn < first and at and at[0] == "auto":
                pts.append(sample(fn, at, xy, alliance, slot=slot))
        for p in paths[slot]:
            xy = p.get("field_pos")
            at = clock.at(int(p.get("frame_num", -1)))
            if xy and at and at[0] == "auto":
                pts.append(sample(p["frame_num"], at, xy, alliance, slot=slot))
        if pts:
            tm.auto_path, tm.auto_slot = pts, slot
            tm.auto_label_share, tm.auto_runner_up_share = round(share, 3), round(runner, 3)
            tm.auto_source = "autonomous_paths"
    for tm in md.teams.values():
        if not tm.auto_path:
            auto = [s for s in tm.samples if s.period == "auto"]
            if auto:
                tm.auto_path, tm.auto_source = auto, "frames"


def load_event(event_key: str, db_path=None) -> tuple:
    """(matches: list[MatchData], skipped: list[{match_key, reason}])."""
    year = event_season_year(event_key)
    conn = sqlite3.connect(str(db_path or default_db_path()))
    try:
        matches, skipped = [], []
        for match_key, rows in sorted(candidate_replays(conn, event_key).items()):
            replay, meta, legacy = pick_replay(rows)
            if replay is None:
                skipped.append({"match_key": match_key, "reason": "no readable meta.json"})
                continue
            md, why = build_match(event_key, match_key, replay, meta, legacy,
                                  match_lineup(conn, match_key), year)
            if md is None:
                skipped.append({"match_key": match_key, "replay_id": replay["id"], "reason": why})
            else:
                matches.append(md)
        return matches, skipped
    finally:
        conn.close()


def source_fingerprint(event_key: str, db_path=None) -> tuple:
    """Cheap cache key: the set of complete replays (+ their meta mtimes)."""
    conn = sqlite3.connect(str(db_path or default_db_path()))
    try:
        out = []
        for match_key, rows in sorted(candidate_replays(conn, event_key).items()):
            for r in rows:
                p = _meta_path(r)
                out.append((r["id"], p.stat().st_mtime if p else 0.0))
        return tuple(out)
    finally:
        conn.close()
