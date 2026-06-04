"""
SQLite database for persisting match results and per-robot scoring data.
"""

import aiosqlite
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

DB_PATH = Path(__file__).parent.parent / "frc_scorer.db"

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
    team TEXT NOT NULL,
    alliance_color TEXT NOT NULL,
    embedding BLOB NOT NULL,
    legibility REAL,
    image BLOB,
    verified INTEGER DEFAULT 0,
    job_id TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_gallery_event ON appearance_gallery(event_key, alliance_color);

CREATE TABLE IF NOT EXISTS match_robot_identity (
    job_id TEXT NOT NULL,
    slot INTEGER NOT NULL,
    slot_label TEXT,
    alliance TEXT,
    team_number TEXT,
    confirmed INTEGER DEFAULT 0,
    source TEXT,
    confidence REAL,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (job_id, slot)
);
CREATE TABLE IF NOT EXISTS match_robot_crop (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    slot INTEGER NOT NULL,
    legibility REAL,
    embedding BLOB,
    image BLOB
);
CREATE INDEX IF NOT EXISTS idx_match_crop ON match_robot_crop(job_id, slot);
"""


async def init_db():
    """Initialize the database and create tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        # Lightweight migrations: add columns to a pre-existing replays table.
        for col_def in ("processing_sec REAL", "timings TEXT"):
            try:
                await db.execute(f"ALTER TABLE replays ADD COLUMN {col_def}")
            except Exception:
                pass  # column already exists
        # Lightweight migrations: add columns to a pre-existing appearance_gallery table.
        for col_def in ("image BLOB", "verified INTEGER DEFAULT 0", "job_id TEXT"):
            try:
                await db.execute(f"ALTER TABLE appearance_gallery ADD COLUMN {col_def}")
            except Exception:
                pass  # column already exists
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
    """Get a match by its TBA key."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM matches WHERE tba_match_key = ?", (tba_match_key,))
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
    """All replays, newest first."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM replays ORDER BY created_at DESC, id DESC")
        return [dict(r) for r in await cursor.fetchall()]


async def get_replay(replay_id: str) -> Optional[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM replays WHERE id = ?", (replay_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def delete_replay(replay_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM replays WHERE id = ?", (replay_id,))
        await db.commit()


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
            "(event_key, team, alliance_color, embedding, legibility, image, verified, job_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (row["event_key"], row["team"], row["alliance_color"],
             row["embedding"], row.get("legibility"), row.get("image"),
             int(row.get("verified", 0)), row.get("job_id")))
        await db.commit()


async def get_gallery_exemplars(event_key: str) -> List[Dict]:
    """All gallery exemplars for an event."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, team, alliance_color, embedding, legibility, image, verified, job_id "
            "FROM appearance_gallery WHERE event_key = ? ORDER BY id", (event_key,))
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


# --- Match robot identities & crops ---

async def upsert_match_robot_identity(row: Dict[str, Any]) -> None:
    """Insert or replace the identity for one robot slot in a job."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO match_robot_identity "
            "(job_id, slot, slot_label, alliance, team_number, confirmed, source, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (row["job_id"], row["slot"], row.get("slot_label"), row.get("alliance"),
             row.get("team_number"), int(row.get("confirmed", 0)), row.get("source"),
             row.get("confidence")))
        await db.commit()


async def insert_match_robot_crop(row: Dict[str, Any]) -> None:
    """Insert one captured bumper crop for a robot slot."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO match_robot_crop (job_id, slot, legibility, embedding, image) "
            "VALUES (?, ?, ?, ?, ?)",
            (row["job_id"], row["slot"], row.get("legibility"), row.get("embedding"),
             row.get("image")))
        await db.commit()


async def get_match_identities(job_id) -> List[Dict]:
    """All robot-slot identities for a job, ordered by slot."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM match_robot_identity WHERE job_id = ? ORDER BY slot", (job_id,))
        return [dict(r) for r in await cur.fetchall()]


async def get_match_crops(job_id, slot) -> List[Dict]:
    """All captured crops for a robot slot, most legible first."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM match_robot_crop WHERE job_id = ? AND slot = ? ORDER BY legibility DESC",
            (job_id, slot))
        return [dict(r) for r in await cur.fetchall()]
