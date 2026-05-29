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
"""


async def init_db():
    """Initialize the database and create tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
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
