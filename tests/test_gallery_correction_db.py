import asyncio
import numpy as np
import pytest
import web.database as database


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    asyncio.run(database.init_db())
    return database


def _b(*v): return np.array(v, dtype=np.float32).tobytes()


def test_match_robot_identity_and_crops_roundtrip(db):
    asyncio.run(db.upsert_match_robot_identity({
        "job_id": "job1", "slot": 3, "slot_label": "R1", "alliance": "red",
        "team_number": "2486", "confirmed": 1, "source": "ocr", "confidence": 0.95}))
    asyncio.run(db.insert_match_robot_crop({
        "job_id": "job1", "slot": 3, "legibility": 0.9,
        "embedding": _b(1, 0, 0), "image": b"\xff\xd8jpeg"}))
    ids = asyncio.run(db.get_match_identities("job1"))
    assert ids[0]["team_number"] == "2486" and ids[0]["slot"] == 3
    crops = asyncio.run(db.get_match_crops("job1", 3))
    assert len(crops) == 1 and crops[0]["image"] == b"\xff\xd8jpeg"


def test_identity_upsert_replaces(db):
    base = {"job_id": "j", "slot": 0, "slot_label": "B1", "alliance": "blue",
            "team_number": "111", "confirmed": 0, "source": "ocr", "confidence": 0.5}
    asyncio.run(db.upsert_match_robot_identity(base))
    asyncio.run(db.upsert_match_robot_identity({**base, "team_number": "222"}))
    ids = asyncio.run(db.get_match_identities("j"))
    assert len(ids) == 1 and ids[0]["team_number"] == "222"


def test_gallery_has_image_and_verified_columns_and_crud(db):
    asyncio.run(db.insert_gallery_exemplar({
        "event_key": "2026x", "team": "1787", "alliance_color": "blue",
        "embedding": _b(1, 0, 0), "legibility": 0.9, "image": b"img", "verified": 1}))
    rows = asyncio.run(db.get_gallery_exemplars("2026x"))
    assert rows[0]["verified"] == 1 and rows[0]["image"] == b"img" and "id" in rows[0]
    gid = rows[0]["id"]
    # relabel + verify toggle
    asyncio.run(db.update_gallery_exemplar(gid, team="254", verified=0))
    rows = asyncio.run(db.get_gallery_exemplars("2026x"))
    assert rows[0]["team"] == "254" and rows[0]["verified"] == 0
    # delete
    asyncio.run(db.delete_gallery_exemplar(gid))
    assert asyncio.run(db.get_gallery_exemplars("2026x")) == []


def test_gallery_migration_adds_columns_to_old_table(tmp_path, monkeypatch):
    import sqlite3
    dbfile = tmp_path / "old.db"
    con = sqlite3.connect(dbfile)
    con.execute("CREATE TABLE appearance_gallery (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "event_key TEXT, team TEXT, alliance_color TEXT, embedding BLOB, legibility REAL, "
                "created_at TEXT)")
    con.commit(); con.close()
    monkeypatch.setattr(database, "DB_PATH", dbfile)
    asyncio.run(database.init_db())  # must ALTER-add image + verified without error
    asyncio.run(database.insert_gallery_exemplar({
        "event_key": "e", "team": "9", "alliance_color": "red",
        "embedding": _b(0, 1, 0), "legibility": 0.5, "image": b"i", "verified": 1}))
    assert asyncio.run(database.get_gallery_exemplars("e"))[0]["verified"] == 1


def test_gallery_migration_backfills_embed_model_on_preexisting_rows(tmp_path, monkeypatch):
    # Rows written by an old schema (no embed_model column at all) predate
    # this feature and were all embedded with dinov2-small, the only backend
    # that ever ran; init_db() must both ALTER-add the column and backfill
    # those rows rather than leaving them NULL.
    import sqlite3
    dbfile = tmp_path / "old_embed_model.db"
    con = sqlite3.connect(dbfile)
    con.execute("CREATE TABLE appearance_gallery (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "event_key TEXT, team TEXT, alliance_color TEXT, embedding BLOB, legibility REAL, "
                "created_at TEXT)")
    con.execute(
        "INSERT INTO appearance_gallery (event_key, team, alliance_color, embedding, legibility) "
        "VALUES ('e', '9', 'red', ?, 0.5)", (_b(0, 1, 0),))
    con.commit(); con.close()

    monkeypatch.setattr(database, "DB_PATH", dbfile)
    asyncio.run(database.init_db())

    con = sqlite3.connect(dbfile)
    row = con.execute("SELECT embed_model FROM appearance_gallery WHERE team = '9'").fetchone()
    con.close()
    assert row[0] == "facebook/dinov2-small"


def test_match_robot_crop_migration_adds_embed_model_column(tmp_path, monkeypatch):
    # match_robot_crop predates embed_model too; init_db() must ALTER-add it
    # without error even on a table that already has the identity_id schema
    # (so the rename/rebuild migration in _migrate_identity_tables is a no-op).
    import sqlite3
    dbfile = tmp_path / "old_crop.db"
    con = sqlite3.connect(dbfile)
    con.execute("CREATE TABLE match_robot_crop (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "job_id TEXT NOT NULL, identity_id INTEGER NOT NULL, legibility REAL, "
                "embedding BLOB, image BLOB)")
    con.execute(
        "INSERT INTO match_robot_crop (job_id, identity_id, legibility, embedding) "
        "VALUES ('job1', 0, 0.9, ?)", (_b(1, 0, 0),))
    con.commit(); con.close()

    monkeypatch.setattr(database, "DB_PATH", dbfile)
    asyncio.run(database.init_db())

    crops = asyncio.run(database.get_match_crops("job1", 0))
    assert len(crops) == 1
    assert crops[0]["embed_model"] == "facebook/dinov2-small"  # backfilled
