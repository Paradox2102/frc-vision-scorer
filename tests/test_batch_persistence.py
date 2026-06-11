"""Batch variants of the finalize-time DB writes.

Finalize used to open one aiosqlite connection (and fsync one commit) per
gallery exemplar / identity / crop — dozens of connections at end of job.
The batch functions write all rows in one connection and one transaction,
matching the single-row functions' SQL exactly.
"""
import asyncio

import numpy as np
import pytest

import web.database as database


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    asyncio.run(database.init_db())
    return database


def _emb_bytes(*vals):
    return np.array(vals, dtype=np.float32).tobytes()


def test_insert_gallery_exemplars_batch(db):
    rows = [
        {"event_key": "2026hop", "team": "4174", "alliance_color": "blue",
         "embedding": _emb_bytes(1, 0), "legibility": 0.9, "job_id": "j1"},
        {"event_key": "2026hop", "team": "401", "alliance_color": "red",
         "embedding": _emb_bytes(0, 1), "legibility": 0.8, "job_id": "j1"},
    ]
    asyncio.run(db.insert_gallery_exemplars(rows))
    got = asyncio.run(db.get_gallery_exemplars("2026hop"))
    assert sorted(r["team"] for r in got) == ["401", "4174"]
    assert all(r["job_id"] == "j1" for r in got)


def test_insert_gallery_exemplars_empty_is_noop(db):
    asyncio.run(db.insert_gallery_exemplars([]))
    assert asyncio.run(db.get_gallery_exemplars("2026hop")) == []


def test_upsert_match_robot_identities_batch(db):
    rows = [
        {"job_id": "j1", "identity_id": 0, "slot": 0, "slot_label": "B1",
         "alliance": "blue", "team_number": "4174", "confirmed": 1,
         "source": "ocr", "confidence": 0.9},
        {"job_id": "j1", "identity_id": 1, "slot": 1, "slot_label": "B2",
         "alliance": "blue", "team_number": None, "confirmed": 0,
         "source": "ocr", "confidence": None},
    ]
    asyncio.run(db.upsert_match_robot_identities(rows))
    got = asyncio.run(db.get_match_identities("j1"))
    assert len(got) == 2
    assert got[0]["team_number"] == "4174"

    # Re-upsert replaces (same semantics as the single-row function).
    rows[1]["team_number"] = "573"
    asyncio.run(db.upsert_match_robot_identities(rows))
    got = asyncio.run(db.get_match_identities("j1"))
    assert len(got) == 2
    assert got[1]["team_number"] == "573"


def test_upsert_match_robot_identities_requires_identity_or_slot(db):
    with pytest.raises(KeyError):
        asyncio.run(db.upsert_match_robot_identities([{"job_id": "j1"}]))


def test_insert_match_robot_crops_batch(db):
    asyncio.run(db.upsert_match_robot_identities([
        {"job_id": "j1", "identity_id": 0, "slot": 0, "slot_label": "B1",
         "alliance": "blue", "team_number": "4174", "confirmed": 1,
         "source": "ocr", "confidence": 0.9},
    ]))
    crops = [
        {"job_id": "j1", "identity_id": 0, "legibility": 0.9,
         "embedding": _emb_bytes(1, 0), "image": b"jpg1"},
        {"job_id": "j1", "identity_id": 0, "legibility": 0.7,
         "embedding": None, "image": b"jpg2"},
    ]
    asyncio.run(db.insert_match_robot_crops(crops))
    got = asyncio.run(db.get_match_crops("j1", 0))
    assert len(got) == 2
