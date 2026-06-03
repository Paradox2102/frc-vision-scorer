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


def test_insert_and_fetch_gallery_exemplars_by_event(db):
    asyncio.run(db.insert_gallery_exemplar({
        "event_key": "2026wamon", "team": "1787", "alliance_color": "blue",
        "embedding": _emb_bytes(1, 0, 0), "legibility": 0.9,
    }))
    asyncio.run(db.insert_gallery_exemplar({
        "event_key": "2026wamon", "team": "254", "alliance_color": "red",
        "embedding": _emb_bytes(0, 1, 0), "legibility": 0.8,
    }))
    asyncio.run(db.insert_gallery_exemplar({
        "event_key": "2026other", "team": "9", "alliance_color": "red",
        "embedding": _emb_bytes(0, 0, 1), "legibility": 0.7,
    }))

    rows = asyncio.run(db.get_gallery_exemplars("2026wamon"))
    assert len(rows) == 2
    teams = sorted(r["team"] for r in rows)
    assert teams == ["1787", "254"]
    # embedding survives round-trip as bytes -> float32
    blue = next(r for r in rows if r["team"] == "1787")
    arr = np.frombuffer(blue["embedding"], dtype=np.float32)
    assert np.allclose(arr, [1, 0, 0])


def test_get_gallery_exemplars_empty_for_unknown_event(db):
    assert asyncio.run(db.get_gallery_exemplars("nope")) == []
