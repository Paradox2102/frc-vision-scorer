import asyncio
import numpy as np
import pytest
from fastapi.testclient import TestClient

import web.database as database


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "t.db")
    asyncio.run(database.init_db())
    from web.app import app
    return TestClient(app), tmp_path


def _seed():
    asyncio.run(database.insert_gallery_exemplar({
        "event_key": "2026x", "team": "1787", "alliance_color": "blue",
        "embedding": np.array([1, 0, 0], np.float32).tobytes(),
        "legibility": 0.7, "image": b"img1", "verified": 0,
    }))
    asyncio.run(database.insert_gallery_exemplar({
        "event_key": "2026x", "team": "254", "alliance_color": "red",
        "embedding": np.array([0, 1, 0], np.float32).tobytes(),
        "legibility": 0.9, "image": b"img2", "verified": 1,
    }))


def test_list_gallery_groups_and_images(client):
    c, _ = client
    _seed()
    r = c.get("/api/gallery?event_key=2026x")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["event_key"] == "2026x"
    ex = body["exemplars"]
    assert len(ex) == 2
    teams = {e["team"] for e in ex}
    assert teams == {"1787", "254"}
    for e in ex:
        assert set(["id", "team", "alliance_color", "verified", "legibility", "image"]).issubset(e.keys())
        assert isinstance(e["verified"], bool)
        assert e["image"].startswith("data:image/jpeg;base64,")
    # No embedding bytes ever leak to the client
    assert "embedding" not in r.text


def test_update_gallery_relabel_and_verify(client):
    c, _ = client
    _seed()
    rows = asyncio.run(database.get_gallery_exemplars("2026x"))
    gid = [e for e in rows if e["team"] == "1787"][0]["id"]
    r = c.post(f"/api/gallery/{gid}", json={"team": "1799", "verified": True})
    assert r.status_code == 200, r.text
    rows = asyncio.run(database.get_gallery_exemplars("2026x"))
    row = [e for e in rows if e["id"] == gid][0]
    assert row["team"] == "1799"
    assert row["verified"] == 1


def test_delete_gallery_exemplar(client):
    c, _ = client
    _seed()
    rows = asyncio.run(database.get_gallery_exemplars("2026x"))
    gid = [e for e in rows if e["team"] == "254"][0]["id"]
    r = c.delete(f"/api/gallery/{gid}")
    assert r.status_code == 200, r.text
    rows = asyncio.run(database.get_gallery_exemplars("2026x"))
    assert gid not in [e["id"] for e in rows]
