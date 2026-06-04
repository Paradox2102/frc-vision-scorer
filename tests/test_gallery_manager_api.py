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


def test_list_gallery_attaches_source_match_roster(client):
    # Each exemplar carries the 6-team roster of the match it was pulled from,
    # so the UI can offer that match as a relabel dropdown.
    c, _ = client
    asyncio.run(database.insert_replay({
        "id": "job-xyz", "event_key": "2026x",
        "blue_teams": '["1787", "254", "118"]',
        "red_teams": '["973", "1678", "2056"]',
    }))
    asyncio.run(database.insert_gallery_exemplar({
        "event_key": "2026x", "team": "1787", "alliance_color": "blue",
        "embedding": np.array([1, 0, 0], np.float32).tobytes(),
        "legibility": 0.7, "image": b"img1", "verified": 0, "job_id": "job-xyz",
    }))
    # An exemplar with no source match -> empty roster, no crash.
    asyncio.run(database.insert_gallery_exemplar({
        "event_key": "2026x", "team": "9999", "alliance_color": "red",
        "embedding": np.array([0, 1, 0], np.float32).tobytes(),
        "legibility": 0.5, "image": b"img2", "verified": 0,
    }))
    body = c.get("/api/gallery?event_key=2026x").json()
    by_team = {e["team"]: e for e in body["exemplars"]}
    assert sorted(by_team["1787"]["match_teams"]) == sorted(
        ["1787", "254", "118", "973", "1678", "2056"])
    assert by_team["9999"]["match_teams"] == []


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


def test_update_gallery_coerces_nonbool_verified(client):
    # A non-bool 'verified' value must not 500 (int("yes") would raise);
    # bool("yes") is True -> stored as 1.
    c, _ = client
    _seed()
    rows = asyncio.run(database.get_gallery_exemplars("2026x"))
    gid = [e for e in rows if e["team"] == "1787"][0]["id"]
    r = c.post(f"/api/gallery/{gid}", json={"verified": "yes"})
    assert r.status_code == 200, r.text
    rows = asyncio.run(database.get_gallery_exemplars("2026x"))
    row = [e for e in rows if e["id"] == gid][0]
    assert row["verified"] == 1


def test_update_gallery_rejects_bad_alliance_color(client):
    # An invalid alliance_color must 400 and leave the row untouched.
    c, _ = client
    _seed()
    rows = asyncio.run(database.get_gallery_exemplars("2026x"))
    gid = [e for e in rows if e["team"] == "1787"][0]["id"]
    r = c.post(f"/api/gallery/{gid}", json={"alliance_color": "green"})
    assert r.status_code == 400, r.text
    assert "alliance_color" in r.json()["error"]
    rows = asyncio.run(database.get_gallery_exemplars("2026x"))
    row = [e for e in rows if e["id"] == gid][0]
    assert row["alliance_color"] == "blue"


def test_update_gallery_empty_body_is_noop(client):
    c, _ = client
    _seed()
    rows = asyncio.run(database.get_gallery_exemplars("2026x"))
    gid = [e for e in rows if e["team"] == "1787"][0]["id"]
    r = c.post(f"/api/gallery/{gid}", json={})
    assert r.status_code == 200, r.text
    rows = asyncio.run(database.get_gallery_exemplars("2026x"))
    row = [e for e in rows if e["id"] == gid][0]
    assert row["team"] == "1787"
    assert row["alliance_color"] == "blue"
    assert row["verified"] == 0


def test_delete_gallery_exemplar(client):
    c, _ = client
    _seed()
    rows = asyncio.run(database.get_gallery_exemplars("2026x"))
    gid = [e for e in rows if e["team"] == "254"][0]["id"]
    r = c.delete(f"/api/gallery/{gid}")
    assert r.status_code == 200, r.text
    rows = asyncio.run(database.get_gallery_exemplars("2026x"))
    assert gid not in [e["id"] for e in rows]
