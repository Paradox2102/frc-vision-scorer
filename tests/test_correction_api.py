import asyncio
import json
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


def _seed(rid="job-1", event_key="2026x", alliance="blue",
          blue_teams='["111","222","333"]', red_teams='["444","555","666"]',
          team_number="111", with_embedding=True):
    asyncio.run(database.insert_replay({
        "id": rid, "created_at": "2026-06-03T00:00:00", "label": "Qual 10",
        "event_key": event_key, "match_key": None, "comp_level": None,
        "blue_score": 70, "red_score": 67,
        "blue_teams": blue_teams, "red_teams": red_teams,
        "video_path": "x/hud.mp4", "meta_path": "x/meta.json",
        "thumb_path": "x/thumb.jpg", "source_video_path": "x.mp4",
        "fps": 8.0, "frame_count": 5, "duration_sec": 0.625,
        "width": 160, "height": 120, "codec": "mp4v", "status": "complete",
    }))
    asyncio.run(database.upsert_match_robot_identity({
        "job_id": rid, "slot": 0, "slot_label": "B1", "alliance": alliance,
        "team_number": team_number, "confirmed": 1, "source": "ocr",
        "confidence": 0.9,
    }))
    emb = np.array([1, 0, 0], np.float32).tobytes() if with_embedding else None
    asyncio.run(database.insert_match_robot_crop({
        "job_id": rid, "slot": 0, "legibility": 0.8,
        "embedding": emb, "image": b"img",
    }))


def test_get_identities_returns_crops_as_data_urls_no_embeddings(client):
    c, _ = client
    _seed()
    r = c.get("/api/replays/job-1/identities")
    assert r.status_code == 200
    body = r.json()
    robots = body["robots"]
    slot0 = [x for x in robots if x["slot"] == 0][0]
    assert slot0["team_number"] == "111"
    assert slot0["crops"]
    for u in slot0["crops"]:
        assert u.startswith("data:image/jpeg;base64,")
    # No embeddings ever leak to the client
    assert "embedding" not in r.text


def test_correct_updates_identity_and_mints_verified_exemplar(client):
    c, _ = client
    _seed()
    r = c.post("/api/replays/job-1/correct", json={"slot": 0, "team_number": "999"})
    assert r.status_code == 200, r.text

    ids = asyncio.run(database.get_match_identities("job-1"))
    slot0 = [x for x in ids if x["slot"] == 0][0]
    assert slot0["team_number"] == "999"

    ex = asyncio.run(database.get_gallery_exemplars("2026x"))
    verified_999 = [e for e in ex if e["team"] == "999" and e["verified"] == 1]
    assert len(verified_999) == 1  # one per crop that had an embedding

    replay = asyncio.run(database.get_replay("job-1"))
    blue = json.loads(replay["blue_teams"])
    assert "999" in [str(t) for t in blue]


def test_correct_skips_gallery_when_no_event_key(client):
    c, _ = client
    _seed(rid="job-2", event_key=None, team_number="111")
    r = c.post("/api/replays/job-2/correct", json={"slot": 0, "team_number": "888"})
    assert r.status_code == 200, r.text

    ids = asyncio.run(database.get_match_identities("job-2"))
    slot0 = [x for x in ids if x["slot"] == 0][0]
    assert slot0["team_number"] == "888"

    # No event_key -> nothing minted anywhere
    assert asyncio.run(database.get_gallery_exemplars("2026x")) == []
    assert asyncio.run(database.get_gallery_exemplars("")) == []
    assert asyncio.run(database.get_gallery_exemplars(None)) == []
