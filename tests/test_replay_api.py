import asyncio
import json
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

import web.database as database


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "t.db")
    asyncio.run(database.init_db())
    from web.app import app
    return TestClient(app), tmp_path


def _make_replay_files(tmp_path, rid="job-1", frames=5, fps=8.0):
    d = tmp_path / rid
    d.mkdir()
    vw = cv2.VideoWriter(str(d / "hud.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for _ in range(frames):
        vw.write(np.zeros((120, 160, 3), np.uint8))
    vw.release()
    (d / "meta.json").write_text(json.dumps({
        "summary": {"frame_count": frames, "fps": fps},
        "frames": [{"i": i, "t": i / fps} for i in range(frames)],
        "events": []}))
    cv2.imwrite(str(d / "thumb.jpg"), np.zeros((120, 160, 3), np.uint8))
    return d


def _insert(database, d, rid="job-1"):
    asyncio.run(database.insert_replay({
        "id": rid, "created_at": "2026-06-03T00:00:00", "label": "Qual 10",
        "event_key": None, "match_key": None, "comp_level": None,
        "blue_score": 70, "red_score": 67, "blue_teams": "[581]", "red_teams": "[1732]",
        "video_path": str(d / "hud.mp4"), "meta_path": str(d / "meta.json"),
        "thumb_path": str(d / "thumb.jpg"), "source_video_path": "x.mp4",
        "fps": 8.0, "frame_count": 5, "duration_sec": 0.625,
        "width": 160, "height": 120, "codec": "mp4v", "status": "complete",
    }))


def test_list_get_thumb_delete(client):
    c, tmp_path = client
    d = _make_replay_files(tmp_path)
    _insert(database, d)

    r = c.get("/api/replays")
    assert r.status_code == 200
    assert any(x["id"] == "job-1" and x["label"] == "Qual 10" for x in r.json())
    item = [x for x in r.json() if x["id"] == "job-1"][0]
    assert item["blue_teams"] == [581] and item["red_teams"] == [1732]
    assert item["thumb_url"] == "/api/replays/job-1/thumb"

    r = c.get("/api/replays/job-1")
    assert r.status_code == 200 and r.json()["summary"]["frame_count"] == 5

    r = c.get("/api/replays/job-1/thumb")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/")

    r = c.delete("/api/replays/job-1")
    assert r.status_code == 200
    assert c.get("/api/replays/job-1").status_code == 404
    assert not d.exists()


def test_get_missing_returns_404(client):
    c, tmp_path = client
    assert c.get("/api/replays/nope").status_code == 404


def test_video_full_get_and_range(client):
    c, tmp_path = client
    d = _make_replay_files(tmp_path, rid="job-2")
    _insert(database, d, rid="job-2")
    size = (d / "hud.mp4").stat().st_size

    # full GET -> 200, whole file
    r = c.get("/api/replays/job-2/video")
    assert r.status_code == 200 and len(r.content) == size

    # ranged GET -> 206 with correct slice + headers
    r = c.get("/api/replays/job-2/video", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206
    assert r.headers["content-range"] == f"bytes 0-99/{size}"
    assert r.headers["accept-ranges"] == "bytes"
    assert len(r.content) == 100


def test_video_missing_returns_404(client):
    c, tmp_path = client
    assert c.get("/api/replays/nope/video").status_code == 404


def test_download_video_and_data_are_attachments(client):
    c, tmp_path = client
    d = _make_replay_files(tmp_path, rid="job-9")
    _insert(database, d, rid="job-9")  # label "Qual 10" -> "Qual_10"

    r = c.get("/api/replays/job-9/download/video")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("video/mp4")
    assert "attachment" in r.headers.get("content-disposition", "")
    assert "Qual_10.mp4" in r.headers["content-disposition"]
    assert len(r.content) == (d / "hud.mp4").stat().st_size

    r = c.get("/api/replays/job-9/download/data")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert "attachment" in r.headers.get("content-disposition", "")
    assert "Qual_10.json" in r.headers["content-disposition"]
    assert json.loads(r.content)["summary"]["frame_count"] == 5


def test_download_missing_returns_404(client):
    c, tmp_path = client
    assert c.get("/api/replays/nope/download/video").status_code == 404
    assert c.get("/api/replays/nope/download/data").status_code == 404
