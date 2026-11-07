"""scripts/gt_annotator_server.py — save/load round trip with server-side
reprojection, path-traversal rejection, and HTTP Range on the video endpoint.
Uses a temp sqlite DB, a tiny generated mp4 and a fake linear calibration, so it
touches neither the real DB nor data/eval/ground_truth."""
import json
import sqlite3

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import scripts.gt_annotator_server as server
from web.field import ground_truth as gt
from web.field.camera_view import CameraView

MATCH = "2099tst_qm1"
EVENT = "2099tst"


class _LinearCal:
    """uv (crop-local px) -> field metres = uv / 100."""
    mode = "fake"
    field_season = "2026-rebuilt-welded"

    def image_to_field(self, uv, z=0.0):
        return (uv[0] / 100.0, uv[1] / 100.0)

    def field_to_image(self, xyz):
        return (xyz[0] * 100.0, xyz[1] * 100.0 - xyz[2] * 100.0)


def _fake_views_for(event_key, match_key):
    views = [CameraView("main", (0, 0, 1920, 723), (), _LinearCal()),
             CameraView("corner_left", (0, 703, 960, 377), (), _LinearCal())]
    return views, {"main": "v-main", "corner_left": "v-cl"}, (16.54, 8.07), "2026-rebuilt-welded"


def _make_video(path, n=12):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 60, (64, 36))
    for i in range(n):
        w.write(np.full((36, 64, 3), i * 10, np.uint8))
    w.release()
    assert path.stat().st_size > 200


@pytest.fixture
def client(tmp_path, monkeypatch):
    vid = tmp_path / "match.mp4"
    _make_video(vid)
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({
        "summary": {"periods": [{"label": "auto", "start_frame": 3, "end_frame": 9}]},
        "frames": [{"frame_num": 4, "robots": [{"slot": "B1", "alliance": "blue", "team_number": "111",
                                                 "field_pos": [1.0, 2.0]}]},
                   {"frame_num": 6, "robots": [{"slot": "B1", "alliance": "blue", "team_number": "111",
                                                 "field_pos": [1.5, 2.5]}]}]}))
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE replays (id TEXT PRIMARY KEY, created_at TEXT, label TEXT, event_key TEXT,
            match_key TEXT, source_video_path TEXT, meta_path TEXT, status TEXT);
        CREATE TABLE matches (id INTEGER PRIMARY KEY, tba_match_key TEXT);
        CREATE TABLE match_robots (id INTEGER PRIMARY KEY, match_id INTEGER, alliance TEXT,
            position INTEGER, team_number TEXT);
    """)
    con.execute("INSERT INTO replays VALUES ('r1','2099-01-01T00:00:00','x',?,?,?,?,'complete')",
                (EVENT, MATCH, str(vid), str(meta)))
    con.execute("INSERT INTO replays VALUES ('r0','2098-01-01T00:00:00','gone',?,?,?,?,'complete')",
                (EVENT, MATCH, str(tmp_path / "missing.mp4"), str(meta)))
    con.execute("INSERT INTO matches VALUES (1, ?)", (MATCH,))
    for i, (a, p, t) in enumerate([("blue", 1, "111"), ("blue", 2, "222"), ("red", 1, "333")]):
        con.execute("INSERT INTO match_robots VALUES (?,?,?,?,?)", (i + 1, 1, a, p, t))
    con.commit()
    con.close()
    gt_dir = tmp_path / "gt"
    monkeypatch.setattr(server, "DB_PATH", db)
    monkeypatch.setattr(server, "GT_DIR", gt_dir)
    monkeypatch.setattr(server, "views_for", _fake_views_for)
    server._videos.clear()
    c = TestClient(server.app)
    c.gt_dir = gt_dir
    return c


def _video_id(client):
    m = client.get("/api/matches").json()["matches"][0]
    return m, m["default_video_id"]


def test_matches_lists_roster_video_and_t0(client):
    m, vid = _video_id(client)
    assert m["match_key"] == MATCH and m["event_key"] == EVENT
    assert [r["team"] for r in m["roster"]["blue"]] == ["111", "222"]
    assert m["roster"]["red"] == [{"team": "333", "station": 1}]
    assert len(m["videos"]) == 1                       # missing file skipped
    v = m["videos"][0]
    assert v["video_id"] == vid and v["suggested_t0_frame"] == 3
    assert v["fps"] == pytest.approx(60.0) and v["frame_count"] == 12 and v["width"] == 64


def test_save_load_round_trip_reprojects_server_side(client):
    _, vid = _video_id(client)
    body = {"video_id": vid, "match_t0_frame": 3, "annotator": "nick", "keyframes": [
        {"frame": 5, "state": "visible", "uv": [300, 200], "src": "follow", "rate": 0.75,
         "field_xy": [99, 99]},                                  # client field_xy ignored
        {"frame": 2, "state": "estimated", "uv": [100, 800], "src": "click"},   # corner view
        {"frame": 7, "state": "unknown", "uv": [1, 1], "src": "click"},
        {"frame": 5, "state": "visible", "uv": [400, 200], "src": "follow", "rate": 0.75},  # dup: last wins
    ]}
    r = client.put(f"/api/gt/{MATCH}/111", json=body)
    assert r.status_code == 200, r.text
    doc = r.json()["doc"]
    assert doc["team"] == "111" and doc["alliance"] == "blue" and doc["station"] == 1
    assert doc["calibration"] == {"main": "v-main", "corner_left": "v-cl"}
    assert doc["video"]["fingerprint"] == vid.replace("-", ":")
    assert doc["video"]["fps"] == pytest.approx(60.0)
    kfs = {k["frame"]: k for k in doc["keyframes"]}
    assert [k["frame"] for k in doc["keyframes"]] == [2, 5, 7]
    assert kfs[5]["field_xy"] == [4.0, 2.0] and kfs[5]["view"] == "main"
    assert kfs[5]["src"] == "follow" and kfs[5]["rate"] == 0.75
    assert kfs[2]["view"] == "corner_left" and kfs[2]["field_xy"] == [1.0, 0.97]
    assert "uv" not in kfs[7] and "field_xy" not in kfs[7]

    got = client.get(f"/api/gt/{MATCH}/111").json()["doc"]
    assert got["keyframes"] == doc["keyframes"]
    assert (client.gt_dir / MATCH / "111.json").exists()
    assert [d["team"] for d in client.get(f"/api/gt/{MATCH}").json()["docs"]] == ["111"]
    st = client.get("/api/matches").json()["matches"][0]["gt_status"]
    assert st["111"]["keyframes"] == 3


def test_save_refused_on_different_video(client, tmp_path):
    _, vid = _video_id(client)
    client.put(f"/api/gt/{MATCH}/111", json={"video_id": vid, "keyframes": [
        {"frame": 1, "state": "visible", "uv": [10, 10], "src": "click"}]})
    doc = gt.load_gt(MATCH, "111", client.gt_dir)
    doc["video"]["fingerprint"] = "1:deadbeef"
    gt.save_gt(doc, client.gt_dir)
    r = client.put(f"/api/gt/{MATCH}/111", json={"video_id": vid, "keyframes": []})
    assert r.status_code == 409


@pytest.mark.parametrize("url", [
    f"/api/gt/{MATCH}/..%2F..%2Fetc",
    f"/api/gt/{MATCH}/a.b",
    "/api/gt/..%2Fx",
    "/api/gt/bad.key/111",
])
def test_gt_path_traversal_rejected(client, url):
    assert client.get(url).status_code in (400, 404)


def test_put_path_traversal_rejected(client, tmp_path):
    _, vid = _video_id(client)
    for team in ("..", "a.json", "x%2F..%2Fy"):
        r = client.put(f"/api/gt/{MATCH}/{team}", json={"video_id": vid, "keyframes": []})
        assert r.status_code in (400, 404, 405), team
    assert sorted(p.name for p in tmp_path.rglob("*.json")) == ["meta.json"]


@pytest.mark.parametrize("video_id", ["..%2F..%2Fetc%2Fpasswd", "match.mp4", "deadbeef", "%2Fetc%2Fpasswd"])
def test_video_allow_list(client, video_id):
    _video_id(client)
    assert client.get(f"/api/video/{video_id}").status_code == 404


def test_video_range_request_returns_206(client):
    _, vid = _video_id(client)
    full = client.get(f"/api/video/{vid}")
    assert full.status_code == 200
    r = client.get(f"/api/video/{vid}", headers={"Range": "bytes=10-109"})
    assert r.status_code == 206
    assert r.headers["content-range"].startswith("bytes 10-109/")
    assert r.content == full.content[10:110]


def test_replay_tracks_and_list(client):
    reps = client.get(f"/api/replays/{MATCH}").json()["replays"]
    assert [r["id"] for r in reps] == ["r1", "r0"]
    t = client.get("/api/replay/r1/tracks").json()
    assert t["slots"] == [{"slot": "B1", "alliance": "blue", "team_number": "111",
                           "pts": [[4, 1.0, 2.0], [6, 1.5, 2.5]]}]
    assert client.get("/api/replay/nope/tracks").status_code == 404


def test_projection_grid_matches_exact_projection():
    views, *_ = _fake_views_for(EVENT, MATCH)
    g = server.projection_grid(views, step=8)
    main = g[0]
    assert main["name"] == "main" and main["nx"] == 1920 // 8 + 2
    i, j = 10, 7
    assert main["xy"][j * main["nx"] + i] == [0.8, 0.56]


def test_project_endpoint(client):
    r = client.post("/api/project", json={"event_key": EVENT, "match_key": MATCH, "uv": [300, 200]})
    assert r.json() == {"view": "main", "field_xy": [3.0, 2.0], "on_field": True}
    assert client.post("/api/project", json={"event_key": "../x", "uv": [1, 1]}).status_code == 400


def test_robot_box_grid(client):
    r = client.get(f"/api/event/{EVENT}/robot_box_grid?match_key={MATCH}")
    assert r.status_code == 200
    d = r.json()
    main = d["grid"][0]
    assert main["name"] == "main" and main["step"] == 32
    nx = main["nx"]
    # footprint 0.95 m -> +-47.5 px; height 0.6 m -> 60 px up; ground point at bottom-centre
    assert main["box"][5 * nx + 10] == [-47.5, -107.5, 47.5, 47.5]
    assert main["box"][5 * nx + (nx - 1)] is None          # x = 19.5 m: off-field
    assert client.get("/api/event/..%2Fx/robot_box_grid").status_code in (400, 404)
