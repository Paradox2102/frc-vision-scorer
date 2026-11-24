"""/calibrate video picker: listing local videos, uploading one, and the
``video_path`` whitelist every frame-grabbing calibration route goes through.

Everything is redirected into tmp_path (footage dirs, upload dir, repo root, the
version store); the replays-table labels are monkeypatched — no real footage, DB,
network or GPU.
"""
import importlib
import os

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from web.field import auto_calibrate as ac
from web.field import store as field_store

HEX = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def env(tmp_path, monkeypatch):
    mod = importlib.import_module("web.app")
    root = tmp_path / "repo"
    footage, videos, uploads = (root / "data" / "test_footage", root / "data" / "videos",
                                root / "web" / "uploads")
    for d in (footage, videos, uploads):
        d.mkdir(parents=True)
    monkeypatch.setattr(mod, "PROJECT_ROOT", root)
    monkeypatch.setattr(mod, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(ac, "FOOTAGE_DIRS", (footage, videos))
    monkeypatch.setattr(ac, "DOWNLOAD_DIR", footage)
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path / "store")
    monkeypatch.setattr(mod, "_calib_median_cache", {})
    monkeypatch.setattr(mod, "_match_stream_url",
                        lambda mk: pytest.fail("must not resolve a stream"))

    async def _rows():
        return []
    monkeypatch.setattr(mod.database, "replay_source_videos", _rows)
    return {"mod": mod, "root": root, "footage": footage, "videos": videos,
            "uploads": uploads, "tmp": tmp_path, "client": TestClient(mod.app)}


def _video(path, n=6, value=None, size=(64, 36), fourcc="mp4v"):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 1, size)
    assert w.isOpened(), f"no {fourcc} writer"
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), value if value is not None else 30 * i,
                        np.uint8))
    w.release()
    return path


def _touch(path, data=b"x" * 64, mtime=None):
    path.write_bytes(data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


# ------------------------------------------------------------------ listing

def test_list_orders_event_first_and_labels(env, monkeypatch):
    f, v, u = env["footage"], env["videos"], env["uploads"]
    _touch(f / "evt_qm10.mp4", mtime=1000)
    _touch(f / "evt_qm2.mp4", mtime=1001)
    _touch(v / "evt_qm2.mp4", mtime=1002)          # shadowed by test_footage's copy
    _touch(f / "other_qm1.mp4", mtime=3000)
    _touch(f / "sim render.webm", mtime=4000)
    _touch(v / "old.mkv", mtime=500)
    _touch(f / "notes.txt")                         # not a video
    _touch(f / ".evt_qm3.mp4.ab12cd34.part")        # an upload in flight
    _touch(f / "empty_qm9.mp4", data=b"")           # empty
    _touch(u / f"video_{HEX}.mp4", mtime=2000)
    _touch(u / "video_short.mp4")                   # not a content-addressed upload
    _touch(u / "calibration_evt_main.json")

    async def _rows():
        return [{"source_video_path": f"/elsewhere/web/uploads/video_{HEX}.mp4",
                 "match_key": "evt_qm1", "label": "Qual 1 rerun"},
                {"source_video_path": "/x/video_ffff.mp4", "match_key": "evt_qm5"}]
    monkeypatch.setattr(env["mod"].database, "replay_source_videos", _rows)

    r = env["client"].get("/api/calibration/videos", params={"event": "evt"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["upload_dir"] == "data/test_footage"
    paths = [x["path"] for x in j["videos"]]
    assert paths == [
        f"web/uploads/video_{HEX}.mp4",            # evt_qm1 (labelled from replays)
        "data/test_footage/evt_qm2.mp4",           # qm2: the one the match reads first
        "data/videos/evt_qm2.mp4",
        "data/test_footage/evt_qm10.mp4",
        "data/test_footage/sim render.webm",       # then the rest, newest first
        "data/test_footage/other_qm1.mp4",
        "data/videos/old.mkv",
    ]
    by = {x["path"]: x for x in j["videos"]}
    up = by[f"web/uploads/video_{HEX}.mp4"]
    assert up["source"] == "upload" and up["match_key"] == "evt_qm1"
    assert up["event_match"] and up["label"] == "Qual 1 rerun" and not up["match_default"]
    assert by["data/test_footage/evt_qm2.mp4"]["match_default"] is True
    assert by["data/videos/evt_qm2.mp4"]["match_default"] is False
    assert by["data/test_footage/other_qm1.mp4"]["match_key"] == "other_qm1"
    assert by["data/test_footage/other_qm1.mp4"]["event_match"] is False
    assert by["data/test_footage/sim render.webm"]["match_key"] is None
    assert by["data/test_footage/evt_qm10.mp4"]["size"] == 64


def test_list_survives_db_errors_and_no_event(env, monkeypatch):
    _touch(env["footage"] / "evt_qm1.mp4")

    async def _boom():
        raise RuntimeError("no replays table")
    monkeypatch.setattr(env["mod"].database, "replay_source_videos", _boom)
    r = env["client"].get("/api/calibration/videos")
    assert r.status_code == 200
    (only,) = r.json()["videos"]
    assert only["event_match"] is False and only["match_default"] is True


# --------------------------------------------------------- video_path whitelist

def test_resolve_allows_footage_dirs_and_uploads(env):
    mod, root = env["mod"], env["root"]
    (env["footage"] / "sub").mkdir()
    for p in (_touch(env["footage"] / "a.mp4"), _touch(env["videos"] / "b.webm"),
              _touch(env["footage"] / "sub" / "c.mkv"),
              _touch(env["uploads"] / f"video_{HEX}.webm")):
        rel = p.relative_to(root).as_posix()
        assert mod._resolve_calib_video_path(rel) == (p.resolve(), None)
        assert mod._resolve_calib_video_path(str(p)) == (p.resolve(), None)


@pytest.mark.parametrize("raw", [
    "data/test_footage/../secret.mp4",
    "data/test_footage/../../etc/passwd.mp4",
    "/etc/passwd",
    "data/secret.mp4",
    "web/uploads/calibration_evt_main.mp4",          # uploads: only video_<md5>.*
    f"web/uploads/sub/video_{HEX}.mp4",
    "https://example.com/data/test_footage/x.mp4",
    "file:///etc/passwd",
])
def test_resolve_rejects_escapes(env, raw):
    (env["root"] / "data" / "secret.mp4").write_bytes(b"x")
    path, err = env["mod"]._resolve_calib_video_path(raw)
    assert path is None and err == "video_path outside allowed directory"


def test_resolve_rejects_symlink_out_of_footage_and_bad_suffix(env):
    mod = env["mod"]
    outside = _touch(env["tmp"] / "outside.mp4")
    (env["footage"] / "link.mp4").symlink_to(outside)
    assert mod._resolve_calib_video_path("data/test_footage/link.mp4")[1] == \
        "video_path outside allowed directory"
    _touch(env["footage"] / "notes.txt")
    assert mod._resolve_calib_video_path("data/test_footage/notes.txt") == \
        (None, "unsupported video extension")
    for bad in ("", None, 5, "a\x00b.mp4"):
        assert mod._resolve_calib_video_path(bad)[0] is None


def test_resolve_follows_symlinked_footage_dir(env, monkeypatch):
    """A worktree's data/test_footage is a symlink into the main checkout: both sides
    of the containment check must be resolved."""
    real = env["tmp"] / "main_checkout_footage"
    real.mkdir()
    p = _touch(real / "evt_qm1.mp4")
    link = env["root"] / "data" / "linked_footage"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(ac, "FOOTAGE_DIRS", (link,))
    assert env["mod"]._resolve_calib_video_path("data/linked_footage/evt_qm1.mp4") == \
        (p.resolve(), None)


# ------------------------------------------------------- frame grabs by video_path

def test_frame_route_reads_picked_video_over_match(env):
    _video(env["footage"] / "evt_qm2.mp4", value=10)        # the match's footage
    _video(env["videos"] / "picked.mp4", value=200)
    c = env["client"]
    r = c.post("/api/calibration/frame", json={
        "timestamp": 2, "camera": "main", "event": "evt",
        "match_key": "evt_qm2", "video_path": "data/videos/picked.mp4"})
    assert r.status_code == 200, r.text
    img = cv2.imdecode(np.frombuffer(__import__("base64").b64decode(r.json()["frame"]),
                                     np.uint8), cv2.IMREAD_COLOR)
    assert img.mean() > 150                                  # the picked video's frame
    r2 = c.post("/api/calibration/frame", json={
        "timestamp": 2, "camera": "main", "event": "evt",
        "video_path": "data/test_footage/../../outside.mp4"})
    assert r2.status_code == 400 and "outside allowed" in r2.json()["error"]
    r3 = c.post("/api/calibration/frame", json={
        "timestamp": 2, "camera": "main", "video_path": "data/videos/missing.mp4"})
    assert r3.status_code == 400 and r3.json()["error"] == "video not found"


def test_aggregate_and_refine_tag_accept_video_path(env):
    _video(env["footage"] / "picked.mp4", n=12, value=90)
    mod = env["mod"]
    info = {}
    frame, err = mod._read_composite_frame(
        {"video_path": "data/test_footage/picked.mp4", "timestamp": 5, "aggregate": True},
        info)
    assert err is None and info["aggregated"] is True and frame.shape[:2] == (36, 64)
    r = env["client"].post("/api/calibration/refine-tag", json={
        "camera": "main", "timestamp": 5, "video_path": "data/test_footage/picked.mp4",
        "u": 10, "v": 10, "id": 3})
    assert r.status_code == 200 and r.json() == {"corners": None}


def test_webm_video_path_reads_frames(env):
    p = env["footage"] / "sim.webm"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"VP80"), 2, (64, 36))
    if not w.isOpened():
        pytest.skip("OpenCV build has no VP8 writer")
    for i in range(8):
        w.write(np.full((36, 64, 3), 25 * i, np.uint8))
    w.release()
    frame, err = env["mod"]._read_composite_frame(
        {"video_path": "data/test_footage/sim.webm", "timestamp": 2})
    assert err is None
    assert abs(int(frame.mean()) - 100) <= 8                 # frame 4 = 2 s at 2 fps


def test_solve_tags_stores_video_path_in_source(env):
    from web.field.field_model import FieldModel
    from tests._field_synth import truth_pinhole_camera, project_tag_corners
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera(D=np.array([-0.12, 0.02, 0.0, 0.0, 0.0]))
    corr = project_tag_corners(cam, fm)
    auto = {str(tid): corr[tid][0].tolist() for tid in corr}
    r = env["client"].post("/api/calibration/solve-tags", json={
        "event": "evt_vid", "camera": "main", "season": "2026-rebuilt-welded",
        "image_size": list(cam.image_size), "auto_tags": auto, "manual_tags": [],
        "start_match": "evt_vid_qm4", "timestamp": 12, "match_key": None,
        "video_path": "data/test_footage/sim.webm"})
    assert r.status_code == 200, r.text
    (v,) = field_store.load_versions("evt_vid", "main")
    assert v["start_match"] == "evt_vid_qm4"                # anchor stays the match
    assert v["source"]["video_path"] == "data/test_footage/sim.webm"
    assert v["source"]["match_key"] is None


# ------------------------------------------------------------------ upload

def _upload(client, name, data, **form):
    return client.post("/api/calibration/videos/upload",
                       files={"file": (name, data, "video/mp4")}, data=form)


def test_upload_sanitises_name_and_refuses_overwrite(env):
    src = _video(env["tmp"] / "src.mp4").read_bytes()
    c = env["client"]
    r = _upload(c, "../../evil name!.MP4", src)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["path"] == "data/test_footage/evil_name.mp4" and j["name"] == "evil_name.mp4"
    assert (env["footage"] / "evil_name.mp4").read_bytes() == src
    assert j["match_key"] is None and j["size"] == len(src) and j["fps"] == 1.0
    r2 = _upload(c, "evil name!.mp4", src)
    assert r2.status_code == 409 and r2.json()["exists"] is True
    r3 = _upload(c, "evil name!.mp4", src, force="true")
    assert r3.status_code == 200
    assert sorted(p.name for p in env["footage"].iterdir()) == ["evil_name.mp4"]


def test_upload_as_match_footage(env):
    src = _video(env["tmp"] / "src.mp4").read_bytes()
    r = _upload(env["client"], "whatever.mp4", src, match_key="evt_qm7")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["path"] == "data/test_footage/evt_qm7.mp4"
    assert j["match_key"] == "evt_qm7" and j["match_default"] is True
    assert ac.local_video_for("evt_qm7") == env["footage"] / "evt_qm7.mp4"


@pytest.mark.parametrize("name,form,err", [
    ("clip.txt", {}, "unsupported video extension"),
    ("clip", {}, "unsupported video extension"),
    ("clip.mp4", {"match_key": "../../x"}, "invalid match key"),
    ("clip.mp4", {"match_key": "evt_qm7/../../x"}, "invalid match key"),
])
def test_upload_rejects_bad_names(env, name, form, err):
    r = _upload(env["client"], name, b"\x00" * 100, **form)
    assert r.status_code == 400 and err in r.json()["error"]
    assert list(env["footage"].iterdir()) == []


def test_upload_rejects_undecodable_and_leaves_nothing(env):
    r = _upload(env["client"], "junk.mp4", b"not a video at all" * 100)
    assert r.status_code == 400 and "not a readable video" in r.json()["error"]
    assert list(env["footage"].iterdir()) == []              # no .part, no file
