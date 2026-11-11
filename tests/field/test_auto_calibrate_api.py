"""Auto-calibration routes, background runner and hooks in web.app.

Never touches the network or real footage: the reference-video resolution and the
per-camera solve are monkeypatched, and the store is redirected to tmp_path.
"""
import asyncio
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from web.field import auto_calibrate as ac
from web.field import store as field_store
from tests._field_synth import truth_pinhole_camera


@pytest.fixture
def app_mod(tmp_path, monkeypatch):
    mod = importlib.import_module("web.app")
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(mod, "_auto_calib_tasks", {})
    monkeypatch.setattr(mod, "_auto_calib_judge", lambda: None)   # never Jev in tests
    return mod


def _client(mod):
    return TestClient(mod.app)


def _ready_record(match_key="evt_qm4", method="full_solve"):
    cal = truth_pinhole_camera()
    return ac.new_record(
        "ready", "solved", match_key=match_key, method=method, timestamp=40.0,
        calibration=cal.to_json(), auto_tags={"6": [[1, 2], [3, 4], [5, 6], [7, 8]]},
        metrics={"n_tags": 6, "rms_px": 1.1}, preview="calibration_auto_evt_main.jpg")


# --------------------------------------------------------------------- status

def test_status_omits_cameras_without_records_and_strips_calibration(app_mod):
    r = _client(app_mod).get("/api/calibration/auto/status", params={"event": "evt"})
    assert r.status_code == 200
    assert r.json() == {"event": "evt", "running": False, "cameras": {}}

    field_store.save_auto_record("evt", "main", _ready_record())
    j = _client(app_mod).get("/api/calibration/auto/status", params={"event": "evt"}).json()
    assert set(j["cameras"]) == {"main"}
    rec = j["cameras"]["main"]
    assert rec["status"] == "ready" and "calibration" not in rec
    assert rec["auto_tags"]["6"][0] == [1, 2]


def test_status_reports_running_and_stale_running_as_failed(app_mod, monkeypatch):
    field_store.save_auto_record("evt", "main", ac.new_record("running", "queued"))
    j = _client(app_mod).get("/api/calibration/auto/status", params={"event": "evt"}).json()
    assert j["running"] is False
    assert j["cameras"]["main"]["status"] == "failed"
    assert "interrupted" in j["cameras"]["main"]["reason"]

    field_store.save_auto_record("evt", "main", ac.new_record("running", "queued"))
    monkeypatch.setitem(app_mod._auto_calib_tasks, "evt",
                        SimpleNamespace(done=lambda: False))
    j = _client(app_mod).get("/api/calibration/auto/status", params={"event": "evt"}).json()
    assert j["running"] is True and j["cameras"]["main"]["status"] == "running"


# -------------------------------------------------------------------- preview

def test_preview_404_then_jpeg(app_mod):
    c = _client(app_mod)
    r = c.get("/api/calibration/auto/preview", params={"event": "evt", "camera": "main"})
    assert r.status_code == 404
    field_store.save_auto_record("evt", "main", _ready_record())
    field_store.save_auto_preview("evt", "main", b"\xff\xd8\xff\xe0fakejpeg")
    r = c.get("/api/calibration/auto/preview", params={"event": "evt", "camera": "main"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content.startswith(b"\xff\xd8")


# -------------------------------------------------------------------- confirm

def test_confirm_without_versions_promotes_with_null_start_match(app_mod):
    field_store.save_auto_record("evt", "main", _ready_record("evt_qm4"))
    r = _client(app_mod).post("/api/calibration/auto/confirm",
                              json={"event": "evt", "camera": "main"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["start_match"] is None and j["version_id"]
    versions = field_store.load_versions("evt", "main")
    assert len(versions) == 1
    v = versions[0]
    assert v["start_match"] is None and v["label"] == "auto (confirmed)"
    assert v["source"]["method"] == "auto" and v["source"]["match_key"] == "evt_qm4"
    assert v["source"]["auto_tags"] == {"6": [[1, 2], [3, 4], [5, 6], [7, 8]]}
    assert v["source"]["manual_tags"] == [] and v["source"]["holdout_ids"] == []
    assert v["source"]["timestamp"] == 40.0 and v["source"]["metrics"]["n_tags"] == 6
    rec = field_store.load_auto_record("evt", "main")
    assert rec["status"] == "confirmed" and rec["resolved_version"] == j["version_id"]
    assert rec["resolved_at"]


def test_confirm_with_versions_anchors_at_reference_and_carries_masks(app_mod):
    base = truth_pinhole_camera()
    base.masks = [(1, 2, 3, 4)]
    base.robot_track_top_crop = 0.2
    base_id = field_store.save_version("evt", "main", base, start_match=None)
    field_store.save_auto_record("evt", "main",
                                 _ready_record("evt_qm9", method="pose_resolve"))
    r = _client(app_mod).post("/api/calibration/auto/confirm",
                              json={"event": "evt", "camera": "main"})
    assert r.status_code == 200, r.text
    assert r.json()["start_match"] == "evt_qm9"
    from web.field.match_order import select_epoch
    versions = field_store.load_versions("evt", "main")
    assert select_epoch(versions, "evt_qm8")["version_id"] == base_id
    cover = select_epoch(versions, "evt_qm9")
    assert cover["version_id"] == r.json()["version_id"]
    assert cover["calibration"].masks == [(1, 2, 3, 4)]
    assert cover["calibration"].robot_track_top_crop == 0.2


@pytest.mark.parametrize("status", ["verified", "failed", "running", "confirmed"])
def test_confirm_refuses_non_ready(app_mod, status):
    rec = _ready_record(); rec["status"] = status
    field_store.save_auto_record("evt", "main", rec)
    r = _client(app_mod).post("/api/calibration/auto/confirm",
                              json={"event": "evt", "camera": "main"})
    assert r.status_code == 400
    assert field_store.load_versions("evt", "main") == []


def test_confirm_refuses_ready_without_calibration(app_mod):
    rec = _ready_record(); rec["calibration"] = None
    field_store.save_auto_record("evt", "main", rec)
    r = _client(app_mod).post("/api/calibration/auto/confirm",
                              json={"event": "evt", "camera": "main"})
    assert r.status_code == 400


def test_confirm_missing_record_404(app_mod):
    r = _client(app_mod).post("/api/calibration/auto/confirm",
                              json={"event": "evt", "camera": "main"})
    assert r.status_code == 404


# -------------------------------------------------------------------- dismiss

def test_dismiss_marks_failed(app_mod):
    field_store.save_auto_record("evt", "main", _ready_record())
    r = _client(app_mod).post("/api/calibration/auto/dismiss",
                              json={"event": "evt", "camera": "main"})
    assert r.status_code == 200
    rec = field_store.load_auto_record("evt", "main")
    assert rec["status"] == "failed" and rec["reason"] == "dismissed — calibrate manually"


# ------------------------------------------------------------------------ run

def test_run_returns_409_while_running(app_mod, monkeypatch):
    monkeypatch.setitem(app_mod._auto_calib_tasks, "evt",
                        SimpleNamespace(done=lambda: False))
    r = _client(app_mod).post("/api/calibration/auto/run", json={"event": "evt"})
    assert r.status_code == 409


def test_run_schedules_requested_camera_with_force(app_mod, monkeypatch):
    calls = []
    monkeypatch.setattr(app_mod, "_schedule_auto_calibrate",
                        lambda ev, matches, cameras=None, force=False:
                        calls.append((ev, cameras, force)) or True)
    r = _client(app_mod).post("/api/calibration/auto/run",
                              json={"event": "evt", "camera": "main", "force": True})
    assert r.status_code == 200, r.text
    assert calls == [("evt", ["main"], True)]
    r = _client(app_mod).post("/api/calibration/auto/run",
                              json={"event": "evt", "camera": "nope"})
    assert r.status_code == 400


# --------------------------------------------------------------------- runner

def _fake_result(status="ready"):
    rec = ac.new_record(status, "fake", match_key="evt_qm1", method="full_solve",
                        calibration=truth_pinhole_camera().to_json())
    return ac.AutoCalibResult(record=rec, calibration=None,
                              overlay=np.zeros((10, 10, 3), np.uint8),
                              median=np.full((10, 10, 3), 7, np.uint8))


def test_runner_download_failure_fails_every_camera(app_mod, monkeypatch, tmp_path):
    monkeypatch.setattr(ac, "FOOTAGE_DIRS", (tmp_path / "none",))
    monkeypatch.setattr(ac, "DOWNLOAD_DIR", tmp_path)

    def boom(video_key, out):
        raise ac.AutoCalibError(f"download of {video_key} failed: ERROR: HTTP Error 403")
    monkeypatch.setattr(ac, "download_match_video", boom)
    monkeypatch.setattr(ac, "run_camera",
                        lambda *a, **k: pytest.fail("must not solve without footage"))
    asyncio.run(app_mod._run_auto_calibrate(
        "evt", [{"key": "evt_qm1", "video_key": "vk"}], ["main", "corner_left"]))
    for cam in ("main", "corner_left"):
        rec = field_store.load_auto_record("evt", cam)
        assert rec["status"] == "failed" and "403" in rec["reason"]


def test_runner_one_camera_failing_does_not_stop_others(app_mod, monkeypatch, tmp_path):
    monkeypatch.setattr(ac, "resolve_reference_video",
                        lambda *a, **k: ("evt_qm1", tmp_path / "evt_qm1.mp4"))
    seen_status = {}

    def fake_run(video, match_key, camera, crop, fm, versions, cfg=None, masks=(), **kw):
        seen_status[camera] = field_store.load_auto_record("evt", camera)["status"]
        if camera == "corner_left":
            raise RuntimeError("boom")
        return _fake_result()
    monkeypatch.setattr(ac, "run_camera", fake_run)
    asyncio.run(app_mod._run_auto_calibrate("evt", [], ["main", "corner_left",
                                                         "corner_right"]))
    assert seen_status == {"main": "running", "corner_left": "running",
                           "corner_right": "running"}
    assert field_store.load_auto_record("evt", "corner_left")["status"] == "failed"
    assert "boom" in field_store.load_auto_record("evt", "corner_left")["reason"]
    for cam in ("main", "corner_right"):
        rec = field_store.load_auto_record("evt", cam)
        assert rec["status"] == "ready"
        assert rec["preview"] == f"calibration_auto_evt_{cam}.jpg"
        assert field_store.auto_preview_path("evt", cam).is_file()
    assert field_store.load_versions("evt", "main") == []      # never writes versions
    assert field_store.load_auto_record("evt", "main")["frame"] == "calibration_auto_evt_main_frame.jpg"
    assert field_store.auto_frame_path("evt", "main").is_file()


def test_runner_skips_settled_unless_forced(app_mod, monkeypatch, tmp_path):
    monkeypatch.setattr(ac, "resolve_reference_video",
                        lambda *a, **k: ("evt_qm1", tmp_path / "evt_qm1.mp4"))
    ran = []
    monkeypatch.setattr(ac, "run_camera",
                        lambda video, mk, camera, *a, **k: ran.append(camera) or _fake_result())
    for st in ("ready", "verified", "confirmed"):
        ran.clear()
        rec = _ready_record(); rec["status"] = st
        field_store.save_auto_record("evt", "main", rec)
        asyncio.run(app_mod._run_auto_calibrate("evt", [], ["main"]))
        assert ran == [] and field_store.load_auto_record("evt", "main")["status"] == st
        asyncio.run(app_mod._run_auto_calibrate("evt", [], ["main"], force=True))
        assert ran == ["main"]
    ran.clear()
    rec = _ready_record(); rec["status"] = "failed"
    field_store.save_auto_record("evt", "main", rec)
    asyncio.run(app_mod._run_auto_calibrate("evt", [], ["main"]))
    assert ran == []                            # a fresh failure is not retried on load
    asyncio.run(app_mod._run_auto_calibrate("evt", [], ["main"], force=True))
    assert ran == ["main"]                      # ... but Re-run (force) does
    ran.clear()
    rec = _ready_record(); rec["status"] = "failed"; rec["created_at"] = "2020-01-01T00:00:00"
    field_store.save_auto_record("evt", "main", rec)
    asyncio.run(app_mod._run_auto_calibrate("evt", [], ["main"]))
    assert ran == ["main"]                      # an old failure is retried


def test_schedule_refuses_second_concurrent_run(app_mod, monkeypatch):
    gate = {}

    async def slow(event, matches, cameras=None, force=False):
        await gate["ev"].wait()
    monkeypatch.setattr(app_mod, "_run_auto_calibrate", slow)

    async def main():
        gate["ev"] = asyncio.Event()
        assert app_mod._schedule_auto_calibrate("evt", []) is True
        assert app_mod._auto_calib_running("evt")
        assert app_mod._schedule_auto_calibrate("evt", []) is False
        gate["ev"].set()
        await app_mod._auto_calib_tasks["evt"]
        await asyncio.sleep(0)
        assert not app_mod._auto_calib_running("evt")
    asyncio.run(main())


# ---------------------------------------------------------------------- hooks

def test_solve_tags_marks_auto_record_corrected(app_mod):
    from web.field.field_model import FieldModel
    from tests._field_synth import project_tag_corners
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    corr = project_tag_corners(cam, fm)
    auto = {str(t): corr[t][0].tolist() for t in corr}
    rec = _ready_record(); rec["status"] = "failed"
    field_store.save_auto_record("evt", "main", rec)
    c = _client(app_mod)
    body = {"event": "evt", "camera": "main", "image_size": list(cam.image_size),
            "auto_tags": auto, "manual_tags": []}
    # dry_run never resolves the record
    assert c.post("/api/calibration/solve-tags", json={**body, "dry_run": True}).status_code == 200
    assert field_store.load_auto_record("evt", "main")["status"] == "failed"
    r = c.post("/api/calibration/solve-tags", json=body)
    assert r.status_code == 200, r.text
    rec = field_store.load_auto_record("evt", "main")
    assert rec["status"] == "confirmed" and rec["reason"] == "corrected manually"
    assert rec["resolved_version"] == field_store.load_versions("evt", "main")[0]["version_id"]


def test_solve_tags_leaves_verified_record_alone(app_mod):
    from web.field.field_model import FieldModel
    from tests._field_synth import project_tag_corners
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    corr = project_tag_corners(cam, fm)
    rec = _ready_record(); rec["status"] = "verified"
    field_store.save_auto_record("evt", "main", rec)
    _client(app_mod).post("/api/calibration/solve-tags", json={
        "event": "evt", "camera": "main", "image_size": list(cam.image_size),
        "auto_tags": {str(t): corr[t][0].tolist() for t in corr}, "manual_tags": []})
    assert field_store.load_auto_record("evt", "main")["status"] == "verified"


def _patch_event_load(app_mod, monkeypatch, event_key="2099tst"):
    # The route does `from tba import ...` / `from database import ...` (web/ is on
    # sys.path when the server runs) — install fake top-level modules for both.
    import sys
    import types
    tba = types.ModuleType("tba")
    database = types.ModuleType("database")
    tba.extract_hub_scores = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "tba", tba)
    monkeypatch.setitem(sys.modules, "database", database)
    matches = [{"key": f"{event_key}_qm1", "comp_level": "qm", "match_number": 1,
                "video_key": "vk", "blue_teams": [], "red_teams": []}]
    event = {"event_key": event_key, "event_name": "Test", "year": 2099,
             "matches": matches, "teams": []}

    async def fake_load(_):
        return event

    async def fake_async(*a, **k):
        return None
    tba.load_event_data = fake_load
    for name in ("upsert_event", "upsert_match", "upsert_match_robot",
                 "get_ocr_regions", "get_hub_positions"):
        setattr(database, name, fake_async)
    monkeypatch.setattr(app_mod, "current_event", None)
    return matches


def test_event_load_schedules_auto_calibration(app_mod, monkeypatch):
    matches = _patch_event_load(app_mod, monkeypatch)
    calls = []
    monkeypatch.setattr(app_mod, "_schedule_auto_calibrate",
                        lambda ev, m, *a, **k: calls.append((ev, m)) or True)
    r = _client(app_mod).post("/api/event/load", json={"event_key": "2099tst"})
    assert r.status_code == 200, r.text
    assert calls == [("2099tst", matches)]


def test_event_load_survives_scheduler_error(app_mod, monkeypatch):
    _patch_event_load(app_mod, monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("scheduler exploded")
    monkeypatch.setattr(app_mod, "_schedule_auto_calibrate", boom)
    r = _client(app_mod).post("/api/event/load", json={"event_key": "2099tst"})
    assert r.status_code == 200 and r.json()["event_key"] == "2099tst"


def test_event_load_hook_respects_knob(app_mod, monkeypatch):
    _patch_event_load(app_mod, monkeypatch)
    monkeypatch.setattr(app_mod, "AUTO_CALIB_ON_LOAD", False)
    monkeypatch.setattr(app_mod, "_schedule_auto_calibrate",
                        lambda *a, **k: pytest.fail("must not schedule"))
    assert _client(app_mod).post("/api/event/load",
                                 json={"event_key": "2099tst"}).status_code == 200


def test_read_composite_frame_prefers_local_match_footage(app_mod, monkeypatch, tmp_path):
    import cv2
    p = tmp_path / "evt_qm3.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 1, (64, 36))
    for i in range(5):
        w.write(np.full((36, 64, 3), 40 * i, np.uint8))
    w.release()
    monkeypatch.setattr(ac, "FOOTAGE_DIRS", (tmp_path,))
    monkeypatch.setattr(app_mod, "_match_stream_url",
                        lambda mk: pytest.fail("must not resolve a stream"))
    frame, err = app_mod._read_composite_frame({"match_key": "evt_qm3", "timestamp": 2})
    assert err is None and frame.shape[:2] == (36, 64)


def test_no_calibration_message_points_to_review_when_ready(app_mod):
    field_store.save_auto_record("evt", "main", _ready_record())
    job = SimpleNamespace(event_key="evt", camera="main", match_key="evt_qm1", error=None)
    assert app_mod.InferenceJob._init_field_tracking(job) is False
    assert "awaiting confirmation at /calibrate" in job.error


# ------------------------------------------------------------- phase 2 routes

def test_auto_frame_route_404_then_jpeg(app_mod):
    c = _client(app_mod)
    assert c.get("/api/calibration/auto/frame",
                 params={"event": "evt", "camera": "main"}).status_code == 404
    rec = _ready_record(); rec["frame"] = field_store.save_auto_frame(
        "evt", "main", b"\xff\xd8\xff\xe0median")
    field_store.save_auto_record("evt", "main", rec)
    r = c.get("/api/calibration/auto/frame", params={"event": "evt", "camera": "main"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content.endswith(b"median")


def test_confirm_carries_located_manual_tags(app_mod):
    rec = _ready_record()
    rec["manual_tags"] = [{"id": 13, "u": 1699.0, "v": 320.3}]
    field_store.save_auto_record("evt", "main", rec)
    r = _client(app_mod).post("/api/calibration/auto/confirm",
                              json={"event": "evt", "camera": "main"})
    assert r.status_code == 200, r.text
    v = field_store.load_versions("evt", "main")[0]
    assert v["source"]["manual_tags"] == [{"id": 13, "u": 1699.0, "v": 320.3}]


def _occluded_video(path, n=40):
    """Background 60 with a bright 'robot' square crossing a minority of frames."""
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1, (64, 48))
    for i in range(n):
        f = np.full((48, 64, 3), 60, np.uint8)
        if i % 5 == 0:
            f[10:30, 10:30] = 250
        w.write(f)
    w.release()


def test_frame_route_aggregate_returns_median_and_caches(app_mod, monkeypatch, tmp_path):
    p = tmp_path / "evt_qm2.mp4"
    _occluded_video(p)
    monkeypatch.setattr(ac, "FOOTAGE_DIRS", (tmp_path,))
    monkeypatch.setattr(app_mod, "_calib_median_cache", {})
    body = {"match_key": "evt_qm2", "timestamp": 20, "aggregate": True}
    info = {}
    f1, err = app_mod._read_composite_frame(body, info)
    assert err is None and info["aggregated"] is True
    assert int(f1[20, 20].mean()) < 100            # the occluder is gone
    f2, _ = app_mod._read_composite_frame(body)
    assert f2 is f1                                 # served from the cache (refine-tag)
    single, _ = app_mod._read_composite_frame({"match_key": "evt_qm2", "timestamp": 20})
    assert int(single[20, 20].mean()) > 200         # frame 20 has the occluder
    r = _client(app_mod).post("/api/calibration/frame",
                              json={**body, "camera": "main"})
    assert r.status_code == 200 and r.json()["aggregated"] is True
