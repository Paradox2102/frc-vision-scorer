"""Per-event broadcast layout inside auto-calibration: the layout step (adopt vs
kept_default), cameras missing from the event's broadcast, source.crop on saved
versions, and the /api/calibration/layout routes. No network, no real video."""
import asyncio
import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from web.field import auto_calibrate as ac
from web.field import composite_layout as cl
from web.field import store as field_store
from tests._field_synth import truth_pinhole_camera

DEFAULT = cl.load_layout()          # global: main (0,0,1920,723), corners y=703

DETECTED_SHIFTED = {               # like 2026cur: corners detected at y=743
    "frame_size": [1920, 1080],
    "cameras": {"main": {"crop": [0, 0, 1920, 723], "masks": []},
                "corner_left": {"crop": [0, 743, 960, 337], "masks": []},
                "corner_right": {"crop": [960, 743, 960, 337], "masks": []}},
    "confidence": {"main": 0.9}, "diag": {}}
DETECTED_HIHO = {                  # main only at the top, scoreboard below
    "frame_size": [1920, 1080],
    "cameras": {"main": {"crop": [0, 0, 1920, 460], "masks": [[10, 10, 50, 20]]}},
    "confidence": {"main": 0.8}, "diag": {}}


def _detector(layout):
    return lambda video: (layout, np.full((1080, 1920, 3), 90, np.uint8))


def _version(event, cam, crop=None):
    src = {"crop": list(crop)} if crop else None
    return field_store.save_version(event, cam, truth_pinhole_camera(), source=src)


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    return tmp_path


# ------------------------------------------------------------- layout step

def test_layout_step_adopts_detection_when_no_calibrations(tmp_store):
    rec = ac.ensure_event_layout("evtH", "v.mp4", detector=_detector(DETECTED_HIHO),
                                 log=lambda m: None)
    assert rec["status"] == "auto" and rec["source"] == "detect"
    lay = cl.load_layout(event_key="evtH")
    assert set(lay) == {"main"} and lay["main"].crop == (0, 0, 1920, 460)
    assert lay["main"].masks == ((10, 10, 50, 20),)
    assert cl.event_layout_preview_path("evtH").is_file()
    assert ac.layout_frame_path("evtH").is_file()


def test_layout_step_keeps_default_under_existing_calibrations(tmp_store):
    _version("evtC", "corner_left")               # solved in the default crop (no source.crop)
    rec = ac.ensure_event_layout("evtC", "v.mp4", detector=_detector(DETECTED_SHIFTED),
                                 log=lambda m: None)
    assert rec["status"] == "kept_default"
    assert "corner_left y 703 vs new 743" in rec["reason"]
    assert rec["detected"]["corner_left"]["crop"] == [0, 743, 960, 337]
    lay = cl.load_layout(event_key="evtC")
    assert lay["corner_left"].crop == DEFAULT["corner_left"].crop      # global honoured


def test_layout_step_uses_recorded_source_crop(tmp_store):
    _version("evtR", "corner_left", crop=(0, 743, 960, 337))   # solved at the new origin
    rec = ac.ensure_event_layout("evtR", "v.mp4", detector=_detector(DETECTED_SHIFTED),
                                 log=lambda m: None)
    assert rec["status"] == "auto"


def test_layout_step_small_origin_change_is_tolerated(tmp_store):
    _version("evtT", "main")
    det = {"frame_size": [1920, 1080],
           "cameras": {"main": {"crop": [3, 2, 1915, 720], "masks": []}}}
    assert ac.ensure_event_layout("evtT", "v.mp4", detector=_detector(det),
                                  log=lambda m: None)["status"] == "auto"


def test_layout_step_runs_once_unless_forced(tmp_store):
    calls = []

    def det(video):
        calls.append(video)
        return DETECTED_HIHO, None
    ac.ensure_event_layout("evtO", "a.mp4", detector=det, log=lambda m: None)
    ac.ensure_event_layout("evtO", "b.mp4", detector=det, log=lambda m: None)
    assert calls == ["a.mp4"]
    ac.ensure_event_layout("evtO", "c.mp4", detector=det, force=True, log=lambda m: None)
    assert calls == ["a.mp4", "c.mp4"]


def test_layout_step_without_main_is_rejected(tmp_store):
    det = {"frame_size": [1920, 1080],
           "cameras": {"corner_left": {"crop": [0, 600, 900, 400], "masks": []}}}
    rec = ac.ensure_event_layout("evtN", "v.mp4", detector=_detector(det), log=lambda m: None)
    assert rec["status"] == "rejected"
    assert cl.load_layout(event_key="evtN")["main"].crop == DEFAULT["main"].crop


def test_layout_conflicts_reports_missing_camera():
    new = cl.layout_to_cameras(DETECTED_HIHO)
    out = ac.layout_conflicts(new, {"corner_left": [{"source": None}]}, DEFAULT)
    assert out and "no corner_left view" in out[0]


# ------------------------------------------------------------------ app side

@pytest.fixture
def app_mod(tmp_store, monkeypatch):
    mod = importlib.import_module("web.app")
    monkeypatch.setattr(mod, "_auto_calib_tasks", {})
    monkeypatch.setattr(mod, "_auto_calib_judge", lambda: None)
    return mod


def _client(mod):
    return TestClient(mod.app)


def test_runner_skips_cameras_missing_from_event_layout(app_mod, monkeypatch, tmp_store):
    monkeypatch.setattr(ac, "resolve_reference_video",
                        lambda *a, **k: ("evtH_qm1", tmp_store / "evtH_qm1.mp4"))
    monkeypatch.setattr(ac, "ensure_event_layout",
                        lambda event, video, **k: cl.save_event_layout(event, DETECTED_HIHO))
    seen = {}

    def fake_run(video, match_key, camera, crop, fm, versions, cfg=None, masks=(), **kw):
        seen[camera] = tuple(crop)
        rec = ac.new_record("ready", "ok", match_key=match_key, method="full_solve",
                            calibration=truth_pinhole_camera().to_json())
        return ac.AutoCalibResult(record=rec, calibration=None, overlay=None)
    monkeypatch.setattr(ac, "run_camera", fake_run)
    asyncio.run(app_mod._run_auto_calibrate("evtH", []))
    assert seen == {"main": (0, 0, 1920, 460)}               # the event's own crop
    assert field_store.load_auto_record("evtH", "main")["crop"] == [0, 0, 1920, 460]
    for cam in ("corner_left", "corner_right"):
        rec = field_store.load_auto_record("evtH", cam)
        assert rec["status"] == "failed"
        assert rec["reason"] == "camera not present in this event's broadcast layout"


def test_runner_layout_failure_falls_back_to_default(app_mod, monkeypatch, tmp_store):
    monkeypatch.setattr(ac, "resolve_reference_video",
                        lambda *a, **k: ("evtF_qm1", tmp_store / "x.mp4"))

    def boom(*a, **k):
        raise RuntimeError("no frames")
    monkeypatch.setattr(ac, "ensure_event_layout", boom)
    seen = []
    monkeypatch.setattr(ac, "run_camera", lambda video, mk, camera, *a, **k: seen.append(camera)
                        or ac.AutoCalibResult(ac.new_record("ready", "ok"), None, None))
    asyncio.run(app_mod._run_auto_calibrate("evtF", [], ["main"]))
    assert seen == ["main"]


def test_solve_tags_records_crop_of_event_layout(app_mod):
    from web.field.field_model import FieldModel
    from tests._field_synth import project_tag_corners
    cl.save_event_layout("evtS", DETECTED_HIHO)
    cam = truth_pinhole_camera()
    corr = project_tag_corners(cam, FieldModel.load("2026-rebuilt-welded"))
    r = _client(app_mod).post("/api/calibration/solve-tags", json={
        "event": "evtS", "camera": "main", "image_size": list(cam.image_size),
        "auto_tags": {str(t): corr[t][0].tolist() for t in corr}, "manual_tags": []})
    assert r.status_code == 200, r.text
    v = field_store.load_versions("evtS", "main")[0]
    assert v["source"]["crop"] == [0, 0, 1920, 460]


def test_masks_route_new_version_records_default_crop(app_mod):
    r = _client(app_mod).post("/api/calibration/masks", json={
        "event": "evtM", "camera": "corner_left", "masks": [[1, 2, 3, 4]]})
    assert r.status_code == 200, r.text
    v = field_store.load_versions("evtM", "corner_left")[0]
    assert v["source"]["crop"] == list(DEFAULT["corner_left"].crop)


def test_confirm_records_run_crop_and_promotes_auto_layout(app_mod):
    cl.save_event_layout("evtP", DETECTED_HIHO, status="auto")
    rec = ac.new_record("ready", "ok", match_key="evtP_qm1", method="full_solve",
                        calibration=truth_pinhole_camera().to_json(), crop=[0, 0, 1920, 460])
    field_store.save_auto_record("evtP", "main", rec)
    r = _client(app_mod).post("/api/calibration/auto/confirm",
                              json={"event": "evtP", "camera": "main"})
    assert r.status_code == 200, r.text
    assert field_store.load_versions("evtP", "main")[0]["source"]["crop"] == [0, 0, 1920, 460]
    assert cl.load_event_layout_record("evtP")["status"] == "confirmed"
    assert cl.load_layout(event_key="evtP")["main"].crop == (0, 0, 1920, 460)


# --------------------------------------------------------------- layout routes

def test_get_layout_without_record_returns_default(app_mod):
    j = _client(app_mod).get("/api/calibration/layout", params={"event": "evtX"}).json()
    assert j["record"] is None and j["has_preview"] is False
    assert j["effective"]["cameras"]["main"]["crop"] == list(DEFAULT["main"].crop)
    assert j["frame_size"] == [1920, 1080]
    assert _client(app_mod).get("/api/calibration/layout/preview",
                                params={"event": "evtX"}).status_code == 404


def test_get_layout_with_record_and_preview(app_mod):
    ac.ensure_event_layout("evtG", "v.mp4", detector=_detector(DETECTED_HIHO), log=lambda m: None)
    c = _client(app_mod)
    j = c.get("/api/calibration/layout", params={"event": "evtG"}).json()
    assert j["record"]["status"] == "auto" and "diag" not in j["record"]
    assert j["effective"]["cameras"] == {"main": {"crop": [0, 0, 1920, 460],
                                                  "masks": [[10, 10, 50, 20]]}}
    r = c.get("/api/calibration/layout/preview", params={"event": "evtG"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_post_layout_confirms_and_refuses_origin_change(app_mod):
    c = _client(app_mod)
    _version("evtL", "main")                                   # default main origin (0,0)
    ok = {"main": {"crop": [0, 0, 1920, 700], "masks": []},
          "corner_left": {"crop": [0, 703, 960, 377]}}
    r = c.post("/api/calibration/layout", json={"event": "evtL", "cameras": ok})
    assert r.status_code == 200, r.text
    assert r.json()["record"]["status"] == "confirmed"
    assert cl.load_layout(event_key="evtL")["main"].crop == (0, 0, 1920, 700)
    moved = {"main": {"crop": [0, 40, 1920, 700]}}
    r = c.post("/api/calibration/layout", json={"event": "evtL", "cameras": moved})
    assert r.status_code == 409 and "main y 0 vs new 40" in r.json()["error"]
    assert cl.load_layout(event_key="evtL")["main"].crop == (0, 0, 1920, 700)
    r = c.post("/api/calibration/layout", json={"event": "evtL", "cameras": moved,
                                                 "force": True})
    assert r.status_code == 200
    assert cl.load_layout(event_key="evtL")["main"].crop == (0, 40, 1920, 700)


def test_post_layout_validation(app_mod):
    c = _client(app_mod)
    assert c.post("/api/calibration/layout", json={"event": "e"}).status_code == 400
    assert c.post("/api/calibration/layout", json={
        "event": "e", "cameras": {"corner_left": {"crop": [0, 0, 10, 10]}}}).status_code == 400
    assert c.post("/api/calibration/layout", json={
        "event": "e", "cameras": {"main": {"crop": [0, 0, -1, 10]}}}).status_code == 400


def test_redetect_route_forces_detection(app_mod, monkeypatch, tmp_store):
    cl.save_event_layout("evtD", DETECTED_HIHO, status="confirmed", source="manual")
    monkeypatch.setattr(ac, "resolve_reference_video",
                        lambda *a, **k: ("evtD_qm1", tmp_store / "evtD_qm1.mp4"))
    monkeypatch.setattr(ac, "detect_with_preview", _detector(DETECTED_SHIFTED))
    r = _client(app_mod).post("/api/calibration/layout/redetect",
                              json={"event": "evtD", "replace_confirmed": True})
    assert r.status_code == 200, r.text
    assert r.json()["record"]["status"] == "auto"
    assert cl.load_layout(event_key="evtD")["corner_left"].crop == (0, 743, 960, 337)


# ------------------------------------------------- review fixes (crop safety)

MAIN_Y40 = {"frame_size": [1920, 1080],
            "cameras": {"main": {"crop": [0, 40, 1920, 680], "masks": []}}}
MAIN_Y0 = {"frame_size": [1920, 1080],
           "cameras": {"main": {"crop": [0, 0, 1920, 700], "masks": []}}}


def test_redetect_keeps_current_event_layout_under_its_calibrations(tmp_store):
    cl.save_event_layout("evtK", MAIN_Y40, status="auto")
    _version("evtK", "main", crop=(0, 40, 1920, 680))        # solved in the event crop
    rec = ac.ensure_event_layout("evtK", "v.mp4", detector=_detector(MAIN_Y0), force=True,
                                 log=lambda m: None)
    assert rec["status"] == "auto"                            # NOT kept_default/global
    assert "current layout kept" in rec["reason"] and "main y 40 vs new 0" in rec["reason"]
    assert rec["detected"]["main"]["crop"] == [0, 0, 1920, 700]
    assert cl.load_layout(event_key="evtK")["main"].crop == (0, 40, 1920, 680)


def test_redetect_never_replaces_confirmed_layout_unless_asked(tmp_store):
    cl.save_event_layout("evtQ", MAIN_Y40, status="confirmed", source="manual")
    rec = ac.ensure_event_layout("evtQ", "v.mp4", detector=_detector(MAIN_Y0), force=True,
                                 log=lambda m: None)
    assert rec["status"] == "confirmed" and rec["source"] == "manual"
    assert cl.load_layout(event_key="evtQ")["main"].crop == (0, 40, 1920, 680)
    rec = ac.ensure_event_layout("evtQ", "v.mp4", detector=_detector(MAIN_Y0), force=True,
                                 replace_confirmed=True, log=lambda m: None)
    assert rec["status"] == "auto"
    assert cl.load_layout(event_key="evtQ")["main"].crop == (0, 0, 1920, 700)


def test_adopting_moved_layout_invalidates_ready_candidates(tmp_store):
    cl.save_event_layout("evtV", MAIN_Y40, status="auto")
    field_store.save_auto_record("evtV", "main", ac.new_record("ready", "ok", crop=[0, 40, 1920, 680]))
    field_store.save_auto_record("evtV", "corner_left", ac.new_record("verified", "ok"))
    ac.ensure_event_layout("evtV", "v.mp4", detector=_detector(MAIN_Y0), force=True,
                           log=lambda m: None)
    r = field_store.load_auto_record("evtV", "main")
    assert r["status"] == "failed" and r["reason"] == "layout changed — re-run"
    assert field_store.load_auto_record("evtV", "corner_left")["status"] == "verified"


def test_confirm_refuses_candidate_solved_in_another_crop(app_mod):
    cl.save_event_layout("evtZ", MAIN_Y40, status="confirmed", source="manual")
    rec = ac.new_record("ready", "ok", match_key="evtZ_qm1", method="full_solve",
                        calibration=truth_pinhole_camera().to_json(), crop=[0, 0, 1920, 700])
    field_store.save_auto_record("evtZ", "main", rec)
    c = _client(app_mod)
    r = c.post("/api/calibration/auto/confirm", json={"event": "evtZ", "camera": "main"})
    assert r.status_code == 409 and "layout changed" in r.json()["error"]
    assert field_store.load_versions("evtZ", "main") == []
    rec["crop"] = [2, 41, 1916, 679]                           # within tolerance
    field_store.save_auto_record("evtZ", "main", rec)
    r = c.post("/api/calibration/auto/confirm", json={"event": "evtZ", "camera": "main"})
    assert r.status_code == 200, r.text


def test_layout_post_invalidates_ready_and_refuses_while_running(app_mod, monkeypatch):
    from types import SimpleNamespace
    field_store.save_auto_record("evtW", "main", ac.new_record("ready", "ok", crop=[0, 0, 1920, 723]))
    c = _client(app_mod)
    monkeypatch.setitem(app_mod._auto_calib_tasks, "evtW", SimpleNamespace(done=lambda: False))
    r = c.post("/api/calibration/layout", json={"event": "evtW", "cameras": MAIN_Y40["cameras"]})
    assert r.status_code == 409
    monkeypatch.delitem(app_mod._auto_calib_tasks, "evtW")
    r = c.post("/api/calibration/layout", json={"event": "evtW", "cameras": MAIN_Y40["cameras"]})
    assert r.status_code == 200, r.text
    rec = field_store.load_auto_record("evtW", "main")
    assert rec["status"] == "failed" and rec["reason"] == "layout changed — re-run"


def test_layout_post_bad_event_is_400_not_500(app_mod):
    r = _client(app_mod).post("/api/calibration/layout",
                              json={"event": "!!!", "cameras": MAIN_Y0["cameras"]})
    assert r.status_code == 400


def test_redetect_route_respects_confirmed_unless_replace(app_mod, monkeypatch, tmp_store):
    cl.save_event_layout("evtE", MAIN_Y40, status="confirmed", source="manual")
    monkeypatch.setattr(ac, "resolve_reference_video",
                        lambda *a, **k: ("evtE_qm1", tmp_store / "evtE_qm1.mp4"))
    monkeypatch.setattr(ac, "detect_with_preview", _detector(MAIN_Y0))
    c = _client(app_mod)
    j = c.post("/api/calibration/layout/redetect", json={"event": "evtE"}).json()
    assert j["record"]["status"] == "confirmed"
    assert j["effective"]["cameras"]["main"]["crop"] == [0, 40, 1920, 680]
    j = c.post("/api/calibration/layout/redetect",
               json={"event": "evtE", "replace_confirmed": True}).json()
    assert j["effective"]["cameras"]["main"]["crop"] == [0, 0, 1920, 700]


def test_skip_reason_ages_failed_records():
    now = 1_000_000_000.0
    fresh = ac.new_record("failed", "x")
    fresh["created_at"] = __import__("time").strftime("%Y-%m-%dT%H:%M:%S",
                                                      __import__("time").localtime(now - 3600))
    old = dict(fresh, created_at=__import__("time").strftime(
        "%Y-%m-%dT%H:%M:%S", __import__("time").localtime(now - 7 * 3600)))
    assert ac.skip_reason(fresh, False, now) and "failed" in ac.skip_reason(fresh, False, now)
    assert ac.skip_reason(old, False, now) is None
    assert ac.skip_reason(fresh, True, now) is None
    assert ac.skip_reason(ac.new_record("ready"), False, now) == "ready"
    assert ac.skip_reason(None, False, now) is None
    assert ac.skip_reason(dict(fresh, created_at="garbage"), False, now) is None


def test_atomic_writers_leave_no_temp_files(tmp_store):
    import threading
    def w(i):
        field_store.save_auto_record("evtA", "main", ac.new_record("running", str(i)))
        cl.save_event_layout("evtA", MAIN_Y0)
    ts = [threading.Thread(target=w, args=(i,)) for i in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not list(tmp_store.glob("*.tmp"))
    assert field_store.load_auto_record("evtA", "main")["status"] == "running"
    assert cl.load_event_layout_record("evtA")["cameras"]["main"]["crop"] == [0, 0, 1920, 700]
