"""Borrowed lenses through web.app: the manual solve (``lens_from`` on
/api/calibration/solve-tags), the auto-calibration run's lender lookup + second
pass, and confirm carrying the lens source into the version. No network, no
real footage; versions live in a tmp UPLOAD_DIR (never web/uploads)."""
import asyncio
import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.field.test_lens_borrow import (HOP_LEFT, HOP_RIGHT_AUTO, HOP_RIGHT_MANUAL,
                                          HOP_RIGHT_SIZE, _lens)
from web.field import auto_calibrate as ac
from web.field import store as field_store
from web.field.calibration import CameraCalibration


@pytest.fixture
def app_mod(tmp_path, monkeypatch):
    mod = importlib.import_module("web.app")
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(mod, "_auto_calib_tasks", {})
    monkeypatch.setattr(mod, "_auto_calib_judge", lambda: None)
    monkeypatch.setattr(mod, "_auto_calib_pick_steady",
                        lambda event, camera, mk, video, *a, **k: (mk, video, [], None))
    # the overlay needs footage; these tests only look at the solve
    monkeypatch.setattr(mod, "_read_composite_frame", lambda body: (None, "no footage"))
    return mod


def _client(mod):
    return TestClient(mod.app)


def _save_left(start_match="2025hop_qm1"):
    return field_store.save_version("2025hop", "corner_left",
                                    CameraCalibration.from_json(HOP_LEFT),
                                    start_match=start_match)


def _body(**kw):
    b = {"event": "2025hop", "camera": "corner_right", "image_size": list(HOP_RIGHT_SIZE),
         "auto_tags": HOP_RIGHT_AUTO, "manual_tags": HOP_RIGHT_MANUAL,
         "start_match": "2025hop_qm1", "timestamp": 21.3, "match_key": "2025hop_qm1"}
    b.update(kw)
    return b


# ------------------------------------------------------------ manual solve

def test_solve_tags_with_lens_from_solves_pose_only_and_records_the_source(app_mod):
    vid = _save_left()
    r = _client(app_mod).post("/api/calibration/solve-tags",
                              json=_body(lens_from="corner_left"))
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["model"] == "pose"
    assert j["lens_from"] == {"camera": "corner_left", "version_id": vid}
    assert j["rms_error"] == pytest.approx(11.7, abs=0.5)
    (v,) = field_store.load_versions("2025hop", "corner_right")
    cal = v["calibration"]
    assert cal.K[0, 0] == pytest.approx(739.97, abs=0.01)
    assert (cal.K[0, 2], cal.K[1, 2]) == (458.0, 173.5)
    assert cal.camera_center[0] == pytest.approx(9.29, abs=0.1)
    assert v["source"]["lens_from"] == "corner_left"
    assert v["source"]["lens_version_id"] == vid
    assert v["source"]["manual_tags"] == HOP_RIGHT_MANUAL       # inputs still recorded


def test_solve_tags_without_lens_from_is_unchanged(app_mod):
    _save_left()
    j = _client(app_mod).post("/api/calibration/solve-tags", json=_body()).json()
    assert j["model"] != "pose" and j["lens_from"] is None
    (v,) = field_store.load_versions("2025hop", "corner_right")
    assert "lens_from" not in v["source"]


@pytest.mark.parametrize("camera,lens_from,needle", [
    ("main", "corner_left", "main"),
    ("corner_right", "main", "main"),
    ("corner_left", "corner_left", "own lens"),
])
def test_solve_tags_rejects_main_and_self(app_mod, camera, lens_from, needle):
    _save_left()
    field_store.save_version("2025hop", "main", CameraCalibration.from_json(HOP_LEFT))
    r = _client(app_mod).post("/api/calibration/solve-tags",
                              json=_body(camera=camera, lens_from=lens_from))
    assert r.status_code == 400 and needle in r.json()["error"]
    assert field_store.load_versions("2025hop", "corner_right") == []


def test_solve_tags_lens_from_needs_a_lender_version(app_mod):
    r = _client(app_mod).post("/api/calibration/solve-tags",
                              json=_body(lens_from="corner_left"))
    assert r.status_code == 400 and "no calibration version" in r.json()["error"]


def test_solve_tags_lens_version_pin(app_mod):
    """A re-preview pins the exact lender version the saved epoch used."""
    old = _save_left(start_match=None)
    _save_left()
    j = _client(app_mod).post("/api/calibration/solve-tags", json=_body(
        lens_from="corner_left", lens_version_id=old, dry_run=True)).json()
    assert j["lens_from"]["version_id"] == old
    r = _client(app_mod).post("/api/calibration/solve-tags", json=_body(
        lens_from="corner_left", lens_version_id="nope", dry_run=True))
    assert r.status_code == 400


# ------------------------------------------------------- auto-calibration run

def _ref(monkeypatch, tmp_path):
    monkeypatch.setattr(ac, "resolve_reference_video",
                        lambda *a, **k: ("2025hop_qm1", tmp_path / "2025hop_qm1.mp4"))
    monkeypatch.setattr(ac, "ensure_event_layout", lambda *a, **k: None)


def _failed(n_tags=2, **kw):
    return ac.AutoCalibResult(record=ac.new_record(
        "failed", "only 2 known tags", method="full_solve",
        metrics={"n_tags": n_tags}, **kw), calibration=None, overlay=None)


def _ready(cal=None):
    cal = cal or _lens(f=740.0, size=(965, 347))
    return ac.AutoCalibResult(record=ac.new_record(
        "ready", "solved", method="full_solve", calibration=cal.to_json(),
        metrics={"n_tags": 6}), calibration=cal, overlay=None)


def test_auto_run_retries_a_failed_corner_once_a_sibling_is_ready(app_mod, monkeypatch,
                                                                  tmp_path):
    _ref(monkeypatch, tmp_path)
    calls = []

    def fake_run(video, mk, cam, crop, fm, versions, cfg=None, lenders=(), **kw):
        calls.append((cam, [(l.camera, l.kind) for l in lenders]))
        if cam == "corner_left":
            return _ready()
        if lenders:
            return _failed(lens_from=lenders[0].camera, lens_lenders_tried=["corner_left"])
        return _failed()
    monkeypatch.setattr(ac, "run_camera", fake_run)
    asyncio.run(app_mod._run_auto_calibrate("2025hop", [], ["corner_right", "corner_left"]))
    assert calls == [("corner_right", []), ("corner_left", []),
                     ("corner_right", [("corner_left", "auto_record")])]
    rec = field_store.load_auto_record("2025hop", "corner_right")
    assert rec["lens_from"] == "corner_left"


def test_auto_run_no_retry_when_the_corner_solves_itself(app_mod, monkeypatch, tmp_path):
    _ref(monkeypatch, tmp_path)
    calls = []

    def fake_run(video, mk, cam, crop, fm, versions, cfg=None, lenders=(), **kw):
        calls.append(cam)
        return _ready()
    monkeypatch.setattr(ac, "run_camera", fake_run)
    asyncio.run(app_mod._run_auto_calibrate("2025hop", [], ["corner_right", "corner_left"]))
    assert calls == ["corner_right", "corner_left"]


def test_auto_run_passes_a_versioned_sibling_and_never_main(app_mod, monkeypatch, tmp_path):
    _ref(monkeypatch, tmp_path)
    vid = _save_left()
    field_store.save_version("2025hop", "main", CameraCalibration.from_json(HOP_LEFT))
    seen = {}

    def fake_run(video, mk, cam, crop, fm, versions, cfg=None, lenders=(), **kw):
        seen.setdefault(cam, []).append([(l.camera, l.version_id) for l in lenders])
        return _failed(lens_lenders_tried=[l.camera for l in lenders])
    monkeypatch.setattr(ac, "run_camera", fake_run)
    asyncio.run(app_mod._run_auto_calibrate("2025hop", [], ["main", "corner_right"]))
    assert seen == {"main": [[]], "corner_right": [[("corner_left", vid)]]}


def test_confirm_carries_the_borrowed_lens_into_the_version(app_mod):
    cal = _lens(f=740.0, size=(916, 347))
    cal.rvec, cal.tvec = np.zeros((3, 1)), np.array([[0.0], [0.0], [5.0]])
    field_store.save_auto_record("2025hop", "corner_right", ac.new_record(
        "ready", "lens borrowed", method="full_solve", match_key="2025hop_qm1",
        calibration=cal.to_json(), lens_from="corner_left", lens_version_id="L1",
        lens_kind="version"))
    r = _client(app_mod).post("/api/calibration/auto/confirm",
                              json={"event": "2025hop", "camera": "corner_right"})
    assert r.status_code == 200, r.text
    (v,) = field_store.load_versions("2025hop", "corner_right")
    assert v["source"]["lens_from"] == "corner_left"
    assert v["source"]["lens_version_id"] == "L1"
