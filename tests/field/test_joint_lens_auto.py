"""Auto-calibration and the shared corner lens: fold_ratio on every record (a note,
never a status change) and the joint-lens pass for corners that fail / fold on
their own. Synthetic footage (a black mp4 + injected detectors) as in
test_lens_borrow; versions / records in a tmp UPLOAD_DIR (never web/uploads)."""
import asyncio

import cv2
import numpy as np
import pytest

from tests._field_synth import project_tag_corners, truth_pinhole_camera
from tests.field.hop2025_fixtures import (HOP_LEFT, HOP_LEFT_SOURCE, HOP_RIGHT_OWN,
                                          HOP_RIGHT_SOURCE, corr_from_source)
from tests.field.test_lens_borrow import FM, GOOD_D, CROP, _detector, _video
from tests.field.test_lens_borrow_api import app_mod  # noqa: F401 (fixture)
from web.field import auto_calibrate as ac
from web.field import calibration as cmod
from web.field import store as field_store
from web.field.calibration import CalibrationError, CameraCalibration

SEASON26 = "2026-rebuilt-welded"
LENS_K = np.array([[1000., 0, 960], [0, 1000., 540], [0, 0, 1]])
LENS_D = np.array([-0.15, 0, 0, 0, 0])
# two corner cameras, same lens, each seeing 4 tags of the red hub (one structure:
# too little spread for its own lens)
CORNERS = {"corner_left": ((16.2, 8.4, 0.7), [9, 10, 11, 2]),
           "corner_right": ((16.2, -0.4, 0.7), [9, 10, 5, 8])}


# ------------------------------------------------------------ fold_ratio notes

def test_fold_note_on_the_real_hop_solves():
    ratio, note = ac.fold_note(CameraCalibration.from_json(HOP_RIGHT_OWN),
                               corr_from_source(HOP_RIGHT_SOURCE))
    assert ratio == pytest.approx(0.856, abs=0.01)
    assert "tag 2" in note and "over-fitted" in note
    ratio, note = ac.fold_note(CameraCalibration.from_json(HOP_LEFT),
                               corr_from_source(HOP_LEFT_SOURCE))
    assert ratio == pytest.approx(0.62, abs=0.01) and note is None
    _r, fixed = ac.fold_note(CameraCalibration.from_json(HOP_RIGHT_OWN),
                             corr_from_source(HOP_RIGHT_SOURCE), lens_fixed=True)
    assert "over-fitted" not in fixed


def test_run_camera_records_fold_ratio_and_a_note_never_a_status_change(tmp_path,
                                                                        monkeypatch):
    cam = truth_pinhole_camera(D=GOOD_D)
    video = _video(tmp_path)
    res = ac.run_camera(video, "evt_qm1", "corner_right", CROP, FM, versions=[],
                        detect_fn=_detector(cam))
    assert res.record["status"] == "ready"
    ratio = res.record["metrics"]["fold_ratio"]
    assert ratio is not None and ratio < cmod.FOLD_RATIO_WARN
    assert "distortion limit" not in res.record["reason"]
    monkeypatch.setattr(cmod, "FOLD_RATIO_WARN", 0.0)          # everything "near fold"
    res = ac.run_camera(video, "evt_qm1", "corner_right", CROP, FM, versions=[],
                        detect_fn=_detector(cam))
    assert res.record["status"] == "ready"                     # validation decides
    assert "distortion limit" in res.record["reason"]


# ------------------------------------------------------------ joint candidates

def _rec(status="failed", n_tags=4, fold=None, **kw):
    return ac.new_record(status, "own", method=kw.pop("method", "full_solve"),
                         metrics={"n_tags": n_tags, "fold_ratio": fold}, **kw)


def test_joint_candidates():
    assert ac.joint_candidate(_rec(), "corner_left")
    assert ac.joint_candidate(_rec("ready", fold=0.9), "corner_left")
    assert not ac.joint_candidate(_rec("ready", fold=0.5), "corner_left")
    assert not ac.joint_candidate(_rec("ready", fold=0.9, lens_from="x"), "corner_left")
    assert not ac.joint_candidate(_rec(), "main")
    assert not ac.joint_candidate(_rec(n_tags=1), "corner_left")
    assert not ac.joint_candidate(_rec(method="pose_resolve"), "corner_left")
    assert not ac.joint_candidate(_rec("verified", method="verify"), "corner_left")
    assert not ac.joint_candidate(None, "corner_left")


# ------------------------------------------------------------------ run_joint

def _joint_inputs(tmp_path, visible=None):
    video = _video(tmp_path)
    cams, cfgs, truth = {}, {}, {}
    for name, (eye, ids) in CORNERS.items():
        cam = truth_pinhole_camera(eye=eye, target=(8, 4, 0), K=LENS_K, D=LENS_D)
        c = project_tag_corners(cam, FM)
        dets = {t: c[t][0] for t in ids}
        shown = {t: dets[t] for t in (visible or ids)}
        rec = ac.new_record("failed", "own lens: tags only span 1.1 m", method="full_solve",
                            match_key="evt_qm1", metrics={"n_tags": len(ids)},
                            auto_tags={str(t): v.round(2).tolist() for t, v in dets.items()},
                            crop=[0, 0, 1920, 1080])
        cams[name] = {"record": rec, "video": video, "median": None,
                      "detect_fn": (lambda d: lambda f: {k: v.copy() for k, v in d.items()})(shown)}
        cfgs[name] = ac.AutoCalibConfig(field_season=SEASON26, drift_max_m=0.3)
        truth[name] = cam
    return cams, cfgs, truth


def test_run_joint_recovers_the_shared_lens_and_validates(tmp_path):
    cams, cfgs, truth = _joint_inputs(tmp_path)
    out = ac.run_joint(cams, FM, cfgs)
    for name, res in out.items():
        rec = res.record
        assert rec["status"] == "ready", rec["reason"]
        assert rec["method"] == ac.JOINT_METHOD and rec["lens_kind"] == "joint"
        assert rec["lens_partners"] == [c for c in CORNERS if c != name]
        assert rec["joint"]["f_px"] == pytest.approx(1000, rel=0.01)
        assert np.linalg.norm(res.calibration.camera_center
                              - truth[name].camera_center) < 0.05
        assert rec["metrics"]["fold_ratio"] < cmod.FOLD_RATIO_WARN
        assert rec["crop"] == [0, 0, 1920, 1080]
        assert "joint lens with" in rec["reason"]


def test_run_joint_stays_honest_when_it_cannot_validate(tmp_path):
    cams, cfgs, _ = _joint_inputs(tmp_path, visible=[9, 10])   # < min_tags per frame
    out = ac.run_joint(cams, FM, cfgs)
    for res in out.values():
        assert res.record["status"] == "failed"
        assert "does not validate" in res.record["reason"]
        assert res.record["calibration"] is not None          # candidate for review


def test_run_joint_needs_two_tags_per_camera(tmp_path):
    cams, cfgs, _ = _joint_inputs(tmp_path)
    cams["corner_left"]["record"]["auto_tags"] = {"9": cams["corner_left"]["record"]
                                                  ["auto_tags"]["9"]}
    with pytest.raises(CalibrationError):
        ac.run_joint(cams, FM, cfgs)


def test_adopt_joint():
    own_failed = _rec("failed")
    own_ready = _rec("ready", fold=0.9)
    ready = ac.new_record("ready", "joint ok", method=ac.JOINT_METHOD, lens_kind="joint")
    failed = ac.new_record("failed", "joint bad", method=ac.JOINT_METHOD,
                           calibration={"x": 1}, joint={"f_px": 700.0})
    out = ac.adopt_joint(own_failed, ready)
    assert out["status"] == "ready" and out["method"] == ac.JOINT_METHOD
    assert out["own_status"] == "failed" and out["own_lens_reason"] == "own"
    out = ac.adopt_joint(own_failed, failed)
    assert out["status"] == "failed" and out["method"] == "full_solve"
    assert "joint bad" in out["reason"] and out["joint_attempt"]["f_px"] == 700.0
    out = ac.adopt_joint(own_ready, failed)
    assert out["status"] == "ready" and out["method"] == "full_solve"


# --------------------------------------------------- the run (web.app wiring)

def _ref(monkeypatch, tmp_path):
    monkeypatch.setattr(ac, "resolve_reference_video",
                        lambda *a, **k: ("2025hop_qm1", tmp_path / "2025hop_qm1.mp4"))
    monkeypatch.setattr(ac, "ensure_event_layout", lambda *a, **k: None)


def _own(status, **kw):
    return ac.AutoCalibResult(record=_rec(status, auto_tags={"1": [[0, 0]] * 4,
                                                             "2": [[0, 0]] * 4},
                                          match_key="2025hop_qm1", **kw),
                              calibration=None, overlay=None)


def test_auto_run_joint_pass_over_failed_corners(app_mod, monkeypatch, tmp_path):
    _ref(monkeypatch, tmp_path)
    monkeypatch.setattr(ac, "run_camera",
                        lambda video, mk, cam, *a, **k: _own("failed"))
    seen = []

    def fake_joint(cams, fm, cfgs, detect_fn=None):
        seen.append(sorted(cams))
        assert all(c["record"]["crop"] for c in cams.values())
        return {c: ac.AutoCalibResult(record=ac.new_record(
            "ready", "joint ok", method=ac.JOINT_METHOD, lens_kind="joint",
            lens_partner=[o for o in cams if o != c][0]), calibration=None, overlay=None)
            for c in cams}
    monkeypatch.setattr(ac, "run_joint", fake_joint)
    asyncio.run(app_mod._run_auto_calibrate("2025hop", [],
                                            ["main", "corner_left", "corner_right"]))
    assert seen == [["corner_left", "corner_right"]]          # never main
    for cam in ("corner_left", "corner_right"):
        rec = field_store.load_auto_record("2025hop", cam)
        assert rec["status"] == "ready" and rec["method"] == ac.JOINT_METHOD
        assert rec["own_status"] == "failed"
    assert field_store.load_auto_record("2025hop", "main")["method"] == "full_solve"


def test_auto_run_no_joint_pass_for_a_single_failed_corner(app_mod, monkeypatch, tmp_path):
    _ref(monkeypatch, tmp_path)
    ready = ac.AutoCalibResult(record=_rec("ready", fold=0.3), calibration=None,
                               overlay=None)
    monkeypatch.setattr(ac, "run_camera", lambda video, mk, cam, *a, **k:
                        _own("failed") if cam == "corner_right" else ready)
    monkeypatch.setattr(ac, "run_joint", lambda *a, **k: pytest.fail("no joint pass"))
    asyncio.run(app_mod._run_auto_calibrate("2025hop", [], ["corner_left", "corner_right"]))


def test_auto_run_joint_failure_is_noted_and_status_kept(app_mod, monkeypatch, tmp_path):
    _ref(monkeypatch, tmp_path)
    monkeypatch.setattr(ac, "run_camera", lambda video, mk, cam, *a, **k:
                        _own("failed") if cam == "corner_right" else _own("ready", fold=0.9))

    def boom(*a, **k):
        raise CalibrationError("no start could be posed")
    monkeypatch.setattr(ac, "run_joint", boom)
    asyncio.run(app_mod._run_auto_calibrate("2025hop", [], ["corner_left", "corner_right"]))
    left = field_store.load_auto_record("2025hop", "corner_left")
    right = field_store.load_auto_record("2025hop", "corner_right")
    assert left["status"] == "ready" and right["status"] == "failed"
    assert "joint lens with corner_right failed" in left["reason"]


def test_confirm_carries_the_joint_lens_into_the_version(app_mod):
    cal = CameraCalibration.from_json(HOP_LEFT)
    field_store.save_auto_record("2025hop", "corner_left", ac.new_record(
        "ready", "joint", method=ac.JOINT_METHOD, match_key="2025hop_qm1",
        calibration=cal.to_json(), lens_kind="joint", lens_partner="corner_right",
        lens_partners=["corner_right"], joint={"f_px": 692.0, "k1": -0.15}))
    from fastapi.testclient import TestClient
    r = TestClient(app_mod.app).post("/api/calibration/auto/confirm",
                                     json={"event": "2025hop", "camera": "corner_left"})
    assert r.status_code == 200, r.text
    (v,) = field_store.load_versions("2025hop", "corner_left")
    assert v["source"]["lens_kind"] == "joint"
    assert v["source"]["lens_partner"] == "corner_right"
    assert v["source"]["joint"]["f_px"] == 692.0
