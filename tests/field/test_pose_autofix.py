"""Auto pose re-solve: pose recovery against fixed intrinsics + guardrail decisions.

All synthetic — no video/GPU. The video-glue layer (resolve_pose_from_video) is a
thin wrapper over these tested units plus OpenCV frame IO.
"""
import numpy as np
import pytest

from web.field.calibration import CameraCalibration, CalibrationError
from web.field.field_model import FieldModel
from web.field.pose_autofix import (PoseFixConfig, resolve_pose, evaluate_fix,
                                     _pose_deltas)
from tests._field_synth import (truth_camera, truth_pinhole_camera,
                                 project_tag_corners)

FM = FieldModel.load("2026-rebuilt-welded")


def test_resolve_pose_recovers_pivoted_pinhole_camera():
    """Camera keeps its lens + optical center but pans; re-solving against the SAME
    intrinsics recovers the new pose from tags."""
    base = truth_pinhole_camera()                      # intrinsics we keep
    moved = truth_pinhole_camera(target=(10.0, 4.03, 0.0))  # panned right, same eye
    corr = project_tag_corners(moved, FM)
    assert len(corr) >= 3
    cand = resolve_pose(corr, base)
    assert cand.mode == "pinhole"
    assert cand.rms_error < 1.0
    assert np.linalg.norm(np.asarray(cand.camera_center) -
                          np.asarray(moved.camera_center)) < 0.05


def test_resolve_pose_recovers_pivoted_fisheye_camera():
    base = truth_camera()
    moved = truth_camera(target=(10.0, 4.03, 0.0))
    corr = project_tag_corners(moved, FM)
    cand = resolve_pose(corr, base)
    assert cand.mode == "camera"
    assert cand.rms_error < 1.0
    assert np.linalg.norm(np.asarray(cand.camera_center) -
                          np.asarray(moved.camera_center)) < 0.05


def test_resolve_pose_rejects_homography_base():
    base = CameraCalibration(mode="homography", H=np.eye(3), image_size=(100, 100),
                             field_season="x")
    corr = project_tag_corners(truth_pinhole_camera(), FM)
    with pytest.raises(CalibrationError):
        resolve_pose(corr, base)


def test_pose_deltas_pure_pivot_has_zero_center_shift():
    base = truth_pinhole_camera()
    pivot = truth_pinhole_camera(target=(10.0, 4.03, 0.0))   # same eye, new aim
    shift, rot_deg = _pose_deltas(base, pivot)
    assert shift < 1e-6
    assert rot_deg > 1.0            # it genuinely rotated


def test_pose_deltas_translation_measured():
    base = truth_pinhole_camera(eye=(8.27, -3.0, 6.0))
    moved = truth_pinhole_camera(eye=(10.27, -3.0, 6.0))     # +2 m in x
    shift, _ = _pose_deltas(base, moved)
    assert abs(shift - 2.0) < 1e-6


def _cand_like(base, **kw):
    """A candidate calibration built from base's intrinsics at a new pose."""
    return truth_pinhole_camera(**kw)


def test_evaluate_fix_accepts_clean_pivot():
    cfg = PoseFixConfig()
    base = truth_pinhole_camera()
    cand = truth_pinhole_camera(target=(10.0, 4.03, 0.0))    # pivot, center unchanged
    res = evaluate_fix(base, cand, best_err_m=0.12, n_tags=6, cfg=cfg)
    assert res.passed
    assert res.center_shift_m < 1e-6
    assert res.summary()["best_err_m"] == 0.12


def test_evaluate_fix_rejects_too_few_tags():
    cfg = PoseFixConfig(min_tags=3)
    base = truth_pinhole_camera()
    cand = truth_pinhole_camera(target=(10.0, 4.03, 0.0))
    res = evaluate_fix(base, cand, best_err_m=0.10, n_tags=2, cfg=cfg)
    assert not res.passed
    assert "tags" in res.reason


def test_evaluate_fix_rejects_error_over_gate():
    cfg = PoseFixConfig(drift_max_m=0.5)
    base = truth_pinhole_camera()
    cand = truth_pinhole_camera(target=(10.0, 4.03, 0.0))
    res = evaluate_fix(base, cand, best_err_m=0.9, n_tags=6, cfg=cfg)
    assert not res.passed
    assert "gate" in res.reason


def test_evaluate_fix_rejects_large_translation():
    """The camera-center guardrail: a big translation reads as a re-placed camera,
    not a pivot, so it must not auto-adopt even if reprojection looks fine."""
    cfg = PoseFixConfig(max_center_shift_m=1.0)
    base = truth_pinhole_camera(eye=(8.27, -3.0, 6.0))
    cand = truth_pinhole_camera(eye=(12.0, -3.0, 6.0))       # moved 3.73 m
    res = evaluate_fix(base, cand, best_err_m=0.10, n_tags=6, cfg=cfg)
    assert not res.passed
    assert "moved" in res.reason
    assert res.center_shift_m > 1.0


def test_evaluate_fix_collects_all_failing_reasons():
    cfg = PoseFixConfig(min_tags=4, drift_max_m=0.5, max_center_shift_m=1.0)
    base = truth_pinhole_camera(eye=(8.27, -3.0, 6.0))
    cand = truth_pinhole_camera(eye=(12.0, -3.0, 6.0))
    res = evaluate_fix(base, cand, best_err_m=0.9, n_tags=2, cfg=cfg)
    assert not res.passed
    # all three tripwires reported, not just the first
    assert "tags" in res.reason and "gate" in res.reason and "moved" in res.reason
