"""Fold conditioning of a lens solve (calibration.fold_conditioning): how close the
solve's used points get to the radius where the fitted distortion folds back.
An over-fitted lens bends itself to absorb click error and pulls its fold radius in
to just past the outermost tag (2025hop corner_right: ratio 0.86)."""
import math

import numpy as np
import pytest

from tests.field.hop2025_fixtures import (HOP_LEFT, HOP_LEFT_SOURCE, HOP_RIGHT_BORROWED,
                                          HOP_RIGHT_OWN, HOP_RIGHT_SOURCE,
                                          corr_from_source)
from web.field import calibration as cmod
from web.field.calibration import CameraCalibration, fold_conditioning, fold_warning


def test_threshold_sits_between_the_good_and_the_overfitted_hop_solves():
    assert 0.62 < cmod.FOLD_RATIO_WARN < 0.756


def test_overfitted_corner_right_is_near_the_fold_at_tag_2():
    cond = fold_conditioning(CameraCalibration.from_json(HOP_RIGHT_OWN),
                             corr_from_source(HOP_RIGHT_SOURCE))
    assert cond.fold_radius == pytest.approx(0.528, abs=0.005)
    assert cond.r_max == pytest.approx(0.452, abs=0.005)
    assert cond.ratio == pytest.approx(0.856, abs=0.01)
    assert cond.tag_id == 2
    assert cond.near_fold


def test_good_corner_left_is_well_conditioned():
    cond = fold_conditioning(CameraCalibration.from_json(HOP_LEFT),
                             corr_from_source(HOP_LEFT_SOURCE))
    assert cond.fold_radius == pytest.approx(0.788, abs=0.005)
    assert cond.r_max == pytest.approx(0.488, abs=0.005)
    assert cond.ratio == pytest.approx(0.62, abs=0.01)
    assert cond.tag_id == 12
    assert not cond.near_fold


def test_borrowed_corner_right_is_below_the_threshold():
    cond = fold_conditioning(CameraCalibration.from_json(HOP_RIGHT_BORROWED),
                             corr_from_source(HOP_RIGHT_SOURCE))
    assert cond.ratio == pytest.approx(0.69, abs=0.01)
    assert not cond.near_fold


def test_a_lens_that_never_folds_has_ratio_zero():
    cal = CameraCalibration.from_json(HOP_LEFT)
    cal.D = np.array([0.05, 0.0, 0.0, 0.0, 0.0])          # k1 > 0: never folds
    cond = fold_conditioning(cal, corr_from_source(HOP_LEFT_SOURCE))
    assert math.isinf(cond.fold_radius) and cond.ratio == 0.0
    assert not cond.near_fold
    assert cond.to_json()["fold_radius"] is None


def test_a_point_behind_the_camera_counts_as_past_the_fold():
    cal = CameraCalibration.from_json(HOP_LEFT)
    corr = corr_from_source(HOP_LEFT_SOURCE)
    R = cal.rvec
    import cv2
    Rm, _ = cv2.Rodrigues(R)
    c = (-Rm.T @ cal.tvec).reshape(3)
    behind = c - 3.0 * (Rm.T @ np.array([0, 0, 1.0]))     # 3 m behind the camera
    corr["m99"] = (np.zeros((1, 2)), behind.reshape(1, 3))
    cond = fold_conditioning(cal, corr)
    assert math.isinf(cond.ratio) and cond.near_fold and cond.key == "m99"
    assert cond.to_json()["ratio"] is None
    assert "behind the camera" in fold_warning(cond)


def test_pose_free_and_empty_inputs_give_none():
    cal = CameraCalibration(mode="homography", H=np.eye(3), image_size=(10, 10))
    assert fold_conditioning(cal, corr_from_source(HOP_LEFT_SOURCE)) is None
    assert fold_conditioning(CameraCalibration.from_json(HOP_LEFT), {}) is None
    assert fold_conditioning(None, {}) is None


def test_warning_text_names_the_tag_and_the_suggestion():
    cond = fold_conditioning(CameraCalibration.from_json(HOP_RIGHT_OWN),
                             corr_from_source(HOP_RIGHT_SOURCE))
    w = fold_warning(cond, suggestion="try borrowing from corner_left")
    assert "tag 2" in w and "r 0.45 of 0.53" in w
    assert "over-fitted" in w and w.endswith("try borrowing from corner_left")
    assert fold_warning(cond).endswith("over-fitted")             # no sibling: no hint
    fixed = fold_warning(cond, lens_fixed=True)
    assert "over-fitted" not in fixed and "tag 2" in fixed
    good = fold_conditioning(CameraCalibration.from_json(HOP_LEFT),
                             corr_from_source(HOP_LEFT_SOURCE))
    assert fold_warning(good, suggestion="x") is None
    assert fold_warning(None) is None
