"""One lens fitted jointly to two corner cameras (web.field.joint_lens) on the
operator's real 2025hop clicks (copied into hop2025_fixtures; never web/uploads)."""
import numpy as np
import pytest

from tests.field.hop2025_fixtures import (HOP_LEFT, HOP_LEFT_SIZE, HOP_LEFT_SOURCE,
                                          HOP_RIGHT_BORROWED, HOP_RIGHT_SIZE,
                                          HOP_RIGHT_SOURCE, SEASON25, corr_from_source)
from web.field import joint_lens as jl
from web.field.calibration import CalibrationError, CameraCalibration, fold_conditioning


def _cams(left_scale=1):
    left = corr_from_source(HOP_LEFT_SOURCE)
    if left_scale != 1:
        left = {k: (img * left_scale, obj) for k, (img, obj) in left.items()}
    return {"corner_right": (corr_from_source(HOP_RIGHT_SOURCE), HOP_RIGHT_SIZE),
            "corner_left": (left, tuple(s * left_scale for s in HOP_LEFT_SIZE))}


def _stored_init():
    L = CameraCalibration.from_json(HOP_LEFT)
    R = CameraCalibration.from_json(HOP_RIGHT_BORROWED)
    return {"f": 740.0, "k1": -0.537,
            "poses": {"corner_left": (L.rvec, L.tvec), "corner_right": (R.rvec, R.tvec)}}


@pytest.mark.parametrize("init", [_stored_init(), None], ids=["stored-init", "no-init"])
def test_2025hop_joint_lens_puts_both_corners_on_the_barge(init):
    res = jl.solve_joint_lens(_cams(), SEASON25, init=init)
    assert res.f_px == pytest.approx(692, rel=0.15)
    assert res.k1 == pytest.approx(-0.151, abs=0.05)
    for cam in ("corner_left", "corner_right"):
        cal = res.calibrations[cam]
        assert 8.2 < cal.camera_center[0] < 9.4, cam             # the barge, both
        assert cal.K[0, 0] == pytest.approx(res.f_px)            # same height: same f
        assert (cal.K[0, 2], cal.K[1, 2]) == (cal.image_size[0] / 2, 173.5)
        assert np.allclose(cal.D, [res.k1, 0, 0, 0, 0])          # k1 only, k2 = 0
    assert res.plausible
    assert res.rms_px == pytest.approx(6.6, abs=0.5)
    assert res.rms_inlier_px == pytest.approx(3.1, abs=0.5)
    assert res.rms_inlier_px < res.rms_px
    # the outliers are the reef-centre clicks cut off at the frame edge
    assert [(o["camera"], o["key"]) for o in res.outliers] == [
        ("corner_right", "m10"), ("corner_left", "m20")]
    assert res.outliers[0]["px"] == pytest.approx(23.5, abs=1.0)
    # tags 1 / 2 / 12 / 13 fit to a few px
    for cam, key in (("corner_right", "a1"), ("corner_right", "m2"),
                     ("corner_left", "a12"), ("corner_left", "a13")):
        assert max(res.residuals[cam][key]) < 4.0, (cam, key)
    # the shared lens is well conditioned on both cameras
    for cam, (corr, _size) in _cams().items():
        assert fold_conditioning(res.calibrations[cam], corr).ratio < 0.5


def test_joint_focal_scales_with_crop_height():
    res = jl.solve_joint_lens(_cams(left_scale=2), SEASON25)
    assert res.reference == "corner_right"
    assert res.f_px == pytest.approx(692, rel=0.02)
    assert res.calibrations["corner_left"].K[0, 0] == pytest.approx(2 * res.f_px)
    assert 8.2 < res.calibrations["corner_left"].camera_center[0] < 9.4


def test_result_json_is_serialisable():
    import json
    j = jl.solve_joint_lens(_cams(), SEASON25).to_json()
    json.dumps(j)
    assert set(j["cameras"]) == {"corner_left", "corner_right"}
    assert j["cameras"]["corner_right"]["residuals"]["m10"][0] > 20


def test_needs_two_cameras_with_two_tags_each():
    cams = _cams()
    with pytest.raises(CalibrationError, match="cameras"):
        jl.solve_joint_lens({"corner_left": cams["corner_left"]}, SEASON25)
    one_tag = {"a1": cams["corner_right"][0]["a1"]}
    with pytest.raises(CalibrationError, match="corner_right"):
        jl.solve_joint_lens({"corner_right": (one_tag, HOP_RIGHT_SIZE),
                             "corner_left": cams["corner_left"]}, SEASON25)
