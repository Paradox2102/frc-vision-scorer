import cv2
import numpy as np
import pytest
from web.field.calibration import Calibrator, CameraCalibration, CalibrationError
from tests._field_synth import look_at, TRUTH_K, IMAGE_SIZE

_RVEC, _TVEC = look_at((8.27, -3.0, 6.0), (8.27, 4.03, 0.0))

def _pinhole_px(x, y, z=0.0):
    """Project a field point through a distortion-free pinhole camera -> pixel."""
    obj = np.array([[[float(x), float(y), float(z)]]], np.float64)
    img, _ = cv2.projectPoints(obj, _RVEC, _TVEC, TRUTH_K, None)
    return tuple(img.reshape(2))

GROUND = [(2,2),(14,2),(14,6),(2,6),(8,4),(5,5),(11,3)]

def test_ground_homography_roundtrip():
    img = np.array([_pinhole_px(x, y) for x, y in GROUND], float)
    fld = np.array(GROUND, float)
    cal = Calibrator.solve_ground_homography(img, fld, IMAGE_SIZE, "2026-rebuilt-welded")
    assert cal.mode == "homography"
    assert cal.rms_error < 0.5  # px, near-exact for a pinhole view of the ground plane
    for x, y in [(6,3),(10,5)]:
        xs, ys = cal.image_to_field(_pinhole_px(x, y))
        assert np.hypot(xs - x, ys - y) < 0.05  # 5 cm

def test_homography_field_to_image_roundtrip():
    img = np.array([_pinhole_px(x, y) for x, y in GROUND], float)
    cal = Calibrator.solve_ground_homography(img, np.array(GROUND, float),
                                             IMAGE_SIZE, "2026-rebuilt-welded")
    uv = _pinhole_px(7.0, 4.0)
    back = cal.field_to_image((7.0, 4.0, 0.0))   # z ignored in homography mode
    assert np.hypot(back[0] - uv[0], back[1] - uv[1]) < 2.0  # px

def test_homography_serialization_roundtrip():
    img = np.array([_pinhole_px(x, y) for x, y in GROUND[:5]], float)
    cal = Calibrator.solve_ground_homography(img, np.array(GROUND[:5], float),
                                             IMAGE_SIZE, "2026-rebuilt-welded")
    c2 = CameraCalibration.from_json(cal.to_json())
    assert c2.mode == "homography"
    np.testing.assert_allclose(c2.H, cal.H, atol=1e-9)
    x, y = c2.image_to_field(_pinhole_px(7.0, 4.0))
    assert np.hypot(x - 7.0, y - 4.0) < 0.05

def test_too_few_points_raises():
    with pytest.raises(CalibrationError):
        Calibrator.solve_ground_homography(np.zeros((3,2)), np.zeros((3,2)),
                                            IMAGE_SIZE, "2026-rebuilt-welded")
