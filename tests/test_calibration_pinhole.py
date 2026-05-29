import numpy as np
import pytest
from web.field.calibration import CameraCalibration, Calibrator, CalibrationError
from web.field.field_model import FieldModel
from tests._field_synth import look_at


def test_pinhole_field_image_roundtrip_on_ground():
    """field_to_image then image_to_field returns the original ground point."""
    K = np.array([[1200., 0., 960.], [0., 1200., 324.], [0., 0., 1.]])
    D = np.array([-0.25, -0.08, 0.0, 0.0, 0.0])           # k1,k2,p1,p2,k3
    rvec, tvec = look_at((8.27, -3.0, 6.0), (8.27, 4.03, 0.0))
    cam = CameraCalibration(mode="pinhole", K=K, D=D, rvec=rvec, tvec=tvec,
                            image_size=(1920, 648),
                            field_season="2026-rebuilt-welded")
    for x, y in [(3., 3.), (8.27, 4.), (13., 5.)]:
        u, v = cam.field_to_image((x, y, 0.0))
        xs, ys = cam.image_to_field((u, v))
        # cv2.undistortPoints inverts distortion iteratively, so the round-trip is
        # exact to ~sub-mm rather than machine epsilon. 1 mm is far below the 10 cm
        # field-tracking requirement.
        assert np.hypot(xs - x, ys - y) < 1e-3
