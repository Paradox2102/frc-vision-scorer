import numpy as np
from web.field.calibration import holdout_field_error, CameraCalibration


def test_holdout_field_error_zero_for_exact():
    # identity-ish homography: field == image (so image_to_field returns input)
    cal = CameraCalibration(mode="homography", H=np.eye(3), image_size=(100, 100))
    held = {7: (np.array([[3.0, 4.0]]), np.array([[3.0, 4.0, 0.0]]))}
    err = holdout_field_error(cal, held)
    assert err < 1e-6
