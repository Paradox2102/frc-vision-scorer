import numpy as np
import pytest
from web.field.calibration import Calibrator, CalibrationError
from web.field.field_model import FieldModel
from tests._field_synth import truth_camera, project_tag_corners, in_frame_points


def test_solve_pose_recovers_camera_within_10cm():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_camera()
    corr = project_tag_corners(cam, fm)
    assert len(corr) >= 2
    solved = Calibrator.solve_pose(corr, K=cam.K, D=cam.D,
                                   image_size=(1920, 1080),
                                   field_season="2026-rebuilt-welded")
    assert solved.rms_error < 1.0  # px
    pts = in_frame_points(cam, [(2., 2.), (8.27, 4.), (14., 2.), (14., 6.)])
    assert len(pts) >= 3
    for x, y in pts:
        uv = cam.field_to_image((x, y, 0.0))
        xs, ys = solved.image_to_field(uv)
        assert np.hypot(xs - x, ys - y) < 0.10  # 10 cm


def test_too_few_tags_raises():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_camera()
    corr = project_tag_corners(cam, fm)
    one = dict(list(corr.items())[:1])  # a single tag (4 coplanar pts) is not enough
    with pytest.raises(CalibrationError):
        Calibrator.solve_pose(one, K=cam.K, D=cam.D, image_size=(1920, 1080),
                              field_season="2026-rebuilt-welded")
