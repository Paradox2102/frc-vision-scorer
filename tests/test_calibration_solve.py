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


# append to tests/test_calibration_solve.py
def test_manual_pose_from_sparse_clicked_points():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_camera()
    # Simulate ~8 clicked points: tag centers (known 3D) projected to pixels.
    items = {}
    for i, tid in enumerate(t for t in fm.tags if t <= 28):
        c3d = fm.tag_corners_3d(tid).mean(axis=0)          # tag center (3,)
        uv = cam.field_to_image(tuple(c3d))
        w, h = cam.image_size
        if 0 <= uv[0] < w and 0 <= uv[1] < h:
            items[f"pt{i}"] = (np.array([uv]), np.array([c3d]))   # (1,2),(1,3)
        if len(items) >= 8:
            break
    pose = Calibrator.solve_pose(items, K=cam.K, D=cam.D,
                                 image_size=(1920, 1080),
                                 field_season="2026-rebuilt-welded")
    for x, y in in_frame_points(cam, [(3., 3.), (13., 5.)]):
        uv = cam.field_to_image((x, y, 0.0))
        xs, ys = pose.image_to_field(uv)
        assert np.hypot(xs - x, ys - y) < 0.10
