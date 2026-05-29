import numpy as np
import pytest
from web.field.calibration import CameraCalibration, Calibrator, CalibrationError
from web.field.field_model import FieldModel
from tests._field_synth import (look_at, truth_pinhole_camera,
                                 project_tag_corners, in_frame_points)


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


def test_solve_camera_from_tags_recovers_pinhole_camera():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()                   # image_size (1920,1080)
    corr = project_tag_corners(cam, fm)            # 4 corners per in-frame tag
    assert len(corr) >= 4
    solved = Calibrator.solve_camera_from_tags(
        corr, image_size=cam.image_size, field_season="2026-rebuilt-welded")
    assert solved.mode == "pinhole"
    assert solved.rms_error < 1.0                  # px
    assert solved.crop_region == (0, 0, cam.image_size[0], cam.image_size[1])
    for x, y in in_frame_points(cam, [(3., 3.), (8.27, 4.), (13., 5.), (6., 6.)]):
        uv = cam.field_to_image((x, y, 0.0))
        xs, ys = solved.image_to_field(uv)
        assert np.hypot(xs - x, ys - y) < 0.10     # 10 cm on the ground plane


def test_solve_camera_from_tags_mixed_corners_and_centers():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    corners = project_tag_corners(cam, fm)
    mixed = {}
    for i, tid in enumerate(sorted(corners)):
        if i % 2 == 0:
            mixed[tid] = corners[tid]                       # 4 corners
        else:
            c3d = fm.tag_corners_3d(tid).mean(axis=0)
            uv = cam.field_to_image(tuple(c3d))
            mixed[tid] = (np.array([uv]), np.array([c3d]))  # 1 center
    solved = Calibrator.solve_camera_from_tags(
        mixed, image_size=cam.image_size, field_season="2026-rebuilt-welded")
    # With the principal point solved (not fixed), a half-single-point set conditions
    # the intrinsics a little more loosely, so RMS sits ~2px. What matters is ground
    # accuracy, which stays well within the 10cm field-tracking budget.
    assert solved.rms_error < 3.0
    for x, y in in_frame_points(cam, [(3., 3.), (8.27, 4.), (13., 5.), (10., 2.)]):
        uv = cam.field_to_image((x, y, 0.0))
        xs, ys = solved.image_to_field(uv)
        assert np.hypot(xs - x, ys - y) < 0.10


def test_solve_camera_from_tags_too_few_points_raises():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    tid = sorted(fm.tags)[0]
    c3d = fm.tag_corners_3d(tid).mean(axis=0)
    one = {tid: (np.array([cam.field_to_image(tuple(c3d))]), np.array([c3d]))}
    with pytest.raises(CalibrationError):
        Calibrator.solve_camera_from_tags(
            one, image_size=cam.image_size, field_season="2026-rebuilt-welded")
