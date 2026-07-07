import numpy as np
import pytest

from web.field.calibration import CameraCalibration
from web.field.coverage import cell_coverage, coverage_grid
from web.field.occlusion import Prism
from tests._field_synth import look_at, TRUTH_K, TRUTH_D


def _homog_cam(image_size=(1000, 500), masks=()):
    # image (0..1000, 0..500) -> field (0..10, 0..5): x/100, y/100
    H = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1.0]])
    return CameraCalibration(mode="homography", H=H, image_size=image_size, masks=list(masks))


class _FakeView:
    def __init__(self, name, cal):
        self.name = name
        self.calibration = cal


class _FakeModel:
    length_m, width_m = 16.54, 8.07


def test_cell_inside_frustum_is_covered():
    views = [_FakeView("main", _homog_cam())]
    w = cell_coverage(5.0, 2.5, views, _FakeModel(), prisms=[])
    assert w["main"] == 1.0


def test_cell_outside_frustum_is_blind():
    views = [_FakeView("main", _homog_cam())]
    w = cell_coverage(14.0, 2.5, views, _FakeModel(), prisms=[])  # maps to x=1400px > 1000
    assert w["main"] == 0.0


def test_cell_inside_mask_is_blind():
    # field (5,2.5) maps to image (500, 250); mask covers that rect.
    views = [_FakeView("main", _homog_cam(masks=[(400, 200, 200, 100)]))]
    w = cell_coverage(5.0, 2.5, views, _FakeModel(), prisms=[])
    assert w["main"] == 0.0


def test_cell_outside_mask_is_unaffected():
    views = [_FakeView("main", _homog_cam(masks=[(0, 0, 10, 10)]))]
    w = cell_coverage(5.0, 2.5, views, _FakeModel(), prisms=[])
    assert w["main"] == 1.0


def test_homography_mode_skips_occlusion_and_close_range():
    # camera_center is None in homography mode; even with a prism placed right
    # over the test cell, occlusion should NOT zero the weight out.
    prism = Prism(footprint=np.array([(4.0, 2.0), (6.0, 2.0), (6.0, 3.0), (4.0, 3.0)]), z_top=3.0)
    views = [_FakeView("main", _homog_cam())]
    w = cell_coverage(5.0, 2.5, views, _FakeModel(), prisms=[prism])
    assert w["main"] == 1.0


def test_corner_view_falloff_applied():
    # corner_left view name should be attenuated by corner_detection_weight past
    # its trusted zone depth (default 4.8m) toward the field midline.
    far_x = _FakeModel.length_m / 2.0  # at the midline + margin, weight -> 0
    views = [_FakeView("corner_left", _homog_cam(image_size=(2000, 2000)))]
    w = cell_coverage(far_x + 2.0, 2.5, views, _FakeModel(), prisms=[])
    assert w["corner_left"] == 0.0


def test_close_range_pose_camera_blind_when_box_exceeds_frac():
    # A pose-based (pinhole) camera looking straight down from very low height
    # right above the cell: expected robot pixel width should blow past the
    # frac backstop, forcing weight to 0 even though the point is in-frustum.
    rvec, tvec = look_at((5.0, 2.5, 0.3), (5.0, 2.5, 0.0))
    cal = CameraCalibration(K=TRUTH_K.copy(), D=TRUTH_D.copy(), rvec=rvec, tvec=tvec,
                            image_size=(1920, 1080), crop_region=None,
                            field_season="2026-rebuilt-welded", mode="camera")
    views = [_FakeView("main", cal)]
    w = cell_coverage(5.0, 2.5, views, _FakeModel(), prisms=[])
    assert w["main"] == 0.0


def test_cell_coverage_sums_multiple_cameras():
    views = [_FakeView("main", _homog_cam()), _FakeView("second", _homog_cam())]
    w = cell_coverage(5.0, 2.5, views, _FakeModel(), prisms=[])
    assert w["main"] == 1.0
    assert w["second"] == 1.0
    assert sum(w.values()) == 2.0


def test_coverage_grid_shape_and_blind_flag():
    views = [_FakeView("main", _homog_cam())]
    xs, ys, grid = coverage_grid(views, _FakeModel(), cell_m=1.0, prisms=[])
    assert len(xs) == pytest.approx(_FakeModel.length_m / 1.0, abs=2)
    assert len(ys) == pytest.approx(_FakeModel.width_m / 1.0, abs=2)
    assert grid.shape == (len(ys), len(xs))
    # A cell far outside the homography's mapped frustum (field x > 10) should
    # be fully blind (coverage 0, since only one camera and it's outside).
    ix_far = int(np.argmin(np.abs(xs - 15.0)))
    assert grid[0, ix_far] == 0.0
