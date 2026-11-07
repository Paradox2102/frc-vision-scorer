"""ground_center_offset_m: push robot ground-contact projections away from
the camera (the box/mask bottom is the robot's NEAR bumper edge, not its
footprint centre). Default 0.0 must be byte-identical."""
import numpy as np
import pytest

from tests._field_synth import truth_pinhole_camera
from web.field.calibration import CameraCalibration, push_from_camera
from web.field.detect_bridge import filter_field_robot_boxes, to_field_detections
from web.field.field_model import FieldModel
from web.job_config import KNOBS, JobConfig


class _FakeCal:
    def __init__(self, center):
        self.camera_center = center


def test_push_direction_and_magnitude():
    cal = _FakeCal(np.array([8.41, -5.24, 3.32]))
    xy = (10.0, 3.0)
    out = push_from_camera(cal, xy, 0.475)
    d_in = np.subtract(xy, (8.41, -5.24))
    d_out = np.subtract(out, (8.41, -5.24))
    # Moved exactly 0.475 m ...
    assert np.hypot(*(np.subtract(out, xy))) == pytest.approx(0.475)
    # ... radially away from the camera's ground point (same bearing, farther).
    assert np.linalg.norm(d_out) == pytest.approx(np.linalg.norm(d_in) + 0.475)
    cross = d_in[0] * d_out[1] - d_in[1] * d_out[0]
    assert cross == pytest.approx(0.0, abs=1e-9)
    # Camera behind the y=0 wall -> a point straight ahead moves +y.
    straight = push_from_camera(cal, (8.41, 2.0), 0.5)
    assert straight == pytest.approx((8.41, 2.5))


def test_corner_camera_pushes_from_its_own_center():
    corner = _FakeCal((0.64, -0.3, 0.7))
    out = push_from_camera(corner, (3.64, 3.7), 0.5)   # 3-4-5 triangle from cam
    assert out == pytest.approx((3.64 + 0.3, 3.7 + 0.4))


def test_passthrough_cases():
    cal = _FakeCal((8.41, -5.24, 3.32))
    xy = (10.0, 3.0)
    assert push_from_camera(cal, xy, 0.0) is xy        # offset 0 -> identity
    assert push_from_camera(cal, xy, -0.3) is xy       # negative -> identity
    assert push_from_camera(cal, None, 0.5) is None    # ray missed carpet
    assert push_from_camera(None, xy, 0.5) is xy       # no calibration
    assert push_from_camera(_FakeCal(None), xy, 0.5) is xy  # pose-free
    # Degenerate: point directly under the camera.
    assert push_from_camera(cal, (8.41, -5.24), 0.5) == (8.41, -5.24)


def test_homography_calibration_passthrough():
    cal = CameraCalibration(mode="homography", H=np.eye(3), image_size=(10, 10))
    assert cal.camera_center is None
    xy = cal.image_to_field((3.0, 4.0))
    assert push_from_camera(cal, xy, 0.475) is xy


def test_real_pinhole_camera_center_used():
    cam = truth_pinhole_camera()
    cc = cam.camera_center
    xy = (8.0, 4.0)
    out = push_from_camera(cam, xy, 0.475)
    d_in = np.hypot(xy[0] - cc[0], xy[1] - cc[1])
    d_out = np.hypot(out[0] - cc[0], out[1] - cc[1])
    assert d_out == pytest.approx(d_in + 0.475)


def test_to_field_detections_offset_shifts_and_zero_is_identical():
    fm = FieldModel.load("2026-rebuilt-welded")
    dims = (fm.length_m, fm.width_m)
    cam = truth_pinhole_camera()
    u, v = cam.field_to_image((8.0, 4.0, 0.0))
    box = (u - 30, v - 120, u + 30, v)
    base = to_field_detections([box], cam, dims, colors=["blue"])
    zero = to_field_detections([box], cam, dims, colors=["blue"],
                               ground_offset_m=0.0)
    assert base[0].pos == zero[0].pos
    shifted = to_field_detections([box], cam, dims, colors=["blue"],
                                  ground_offset_m=0.475)
    assert len(shifted) == 1
    assert shifted[0].pos == pytest.approx(
        push_from_camera(cam, base[0].pos, 0.475))


def test_filter_on_field_uses_shifted_point():
    """A contact just inside the far wall margin, pushed past it, is dropped:
    the on-field check runs on the robot centre the detection would carry."""
    fm = FieldModel.load("2026-rebuilt-welded")
    dims = (fm.length_m, fm.width_m)
    cam = truth_pinhole_camera()
    cc = cam.camera_center
    # A point 0.3 m inside the +y margin (W + 0.5 - 0.3), on the camera's x so
    # the push is (almost) pure +y.
    y_edge = fm.width_m + 0.5 - 0.3
    u, v = cam.field_to_image((float(cc[0]), y_edge, 0.0))
    box = (u - 10, v - 40, u + 10, v)
    assert filter_field_robot_boxes([box], cam, dims) == [0]
    assert filter_field_robot_boxes([box], cam, dims, ground_offset_m=0.0) == [0]
    assert filter_field_robot_boxes([box], cam, dims, ground_offset_m=0.475) == []


def test_ray_zone_test_keeps_raw_contact():
    """The ray-beyond-contact gate stays on the RAW contact: a real robot
    whose raw contact sits ~3.2 m short of a zone must not be newly blocked
    because the shifted point's 3 m look-ahead now reaches the zone."""
    fm = FieldModel.load("2026-rebuilt-welded")
    dims = (fm.length_m, fm.width_m)
    cam = truth_pinhole_camera(eye=(0.64, -0.3, 0.7), target=(1.3, 6.5, 0.0))
    cc = cam.camera_center
    contact = np.array([1.0, 2.0])
    ray = contact - np.asarray(cc[:2], float)
    ray /= np.linalg.norm(ray)
    # Zone (0.4 m square) centred 3.3 m beyond the raw contact along the ray.
    zc = contact + 3.3 * ray
    zone = [(zc[0] - 0.2, zc[1] - 0.2, zc[0] + 0.2, zc[1] + 0.2)]
    u, v = cam.field_to_image((float(contact[0]), float(contact[1]), 0.0))
    box = (u - 20, v - 60, u + 20, v)
    assert filter_field_robot_boxes([box], cam, dims, ray_blocked_zones=zone) == [0]
    assert filter_field_robot_boxes(
        [box], cam, dims, ray_blocked_zones=zone, ground_offset_m=0.475) == [0]


def test_knob_default_env_and_override(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    assert JobConfig().ground_center_offset_m == 0.0
    monkeypatch.setenv("GROUND_CENTER_OFFSET_M", "0.3")
    assert JobConfig().ground_center_offset_m == pytest.approx(0.3)
    # Per-job config_json override (A/B arm) beats env.
    assert JobConfig({"ground_center_offset_m": 0.475}).ground_center_offset_m \
        == pytest.approx(0.475)
    assert JobConfig({"ground_center_offset_m": "0.475"}).ground_center_offset_m \
        == pytest.approx(0.475)
