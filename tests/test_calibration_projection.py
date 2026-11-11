import numpy as np
from tests._field_synth import truth_camera, in_frame_points

def test_field_to_image_to_field_roundtrip():
    cam = truth_camera()
    pts = in_frame_points(cam, [(2.0, 2.0), (8.27, 4.0), (14.0, 2.0), (14.0, 6.0)])
    assert len(pts) >= 3                                  # honest: these are imaged
    for x, y in pts:
        uv = cam.field_to_image((x, y, 0.0))
        xy = cam.image_to_field(uv)
        assert np.linalg.norm(np.array(xy) - np.array([x, y])) < 0.01  # 1 cm

def test_image_to_field_lands_on_ground_plane():
    cam = truth_camera()
    uv = cam.field_to_image((6.0, 3.0, 0.0))
    x, y = cam.image_to_field(uv)
    assert abs(x - 6.0) < 0.01 and abs(y - 3.0) < 0.01

def test_image_to_field_above_horizon_returns_none():
    """Pixels whose back-projected ray does not intersect the ground plane
    in front of the camera must return None rather than a bogus coordinate.

    Camera geometry note: truth_camera is at (8.27, -3.0, 6.0) looking across
    the field.  Most fisheye-distorted pixels have world ray_z < 0 (pointing
    down) and land on the extended ground plane behind the field boundary.
    The far corner of the image (1920, 1080) is past the fisheye extreme where
    the undistorted ray flips to world ray_z > 0 (pointing upward), giving
    s <= 0 — no forward intersection with z=0.  This is the guard case.
    A normal in-field ground point (e.g. (6, 3, 0) projected then
    back-projected) must still return a valid (x, y) tuple.
    """
    cam = truth_camera()

    # (a) Top-right corner (1920, 0) of the (upright) camera: past the fisheye
    # extreme, the back-projected ray has world_z > 0 (points upward), so s <= 0
    # → None. (Before the synthetic camera was made upright this was the
    # bottom-right corner — the old look_at rolled the camera 180 deg.)
    result_above = cam.image_to_field((1920.0, 0.0))
    assert result_above is None, (
        f"Expected None for no-intersection pixel (1920, 0), got {result_above}"
    )

    # (b) A normal in-field ground point must still round-trip correctly.
    uv = cam.field_to_image((6.0, 3.0, 0.0))
    result_ground = cam.image_to_field(uv)
    assert result_ground is not None, "Expected a valid tuple for in-field ground point"
    assert isinstance(result_ground, tuple) and len(result_ground) == 2
    assert abs(result_ground[0] - 6.0) < 0.01 and abs(result_ground[1] - 3.0) < 0.01
