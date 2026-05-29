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
