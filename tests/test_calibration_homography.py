import numpy as np
from web.field.calibration import Calibrator, CameraCalibration
from tests._field_synth import truth_camera, IMAGE_SIZE

def _pinhole_camera():
    # zero distortion so a ground-plane homography is geometrically exact
    return truth_camera(D=np.zeros(4))

def test_ground_homography_roundtrip():
    cam = _pinhole_camera()
    pts_xy = [(2,2),(14,2),(14,6),(2,6),(8,4),(5,5),(11,3)]   # spread ground points
    img = np.array([cam.field_to_image((x,y,0.0)) for x,y in pts_xy], float)
    fld = np.array(pts_xy, float)
    cal = Calibrator.solve_ground_homography(img, fld, IMAGE_SIZE, "2026-rebuilt-welded")
    assert cal.mode == "homography"
    assert cal.rms_error < 0.5  # px (near-exact for an undistorted camera)
    for (x,y) in [(6,3),(10,5)]:
        uv = cam.field_to_image((x,y,0.0))
        xs,ys = cal.image_to_field(uv)
        assert np.hypot(xs-x, ys-y) < 0.05  # 5 cm

def test_homography_field_to_image_roundtrip():
    cam = _pinhole_camera()
    pts_xy = [(2,2),(14,2),(14,6),(2,6),(8,4),(5,5),(11,3)]
    img = np.array([cam.field_to_image((x,y,0.0)) for x,y in pts_xy], float)
    cal = Calibrator.solve_ground_homography(img, np.array(pts_xy,float), IMAGE_SIZE, "2026-rebuilt-welded")
    uv = cam.field_to_image((7.0, 4.0, 0.0))
    back = cal.field_to_image((7.0, 4.0, 0.0))   # z ignored in homography mode
    assert np.hypot(back[0]-uv[0], back[1]-uv[1]) < 2.0  # px

def test_homography_serialization_roundtrip():
    cam = _pinhole_camera()
    pts_xy = [(2,2),(14,2),(14,6),(2,6),(8,4)]
    img = np.array([cam.field_to_image((x,y,0.0)) for x,y in pts_xy], float)
    cal = Calibrator.solve_ground_homography(img, np.array(pts_xy,float), IMAGE_SIZE, "2026-rebuilt-welded")
    c2 = CameraCalibration.from_json(cal.to_json())
    assert c2.mode == "homography"
    np.testing.assert_allclose(c2.H, cal.H, atol=1e-9)
    x,y = c2.image_to_field(cam.field_to_image((7.0,4.0,0.0)))
    assert np.hypot(x-7.0, y-4.0) < 0.05

def test_too_few_points_raises():
    import pytest
    from web.field.calibration import CalibrationError
    with pytest.raises(CalibrationError):
        Calibrator.solve_ground_homography(np.zeros((3,2)), np.zeros((3,2)),
                                            IMAGE_SIZE, "2026-rebuilt-welded")
