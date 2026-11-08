"""click_height_m: GT clicks project onto z = click height, removing the
bumper-height parallax (points move TOWARD the camera, more when far away)."""
from types import SimpleNamespace

import cv2
import numpy as np

from web.field import ground_truth as gt
from web.field.calibration import CameraCalibration


def _look_at_cal(cam=(8.0, -5.0, 3.3), target=(8.0, 4.0, 0.0), size=(1920, 723)):
    cam, target = np.array(cam, float), np.array(target, float)
    z = target - cam; z /= np.linalg.norm(z)                     # camera forward
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)          # right
    y = np.cross(z, x)                                            # down
    R = np.vstack([x, y, z])
    rvec, _ = cv2.Rodrigues(R)
    tvec = (-R @ cam).reshape(3, 1)
    K = np.array([[1300.0, 0, size[0] / 2], [0, 1300.0, size[1] / 2], [0, 0, 1]])
    return CameraCalibration(K=K, D=np.zeros(5), rvec=rvec, tvec=tvec, image_size=size,
                             field_season="test", mode="pinhole")


def _views():
    return [SimpleNamespace(name="main", crop=(0, 0, 1920, 723), calibration=_look_at_cal())]


def test_click_height_pulls_toward_camera_more_when_far():
    views, cal = _views(), _views()[0].calibration
    shifts = []
    for y in (2.0, 7.0):
        uv = cal.field_to_image((8.0, y, 0.0))
        at0 = gt.project_uv(views, uv, z=0.0)["field_xy"]
        at10 = gt.project_uv(views, uv, z=0.10)["field_xy"]
        assert at10[1] < at0[1]                      # toward the camera (camera at y=-5)
        shifts.append(at0[1] - at10[1])
    # parallax = h * horizontal distance / camera height: 0.1*7/3.3 vs 0.1*12/3.3
    assert abs(shifts[0] - 0.1 * 7 / 3.3) < 0.01 and abs(shifts[1] - 0.1 * 12 / 3.3) < 0.01


def test_reproject_doc_uses_doc_click_height_and_legacy_zero():
    views, cal = _views(), _views()[0].calibration
    uv = list(cal.field_to_image((8.0, 6.0, 0.0)))
    base = gt.new_gt("2026x_qm1", "2026x", "1", "blue", 1,
                     {"name": "v.mp4", "fingerprint": "1:a", "fps": 60})
    base["keyframes"] = [{"frame": 1, "state": "visible", "uv": uv}]
    assert base["click_height_m"] == gt.DEFAULT_CLICK_HEIGHT_M
    legacy = dict(base); legacy.pop("click_height_m")
    y_legacy = gt.reproject_doc(legacy, views)["keyframes"][0]["field_xy"][1]
    y_new = gt.reproject_doc(base, views)["keyframes"][0]["field_xy"][1]
    assert abs(y_legacy - 6.0) < 1e-3                # no field -> carpet plane, as before
    assert y_new < y_legacy - 0.05
