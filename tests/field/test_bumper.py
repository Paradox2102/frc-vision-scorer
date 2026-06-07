import cv2
import numpy as np

from web.field.bumper import bumper_band_box, bumper_crop


def _robot_crop(body_bgr, bumper_bgr):
    img = np.full((120, 120, 3), 25, np.uint8)
    img[10:65, 20:100] = body_bgr
    img[72:98, 16:104] = bumper_bgr
    cv2.rectangle(img, (42, 77), (52, 93), (255, 255, 255), -1)
    cv2.rectangle(img, (68, 77), (78, 93), (255, 255, 255), -1)
    return img


def test_bumper_crop_returns_lower_band_not_full_robot():
    img = _robot_crop((220, 40, 40), (35, 35, 220))
    crop = bumper_crop(img, (0, 0, 120, 120))

    assert crop is not None
    assert crop.shape[0] < img.shape[0]
    assert crop.shape[1] < img.shape[1]


def test_bumper_band_box_clamps_to_frame():
    img = np.zeros((100, 100, 3), np.uint8)
    band = bumper_band_box(img, (-20, -10, 130, 120))
    assert band is not None
    assert all(0 <= value <= 100 for value in band)
