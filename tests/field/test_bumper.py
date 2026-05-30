import cv2
import numpy as np

from web.field.bumper import bumper_crop, classify_bumper_color


def _robot_crop(body_bgr, bumper_bgr):
    img = np.full((120, 120, 3), 25, np.uint8)
    img[10:65, 20:100] = body_bgr
    img[72:98, 16:104] = bumper_bgr
    cv2.rectangle(img, (42, 77), (52, 93), (255, 255, 255), -1)
    cv2.rectangle(img, (68, 77), (78, 93), (255, 255, 255), -1)
    return img


def test_bumper_color_ignores_opposite_colored_body():
    blue_body_red_bumper = _robot_crop((220, 40, 40), (35, 35, 220))
    red_body_blue_bumper = _robot_crop((35, 35, 220), (220, 45, 35))

    assert classify_bumper_color(blue_body_red_bumper, (0, 0, 120, 120)).alliance == "red"
    assert classify_bumper_color(red_body_blue_bumper, (0, 0, 120, 120)).alliance == "blue"


def test_bumper_crop_returns_lower_band_not_full_robot():
    img = _robot_crop((220, 40, 40), (35, 35, 220))
    crop = bumper_crop(img, (0, 0, 120, 120))

    assert crop is not None
    assert crop.shape[0] < img.shape[0]
    assert crop.shape[1] < img.shape[1]
    assert classify_bumper_color(img, (0, 0, 120, 120)).alliance == "red"
