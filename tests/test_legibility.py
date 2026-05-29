import cv2
import numpy as np

from web.legibility import legibility_score


def _good_red_bumper_crop():
    """Sharp red bumper with white digit-like blobs — high legibility."""
    img = np.full((80, 200, 3), 30, np.uint8)             # dark background
    img[50:75, 20:180] = (40, 40, 220)                    # red bumper band (BGR)
    # White "digits"
    cv2.rectangle(img, (40, 55), (60, 70), (255, 255, 255), -1)
    cv2.rectangle(img, (90, 55), (110, 70), (255, 255, 255), -1)
    cv2.rectangle(img, (140, 55), (160, 70), (255, 255, 255), -1)
    return img


def _blurry_red_bumper_crop():
    sharp = _good_red_bumper_crop()
    return cv2.GaussianBlur(sharp, (15, 15), 0)


def _back_facing_crop():
    """Uniform mid-gray — no bumper color, no text."""
    return np.full((80, 200, 3), 128, np.uint8)


def test_invalid_crop_returns_zero():
    assert legibility_score(None) == 0.0
    assert legibility_score(np.zeros((0, 0, 3), np.uint8)) == 0.0
    assert legibility_score(np.zeros((4, 4, 3), np.uint8)) == 0.0   # too small


def test_no_bumper_color_scores_zero():
    assert legibility_score(_back_facing_crop()) == 0.0


def test_good_scores_above_blurry_scores_above_bad():
    good = legibility_score(_good_red_bumper_crop())
    blurry = legibility_score(_blurry_red_bumper_crop())
    bad = legibility_score(_back_facing_crop())
    assert good > blurry > bad >= 0.0
    assert good > 0.0
