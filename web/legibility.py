"""Pure crop scoring for OCR frame selection.

`legibility_score(crop_bgr)` returns a non-negative float estimating how
likely a Qwen-VL pass on this bumper crop will yield a valid roster read.
Higher is better. Combines three signals:

  1. **color strength** — fraction of pixels matching red or blue bumper HSV
     ranges (reuses the band tuned for `detect_bumper_color`).
  2. **text contrast** — fraction of near-white pixels inside / next to the
     colored band (the digits sit on the bumper).
  3. **sharpness** — variance of the Laplacian; rejects motion blur.

No app state, no I/O — easy to unit-test with synthetic crops.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_CROP_PX = 8
WHITE_THRESHOLD = 200          # gray-value threshold for "near white" pixels
SHARPNESS_NORMALIZER = 200.0   # ~typical Laplacian variance on a sharp bumper
DILATE_PX = 5                  # how far to dilate the color mask to capture nearby text


def _color_masks(hsv: np.ndarray):
    red1 = cv2.inRange(hsv, np.array([0, 45, 40]), np.array([16, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 45, 40]), np.array([180, 255, 255]))
    blue = cv2.inRange(hsv, np.array([88, 45, 40]), np.array([140, 255, 255]))
    return red1 | red2, blue


def legibility_score(crop_bgr) -> float:
    if crop_bgr is None:
        return 0.0
    if not isinstance(crop_bgr, np.ndarray) or crop_bgr.ndim != 3 or crop_bgr.size == 0:
        return 0.0
    h, w = crop_bgr.shape[:2]
    if h < MIN_CROP_PX or w < MIN_CROP_PX:
        return 0.0

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    red_mask, blue_mask = _color_masks(hsv)
    total = float(h * w)
    color_strength = max(cv2.countNonZero(red_mask), cv2.countNonZero(blue_mask)) / total
    if color_strength == 0.0:
        return 0.0

    # Dilate the dominant-color mask to include adjacent white text pixels.
    color_mask = red_mask if cv2.countNonZero(red_mask) >= cv2.countNonZero(blue_mask) else blue_mask
    band = cv2.dilate(color_mask, np.ones((DILATE_PX, DILATE_PX), np.uint8), iterations=1)
    band_pixels = max(1, cv2.countNonZero(band))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    white_in_band = cv2.countNonZero(((gray > WHITE_THRESHOLD).astype(np.uint8)) & (band > 0).astype(np.uint8))
    text_contrast = white_in_band / band_pixels

    sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / SHARPNESS_NORMALIZER, 1.0)

    return float(color_strength * text_contrast * sharpness)
