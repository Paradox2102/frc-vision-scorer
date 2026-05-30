"""Bumper-focused image helpers.

Alliance color must come from the bumper band, not the whole robot body. FRC
robots often have painted mechanisms that are unrelated to the current match
alliance.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BumperColor:
    alliance: str
    confidence: float
    red_ratio: float
    blue_ratio: float
    crop_box: tuple[int, int, int, int]


def _clamp_box(frame: np.ndarray, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bumper_band_box(frame: np.ndarray, box) -> tuple[int, int, int, int] | None:
    """Return the likely bumper band area inside a robot box.

    The band is intentionally low and horizontally trimmed to avoid painted
    superstructure, intakes, and floor/ball spill at the edges.
    """
    clamped = _clamp_box(frame, box)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    bw, bh = x2 - x1, y2 - y1
    bx1 = x1 + int(bw * 0.08)
    bx2 = x2 - int(bw * 0.08)
    by1 = y1 + int(bh * 0.45)
    by2 = y1 + int(bh * 0.96)
    if bx2 <= bx1 or by2 <= by1:
        return clamped
    return bx1, by1, bx2, by2


def bumper_crop(frame: np.ndarray, box) -> np.ndarray | None:
    band = bumper_band_box(frame, box)
    if band is None:
        return None
    x1, y1, x2, y2 = band
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def _color_masks(hsv: np.ndarray):
    red1 = cv2.inRange(hsv, np.array([0, 45, 35]), np.array([16, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 45, 35]), np.array([180, 255, 255]))
    blue = cv2.inRange(hsv, np.array([88, 45, 35]), np.array([140, 255, 255]))
    return red1 | red2, blue


def _largest_rectangular_band_ratio(mask: np.ndarray) -> float:
    """Score colored pixels that form a bumper-like horizontal component."""
    if mask.size == 0:
        return 0.0
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = 0.0
    total = float(mask.shape[0] * mask.shape[1])
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area <= 0 or w < 6 or h < 3:
            continue
        aspect = w / max(1, h)
        # Bumper faces usually look like low horizontal bands in the crop. Keep
        # square-ish fragments too for angled/partially occluded sides.
        if aspect < 0.7:
            continue
        fill = area / float(w * h)
        if fill < 0.25:
            continue
        best = max(best, area / total)
    return best


def classify_bumper_color(frame: np.ndarray, box) -> BumperColor:
    band = bumper_band_box(frame, box)
    if band is None:
        return BumperColor("unknown", 0.0, 0.0, 0.0, (0, 0, 0, 0))
    x1, y1, x2, y2 = band
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return BumperColor("unknown", 0.0, 0.0, 0.0, band)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_mask, blue_mask = _color_masks(hsv)
    total = float(crop.shape[0] * crop.shape[1])
    red_ratio = cv2.countNonZero(red_mask) / total
    blue_ratio = cv2.countNonZero(blue_mask) / total

    red_band = _largest_rectangular_band_ratio(red_mask)
    blue_band = _largest_rectangular_band_ratio(blue_mask)
    red_score = max(red_ratio, red_band * 1.5)
    blue_score = max(blue_ratio, blue_band * 1.5)

    min_score = 0.015
    dominance = 1.25
    if red_score > min_score and red_score > blue_score * dominance:
        conf = min(1.0, (red_score - blue_score) / max(red_score, 1e-6))
        return BumperColor("red", float(conf), float(red_ratio), float(blue_ratio), band)
    if blue_score > min_score and blue_score > red_score * dominance:
        conf = min(1.0, (blue_score - red_score) / max(blue_score, 1e-6))
        return BumperColor("blue", float(conf), float(red_ratio), float(blue_ratio), band)
    return BumperColor("unknown", 0.0, float(red_ratio), float(blue_ratio), band)
