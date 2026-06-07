"""Bumper-focused geometry helpers."""
from __future__ import annotations

import numpy as np


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
