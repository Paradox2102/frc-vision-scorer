"""Detect 36h11 AprilTags -> {tag_id: (4,2) image corners}."""
from __future__ import annotations
import cv2
import numpy as np

# A pupil_apriltags Detector holds C-level state and is NOT thread-safe: with
# MAX_CONCURRENT_JOBS=2 two jobs ran the per-match calibration check at once
# through the old module-level singleton and the server died with no traceback
# (Nov 13 pod). Each thread now gets its own detector (tag_locator's immortal
# per-thread cache; same params -> identical detections).
# quad_decimate=1: the default (2) halves the image before finding tag outlines and
# misses small tags (distant corner-camera tags ~25 px wide decode only at 1).
_DETECTOR_PARAMS = {"quad_decimate": 1.0}


def _detector():
    from web.field.tag_locator import thread_detector  # lazy: tag_locator imports calibration
    return thread_detector(**_DETECTOR_PARAMS)


def detect_tags(frame_bgr: np.ndarray) -> dict[int, np.ndarray]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    out: dict[int, np.ndarray] = {}
    for d in _detector().detect(gray):
        out[int(d.tag_id)] = np.asarray(d.corners, dtype=float)  # (4,2)
    return out


def detect_tags_in_roi(
    frame_bgr: np.ndarray,
    click_xy: tuple[float, float],
    target_id: int,
    half: int = 60,
    upscale: int = 4,
) -> np.ndarray | None:
    """Run tag detection in a small upscaled window around a click and return the
    target tag's corners in FULL-FRAME pixel coords, or None.

    Used by the calibration UI's "manual locate -> auto refine": the operator
    clicks roughly on a tag the frame-wide detector missed; we zoom that patch so
    the detector can decode it. We require the decoded id to equal target_id, so a
    stray neighbouring decode can't create a mislabeled correspondence.
    """
    h, w = frame_bgr.shape[:2]
    cx, cy = int(round(click_xy[0])), int(round(click_xy[1]))
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(w, cx + half), min(h, cy + half)
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    up = cv2.resize(roi, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    det = detect_tags(up)
    corners = det.get(int(target_id))
    if corners is None:
        return None
    # map upscaled-ROI corners back to full-frame: /upscale, + ROI offset
    return np.asarray(corners, float) / float(upscale) + np.array([x0, y0], float)
