"""Detect 36h11 AprilTags -> {tag_id: (4,2) image corners}."""
from __future__ import annotations
import cv2
import numpy as np
from pupil_apriltags import Detector

_detector = Detector(families="tag36h11")


def detect_tags(frame_bgr: np.ndarray) -> dict[int, np.ndarray]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    out: dict[int, np.ndarray] = {}
    for d in _detector.detect(gray):
        out[int(d.tag_id)] = np.asarray(d.corners, dtype=float)  # (4,2)
    return out
