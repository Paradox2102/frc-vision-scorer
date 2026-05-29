import cv2
import numpy as np
from web.field.apriltag_detect import detect_tags


def _render_tag(tag_id: int, px: int = 240, canvas: int = 480) -> np.ndarray:
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(d, tag_id, px)
    img = np.full((canvas, canvas), 255, np.uint8)
    off = (canvas - px) // 2
    img[off:off+px, off:off+px] = marker
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_detects_rendered_tag():
    img = _render_tag(5)
    dets = detect_tags(img)
    assert 5 in dets
    assert dets[5].shape == (4, 2)


def test_corner_order_is_ccw_from_bottom_left():
    # Document & lock the detector's corner order against a centered, axis-aligned tag.
    img = _render_tag(5, px=240, canvas=480)
    corners = detect_tags(img)[5]
    cx, cy = corners.mean(axis=0)
    quad = [(1 if x > cx else 0, 1 if y > cy else 0) for x, y in corners]
    # Pinned from empirical probe (Step 4): pupil_apriltags corner order is
    # [top-right, top-left, bottom-left, bottom-right] in image coordinates
    # (y increases downward), i.e. CW from top-right.
    assert quad == [(1, 0), (0, 0), (0, 1), (1, 1)]
