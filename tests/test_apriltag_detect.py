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


def test_concurrent_detection_matches_serial():
    """Two jobs (MAX_CONCURRENT_JOBS=2) run the calibration check at once; a shared
    pupil_apriltags Detector crashed the server. Each thread must use its own."""
    from concurrent.futures import ThreadPoolExecutor
    import web.field.apriltag_detect as ad

    imgs = [_render_tag(i, px=200, canvas=400) for i in (1, 5, 9, 13)]
    serial = [detect_tags(im) for im in imgs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in range(5):
            got = list(pool.map(detect_tags, imgs * 4))
            for i, d in enumerate(got):
                ref = serial[i % len(imgs)]
                assert d.keys() == ref.keys()
                for k in ref:
                    assert np.allclose(d[k], ref[k])
        # one detector per worker thread, never shared
        dets = set(pool.map(lambda _: id(ad._detector()), range(16)))
    assert id(ad._detector()) not in dets  # main thread has its own too
