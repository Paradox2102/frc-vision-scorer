"""Crops for the window-label stage (web/field/window_labels.py): where a
tracked robot is in the main view, the tiles cut around it, the mosaics sent
to the reader, and a sequential frame reader.

All pixel coordinates here are MAIN-VIEW pixels (the main camera's crop of
the composite frame), which is what ``CameraCalibration.field_to_image``
returns."""
from __future__ import annotations

from typing import Iterable, Iterator, Optional

import cv2
import numpy as np

from web.field.calibration import push_from_camera

TILE_PX = 240
# Crop side = base + gain * projected width of a 0.95 m bumper. Fitted to the
# Nov 9 prototype's crops: within +-15% for >99% of 1,447 qm8/qm11 positions.
CROP_BASE_PX, CROP_WIDTH_GAIN = 150.0, 0.5
CROP_MIN_PX, CROP_MAX_PX = 120, 420
TARGET_X, TARGET_Y = 0.50, 0.62
ROBOT_CENTRE_OFFSET_M = 0.475     # near bumper edge -> footprint centre
HALF_BUMPER_M = 0.475
TARGET_Z_M = 0.10
VIEW_MARGIN_PX = 40
WIDE_SEND_PX, ZOOM_SEND_PX, GRID_CELL_PX = 480, 504, 480
ZOOM_ROWS, ZOOM_COLS = (44, 212), (36, 204)   # central 70% of a 240 px tile
LABEL_W = 96
_GREY, _DARK = (128, 128, 128), (30, 30, 30)


def target(cal, field_pos, ground_center_offset_m: float = 0.0) -> Optional[tuple]:
    """(u, v, crop side px) for a slot's ``field_pos``, or None when it cannot
    be projected. Positions are tracked at the near bumper edge unless the
    job's ground_center_offset_m already pushed them, so only the remainder
    of the half-robot offset is applied."""
    try:
        offset = max(0.0, ROBOT_CENTRE_OFFSET_M - float(ground_center_offset_m or 0.0))
        xy = push_from_camera(cal, (float(field_pos[0]), float(field_pos[1])), offset)
        u, v = cal.field_to_image((xy[0], xy[1], TARGET_Z_M))
        spans = []
        for dx, dy in ((HALF_BUMPER_M, 0.0), (0.0, HALF_BUMPER_M)):
            a = cal.field_to_image((xy[0] - dx, xy[1] - dy, TARGET_Z_M))
            b = cal.field_to_image((xy[0] + dx, xy[1] + dy, TARGET_Z_M))
            spans.append(float(np.hypot(a[0] - b[0], a[1] - b[1])))
    except Exception:
        return None
    if not np.all(np.isfinite([u, v] + spans)):
        return None
    side = CROP_BASE_PX + CROP_WIDTH_GAIN * max(spans)
    return float(u), float(v), int(round(min(CROP_MAX_PX, max(CROP_MIN_PX, side))))


def in_view(u: float, v: float, view_wh: tuple) -> bool:
    w, h = view_wh
    return 0 <= u <= w and -VIEW_MARGIN_PX <= v <= h + VIEW_MARGIN_PX


def _ticks(tile: np.ndarray, tx: int, ty: int) -> None:
    n = TILE_PX
    # a target just outside the view (in_view's margins) lies off the tile:
    # clamp so all four ticks are still drawn, pointing at the nearest edge
    tx, ty = min(max(int(tx), 0), n - 1), min(max(int(ty), 0), n - 1)
    for px, py, dx, dy in ((tx, 0, 0, 1), (tx, n - 1, 0, -1), (0, ty, 1, 0), (n - 1, ty, -1, 0)):
        if 0 <= px < n and 0 <= py < n:
            tri = np.array([(px + dx * 12, py + dy * 12), (px - dy * 6, py - dx * 6),
                            (px + dy * 6, py + dx * 6)], np.int32)
            cv2.fillConvexPoly(tile, tri, (255, 255, 255))
            cv2.polylines(tile, [tri], True, (0, 0, 0), 1)


def wide_tile(view_img: np.ndarray, u: float, v: float, side: int) -> tuple:
    """Square crop of ``side`` px around (u, v), resized to TILE_PX, with tick
    marks pointing at the target. Returns (tile, (tx, ty)): the target's
    position in the tile after clamping the crop to the view."""
    h, w = view_img.shape[:2]
    s = int(max(2, min(side, w, h)))
    x0 = max(0, min(w - s, int(u - s * TARGET_X)))
    y0 = max(0, min(h - s, int(v - s * TARGET_Y)))
    tile = cv2.resize(view_img[y0:y0 + s, x0:x0 + s], (TILE_PX, TILE_PX),
                      interpolation=cv2.INTER_CUBIC)
    tx, ty = int((u - x0) / s * TILE_PX), int((v - y0) / s * TILE_PX)
    _ticks(tile, tx, ty)
    return tile, (tx, ty)


def zoom_tile(tile: np.ndarray) -> np.ndarray:
    crop = tile[ZOOM_ROWS[0]:ZOOM_ROWS[1], ZOOM_COLS[0]:ZOOM_COLS[1]]
    return cv2.resize(crop, (ZOOM_SEND_PX, ZOOM_SEND_PX), interpolation=cv2.INTER_CUBIC)


def send_wide(tile: np.ndarray) -> np.ndarray:
    return cv2.resize(tile, (WIDE_SEND_PX, WIDE_SEND_PX), interpolation=cv2.INTER_CUBIC)


def grid_image(tiles: list) -> np.ndarray:
    """2x2 mosaic of up to four tiles; missing cells are grey."""
    c = GRID_CELL_PX
    cells = [cv2.resize(t, (c, c), interpolation=cv2.INTER_CUBIC) for t in tiles[:4]]
    cells += [np.full((c, c, 3), _GREY, np.uint8)] * (4 - len(cells))
    return np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:4])])


def lineup_card(rows: list, per_row: int = 6) -> np.ndarray:
    """``rows``: [(letter, [tiles])]. One row per robot: letter, then tiles."""
    out = []
    for letter, tiles in rows:
        lab = np.full((TILE_PX, LABEL_W, 3), _DARK, np.uint8)
        cv2.putText(lab, str(letter), (20, TILE_PX // 2 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (255, 255, 255), 3)
        cells = list(tiles[:per_row])
        cells += [np.full((TILE_PX, TILE_PX, 3), _DARK, np.uint8)] * (per_row - len(cells))
        out.append(np.hstack([lab] + cells))
    return np.vstack(out)


def jpeg(img: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encode failed")
    return buf.tobytes()


def iter_view_frames(video_path, frame_nums: Iterable, view: tuple) -> Iterator[tuple]:
    """Yield (frame_num, main-view image) for the wanted source frames, in
    order, from ONE sequential pass (grab every frame, retrieve only wanted
    ones; seeking per frame is ~5x slower and not frame-accurate). ``view`` is
    the main camera's (x, y, w, h) crop of the composite. Raises OSError when
    the video cannot be opened."""
    wanted = sorted({int(n) for n in frame_nums})
    if not wanted:
        return
    x, y, w, h = view
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise OSError(f"cannot open video {video_path}")
        want, last, i = set(wanted), wanted[-1], 0
        while i <= last:
            if not cap.grab():
                break
            if i in want:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    yield i, frame[y:y + h, x:x + w]
            i += 1
    finally:
        cap.release()
