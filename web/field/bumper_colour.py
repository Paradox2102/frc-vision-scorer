"""Per-frame bumper colour read for a tracker input (job knob bumper_color_read).

The pipeline's detection colour is often missing ("unknown": selective-mode SAM
tracks are TBD with no colour) or stale (a SAM track's colour is fixed at seed
and its mask can later slide onto a robot of the other alliance). This module
re-reads the colour from the MAIN view pixels every frame:

1. take the input's field ground point (the pipeline ``pos`` = near-bumper-edge
   carpet point) and push it ``offset_m`` AWAY from the camera to the robot
   centre (``calibration.push_from_camera``);
2. project a 0.95 m x 0.95 m footprint band at z 0.03-0.16 m (the bumper
   height) into the main image and take its convex hull;
3. count HSV red (H<8 or >170, S>110, V>70) vs blue (100<H<128, S>110, V>50)
   pixels inside the hull, EXCLUDING pixels that are red/blue in a robot-free
   background (median of ~40 frames across the match, dilated 5x5) — that
   removes the alliance ramps / tape / station walls;
4. red/blue if max(r, b) >= 0.06 and |r - b| >= 0.04, else unknown.

Prototype: docs/notes/bumper_colour_prototype.py (qm8 hand GT: own half 74%
correct / 0.1% wrong / 26% undecided; opponent half 32% / 2.3% / 66%).

Pure functions (numpy + cv2 only); no per-track memory by design — the
prototype showed per-SAM-track colour memory is harmful (tracks swap robots).
"""
from __future__ import annotations

import cv2
import numpy as np

from web.field.calibration import push_from_camera

# Robot footprint half-size (0.95 m square) and bumper band heights (m).
HALF_FOOTPRINT_M = 0.475
BAND_Z0_M = 0.03
BAND_Z1_M = 0.16
# Default near-edge -> centre push (m); see ground_center_offset_m.
DEFAULT_OFFSET_M = 0.475
# Decision thresholds (fractions of the non-background hull pixels).
MIN_FRAC = 0.06
MIN_MARGIN = 0.04
# Fewer usable (non-background, on-image) pixels than this -> unknown.
MIN_PIXELS = 15
# A hull vertex further than this many pixels off-image is a degenerate
# projection (grazing ray); treat as unreadable.
_MAX_ABS_PX = 1e5


def colour_masks(hsv: np.ndarray):
    """(red, blue) boolean masks for an HSV uint8 image (any leading shape)."""
    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]
    sat = S > 110
    red = ((H < 8) | (H > 170)) & sat & (V > 70)
    blue = (H > 100) & (H < 128) & sat & (V > 50)
    return red, blue


def build_background(frames_bgr) -> np.ndarray | None:
    """Static-colour mask from robot-free-ish main-view frames.

    ``frames_bgr``: iterable of equally-sized HxWx3 uint8 BGR frames (~40
    sampled evenly across the match). Robots move, the field doesn't, so the
    per-pixel median is (mostly) the empty field; its red/blue pixels (alliance
    ramps, tape, driver-station walls), dilated 5x5, are returned as an HxW bool
    mask to exclude from every read. None when no frames are given."""
    frames = [f for f in frames_bgr if f is not None]
    if not frames:
        return None
    shape = frames[0].shape
    frames = [f for f in frames if f.shape == shape]
    stack = np.stack(frames, axis=0)
    # Median per pixel, row-chunked to bound the temporary sort buffers.
    med = np.empty(shape, np.uint8)
    step = 64
    for y0 in range(0, shape[0], step):
        med[y0:y0 + step] = np.median(stack[:, y0:y0 + step], axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)
    red, blue = colour_masks(hsv)
    return cv2.dilate((red | blue).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def _behind_camera(cal, pts_xyz) -> bool:
    rvec = getattr(cal, "rvec", None)
    tvec = getattr(cal, "tvec", None)
    if rvec is None or tvec is None:
        return True
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    cam = (R @ np.asarray(pts_xyz, np.float64).T) + np.asarray(tvec, np.float64).reshape(3, 1)
    return bool(np.any(cam[2] <= 1e-6))


def band_poly(cal, centre_xy, z0=BAND_Z0_M, z1=BAND_Z1_M, half=HALF_FOOTPRINT_M):
    """Convex hull (Nx1x2 int32, image px) of the robot's bumper band projected
    into ``cal``'s image, or None (no pose / behind camera / degenerate)."""
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    pts = [(cx + dx, cy + dy, z) for z in (z0, z1)
           for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half))]
    if _behind_camera(cal, pts):
        return None
    try:
        P = np.array([cal.field_to_image(p) for p in pts], np.float64)
    except Exception:
        return None
    if not np.all(np.isfinite(P)) or np.any(np.abs(P) > _MAX_ABS_PX):
        return None
    return cv2.convexHull(P.astype(np.float32)).astype(np.int32)


def read_bumper_colour(main_bgr, cal, ground_pos, bg_mask=None,
                       offset_m=DEFAULT_OFFSET_M):
    """Read one input's bumper colour from the main-view image.

    ``main_bgr``: the main view's HxWx3 BGR pixels (calibration coordinates).
    Only the hull's bounding box is converted to HSV (cheap per call).
    ``ground_pos``: pipeline field point (near-bumper-edge carpet point); it is
    pushed ``offset_m`` away from the camera (0 = already the robot centre).
    ``bg_mask``: HxW bool static-colour mask from :func:`build_background`
    (None = no exclusion).

    Returns ``(colour, r_frac, b_frac)`` with colour in red/blue/unknown;
    unknown (fractions 0) when the camera has no pose, the band is behind the
    camera / off-image, or too few usable pixels remain."""
    if main_bgr is None or cal is None or ground_pos is None:
        return "unknown", 0.0, 0.0
    if getattr(cal, "camera_center", None) is None:
        return "unknown", 0.0, 0.0    # pose-free (homography) view: no z
    centre = push_from_camera(cal, ground_pos, offset_m) if offset_m and offset_m > 0 else ground_pos
    poly = band_poly(cal, centre)
    if poly is None:
        return "unknown", 0.0, 0.0
    h, w = main_bgr.shape[:2]
    x, y, bw, bh = cv2.boundingRect(poly)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + bw, w), min(y + bh, h)
    if x1 <= x0 or y1 <= y0:
        return "unknown", 0.0, 0.0
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(m, poly - np.array([x0, y0], np.int32), 255)
    keep = m > 0
    if bg_mask is not None:
        keep &= ~bg_mask[y0:y1, x0:x1]
    n = int(keep.sum())
    if n < MIN_PIXELS:
        return "unknown", 0.0, 0.0
    hsv = cv2.cvtColor(np.ascontiguousarray(main_bgr[y0:y1, x0:x1]), cv2.COLOR_BGR2HSV)
    red, blue = colour_masks(hsv[keep])
    r = float(red.mean())
    b = float(blue.mean())
    if max(r, b) < MIN_FRAC or abs(r - b) < MIN_MARGIN:
        return "unknown", r, b
    return ("red" if r > b else "blue"), r, b


class BumperColourStats:
    """Per-job counters for the bumper colour read (one instance per job)."""

    KEYS = ("reads", "red", "blue", "undecided", "agree", "flipped",
            "unknown_to_known", "cleared", "early_reads")

    def __init__(self):
        self.c = {k: 0 for k in self.KEYS}

    def as_dict(self):
        return dict(self.c)


def apply_bumper_colour(mode, detections, main_bgr, cal, bg_mask, offset_m,
                        stats: BumperColourStats | None = None):
    """Apply the read to ``detections`` (FieldDetection-like, mutated IN PLACE
    so every holder of the object — fused observation, tracker input, frame
    evidence — sees the same colour).

    Only main-backed detections (``box is not None``) are read; corner-only
    ones are untouched. ``mode``: "off" (no-op), "override" (a decided read
    replaces the colour; undecided keeps the pipeline colour), "strict" (the
    colour becomes the read, unknown included). Returns the number changed."""
    if mode not in ("override", "strict") or main_bgr is None:
        return 0
    changed = 0
    for d in detections:
        if getattr(d, "box", None) is None:
            continue
        read, _r, _b = read_bumper_colour(main_bgr, cal, d.pos, bg_mask, offset_m)
        pipe = d.color if d.color in ("red", "blue") else "unknown"
        if mode == "override":
            new = read if read != "unknown" else d.color
        else:
            new = read
        if stats is not None:
            c = stats.c
            c["reads"] += 1
            c[read if read != "unknown" else "undecided"] += 1
            if read != "unknown":
                if pipe == read:
                    c["agree"] += 1
                elif pipe == "unknown":
                    c["unknown_to_known"] += 1
                else:
                    c["flipped"] += 1
            elif mode == "strict" and pipe != "unknown":
                c["cleared"] += 1
        if new != d.color:
            d.color = new
            changed += 1
    return changed
