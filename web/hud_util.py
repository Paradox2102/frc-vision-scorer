"""HUD drawing helpers kept import-light so tests don't boot web.app.

"Is it tracked?" has three different answers in the pipeline — holds a roster
slot, has a current mask, drawn in the HUD — and they used to diverge
silently: a coasting track (position alive, mask gone) showed on the field
map but drew nothing on camera, indistinguishable from a lost robot. Ghost
boxes close that gap: the last known box, dashed, so a viewer can tell
"coasting here" from "never seen".
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def select_coasting_ghosts(track_outs: Sequence) -> list[tuple[int, tuple]]:
    """Return (slot_id, box) for tracks coasting with a last known box.

    Visible tracks are drawn solid by the regular path; empty slots and
    coasting tracks that never had a box have nothing drawable.
    """
    return [
        (t.slot_id, t.box)
        for t in track_outs
        if t.state == "coasting" and t.box is not None
    ]


def encode_seed_thumbnail(frame_bgr: np.ndarray, boxes: Sequence[tuple],
                          max_width: int = 480) -> str:
    """Annotated, resized JPEG (base64) of a tracking seed frame.

    Used by the tracking-phase progress panel: each camera view shows the
    prompt frame it was seeded from with the seeded robot boxes drawn, so
    the operator can see what the tracker started with while the live
    overlay feed is still minutes away.
    """
    import base64

    img = frame_bgr.copy()
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 220, 60), 2)
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        img = cv2.resize(img, (max_width, max(1, int(h * scale))))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def draw_dashed_box(img: np.ndarray, box: tuple, color: tuple,
                    thickness: int = 2, dash: int = 8, gap: int = 6) -> None:
    """Draw a dashed rectangle in place (cv2 has no native dashed stroke)."""
    x1, y1, x2, y2 = (int(v) for v in box)

    def _dashed_line(p0, p1):
        length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if length <= 0:
            return
        ux = (p1[0] - p0[0]) / length
        uy = (p1[1] - p0[1]) / length
        pos = 0.0
        while pos < length:
            end = min(pos + dash, length)
            a = (int(p0[0] + ux * pos), int(p0[1] + uy * pos))
            b = (int(p0[0] + ux * end), int(p0[1] + uy * end))
            cv2.line(img, a, b, color, thickness)
            pos = end + gap

    _dashed_line((x1, y1), (x2, y1))
    _dashed_line((x2, y1), (x2, y2))
    _dashed_line((x2, y2), (x1, y2))
    _dashed_line((x1, y2), (x1, y1))
