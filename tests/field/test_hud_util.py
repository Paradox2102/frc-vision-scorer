from __future__ import annotations

import numpy as np

from web.field.tracker import TrackOut
from web.hud_util import select_coasting_ghosts, draw_dashed_box


def _t(slot_id, state, box):
    return TrackOut(slot_id=slot_id, alliance="blue", pos=(1.0, 2.0),
                    vel=(0.0, 0.0), box=box, state=state)


def test_select_coasting_ghosts_filters_states():
    tracks = [
        _t(0, "visible", (1, 2, 3, 4)),    # drawn solid elsewhere — not a ghost
        _t(1, "coasting", (5, 6, 9, 9)),   # ghost
        _t(2, "coasting", None),           # no last box — nothing to draw
        _t(3, "empty", (0, 0, 1, 1)),      # empty slot — not a ghost
    ]
    assert select_coasting_ghosts(tracks) == [(1, (5, 6, 9, 9))]


def test_draw_dashed_box_draws_with_gaps():
    img = np.zeros((50, 50, 3), np.uint8)
    draw_dashed_box(img, (5, 5, 45, 45), (0, 255, 0), thickness=1)
    drawn = int((img > 0).any(axis=2).sum())
    assert drawn > 0
    # A solid 1px rectangle perimeter would be ~160 px; dashes must leave gaps.
    assert drawn < 140


def test_encode_seed_thumbnail_returns_decodable_jpeg_with_boxes():
    import base64
    import cv2

    frame = np.full((200, 400, 3), 40, np.uint8)
    b64 = __import__('web.hud_util', fromlist=['encode_seed_thumbnail']).encode_seed_thumbnail(
        frame, [(10, 10, 80, 90), (200, 50, 280, 150)], max_width=320)
    raw = base64.b64decode(b64)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    assert img.shape[1] <= 320                       # resized to fit
    plain = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    # Boxes were drawn: the thumbnail is not uniform any more.
    assert int(img.std()) > 0


def test_encode_seed_thumbnail_handles_no_boxes():
    from web.hud_util import encode_seed_thumbnail
    frame = np.full((100, 200, 3), 40, np.uint8)
    b64 = encode_seed_thumbnail(frame, [], max_width=320)
    assert isinstance(b64, str) and len(b64) > 0
