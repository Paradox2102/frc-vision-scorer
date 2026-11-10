"""web/hud_draw.py helpers that the golden image does not pin: flag snapshots,
the view crop, aHash, and the mask polygon round trip."""
import dataclasses

import cv2
import numpy as np

from web import hud_draw
from web.field.camera_view import CameraView


def test_draw_flags_reads_env_with_the_app_defaults():
    assert hud_draw.draw_flags({}) == hud_draw.HudFlags()
    f = hud_draw.draw_flags({"HUD_DRAW_ROBOT_BOX": "1", "HUD_DRAW_ROBOT_MASK": "false",
                             "HUD_DRAW_BALL_BOXES": "0"})
    assert f.robot_box is True and f.robot_mask is False and f.ball_boxes is False


def test_flags_round_trip_through_a_dict():
    f = hud_draw.HudFlags(robot_box=True, coasting_ghost=False)
    assert hud_draw.flags_from_dict(dataclasses.asdict(f)) == f
    assert hud_draw.flags_from_dict({"robot_box": 1, "unknown": True}) == hud_draw.HudFlags(robot_box=True)
    assert hud_draw.flags_from_dict(None) == hud_draw.HudFlags()


def test_app_live_flags_match_the_env_snapshot():
    import web.app as app_mod
    assert app_mod._live_hud_flags() == hud_draw.draw_flags()


def test_extract_view_matches_camera_view_extract():
    img = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    v = CameraView(name="c", crop=(10, 5, 40, 30), masks=((2, 3, 5, 4),), calibration=None)
    assert np.array_equal(hud_draw.extract_view(img, v.crop, v.masks), v.extract(img))


def test_ahash_is_16_hex_and_tracks_the_image():
    a = np.zeros((40, 40, 3), np.uint8)
    a[:, :20] = 255
    b = a[:, ::-1].copy()
    ha, hb = hud_draw.ahash(a), hud_draw.ahash(b)
    assert len(ha) == 16 and int(ha, 16) >= 0
    assert hud_draw.ahash(a.copy()) == ha
    assert hud_draw.ahash_distance(ha, ha) == 0
    assert hud_draw.ahash_distance(ha, hb) == 64


def test_mask_poly_round_trip_keeps_the_largest_component():
    m = np.zeros((100, 120), bool)
    cv2.rectangle(m.view(np.uint8), (10, 10), (60, 70), 1, -1)
    m[90:95, 100:105] = True                       # small second component: dropped
    poly = hud_draw.mask_poly(m)
    back = hud_draw.poly_mask(poly, m.shape)
    assert back[40, 30] and not back[92, 102]
    assert abs(int(back.sum()) - 51 * 61) < 200


def test_mask_poly_is_capped_at_64_points():
    m = np.zeros((400, 400), np.uint8)
    pts = [(200 + int(150 * np.cos(a) * (1 + 0.3 * (k % 2))),
            200 + int(150 * np.sin(a) * (1 + 0.3 * (k % 2))))
           for k, a in enumerate(np.linspace(0, 2 * np.pi, 300, endpoint=False))]
    cv2.fillPoly(m, [np.array(pts, np.int32)], 1)
    poly = hud_draw.mask_poly(m.astype(bool))
    assert 3 <= len(poly) <= hud_draw.MASK_POLY_MAX_POINTS


def test_mask_poly_none_and_empty():
    assert hud_draw.mask_poly(None) is None
    assert hud_draw.mask_poly(np.zeros((5, 5), bool)) is None
    assert hud_draw.poly_mask(None, (5, 5)) is None
