"""web/hud_draw.py helpers that the golden image does not pin: flag snapshots,
the view crop, the main-view thumbnail, and the mask polygon round trip."""
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
    assert hud_draw.flags_from_dict(hud_draw.flags_to_dict(f)) == f
    assert hud_draw.flags_from_dict({"robot_box": 1, "unknown": True}) == hud_draw.HudFlags(robot_box=True)
    assert hud_draw.flags_from_dict(None) == hud_draw.HudFlags()


def test_app_live_flags_match_the_env_snapshot():
    import web.app as app_mod
    assert app_mod._live_hud_flags() == hud_draw.draw_flags()


def test_extract_view_matches_camera_view_extract():
    img = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    v = CameraView(name="c", crop=(10, 5, 40, 30), masks=((2, 3, 5, 4),), calibration=None)
    assert np.array_equal(hud_draw.extract_view(img, v.crop, v.masks), v.extract(img))


def test_main_thumb_is_a_32x18_grey_thumbnail_and_tracks_the_image():
    a = np.zeros((40, 64, 3), np.uint8)
    a[:, :32] = 255
    b = a[:, ::-1].copy()
    ta, tb = hud_draw.main_thumb(a), hud_draw.main_thumb(b)
    assert hud_draw.thumb_array(ta).shape == (18, 32)
    assert hud_draw.main_thumb(a.copy()) == ta
    assert hud_draw.thumb_diff(ta, ta) == 0
    assert hud_draw.thumb_diff(ta, tb) == 255
    assert hud_draw.thumb_diff(ta, "not base64!") == 255      # malformed = mismatch
    assert hud_draw.thumb_diff(ta, None) == 255


def _near_static_video(path, n=40, step=1):
    """A static textured background with one small blob moving ``step`` px
    per frame: the hard case for an alignment check (fixed camera)."""
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.integers(60, 120, (240, 320, 3), dtype=np.uint8), (7, 7), 0)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (320, 240))
    for i in range(n):
        f = base.copy()
        cv2.circle(f, (40 + step * i, 80), 8, (230, 230, 230), -1)
        vw.write(f)
    vw.release()


def _decode_thumbs(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        out.append(hud_draw.main_thumb(img[:160, :]))
    cap.release()
    return out


def test_thumb_check_flags_one_and_two_frame_offsets_on_a_near_static_video(tmp_path):
    from web.hud_render import THUMB_MAX_DIFF
    p = tmp_path / "near_static.mp4"
    _near_static_video(p)
    live, again = _decode_thumbs(p), _decode_thumbs(p)
    n = len(live) - 2

    def mismatch_share(off):
        return sum(hud_draw.thumb_diff(live[i], again[i + off]) > THUMB_MAX_DIFF
                   for i in range(n)) / n

    assert mismatch_share(0) == 0.0          # the same decode is bit-identical
    assert mismatch_share(1) > 0.5           # far above the 2% abort budget
    assert mismatch_share(2) > 0.5


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


def test_slot_id_from_label_inverts_slot_label():
    assert [hud_draw.slot_id_from_label(hud_draw.slot_label(i)) for i in range(6)] == list(range(6))
    assert hud_draw.slot_id_from_label(None) is None
    assert hud_draw.slot_id_from_label("X1") is None and hud_draw.slot_id_from_label("B4") is None
