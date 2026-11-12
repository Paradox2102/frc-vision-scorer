"""Visibility-checked field->image projection (CameraCalibration.project_checked /
in_view / field_to_image_visible) and the distortion fold radius.

A wide lens's radial polynomial folds back: past the fold radius a point far
outside the field of view maps INTO the image (2025hop corner_left, k1 = -0.537:
tag 16 at x_n = -1.37 lands on the principal point). Visibility must reject it,
along with points behind or at the camera."""
import math

import cv2
import numpy as np
import pytest

from web.field.calibration import (CameraCalibration, NEAR_LIMIT_M,
                                   distortion_fold_radius)
from web.field.field_model import FieldModel


def hop_corner_left() -> CameraCalibration:
    """2025hop corner_left, version 20261111-232802-934489 (1.65 px over tags
    12, 13, 20, 21, 22): mounted on the barge at (8.98, 3.96, 2.14) m looking at
    the blue wall."""
    f = 739.967700055877
    return CameraCalibration(
        K=np.array([[f, 0.0, 482.5], [0.0, f, 173.5], [0.0, 0.0, 1.0]]),
        D=np.array([-0.5373262887776691, 0.0, 0.0, 0.0, 0.0]),
        rvec=np.array([1.3426460943393659, 1.3867128809160423,
                       -1.0255371138805007]).reshape(3, 1),
        tvec=np.array([-4.251873273284275, -0.22737555519031116,
                       9.097189557677899]).reshape(3, 1),
        image_size=(965, 347), crop_region=(0, 0, 965, 347),
        field_season="2025-reefscape-welded", rms_error=1.6495825784268812,
        mode="pinhole")


# ------------------------------------------------------------------ fold radius

@pytest.mark.parametrize("k1", [-0.537, -0.2, -1.5])
def test_fold_radius_k1_only_is_analytic(k1):
    # r(1 + k1 r^2) peaks where 1 + 3 k1 r^2 = 0
    assert distortion_fold_radius([k1, 0, 0, 0, 0], "pinhole") == pytest.approx(
        math.sqrt(-1.0 / (3.0 * k1)), rel=1e-6)


def test_fold_radius_monotonic_lens_is_infinite():
    assert distortion_fold_radius([0.1, 0.01, 0, 0, 0], "pinhole") == math.inf
    assert distortion_fold_radius([0, 0, 0, 0, 0], "pinhole") == math.inf
    assert distortion_fold_radius(None, "pinhole") == math.inf


def test_fold_radius_uses_k2_and_k3():
    # k2 < 0 alone folds too: 1 + 5 k2 r^4 = 0
    k2 = -0.05
    assert distortion_fold_radius([0, k2, 0, 0, 0], "pinhole") == pytest.approx(
        (-1.0 / (5.0 * k2)) ** 0.25, rel=1e-6)
    k3 = -0.01      # D[4]: 1 + 7 k3 r^6 = 0
    assert distortion_fold_radius([0, 0, 0, 0, k3], "pinhole") == pytest.approx(
        (-1.0 / (7.0 * k3)) ** (1 / 6), rel=1e-6)


def test_fold_radius_fisheye():
    # theta_d = theta (1 + k1 theta^2) peaks at theta^2 = -1/(3 k1); r = tan(theta)
    k1 = -0.3
    th = math.sqrt(-1.0 / (3.0 * k1))
    assert distortion_fold_radius([k1, 0, 0, 0], "camera") == pytest.approx(
        math.tan(th), rel=1e-6)
    # a typical mildly-distorted fisheye never folds before 90 degrees
    assert distortion_fold_radius([0.05, -0.01, 0.0, 0.0], "camera") == math.inf


# ------------------------------------------------------------- 2025hop corner_left

def test_hop_corner_left_tag_visibility():
    cal = hop_corner_left()
    fm = FieldModel.load(cal.field_season)
    vis = {t for t in fm.tags
           if cal.field_to_image_visible(tuple(fm.tag_translation(t))) is not None}
    assert {12, 13, 20, 21, 22} <= vis
    assert not ({16, 3, 4, 14, 15} & vis)
    # the bug being fixed: tag 16 really does fold onto the principal point
    u, v = cal.field_to_image(tuple(fm.tag_translation(16)))
    assert abs(u - 482.5) < 10 and abs(v - 173.5) < 10


def test_in_view_batch_matches_single():
    cal = hop_corner_left()
    fm = FieldModel.load(cal.field_season)
    ids = sorted(fm.tags)
    P = np.array([fm.tag_translation(t) for t in ids])
    ok = cal.in_view(P)
    single = [cal.field_to_image_visible(tuple(p)) is not None for p in P]
    assert ok.tolist() == single
    uv, ok2 = cal.project_checked(P)
    assert ok2.tolist() == single
    for t, (u, v), good in zip(ids, uv, ok2):
        if good:
            assert (u, v) == pytest.approx(cal.field_to_image(tuple(fm.tag_translation(t))))


def test_frame_margin_and_no_frame_check():
    cal = hop_corner_left()
    fm = FieldModel.load(cal.field_season)
    p21 = tuple(fm.tag_translation(21))          # v = 342 of 347: near the bottom
    u, v = cal.field_to_image(p21)
    assert cal.field_to_image_visible(p21) is not None
    # a point just below the frame: rejected at margin 0, kept with margin 50
    # or with the frame test off (margin_px=None: depth + fold only)
    below = None
    for dz in np.linspace(0, -1.0, 101):
        q = (p21[0], p21[1], p21[2] + dz)
        uq, vq = cal.field_to_image(q)
        if 347 + 5 < vq < 347 + 40:
            below = q
            break
    assert below is not None
    assert cal.field_to_image_visible(below) is None
    assert cal.field_to_image_visible(below, margin_px=50) is not None
    assert cal.field_to_image_visible(below, margin_px=None) is not None


def test_near_limit_rejects_points_at_the_camera():
    cal = hop_corner_left()
    c = cal.camera_center
    R, _ = cv2.Rodrigues(cal.rvec)
    # a point straight down the optical axis, closer than the near limit
    p = c + R[2] * (NEAR_LIMIT_M * 0.5)
    assert cal.field_to_image_visible(tuple(p), margin_px=None) is None
    p = c + R[2] * 3.0
    assert cal.field_to_image_visible(tuple(p)) is not None


def test_homography_mode_keeps_frame_only_behaviour():
    H = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1.0]])   # image px -> field m
    cal = CameraCalibration(mode="homography", H=H, image_size=(800, 600))
    assert cal.field_to_image_visible((2.0, 3.0, 0.0)) == pytest.approx((200.0, 300.0))
    assert cal.field_to_image_visible((9.0, 3.0, 0.0)) is None          # u = 900
    assert cal.field_to_image_visible((9.0, 3.0, 0.0), margin_px=None) == pytest.approx(
        (900.0, 300.0))


def test_explicit_image_size_overrides():
    cal = hop_corner_left()
    fm = FieldModel.load(cal.field_season)
    p13 = tuple(fm.tag_translation(13))       # u ~ 749
    assert cal.field_to_image_visible(p13) is not None
    assert cal.field_to_image_visible(p13, image_size=(700, 347)) is None


# --------------------------------------------------------------------- wireframe

def _drawn_segments(monkeypatch, cal, fm, size):
    """Every (p, q) pixel segment draw_field_wireframe hands to cv2.polylines."""
    from web import hud_draw
    segs = []
    real = cv2.polylines

    def spy(img, pts, closed, color, *a, **k):
        for poly in pts:
            poly = np.asarray(poly, float).reshape(-1, 2)
            segs.extend(zip(poly[:-1], poly[1:]))
        return real(img, pts, closed, color, *a, **k)
    monkeypatch.setattr(hud_draw.cv2, "polylines", spy)
    img = np.zeros((size[1], size[0], 3), np.uint8)
    hud_draw.draw_field_wireframe(img, cal, fm)
    return segs, img


def test_hop_wireframe_has_no_streaks(monkeypatch):
    """The barge the camera sits on, the red half behind it and the far-off-axis
    blue-wall ends must not fold / mirror into long streaks across the image."""
    cal = hop_corner_left()
    fm = FieldModel.load(cal.field_season)
    segs, img = _drawn_segments(monkeypatch, cal, fm, cal.image_size)
    assert segs                                   # the visible field still draws
    diag = math.hypot(*cal.image_size)
    longest = max(float(np.linalg.norm(q - p)) for p, q in segs)
    assert longest < 0.6 * diag, longest
    # the in-view structures still draw: reef and coral stations have pixels
    from web import hud_draw
    for g in ("reef", "coral_station"):
        col = np.array(hud_draw.WIREFRAME_GROUP_COLORS[g], np.uint8)
        assert np.all(img == col, axis=2).sum() > 50, g


# ------------------------------------------------------- predicted / visible tags

def test_hop_visible_and_predicted_tag_lists_drop_folded_tags():
    from web.field import auto_calibrate as ac
    from web.field import tag_locator as tl
    import web.app as app_mod
    cal = hop_corner_left()
    fm = FieldModel.load(cal.field_season)
    bad = {16, 3, 4, 14, 15}
    good = {12, 13, 20, 21, 22}
    vis = set(ac.visible_model_tags(cal, fm, cal.image_size))
    assert good <= vis and not (bad & vis)
    pred = {t for t in fm.tags if tl.predict_tag(cal, fm, t, cal.image_size) is not None}
    assert good <= pred and not (bad & pred)
    listed = {int(t) for t in app_mod._predicted_tags(cal, fm, cal.image_size)}
    assert listed == pred
    assert ac._project(cal, fm.tag_translation(16)) is None
    assert ac._project(cal, fm.tag_translation(12)) is not None


def test_hop_review_overlay_labels_no_folded_tag():
    """render_overlay's orange/green tag-centre labels: none on tag 16's folded
    position at the principal point."""
    from web.field import auto_calibrate as ac
    cal = hop_corner_left()
    fm = FieldModel.load(cal.field_season)
    frame = np.zeros((347, 965, 3), np.uint8)
    img = ac.render_overlay(frame, cal, fm, {}, [12, 13])
    # the orange (unused) dot for tag 16 would sit at ~(487, 174)
    win = img[164:184, 477:497]
    assert not np.all(win == np.array(ac.ORANGE, np.uint8), axis=2).any()
