"""Same-view registration (web.field.view_registration): ORB + RANSAC homography
between two crops of one camera. Synthetic textured images; no video."""
import cv2
import numpy as np

from web.field import view_registration as vr

W, H = 900, 340


def _scene(seed=0, size=(1000, 420)):
    """A textured 'field' scene: random rectangles + circles + a little noise."""
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.full((h, w, 3), 90, np.uint8)
    for _ in range(260):
        c = tuple(int(v) for v in rng.integers(0, 256, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(6, 40)), y + int(rng.integers(6, 40))),
                          c, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(4, 22)), c, -1)
    noise = rng.integers(-6, 7, img.shape)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


def _crop(scene, dx=0, dy=0, x0=40, y0=40):
    return scene[y0 + dy:y0 + dy + H, x0 + dx:x0 + dx + W].copy()


def test_identical_view_registers_near_zero():
    s = _scene()
    r = vr.register_views(_crop(s), _crop(s))
    assert r["ok"], r
    assert r["disp_px"] < 0.5
    assert r["inliers"] >= 60 and r["matches"] >= r["inliers"]


def test_small_shift_registers_with_matching_displacement():
    s = _scene()
    r = vr.register_views(_crop(s), _crop(s, dx=3))
    assert r["ok"], r
    assert 2.0 < r["disp_px"] < 4.0
    assert vr.same_view(_crop(s), _crop(s, dx=3), max_px=8.0)["same"]


def test_large_shift_fails_the_gate():
    s = _scene()
    r = vr.register_views(_crop(s), _crop(s, dx=25))
    # registration itself may succeed (it is the same scene) but the gate fails
    assert not r["ok"] or r["disp_px"] > 20
    sv = vr.same_view(_crop(s), _crop(s, dx=25), max_px=8.0)
    assert not sv["same"] and "px" in sv["reason"]


def test_different_scene_fails():
    sv = vr.same_view(_crop(_scene(0)), _crop(_scene(1)), max_px=8.0)
    assert not sv["same"]
    assert not sv["ok"] or sv["disp_px"] > 8.0


def test_shape_mismatch_is_not_ok():
    s = _scene()
    r = vr.register_views(_crop(s), s[:300, :800].copy())
    assert not r["ok"] and "shape" in r["reason"]


def test_textureless_is_not_ok():
    flat = np.full((H, W, 3), 50, np.uint8)
    r = vr.register_views(flat, flat)
    assert not r["ok"] and r["reason"]


def test_localized_static_texture_is_not_enough():
    """A small static patch (broadcast graphic) on a flat image: plenty of inliers
    at identity, but they cover a tiny part of the crop -> not ok."""
    s = _scene()
    a = np.full((H, W, 3), 60, np.uint8)
    a[10:110, 10:200] = s[:100, :190]
    r = vr.register_views(a, a.copy())
    assert not r["ok"] and "spread" in r["reason"]
