"""Composite layout detection on synthetic broadcast composites, plus the
per-event layout store in web.field.composite_layout."""

import numpy as np
import pytest

from web.field import layout_detect as ld
from web.field import composite_layout as cl

W, H = 960, 540


def _camera_scene(rng, h, w):
    """A static camera scene: smooth random texture in the dark-grey range."""
    small = rng.integers(25, 95, size=(max(2, h // 16), max(2, w // 16), 3)).astype(np.float32)
    import cv2
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _stone(rng, h, w):
    """A textured, static broadcast strip (tan, high-frequency texture)."""
    base = np.array([120, 170, 205], np.float32)          # BGR tan
    tex = rng.normal(0, 18, size=(h, w, 1)).astype(np.float32)
    return np.clip(base + tex, 0, 255)


def _render(rng, views, statics, overlays, n=24):
    """views: {name: (x, y, w, h)} camera tiles with moving blobs + noise.
    statics: [(x, y, w, h, img)] static textured graphics (tiny noise each frame).
    overlays: [(x, y, w, h, img)] flat static graphics, bit-identical every frame."""
    import cv2
    scenes = {k: _camera_scene(rng, r[3], r[2]) for k, r in views.items()}
    frames = []
    for _ in range(n):
        f = np.zeros((H, W, 3), np.float32)
        for k, (x, y, w, h) in views.items():
            tile = scenes[k].copy()
            for _ in range(70):                            # robots/balls moving
                cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
                col = tuple(float(c) for c in rng.integers(150, 255, size=3))
                cv2.circle(tile, (cx, cy), int(rng.integers(8, 16)), col, -1)
            tile += rng.normal(0, 3, size=tile.shape)       # sensor noise
            f[y:y + h, x:x + w] = tile
        for (x, y, w, h, img) in statics:
            f[y:y + h, x:x + w] = img + rng.normal(0, 1.5, size=img.shape)  # encoder noise
        f = np.clip(f, 0, 255).astype(np.uint8)
        for (x, y, w, h, img) in overlays:
            f[y:y + h, x:x + w] = img
        frames.append(f)
    return frames


def _scoreboard(w, h):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (40, 40, 40)
    img[:, : w // 3] = (200, 90, 20)                      # blue block
    img[:, 2 * w // 3:] = (30, 30, 210)                    # red block
    img[h // 4: 3 * h // 4, w // 3 + 10: 2 * w // 3 - 10] = (250, 250, 250)
    for i in range(6):                                     # "text"
        x = 12 + i * (w // 7)
        img[h // 3: 2 * h // 3, x: x + 8] = (255, 255, 255)
    img[:2], img[-2:], img[:, :2], img[:, -2:] = 230, 230, 230, 230   # box outline
    return img


def _close(got, exp, tol=4):
    return all(abs(a - b) <= tol for a, b in zip(
        [got[0], got[1], got[0] + got[2], got[1] + got[3]],
        [exp[0], exp[1], exp[0] + exp[2], exp[1] + exp[3]]))


def test_three_view_composite_with_separator_pillar_and_overlays():
    rng = np.random.default_rng(1)
    sep_y0, sep_y1 = 352, 374           # separator strip
    pil_x0, pil_x1 = 468, 494           # pillar between the lower views
    views = {"main": (0, 0, W, sep_y0),
             "corner_left": (0, sep_y1, pil_x0, H - sep_y1),
             "corner_right": (pil_x1, sep_y1, W - pil_x1, H - sep_y1)}
    statics = [(0, sep_y0, W, sep_y1 - sep_y0, _stone(rng, sep_y1 - sep_y0, W)),
               (pil_x0, sep_y1, pil_x1 - pil_x0, H - sep_y1, _stone(rng, H - sep_y1, pil_x1 - pil_x0))]
    sb = (240, 0, 480, 60)               # scoreboard over the top of main
    banner = (300, 492, 360, 30)         # event banner across both lower views
    banner_img = np.full((banner[3], banner[2], 3), 30, np.uint8)
    banner_img[8:22, 40:320] = (240, 240, 240)
    overlays = [(*sb, _scoreboard(sb[2], sb[3])), (*banner, banner_img)]
    frames = _render(rng, views, statics, overlays)

    out = ld.detect_layout_from_frames(frames)
    assert out["frame_size"] == [W, H]
    cams = out["cameras"]
    assert set(cams) == {"main", "corner_left", "corner_right"}
    for name, exp in views.items():
        assert _close(cams[name]["crop"], exp), (name, cams[name]["crop"], exp)
        assert 0.0 < out["confidence"][name] <= 1.0
    # the scoreboard is masked on main, crop-local, and the mask stays near it
    masks = cams["main"]["masks"]
    assert any(m[0] <= sb[0] + 6 and m[0] + m[2] >= sb[0] + sb[2] - 6 and m[1] <= 6
               and m[1] + m[3] >= sb[3] - 6 for m in masks), masks
    for m in masks:                         # conservative: no mask far into the field
        assert m[1] + m[3] <= sb[3] + 12
    # the banner shows up as a mask on at least one lower view, crop-local
    lx, ly = cams["corner_left"]["crop"][:2]
    assert any(abs((lx + m[0]) - banner[0]) <= 6 and abs((ly + m[1]) - banner[1]) <= 6
               for m in cams["corner_left"]["masks"]), cams["corner_left"]["masks"]


def test_full_width_scoreboard_band_at_bottom_and_two_lower_views():
    """hiho-style: short main, separator, lower views, scoreboard band at the
    bottom of the frame (not an overlay)."""
    rng = np.random.default_rng(2)
    views = {"main": (0, 0, W, 230),
             "corner_left": (0, 258, 466, 154),
             "corner_right": (492, 258, W - 492, 154)}
    statics = [(0, 230, W, 28, _stone(rng, 28, W)),
               (466, 258, 26, 154, _stone(rng, 154, 26))]
    overlays = [(0, 412, W, H - 412, _scoreboard(W, H - 412))]
    frames = _render(rng, views, statics, overlays)
    cams = ld.detect_layout_from_frames(frames)["cameras"]
    assert set(cams) == {"main", "corner_left", "corner_right"}
    for name, exp in views.items():
        assert _close(cams[name]["crop"], exp), (name, cams[name]["crop"], exp)


def test_single_camera_below_scoreboard_band_has_no_corners():
    rng = np.random.default_rng(3)
    views = {"main": (0, 128, W, H - 128)}
    overlays = [(0, 0, W, 128, _scoreboard(W, 128))]
    frames = _render(rng, views, [], overlays)
    out = ld.detect_layout_from_frames(frames)
    assert set(out["cameras"]) == {"main"}
    assert _close(out["cameras"]["main"]["crop"], views["main"])


def test_tiled_views_without_separator_and_centre_panel():
    """txama-style: main directly above the lower row (no strip), lower row =
    camera | static centre panel | camera."""
    rng = np.random.default_rng(4)
    views = {"main": (0, 0, W, 330),
             "corner_left": (0, 330, 318, H - 330),
             "corner_right": (642, 330, W - 642, H - 330)}
    panel = np.full((H - 330, 324, 3), 245, np.uint8)
    panel[30:60, 30:290] = (40, 40, 200)
    panel[120:150, 30:200] = (200, 120, 20)
    overlays = [(318, 330, 324, H - 330, panel)]
    frames = _render(rng, views, [], overlays)
    cams = ld.detect_layout_from_frames(frames)["cameras"]
    for name, exp in views.items():
        assert name in cams, cams
        assert _close(cams[name]["crop"], exp), (name, cams[name]["crop"], exp)


def test_outlier_frames_are_rejected():
    rng = np.random.default_rng(5)
    views = {"main": (0, 0, W, 352), "corner_left": (0, 374, 468, H - 374),
             "corner_right": (494, 374, W - 494, H - 374)}
    statics = [(0, 352, W, 22, _stone(rng, 22, W)), (468, 374, 26, H - 374, _stone(rng, H - 374, 26))]
    frames = _render(rng, views, statics, [])
    frames[3] = np.full_like(frames[0], 250)      # full-screen results graphic
    frames[7] = np.full_like(frames[0], 10)
    out = ld.detect_layout_from_frames(frames)
    assert {3, 7} <= set(out["diag"]["rejected"])
    assert _close(out["cameras"]["main"]["crop"], views["main"])


# --------------------------------------------------------------------------
# per-event layout store
# --------------------------------------------------------------------------

@pytest.fixture
def uploads(tmp_path, monkeypatch):
    from web.field import store
    monkeypatch.setattr(store, "UPLOAD_DIR", tmp_path)
    return tmp_path


_DETECTED = {
    "frame_size": [1920, 1080],
    "cameras": {"main": {"crop": [0, 0, 1920, 460], "masks": [[400, 0, 1000, 60]]},
                "corner_left": {"crop": [0, 515, 935, 310], "masks": []}},
    "confidence": {"main": 0.9, "corner_left": 0.7},
    "diag": {"n_used": 40},
}


def test_load_layout_without_event_is_global_default(uploads):
    default = cl.load_layout()
    assert default["main"].crop == (0, 0, 1920, 723)
    assert cl.load_layout(event_key="2026nope") == default


def test_saved_event_layout_is_loaded_for_that_event_only(uploads):
    p = cl.save_event_layout("2026hiho", _DETECTED, status="auto", source="detect")
    assert p.name == "composite_layout_2026hiho.json" and p.parent == uploads
    lay = cl.load_layout(event_key="2026hiho")
    assert set(lay) == {"main", "corner_left"}
    assert lay["main"].crop == (0, 0, 1920, 460)
    assert lay["main"].masks == ((400, 0, 1000, 60),)
    assert cl.load_layout(event_key="2026cur")["main"].crop == (0, 0, 1920, 723)
    assert cl.load_layout()["main"].crop == (0, 0, 1920, 723)     # positional default kept


def test_event_record_keeps_status_confidence_and_diag(uploads):
    cl.save_event_layout("2026hiho", _DETECTED, status="confirmed", source="manual")
    rec = cl.load_event_layout_record("2026hiho")
    assert rec["status"] == "confirmed" and rec["source"] == "manual"
    assert rec["confidence"] == {"main": 0.9, "corner_left": 0.7}
    assert rec["diag"] == {"n_used": 40}
    assert rec["event_key"] == "2026hiho" and rec["saved_at"]
    assert cl.load_event_layout_record("2026other") is None


def test_rejected_or_mainless_record_falls_back_to_default(uploads):
    cl.save_event_layout("2026hiho", _DETECTED, status="rejected")
    assert cl.load_layout(event_key="2026hiho")["main"].crop == (0, 0, 1920, 723)
    no_main = {"cameras": {"corner_left": {"crop": [0, 515, 935, 310]}}}
    cl.save_event_layout("2026x", no_main, status="auto")
    assert cl.load_layout(event_key="2026x")["main"].crop == (0, 0, 1920, 723)


def test_corrupt_record_falls_back_to_default(uploads):
    cl.event_layout_path("2026bad").write_text("{not json")
    assert cl.load_event_layout_record("2026bad") is None
    assert cl.load_layout(event_key="2026bad")["main"].crop == (0, 0, 1920, 723)


def test_event_key_is_made_filename_safe(uploads):
    cl.save_event_layout("../2026 hi/ho", _DETECTED)
    assert (uploads / "composite_layout_2026hiho.json").exists()
    assert cl.load_layout(event_key="../2026 hi/ho")["main"].crop == (0, 0, 1920, 460)
    with pytest.raises(ValueError):
        cl.save_event_layout("../ /", _DETECTED)


def test_save_validates_status_and_crops(uploads):
    with pytest.raises(ValueError):
        cl.save_event_layout("2026hiho", _DETECTED, status="maybe")
    bad = {"cameras": {"main": {"crop": [0, 0, 0, 100]}}}
    with pytest.raises(ValueError):
        cl.save_event_layout("2026hiho", bad)


def test_layout_to_cameras_accepts_detect_output():
    cams = cl.layout_to_cameras(_DETECTED)
    assert cams["corner_left"] == cl.CameraLayout("corner_left", (0, 515, 935, 310), ())


def test_load_views_uses_event_layout(uploads, monkeypatch):
    from web.field import camera_view as cv_mod
    cl.save_event_layout("2026hiho", _DETECTED)
    monkeypatch.setattr(cv_mod.field_store, "load_versions", lambda ek, cam: [])
    monkeypatch.setattr(cv_mod.field_store, "load", lambda ek, cam: object())
    views = {v.name: v for v in cv_mod.load_views("2026hiho")}
    assert set(views) == {"main", "corner_left"}
    assert views["main"].crop == (0, 0, 1920, 460)
    other = {v.name: v for v in cv_mod.load_views("2026cur")}
    assert other["main"].crop == (0, 0, 1920, 723) and "corner_right" in other
