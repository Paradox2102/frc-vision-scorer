import numpy as np
from web.field.camera_view import CameraView

def _frame():
    return np.full((1080, 1920, 3), 255, np.uint8)

def test_extract_crops_to_region():
    v = CameraView(name="corner_left", crop=(0, 648, 960, 432), masks=(), calibration=None)
    out = v.extract(_frame())
    assert out.shape == (432, 960, 3)

def test_extract_blanks_masks():
    v = CameraView(name="corner_right", crop=(960, 648, 960, 432),
                   masks=((10, 20, 30, 40),), calibration=None)
    out = v.extract(_frame())
    # masked rect is zeroed; a pixel outside it is untouched
    assert out[20:60, 10:40].sum() == 0
    assert out[0, 0].sum() == 255 * 3

def test_extract_is_a_copy():
    f = _frame()
    v = CameraView(name="main", crop=(0, 0, 1920, 723), masks=((0, 0, 5, 5),), calibration=None)
    v.extract(f)
    assert f[0, 0].sum() == 255 * 3  # masking did not mutate the source frame


from web.field import camera_view as cv_mod

def test_load_views_includes_only_calibrated(monkeypatch):
    from web.field.composite_layout import CameraLayout
    layout = {
        "main": CameraLayout("main", (0, 0, 1920, 723), ()),
        "corner_left": CameraLayout("corner_left", (0, 648, 960, 432), ()),
    }
    monkeypatch.setattr(cv_mod, "load_layout", lambda: layout)
    # main calibrated, corner_left not
    monkeypatch.setattr(cv_mod.field_store, "load",
                        lambda ek, cam: object() if cam == "main" else None)
    views = cv_mod.load_views("evt")
    assert [v.name for v in views] == ["main"]

def test_load_views_orders_main_first(monkeypatch):
    from web.field.composite_layout import CameraLayout
    layout = {
        "corner_left": CameraLayout("corner_left", (0, 648, 960, 432), ()),
        "main": CameraLayout("main", (0, 0, 1920, 723), ()),
    }
    monkeypatch.setattr(cv_mod, "load_layout", lambda: layout)
    monkeypatch.setattr(cv_mod.field_store, "load", lambda ek, cam: object())
    views = cv_mod.load_views("evt")
    assert views[0].name == "main"
