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
