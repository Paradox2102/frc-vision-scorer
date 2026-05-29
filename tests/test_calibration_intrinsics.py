import numpy as np
import pytest
from web.field.calibration import Calibrator, CalibrationError
from web.field.field_model import FieldModel
from tests._field_synth import truth_camera, TRUTH_K, IMAGE_SIZE


def _view_from(eye, fm):
    cam = truth_camera(eye=eye)
    w, h = IMAGE_SIZE
    obj, img = [], []
    for tid in fm.tags:
        c3d = fm.tag_corners_3d(tid)
        uv = np.array([cam.field_to_image(p) for p in c3d])
        if np.all((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)):
            obj.append(c3d); img.append(uv)
    return (np.vstack(obj), np.vstack(img)) if obj else None


def test_intrinsics_recovered_from_multiple_viewpoints():
    fm = FieldModel.load("2026-rebuilt-welded")
    eyes = [(8.27,-3,6), (4,-2,5), (12,-2,5), (8.27,-5,8), (2,-1,4), (14,-1,4)]
    views = [v for v in (_view_from(e, fm) for e in eyes) if v is not None]
    assert len(views) >= 4
    K, D = Calibrator.calibrate_intrinsics(views, IMAGE_SIZE)
    assert abs(K[0,0] - TRUTH_K[0,0]) < 30 and abs(K[1,1] - TRUTH_K[1,1]) < 30  # ~3%
    assert abs(K[0,2] - TRUTH_K[0,2]) < 20 and abs(K[1,2] - TRUTH_K[1,2]) < 20


def test_intrinsics_needs_enough_viewpoints():
    fm = FieldModel.load("2026-rebuilt-welded")
    one = [_view_from((8.27,-3,6), fm)]
    with pytest.raises(CalibrationError):
        Calibrator.calibrate_intrinsics(one, IMAGE_SIZE)
