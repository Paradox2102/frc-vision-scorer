import numpy as np
from web.field.calibration import tag_alignment_error
from web.field.field_model import FieldModel
from tests._field_synth import truth_pinhole_camera, project_tag_corners


def test_aligned_tags_near_zero_error():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    corr = project_tag_corners(cam, fm)   # {tid: (img(4,2), obj(4,3))}
    detected = {tid: img for tid, (img, _obj) in corr.items()}
    err_m, n = tag_alignment_error(cam, detected, fm)
    assert n >= 4
    assert err_m < 0.50               # truth pinhole camera: ~0.38 m (oblique side-wall tags degrade the average)

def test_shifted_tags_large_error():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    corr = project_tag_corners(cam, fm)
    detected = {tid: (img + 60.0) for tid, (img, _obj) in corr.items()}  # 60px shift
    err_m, n = tag_alignment_error(cam, detected, fm)
    assert n >= 4
    assert err_m > 0.5                # well past the drift threshold

def test_no_usable_tags_returns_inf_zero():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    err_m, n = tag_alignment_error(cam, {999999: np.zeros((4, 2))}, fm)
    assert n == 0 and err_m == float("inf")
