import numpy as np
from web.field.field_model import FieldModel
from web.field.detect_bridge import to_field_detections, field_adjacent_persons
from tests._field_synth import truth_pinhole_camera


def _color_stub(frame, box):
    return "blue"


def test_box_maps_to_field_point():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    gx, gy = 8.0, 4.0
    u, v = cam.field_to_image((gx, gy, 0.0))     # ground point
    box = (u - 30, v - 120, u + 30, v)           # bottom-center == (u, v)
    dets = to_field_detections([box], cam, frame=None, field_dims=(fm.length_m, fm.width_m),
                               color_fn=_color_stub)
    assert len(dets) == 1
    assert np.hypot(dets[0].pos[0] - gx, dets[0].pos[1] - gy) < 0.10
    assert dets[0].color == "blue"


def test_off_field_box_dropped():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    u, v = cam.field_to_image((30.0, 4.0, 0.0))  # way off field (x=30 > 16.5)
    box = (u - 30, v - 120, u + 30, v)
    dets = to_field_detections([box], cam, None, (fm.length_m, fm.width_m), _color_stub)
    assert dets == []


def test_field_adjacent_person_filter():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    u, v = cam.field_to_image((8.0, -0.3, 0.0))   # just outside near sideline
    near = (u - 25, v - 150, u + 25, v)
    uc, vc = cam.field_to_image((40.0, 4.0, 0.0))  # crowd: feet far off field
    crowd = (uc - 25, vc - 150, uc + 25, vc)
    keep = field_adjacent_persons([near, crowd], cam, (fm.length_m, fm.width_m), margin=2.0)
    assert near in keep and crowd not in keep
