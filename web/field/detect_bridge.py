"""Adapter: YOLO image boxes -> field-space detections / field-adjacent persons.

Import-light (no app.py import): the bumper-color reader is injected as color_fn so
this stays unit-testable with a stub."""
from __future__ import annotations
from web.field.tracker import FieldDetection


def _bottom_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def _on_field(xy, dims, margin):
    L, W = dims
    return -margin <= xy[0] <= L + margin and -margin <= xy[1] <= W + margin


def to_field_detections(robot_boxes, calibration, frame, field_dims, color_fn, margin=1.0):
    out = []
    for box in robot_boxes:
        uv = _bottom_center(box)
        xy = calibration.image_to_field(uv)
        if xy is None or not _on_field(xy, field_dims, margin):
            continue
        color = color_fn(frame, box)
        out.append(FieldDetection(pos=xy, color=color, box=tuple(box), score=1.0))
    return out


def field_adjacent_persons(person_boxes, calibration, field_dims, margin=2.0):
    keep = []
    for box in person_boxes:
        xy = calibration.image_to_field(_bottom_center(box))
        if xy is not None and _on_field(xy, field_dims, margin):
            keep.append(box)
    return keep
