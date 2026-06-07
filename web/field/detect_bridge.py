"""Adapter: YOLO image boxes -> field-space detections / field-adjacent persons."""
from __future__ import annotations
import os
from web.field.tracker import FieldDetection

# Field-boundary slack (m) when deciding a detection's footprint is "on the field".
# Tightened from the original 1.0 m, which let field-edge clutter (scoreboard side
# panels, scorekeeper tablets) project just inside the boundary and seed phantom tracks.
DET_FIELD_MARGIN_M = float(os.environ.get("DET_FIELD_MARGIN_M", "0.5"))
# Max plausible carpet footprint width (m) of a detection's bottom edge. An FRC robot
# frame is <=0.92 m; with bumpers ~1.1 m, ~1.5 m on the diagonal. Anything implying a
# wider ground span is not a robot (banner, hub panel, half-frame false positive).
DET_MAX_ROBOT_WIDTH_M = float(os.environ.get("DET_MAX_ROBOT_WIDTH_M", "2.0"))
# Backstop for poor/absent calibration: reject a box wider or taller than this fraction
# of the frame. A real robot in a normal field view is well under half the frame.
DET_MAX_BOX_FRAC = float(os.environ.get("DET_MAX_BOX_FRAC", "0.5"))
# The field-footprint width check is only trustworthy at near range. As a detection
# approaches the horizon in an oblique (corner) camera, the inverse ground-plane
# projection becomes ill-conditioned and inflates the apparent footprint — which was
# wrongly rejecting real, distant robots. Beyond this camera-to-detection ground
# distance (m), skip the width check and rely on the image-fraction backstop instead.
DET_FIELD_WIDTH_MAX_RANGE_M = float(os.environ.get("DET_FIELD_WIDTH_MAX_RANGE_M", "8.0"))


def _bottom_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def _on_field(xy, dims, margin):
    L, W = dims
    return -margin <= xy[0] <= L + margin and -margin <= xy[1] <= W + margin


def _field_footprint_width(box, calibration):
    """Carpet width (m) implied by the box's bottom edge, or None if either bottom
    corner can't be projected to the ground plane (ray behind camera / above horizon)."""
    x1, _y1, x2, y2 = box
    bl = calibration.image_to_field((x1, y2))
    br = calibration.image_to_field((x2, y2))
    if bl is None or br is None:
        return None
    return ((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2) ** 0.5


def _camera_ground_dist(calibration, xy):
    """Ground-plane distance (m) from the camera to field point xy, or None when the
    calibration has no pose (homography) or xy is unavailable."""
    cc = getattr(calibration, "camera_center", None)
    if cc is None or xy is None:
        return None
    return ((xy[0] - cc[0]) ** 2 + (xy[1] - cc[1]) ** 2) ** 0.5


def _too_large(box, calibration, field_xy, frame_shape, max_field_width,
               max_box_frac, max_range):
    """True if the box is implausibly large to be a robot. Two independent layers:
    a carpet-footprint cap (only applied at near range, where the ground projection
    is well-conditioned) and an image-fraction backstop (calibration-independent)."""
    dist = _camera_ground_dist(calibration, field_xy)
    if dist is not None and dist <= max_range:
        width_m = _field_footprint_width(box, calibration)
        if width_m is not None and width_m > max_field_width:
            return True
    if frame_shape is not None:
        fh, fw = frame_shape[0], frame_shape[1]
        x1, y1, x2, y2 = box
        if fw and fh and ((x2 - x1) / fw > max_box_frac or (y2 - y1) / fh > max_box_frac):
            return True
    return False


def to_field_detections(robot_boxes, calibration, field_dims, colors,
                        margin=DET_FIELD_MARGIN_M, frame_shape=None,
                        max_field_width=DET_MAX_ROBOT_WIDTH_M,
                        max_box_frac=DET_MAX_BOX_FRAC,
                        width_max_range=DET_FIELD_WIDTH_MAX_RANGE_M,
                        rejected_out=None):
    """Project image boxes to field-space detections, dropping off-field and
    implausibly-large boxes. If `rejected_out` is a list, each dropped box is
    appended as {"box", "reason", "field_xy"} for hard-negative mining."""
    out = []
    for idx, box in enumerate(robot_boxes):
        uv = _bottom_center(box)
        xy = calibration.image_to_field(uv)
        if xy is None or not _on_field(xy, field_dims, margin):
            if rejected_out is not None:
                rejected_out.append({"box": tuple(float(v) for v in box),
                                     "reason": "off_field", "field_xy": xy})
            continue
        if _too_large(box, calibration, xy, frame_shape, max_field_width,
                      max_box_frac, width_max_range):
            if rejected_out is not None:
                rejected_out.append({"box": tuple(float(v) for v in box),
                                     "reason": "too_large", "field_xy": xy})
            continue
        color = colors[idx] if idx < len(colors) and colors[idx] else "unknown"
        out.append(FieldDetection(pos=xy, color=color, box=tuple(box), score=1.0))
    return out


def field_adjacent_persons(person_boxes, calibration, field_dims, margin=2.0):
    keep = []
    for box in person_boxes:
        xy = calibration.image_to_field(_bottom_center(box))
        if xy is not None and _on_field(xy, field_dims, margin):
            keep.append(box)
    return keep
