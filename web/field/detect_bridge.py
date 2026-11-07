"""Adapter: YOLO image boxes -> field-space detections / field-adjacent persons."""
from __future__ import annotations
import math
import os
from web.field.calibration import push_from_camera
from web.field.tracker import FieldDetection
from web.field.zones import point_in_zones, ray_enters_zones

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
# The oblique corner-camera projections are only reliable in their nearby
# alliance zone. The 2026 bump/trench boundary is roughly 4.7 m from each end.
CORNER_ALLIANCE_ZONE_DEPTH_M = float(
    os.environ.get("CORNER_ALLIANCE_ZONE_DEPTH_M", "4.8")
)
# Beyond the trusted band, corner evidence decays exponentially with this scale
# rather than vanishing — the old hard cutoff left center field covered by the
# main camera alone (hub-occluded in places), so a midfield robot the main view
# missed was invisible system-wide even while a corner camera tracked it.
CORNER_ZONE_FALLOFF_M = float(os.environ.get("CORNER_ZONE_FALLOFF_M", "2.0"))
# Corner evidence is hard-zeroed past the field midline plus this margin: the
# far half is extreme range for an oblique corner view and mostly hub-occluded.
CORNER_MIDLINE_MARGIN_M = float(os.environ.get("CORNER_MIDLINE_MARGIN_M", "1.0"))
# Range-stable width backstop. The footprint check above is disabled past
# DET_FIELD_WIDTH_MAX_RANGE_M because back-projecting box corners to the carpet is
# ill-conditioned near the horizon — which leaves long-range boxes with NO width
# sanity. A SAM2 mask that leaks horizontally onto adjacent field structure then
# passes as a robot far wider than physically possible. Instead, FORWARD-project a
# max-robot-width carpet segment at the detection's field position (well-conditioned
# at any range) and reject a box wider than this factor times that expected width.
# 3.0 clears a genuine robot (and the ~2x far box guarded by the oblique-range
# regression test) with margin, while catching the ~3.5x+ leaks.
DET_MAX_WIDTH_PX_FACTOR = float(os.environ.get("DET_MAX_WIDTH_PX_FACTOR", "3.0"))
# Scope of the image-fraction backstop. "always": apply to every box (legacy).
# "uncalibrated": apply only when the pose-based forward-projected width check
# could not run — its documented purpose. With a full pose the geometric checks
# bound plausibility, and a robot close to a low corner camera legitimately
# spans most of the short corner panel's height (qm10: 4027 passed every
# pose-based check and was silently dropped at 84% panel height).
DET_FRAC_BACKSTOP = os.environ.get("DET_FRAC_BACKSTOP", "always")


def _bottom_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def _on_field(xy, dims, margin):
    L, W = dims
    return -margin <= xy[0] <= L + margin and -margin <= xy[1] <= W + margin


def corner_detection_weight(
    view_name,
    xy,
    field_dims,
    zone_depth=CORNER_ALLIANCE_ZONE_DEPTH_M,
    falloff_m=CORNER_ZONE_FALLOFF_M,
    midline_margin_m=CORNER_MIDLINE_MARGIN_M,
):
    """Confidence weight (0..1) of a corner camera's evidence at field point xy.

    1.0 within ``zone_depth`` of the camera's own end wall (the band where the
    oblique ground projection is well-conditioned), decaying exponentially with
    ``falloff_m`` beyond it, and 0.0 past the field midline + margin. Non-corner
    views are never attenuated. The weight multiplies the detection's fusion
    influence (position average, color vote, embedding average) so distant
    corner evidence supplements the main camera without overriding it.
    """
    if view_name not in ("corner_left", "corner_right"):
        return 1.0
    if xy is None:
        return 0.0
    field_length = float(field_dims[0])
    x = float(xy[0])
    dist_from_end = x if view_name == "corner_left" else field_length - x
    if dist_from_end > field_length / 2.0 + float(midline_margin_m):
        return 0.0
    if dist_from_end <= float(zone_depth):
        return 1.0
    return math.exp(-(dist_from_end - float(zone_depth)) / float(falloff_m))


def corner_detection_in_alliance_zone(
    view_name,
    xy,
    field_dims,
    zone_depth=CORNER_ALLIANCE_ZONE_DEPTH_M,
):
    """Return whether a camera may contribute this projected field position.

    Thin wrapper over :func:`corner_detection_weight` — true wherever the
    weight is nonzero (anywhere up to the midline + margin for corner views).
    """
    return corner_detection_weight(view_name, xy, field_dims, zone_depth) > 0.0


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


def _expected_robot_pixel_width(field_xy, calibration, robot_width_m):
    """Pixel width of a ``robot_width_m`` carpet segment at ``field_xy``, oriented
    broadside to the camera (worst-case widest as the camera sees it).

    Returns None if the calibration cannot forward-project (no pose / no method /
    degenerate geometry). Unlike back-projecting box corners (image->field), this
    forward projection (field->image) has no horizon singularity, so it stays
    well-conditioned at long range — exactly where the footprint check is disabled.
    """
    cc = getattr(calibration, "camera_center", None)
    f2i = getattr(calibration, "field_to_image", None)
    if cc is None or f2i is None or field_xy is None:
        return None
    dx = float(field_xy[0]) - float(cc[0])
    dy = float(field_xy[1]) - float(cc[1])
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-6:
        return None
    ux, uy = -dy / n, dx / n  # unit vector broadside to the camera viewing ray
    half = float(robot_width_m) / 2.0
    try:
        left = f2i((field_xy[0] - ux * half, field_xy[1] - uy * half, 0.0))
        right = f2i((field_xy[0] + ux * half, field_xy[1] + uy * half, 0.0))
        if left is None or right is None:
            return None
        lx, ly, rx, ry = float(left[0]), float(left[1]), float(right[0]), float(right[1])
    except (TypeError, ValueError, IndexError):
        return None
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5


# Public alias: the size-prior hijack cull (sam2 wiring in app.py) consumes the
# forward-projected expectation; keep the private name for in-module callers.
expected_robot_pixel_width = _expected_robot_pixel_width


def box_exceeds_frac(box, frame_shape, max_box_frac):
    """True if box spans more than max_box_frac of the frame in either dimension.
    Calibration-independent backstop against oversized detections."""
    if frame_shape is None:
        return False
    fh, fw = frame_shape[0], frame_shape[1]
    x1, y1, x2, y2 = box
    return bool(fw and fh and (
        (x2 - x1) / fw > max_box_frac or (y2 - y1) / fh > max_box_frac
    ))


def _too_large(box, calibration, field_xy, frame_shape, max_field_width,
               max_box_frac, max_range, width_px_factor=DET_MAX_WIDTH_PX_FACTOR,
               frac_backstop=None):
    """True if the box is implausibly large to be a robot. Three independent layers:
    a carpet-footprint cap (only applied at near range, where the ground projection
    is well-conditioned), a forward-projected expected-width cap (well-conditioned at
    ANY range — catches leaked/oversized boxes the footprint cap can't at distance),
    and an image-fraction backstop (calibration-independent; see DET_FRAC_BACKSTOP
    for when it applies)."""
    if frac_backstop is None:
        frac_backstop = DET_FRAC_BACKSTOP
    dist = _camera_ground_dist(calibration, field_xy)
    if dist is not None and dist <= max_range:
        width_m = _field_footprint_width(box, calibration)
        if width_m is not None and width_m > max_field_width:
            return True
    expected_px = _expected_robot_pixel_width(field_xy, calibration, max_field_width)
    if expected_px is not None and expected_px > 0:
        if (box[2] - box[0]) > expected_px * width_px_factor:
            return True
        if frac_backstop == "uncalibrated":
            # Pose-based width check ran: it owns plausibility; the frame-
            # fraction backstop would wrongly kill tall close-range boxes in
            # the short corner panels.
            return False
    if box_exceeds_frac(box, frame_shape, max_box_frac):
        return True
    return False


def to_field_detections(robot_boxes, calibration, field_dims, colors,
                        margin=DET_FIELD_MARGIN_M, frame_shape=None,
                        max_field_width=DET_MAX_ROBOT_WIDTH_M,
                        max_box_frac=DET_MAX_BOX_FRAC,
                        width_max_range=DET_FIELD_WIDTH_MAX_RANGE_M,
                        blocked_zones=None,
                        rejected_out=None,
                        frac_backstop=None,
                        ground_offset_m: float = 0.0):
    """Project image boxes to field-space detections, dropping off-field and
    implausibly-large boxes. If `rejected_out` is a list, each dropped box is
    appended as {"box", "reason", "field_xy"} for hard-negative mining.

    ``ground_offset_m`` > 0 pushes each projected bottom-centre that far away
    from the camera (``calibration.push_from_camera``): the box bottom is the
    robot's near bumper edge, not its footprint centre. The on-field and
    too-large checks run on the SHIFTED point — the robot's true centre — so
    they agree with the position the detection carries downstream.

    ``blocked_zones`` is accepted for call-site compatibility but detections
    inside a zone are KEPT: zones gate slot *seeding* (FieldTracker._assign),
    not detection. Dropping here killed existing tracks that drove into a zone
    and starved zone-parked robots of the detector corroboration that protects
    them from the SAM2 hijack cull / capped-roster eviction."""
    del blocked_zones  # seed-time concern; see docstring
    out = []
    for idx, box in enumerate(robot_boxes):
        uv = _bottom_center(box)
        xy = push_from_camera(calibration, calibration.image_to_field(uv),
                              ground_offset_m)
        if xy is None or not _on_field(xy, field_dims, margin):
            if rejected_out is not None:
                rejected_out.append({"box": tuple(float(v) for v in box),
                                     "reason": "off_field", "field_xy": xy})
            continue
        if _too_large(box, calibration, xy, frame_shape, max_field_width,
                      max_box_frac, width_max_range, frac_backstop=frac_backstop):
            if rejected_out is not None:
                rejected_out.append({"box": tuple(float(v) for v in box),
                                     "reason": "too_large", "field_xy": xy})
            continue
        color = colors[idx] if idx < len(colors) and colors[idx] else "unknown"
        out.append(FieldDetection(pos=xy, color=color, box=tuple(box), score=1.0))
    return out


def filter_field_robot_boxes(
    robot_boxes,
    calibration,
    field_dims,
    frame_shape=None,
    margin=DET_FIELD_MARGIN_M,
    blocked_zones=None,
    ray_blocked_zones=None,
    frac_backstop=None,
    ground_offset_m: float = 0.0,
):
    """Return indices of boxes whose ground contact is a plausible field robot.

    This is the geometry-only counterpart to ``to_field_detections`` for paths
    such as SAM2 prompting that must retain their original box/color metadata.

    ``ray_blocked_zones``: zones additionally tested along the viewing ray
    BEYOND the contact point (see ``zones.ray_enters_zones``). Pass only for
    low cameras, whose occluded-base contacts land short of the structure.

    ``ground_offset_m`` > 0: on-field, ``blocked_zones`` and too-large checks
    use the camera-pushed robot centre (same point ``to_field_detections``
    emits). The ray-beyond-contact test keeps the RAW image contact: it is a
    signature of where the observed bottom edge lands (its 3 m look-ahead was
    tuned on raw contacts), and pushing its origin would extend the look-ahead
    and newly block real robots standing ~3 m in front of a structure.
    """
    cam_xy = None
    if ray_blocked_zones and calibration.camera_center is not None:
        cam_xy = (float(calibration.camera_center[0]),
                  float(calibration.camera_center[1]))
    kept = []
    for idx, box in enumerate(robot_boxes):
        contact = calibration.image_to_field(_bottom_center(box))
        xy = push_from_camera(calibration, contact, ground_offset_m)
        if xy is None or not _on_field(xy, field_dims, margin):
            continue
        if point_in_zones(xy, blocked_zones):
            continue
        if cam_xy is not None and ray_enters_zones(cam_xy, contact, ray_blocked_zones):
            continue
        if _too_large(
            box,
            calibration,
            xy,
            frame_shape,
            DET_MAX_ROBOT_WIDTH_M,
            DET_MAX_BOX_FRAC,
            DET_FIELD_WIDTH_MAX_RANGE_M,
            frac_backstop=frac_backstop,
        ):
            continue
        kept.append(idx)
    return kept


def field_adjacent_persons(person_boxes, calibration, field_dims, margin=2.0):
    keep = []
    for box in person_boxes:
        xy = calibration.image_to_field(_bottom_center(box))
        if xy is not None and _on_field(xy, field_dims, margin):
            keep.append(box)
    return keep
