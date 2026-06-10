import numpy as np
from web.field.field_model import FieldModel
from web.field.detect_bridge import (
    corner_detection_in_alliance_zone,
    field_adjacent_persons,
    to_field_detections,
)
from tests._field_synth import truth_pinhole_camera


def test_box_maps_to_field_point():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    gx, gy = 8.0, 4.0
    u, v = cam.field_to_image((gx, gy, 0.0))     # ground point
    box = (u - 30, v - 120, u + 30, v)           # bottom-center == (u, v)
    dets = to_field_detections(
        [box], cam, (fm.length_m, fm.width_m), colors=["blue"]
    )
    assert len(dets) == 1
    assert np.hypot(dets[0].pos[0] - gx, dets[0].pos[1] - gy) < 0.10
    assert dets[0].color == "blue"


def test_off_field_box_dropped():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    u, v = cam.field_to_image((30.0, 4.0, 0.0))  # way off field (x=30 > 16.5)
    box = (u - 30, v - 120, u + 30, v)
    dets = to_field_detections(
        [box], cam, (fm.length_m, fm.width_m), colors=["blue"]
    )
    assert dets == []


def test_oversized_field_footprint_dropped():
    """A box whose bottom edge spans ~5 m of carpet implies a 5 m-wide 'robot' —
    physically impossible (FRC robot + bumpers ~1.1 m), so it must be rejected even
    though its bottom-center lands on the field."""
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    ul, vl = cam.field_to_image((6.0, 4.0, 0.0))   # bottom-left  ~ field x=6
    ur, vr = cam.field_to_image((11.0, 4.0, 0.0))  # bottom-right ~ field x=11
    y2 = max(vl, vr)
    box = (min(ul, ur), y2 - 120, max(ul, ur), y2)
    dets = to_field_detections(
        [box], cam, (fm.length_m, fm.width_m), colors=["blue"]
    )
    assert dets == []


def test_far_oblique_detection_not_width_rejected():
    """Regression: at long range the ground-plane projection is ill-conditioned and
    inflates the apparent footprint, so the field-width filter must NOT fire there —
    it was killing real robots in the oblique corner cameras. The image-fraction
    backstop still guards genuinely-huge boxes."""
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    # bottom corners 4 m apart on carpet, far downrange (camDist > the near-range gate)
    ul, vl = cam.field_to_image((6.0, 7.8, 0.0))
    ur, vr = cam.field_to_image((10.0, 7.8, 0.0))
    y2 = max(vl, vr)
    box = (min(ul, ur), y2 - 60, max(ul, ur), y2)
    dets = to_field_detections(
        [box], cam, (fm.length_m, fm.width_m), colors=["blue"],
        frame_shape=(1080, 1920, 3),
    )
    assert len(dets) == 1   # kept: too far to trust the width measurement


def test_oversized_image_fraction_dropped():
    """A box spanning >50% of the frame width is rejected via the image-space backstop
    (defense for cameras with poor calibration where field footprint can't be trusted)."""
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    u, v = cam.field_to_image((8.0, 4.0, 0.0))
    box = (u - 600, v - 120, u + 600, v)           # 1200 px wide == 62% of 1920
    dets = to_field_detections(
        [box], cam, (fm.length_m, fm.width_m), colors=["blue"],
        frame_shape=(1080, 1920, 3),
    )
    assert dets == []


def test_far_leaked_wide_box_dropped():
    """A SAM2 mask that leaks horizontally onto adjacent field structure produces a
    box several times wider than any robot could appear at that range. Beyond the
    near-range footprint cutoff and under 50% of frame, the OLD checks let it through.
    The forward-projected expected-width check (well-conditioned at range) rejects it.

    Mirrors the real failure: R1 at ~14 m with a box ~3.6x the widest a 2 m robot
    could project to. The 4 m far box in test_far_oblique_detection_not_width_rejected
    (~2x) must still be kept — this only fires on gross leaks.
    """
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    # ~7.2 m-wide bottom edge at a far point (≈3.5x a 2 m broadside robot here)
    ul, vl = cam.field_to_image((5.0, 7.8, 0.0))
    ur, vr = cam.field_to_image((12.2, 7.8, 0.0))
    y2 = max(vl, vr)
    box = (min(ul, ur), y2 - 60, max(ul, ur), y2)
    # sanity: under the 50% frame backstop, so only the new check can catch it
    assert (max(ul, ur) - min(ul, ur)) / 1920.0 < 0.5
    dets = to_field_detections(
        [box], cam, (fm.length_m, fm.width_m), colors=["red"],
        frame_shape=(1080, 1920, 3),
    )
    assert dets == []


def test_normal_robot_footprint_kept():
    """A box implying a ~0.9 m footprint on the carpet survives the size filter."""
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    ul, vl = cam.field_to_image((8.0, 4.0, 0.0))
    ur, vr = cam.field_to_image((8.9, 4.0, 0.0))
    y2 = max(vl, vr)
    box = (min(ul, ur), y2 - 120, max(ul, ur), y2)
    dets = to_field_detections(
        [box], cam, (fm.length_m, fm.width_m), colors=["red"],
        frame_shape=(1080, 1920, 3),
    )
    assert len(dets) == 1


def test_rejected_out_records_reasons():
    """rejected_out collects dropped boxes with a reason, for hard-negative mining."""
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    # an off-field box
    uo, vo = cam.field_to_image((30.0, 4.0, 0.0))
    off = (uo - 30, vo - 120, uo + 30, vo)
    # an oversized (on-field bottom-center) box
    ul, vl = cam.field_to_image((6.0, 4.0, 0.0))
    ur, vr = cam.field_to_image((11.0, 4.0, 0.0))
    y2 = max(vl, vr)
    big = (min(ul, ur), y2 - 120, max(ul, ur), y2)
    # a normal robot
    un, vn = cam.field_to_image((8.0, 4.0, 0.0))
    ok = (un - 30, vn - 120, un + 30, vn)

    rejected = []
    dets = to_field_detections(
        [off, big, ok], cam, (fm.length_m, fm.width_m),
        colors=["blue", "blue", "red"], rejected_out=rejected,
    )
    assert len(dets) == 1
    reasons = sorted(r["reason"] for r in rejected)
    assert reasons == ["off_field", "too_large"]
    assert all("box" in r for r in rejected)


def test_missing_color_becomes_unknown():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    u, v = cam.field_to_image((8.0, 4.0, 0.0))
    box = (u - 30, v - 120, u + 30, v)
    dets = to_field_detections(
        [box], cam, (fm.length_m, fm.width_m), colors=[None]
    )
    assert dets[0].color == "unknown"


def test_detection_inside_blocked_structure_zone_is_kept():
    """Zones gate SEEDING (FieldTracker._assign), not detection: a robot that
    drives into a zone must keep feeding its existing track, and a zone-parked
    robot must keep corroborating its SAM2 mask (else eviction would see it as
    detector-absent). Zone hits are also NOT logged to rejected_out — a parked
    robot logged as a hard negative would poison the mining set."""
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    u, v = cam.field_to_image((8.0, 4.0, 0.0))
    box = (u - 30, v - 120, u + 30, v)
    rejected = []

    dets = to_field_detections(
        [box],
        cam,
        (fm.length_m, fm.width_m),
        colors=["blue"],
        blocked_zones=[(7.5, 3.5, 8.5, 4.5)],
        rejected_out=rejected,
    )

    assert len(dets) == 1
    assert dets[0].color == "blue"
    assert rejected == []


def test_field_adjacent_person_filter():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    u, v = cam.field_to_image((8.0, -0.3, 0.0))   # just outside near sideline
    near = (u - 25, v - 150, u + 25, v)
    uc, vc = cam.field_to_image((40.0, 4.0, 0.0))  # crowd: feet far off field
    crowd = (uc - 25, vc - 150, uc + 25, vc)
    keep = field_adjacent_persons([near, crowd], cam, (fm.length_m, fm.width_m), margin=2.0)
    assert near in keep and crowd not in keep


def test_box_exceeds_frac_true_when_wide():
    from web.field.detect_bridge import box_exceeds_frac
    # box width 700 / frame width 1000 = 0.7 > 0.5
    assert box_exceeds_frac((0, 0, 700, 50), (480, 1000, 3), 0.5) is True


def test_box_exceeds_frac_false_when_small():
    from web.field.detect_bridge import box_exceeds_frac
    assert box_exceeds_frac((0, 0, 100, 100), (480, 1000, 3), 0.5) is False


def test_box_exceeds_frac_none_frame_is_false():
    from web.field.detect_bridge import box_exceeds_frac
    assert box_exceeds_frac((0, 0, 999, 999), None, 0.5) is False


def test_filter_field_robot_boxes_rejects_off_field_and_oversized():
    from web.field.detect_bridge import filter_field_robot_boxes

    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    u_ok, v_ok = cam.field_to_image((8.0, 4.0, 0.0))
    u_off, v_off = cam.field_to_image((30.0, 4.0, 0.0))
    boxes = [
        (u_ok - 30, v_ok - 120, u_ok + 30, v_ok),
        (u_off - 30, v_off - 120, u_off + 30, v_off),
        (u_ok - 600, v_ok - 120, u_ok + 600, v_ok),
    ]

    assert filter_field_robot_boxes(
        boxes,
        cam,
        (fm.length_m, fm.width_m),
        frame_shape=(1080, 1920, 3),
    ) == [0]


def test_corner_views_contribute_up_to_midline_margin():
    """The old hard 4.8 m cutoff left center field covered by only the main
    camera. Corner evidence is now admitted with a decaying weight out to the
    midline (+margin); only the far half stays out of bounds."""
    dims = (16.541, 8.069)

    assert corner_detection_in_alliance_zone("corner_left", (4.7, 4.0), dims)
    assert corner_detection_in_alliance_zone("corner_left", (8.2, 4.0), dims)
    assert not corner_detection_in_alliance_zone("corner_left", (10.0, 4.0), dims)
    assert corner_detection_in_alliance_zone("corner_right", (11.8, 4.0), dims)
    assert corner_detection_in_alliance_zone("corner_right", (8.3, 4.0), dims)
    assert not corner_detection_in_alliance_zone("corner_right", (6.5, 4.0), dims)


def test_non_corner_views_remain_global():
    dims = (16.541, 8.069)

    assert corner_detection_in_alliance_zone("main", (8.0, 4.0), dims)
    assert corner_detection_in_alliance_zone("other", (8.0, 4.0), dims)


def test_corner_detection_weight_falloff_shape():
    from web.field.detect_bridge import corner_detection_weight
    dims = (16.541, 8.069)

    # Full trust within the near alliance band.
    assert corner_detection_weight("corner_left", (3.0, 4.0), dims) == 1.0
    assert corner_detection_weight("corner_left", (4.7, 4.0), dims) == 1.0
    # Monotone decay beyond the band toward the midline.
    w6 = corner_detection_weight("corner_left", (6.0, 4.0), dims)
    w8 = corner_detection_weight("corner_left", (8.2, 4.0), dims)
    assert 0.0 < w8 < w6 < 1.0
    # Hard zero past midline + margin (far half stays out of bounds).
    assert corner_detection_weight("corner_left", (9.5, 4.0), dims) == 0.0
    # Mirrored for corner_right (distance measured from the far end wall).
    assert corner_detection_weight("corner_right", (13.0, 4.0), dims) == 1.0
    w10 = corner_detection_weight("corner_right", (10.0, 4.0), dims)
    w83 = corner_detection_weight("corner_right", (8.3, 4.0), dims)
    assert 0.0 < w83 < w10 < 1.0
    assert corner_detection_weight("corner_right", (7.0, 4.0), dims) == 0.0
    # Non-corner views are never attenuated.
    assert corner_detection_weight("main", (8.0, 4.0), dims) == 1.0
    # Unprojectable position contributes nothing.
    assert corner_detection_weight("corner_left", None, dims) == 0.0
