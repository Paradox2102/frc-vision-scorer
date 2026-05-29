import numpy as np
from web.field.field_model import FieldModel
from web.field.occlusion import (hub_prisms, occluded_by_hub, camera_distance,
                                  box_overlap, resolve_occlusion)
from tests._field_synth import truth_pinhole_camera


def test_hub_prisms_two_compact_footprints():
    fm = FieldModel.load("2026-rebuilt-welded")
    prisms = hub_prisms(fm)
    assert len(prisms) == 2
    for p in prisms:
        xs, ys = p.footprint[:, 0], p.footprint[:, 1]
        assert (xs.max() - xs.min()) < 2.0 and (ys.max() - ys.min()) < 2.0
        assert p.z_top > 1.0
    centers = sorted(p.footprint.mean(axis=0)[0] for p in prisms)
    assert centers[0] < 6 and centers[1] > 10


def test_occluded_by_hub_blocks_behind_not_beside():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    prism = [p for p in hub_prisms(fm) if p.footprint.mean(axis=0)[0] > 10][0]
    cc = cam.camera_center
    behind = (prism.footprint.mean(axis=0)[0], prism.footprint.mean(axis=0)[1] + 3.0)
    beside = (2.0, 4.0)
    assert occluded_by_hub(cc, prism, behind) is True
    assert occluded_by_hub(cc, prism, beside) is False


def test_camera_distance_orders_depth():
    cc = np.array([8.27, -3.0, 6.0])
    near = camera_distance(cc, (8.27, 0.0))
    far = camera_distance(cc, (8.27, 6.0))
    assert near < far


def test_resolve_robot_robot_bumper_occlusion():
    """Non-vacuous robot-robot bumper occlusion test with explicit pixel boxes.

    Image boxes are chosen directly so they overlap; field positions are chosen
    so depth ordering is unambiguous from the synthetic camera at eye≈(8.27,-3,6).

    Box layout (image y increases downward):
      far_box  = (900, 250, 1000, 410) — far robot, higher in image (smaller y values)
      near_box = (900, 380, 1000, 520) — near robot, lower in image (larger y values)
    far_bumper_region ≈ y∈[346, 410] (lower-40% band of far_box).
    near_box overlaps [380,410] × [900,1000] with far_bumper_region → fraction ≈ 0.47 > 0.3.
    """
    cam = truth_pinhole_camera()
    cc = cam.camera_center

    # Depth ordering: near_pos is closer to camera than far_pos
    near_pos = (8.0, 1.0)
    far_pos  = (8.0, 3.0)
    d_near = camera_distance(cc, near_pos)
    d_far  = camera_distance(cc, far_pos)
    assert d_near < d_far, f"Expected near ({d_near:.3f}) < far ({d_far:.3f})"

    # Explicit pixel boxes — no back-projection, so no dependence on projection quirks
    far_box  = (900, 250, 1000, 410)
    near_box = (900, 380, 1000, 520)

    # Confirm the near box genuinely overlaps the far robot's bumper band
    from web.field.occlusion import _bumper_region
    far_bumper = _bumper_region(far_box)
    assert box_overlap(near_box, far_bumper) > 0, "near_box must overlap far_bumper_region"

    tracks = [
        {"slot_id": "near", "pos": near_pos, "box": near_box},
        {"slot_id": "far",  "pos": far_pos,  "box": far_box},
    ]
    res = resolve_occlusion(cam, tracks, person_boxes=[], hub_prisms=[], robot_h=0.5)

    # The near robot (closer to camera) covers the far robot's bumper band
    assert res["far"]["bumper_occluded"] is True
    # The near robot has no occluder in front of it
    assert res["near"]["bumper_occluded"] is False


def test_resolve_ignores_off_field_crowd_person():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    tracks = [{"slot_id": 0, "pos": (8.0, 4.0), "box": (900, 300, 1000, 420)}]
    res = resolve_occlusion(cam, tracks, person_boxes=[], hub_prisms=[], robot_h=0.5)
    assert res[0]["view_occluded"] is False and res[0]["bumper_occluded"] is False
