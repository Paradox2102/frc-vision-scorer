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
    assert occluded_by_hub(cc, prism, behind, robot_h=0.5) is True
    assert occluded_by_hub(cc, prism, beside, robot_h=0.5) is False


def test_camera_distance_orders_depth():
    cc = np.array([8.27, -3.0, 6.0])
    near = camera_distance(cc, (8.27, 0.0))
    far = camera_distance(cc, (8.27, 6.0))
    assert near < far


def test_resolve_robot_robot_bumper_occlusion():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    near_xy = (8.0, 0.5); far_xy = (8.0, 3.0)
    def box_at(xy):
        u, v = cam.field_to_image((xy[0], xy[1], 0.3))
        return (u - 40, v - 80, u + 40, v + 10)
    tracks = [{"slot_id": 0, "pos": near_xy, "box": box_at(near_xy)},
              {"slot_id": 1, "pos": far_xy, "box": box_at(far_xy)}]
    res = resolve_occlusion(cam, tracks, person_boxes=[], hub_prisms=[], robot_h=0.5)
    if box_overlap(tracks[0]["box"], tracks[1]["box"]) > 0:
        assert res[1]["bumper_occluded"] is True
        assert res[0]["bumper_occluded"] is False


def test_resolve_ignores_off_field_crowd_person():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    tracks = [{"slot_id": 0, "pos": (8.0, 4.0), "box": (900, 300, 1000, 420)}]
    res = resolve_occlusion(cam, tracks, person_boxes=[], hub_prisms=[], robot_h=0.5)
    assert res[0]["view_occluded"] is False and res[0]["bumper_occluded"] is False
