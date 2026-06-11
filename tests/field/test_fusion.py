from web.field.fusion import fuse_detections, FusedObservation
from web.field.tracker import FieldDetection


def _det(x, y, color="blue", box=(0, 0, 10, 10)):
    return FieldDetection(pos=(x, y), color=color, box=box)


def test_single_camera_passthrough():
    per_cam = {"main": [(_det(1.0, 2.0), b"crop")]}
    fused = fuse_detections(per_cam, weights={"main": 1.0}, radius_m=0.6)
    assert len(fused) == 1
    obs = fused[0]
    assert isinstance(obs, FusedObservation)
    assert abs(obs.detection.pos[0] - 1.0) < 1e-9
    assert obs.evidence == [("main", b"crop", 1.0)]


def test_merges_two_cameras_same_robot():
    main_box = (10, 20, 30, 40)
    corner_box = (400, 200, 700, 500)
    per_cam = {
        "main": [(_det(5.0, 3.0, "blue", main_box), b"m")],
        "corner_left": [(_det(5.2, 3.1, "blue", corner_box), b"c")],
    }
    fused = fuse_detections(per_cam, weights={"main": 1.0, "corner_left": 2.5}, radius_m=0.6)
    assert len(fused) == 1
    obs = fused[0]
    assert obs.detection.pos == (5.0, 3.0)
    assert obs.detection.box == main_box
    assert {e[0] for e in obs.evidence} == {"main", "corner_left"}


def test_corner_only_detection_has_no_main_view_box():
    per_cam = {
        "corner_left": [(_det(3.0, 2.0, "blue", (400, 200, 700, 500)), b"c")],
    }
    fused = fuse_detections(
        per_cam,
        weights={"corner_left": 2.5},
        radius_m=0.6,
    )

    assert len(fused) == 1
    assert fused[0].detection.pos == (3.0, 2.0)
    assert fused[0].detection.box is None


def test_keeps_distinct_robots_separate():
    per_cam = {"main": [(_det(1, 1), b"a"), (_det(8, 7), b"b")]}
    fused = fuse_detections(per_cam, weights={"main": 1.0}, radius_m=0.6)
    assert len(fused) == 2


def test_merges_same_camera_duplicate_masks_on_one_robot():
    first = FieldDetection(
        pos=(6.70, 3.40), color="red", box=(690, 410, 890, 535), track_id=1)
    duplicate = FieldDetection(
        pos=(6.74, 3.43), color="red", box=(700, 418, 885, 530), track_id=3)
    fused = fuse_detections(
        {"main": [(first, b"a"), (duplicate, b"b")]},
        weights={"main": 1.0},
    )

    assert len(fused) == 1
    assert fused[0].detection.color == "red"
    assert fused[0].members == [("main", 1)]


def test_keeps_nearby_same_camera_robots_when_boxes_do_not_overlap():
    left = _det(6.70, 3.40, "red", (600, 400, 700, 520))
    right = _det(6.95, 3.42, "red", (720, 400, 820, 520))

    fused = fuse_detections(
        {"main": [(left, b"a"), (right, b"b")]},
        weights={"main": 1.0},
    )

    assert len(fused) == 2


def test_same_camera_duplicate_cluster_uses_color_majority():
    box = (690, 410, 890, 535)
    per_cam = {
        "main": [
            (FieldDetection((6.70, 3.40), "red", box, track_id=1), b"a"),
            (FieldDetection((6.72, 3.41), "blue", box, track_id=2), b"b"),
            (FieldDetection((6.74, 3.43), "red", box, track_id=3), b"c"),
        ],
    }

    fused = fuse_detections(per_cam, weights={"main": 1.0})

    assert len(fused) == 1
    assert fused[0].detection.color == "red"


def test_color_vote_resolves_unknown():
    per_cam = {
        "main": [(_det(2, 2, "unknown"), b"m")],
        "corner_left": [(_det(2.1, 2.0, "blue"), b"c")],
    }
    fused = fuse_detections(per_cam, weights={"main": 1.0, "corner_left": 2.5}, radius_m=0.6)
    assert fused[0].detection.color == "blue"


def test_members_single_camera():
    # per-camera FieldDetection carries its SAM2 obj id as track_id
    d = FieldDetection(pos=(1.0, 2.0), color="blue", box=(0, 0, 10, 10), track_id=1)
    per_cam = {"main": [(d, b"crop")]}
    fused = fuse_detections(per_cam, weights={"main": 1.0}, radius_m=0.6)
    assert fused[0].members == [("main", 1)]
    assert fused[0].track_id is None  # registry assigns it later; default is None


def test_members_two_cameras_same_robot():
    dm = FieldDetection(pos=(5.0, 3.0), color="blue", box=(0, 0, 10, 10), track_id=2)
    dc = FieldDetection(pos=(5.2, 3.1), color="blue", box=(0, 0, 10, 10), track_id=1)
    per_cam = {"main": [(dm, b"m")], "corner_left": [(dc, b"c")]}
    fused = fuse_detections(per_cam, weights={"main": 1.0, "corner_left": 2.5}, radius_m=0.6)
    assert len(fused) == 1
    assert set(fused[0].members) == {("main", 2), ("corner_left", 1)}


def test_hub_occluded_main_yields_position_to_visible_corner():
    per_cam = {
        "main": [(_det(5.0, 3.0), b"m")],
        "corner_left": [(_det(5.3, 3.1), b"c")],
    }

    fused = fuse_detections(
        per_cam,
        weights={"main": 1.0, "corner_left": 2.5},
        is_hub_occluded=lambda cam, _pos: cam == "main",
    )

    assert fused[0].detection.pos == (5.3, 3.1)
    assert fused[0].detection.box == (0, 0, 10, 10)
    assert {camera for camera, _crop, _weight in fused[0].evidence} == {
        "main", "corner_left",
    }


def test_visible_main_remains_position_reference():
    per_cam = {
        "main": [(_det(5.0, 3.0), b"m")],
        "corner_left": [(_det(5.3, 3.1), b"c")],
    }

    fused = fuse_detections(
        per_cam,
        weights={"main": 1.0, "corner_left": 2.5},
        is_hub_occluded=lambda _cam, _pos: False,
    )

    assert fused[0].detection.pos == (5.0, 3.0)


def test_all_hub_occluded_views_fall_back_to_existing_main_preference():
    per_cam = {
        "main": [(_det(5.0, 3.0), b"m")],
        "corner_left": [(_det(5.3, 3.1), b"c")],
    }

    fused = fuse_detections(
        per_cam,
        weights={"main": 1.0, "corner_left": 2.5},
        is_hub_occluded=lambda _cam, _pos: True,
    )

    assert fused[0].detection.pos == (5.0, 3.0)


def test_member_weight_scales_with_detection_score():
    """A detection's score (e.g. corner-camera falloff weight) attenuates its
    fusion influence: position averaging and color votes both follow it."""
    # Two corner detections of the same robot, conflicting colors. The full-score
    # member must win the color vote despite equal camera weights.
    per_cam = {
        "corner_left": [(FieldDetection(pos=(8.0, 4.0), color="red", box=None, score=1.0), b"a")],
        "corner_right": [(FieldDetection(pos=(8.4, 4.0), color="blue", box=None, score=0.2), b"b")],
    }
    fused = fuse_detections(per_cam, weights={"corner_left": 1.0, "corner_right": 1.0}, radius_m=0.6)
    assert len(fused) == 1
    obs = fused[0]
    assert obs.detection.color == "red"
    # Weighted position: (8.0*1.0 + 8.4*0.2) / 1.2 = 8.0667 — pulled mostly to
    # the full-score member, not the midpoint (8.2).
    assert abs(obs.detection.pos[0] - 8.0667) < 1e-3
    # Evidence carries the effective (camera * score) weight.
    weights = {cam: w for (cam, _crop, w) in obs.evidence}
    assert abs(weights["corner_left"] - 1.0) < 1e-9
    assert abs(weights["corner_right"] - 0.2) < 1e-9


def test_default_score_keeps_existing_weighting():
    """score defaults to 1.0, so paths that never set it are unchanged."""
    per_cam = {
        "main": [(FieldDetection(pos=(5.0, 3.0), color="blue", box=(1, 2, 3, 4)), b"m")],
        "corner_left": [(FieldDetection(pos=(5.2, 3.1), color="blue", box=None), b"c")],
    }
    fused = fuse_detections(per_cam, weights={"main": 1.0, "corner_left": 2.5}, radius_m=0.6)
    assert fused[0].evidence == [("main", b"m", 1.0), ("corner_left", b"c", 2.5)]
