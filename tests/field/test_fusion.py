from web.field.fusion import (
    fuse_detections, FusedObservation, merge_same_id_observations)
from web.field.tracker import FieldDetection


def _det(x, y, color="blue", box=(0, 0, 10, 10)):
    return FieldDetection(pos=(x, y), color=color, box=box)


def _obs(pos, box, tid, cam):
    det = FieldDetection(pos=pos, color="blue", box=box, track_id=tid)
    return FusedObservation(detection=det, evidence=[(cam, b"crop", 1.0)],
                            members=[(cam, tid)], track_id=tid)


def test_merge_collapses_same_track_id_prefers_main_geometry():
    # The qm8 bug: one robot, two fused observations sharing track_id=1 — a
    # main-backed one and a corner-only duplicate whose position has slid away.
    a = _obs((6.0, 3.4), (0, 0, 10, 10), 1, "main")       # real robot (main box)
    b = _obs((7.1, 4.26), None, 1, "corner_left")         # corner-only duplicate
    c = _obs((2.4, 5.1), (0, 0, 5, 5), 3, "main")         # a different robot
    merged, _ = merge_same_id_observations([a, b, c])
    assert len(merged) == 2                                # duplicate collapsed
    m1 = next(o for o in merged if o.track_id == 1)
    assert m1.detection.pos == (6.0, 3.4)                  # main geometry kept
    assert m1.detection.box == (0, 0, 10, 10)
    # corner identity/bearing still contributes (members + evidence unioned)
    assert set(m1.members) == {("main", 1), ("corner_left", 1)}
    assert len(m1.evidence) == 2


def test_merge_main_guard_keeps_two_main_backed_distinct_robots():
    # Two DISTINCT main-camera detections sharing a mis-assigned id (the red
    # collapse). With main_guard they stay separate so the tracker slots both;
    # without it they wrongly collapse.
    a = _obs((6.0, 3.4), (0, 0, 10, 10), 1, "main")
    b = _obs((9.4, 4.3), (0, 0, 10, 10), 1, "main")
    assert len(merge_same_id_observations([a, b], main_guard=True)[0]) == 2
    assert len(merge_same_id_observations([a, b], main_guard=False)[0]) == 1


def test_merge_main_guard_still_collapses_corner_duplicate():
    # A genuine corner duplicate (1 main + 1 corner-only) still collapses under
    # the guard — that is one robot, not two.
    a = _obs((6.0, 3.4), (0, 0, 10, 10), 1, "main")
    b = _obs((7.1, 4.26), None, 1, "corner_left")
    assert len(merge_same_id_observations([a, b], main_guard=True)[0]) == 1


def test_merge_passthrough_when_ids_distinct():
    a = _obs((6.0, 3.4), (0, 0, 10, 10), 1, "main")
    c = _obs((2.4, 5.1), (0, 0, 5, 5), 3, "main")
    merged, _ = merge_same_id_observations([a, c])
    assert len(merged) == 2


def test_merge_single_corner_only_obs_survives_for_handoff():
    # Main occluded, only the corner sees the robot -> one corner-only obs must
    # pass through so the tracker keeps the slot (the handoff case).
    b = _obs((7.0, 4.0), None, 1, "corner_left")
    merged, _ = merge_same_id_observations([b])
    assert len(merged) == 1 and merged[0].detection.box is None
    assert merged[0].track_id == 1


def test_merge_keeps_evidence_list_aligned():
    a = _obs((6.0, 3.4), (0, 0, 10, 10), 1, "main")
    b = _obs((7.1, 4.26), None, 1, "corner_left")
    c = _obs((2.4, 5.1), (0, 0, 5, 5), 3, "main")
    ev = ["evA", "evB", "evC"]
    merged, merged_ev = merge_same_id_observations([a, b, c], ev)
    assert len(merged_ev) == len(merged) == 2
    # the id-1 group keeps its main member's evidence ("evA")
    i1 = next(i for i, o in enumerate(merged) if o.track_id == 1)
    assert merged_ev[i1] == "evA"


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


# --- ray-aware cross-camera gating -----------------------------------------
# A low oblique camera projecting a mask bottom to the carpet has anisotropic
# error: bearing stays accurate while range error grows with distance (the
# visible bottom sits above true ground contact, pushing the estimate AWAY
# from the camera). Geometry below is measured from 2026hop qm24: corner_right
# at (15.82, -0.23), corner_left at (0.64, -0.30).

CENTERS = {"corner_left": (0.64, -0.30), "corner_right": (15.82, -0.23)}
W2 = {"main": 1.0, "corner_left": 2.5, "corner_right": 2.5}


def test_ray_gate_merges_along_ray_range_error():
    # Measured split pair: corner_right saw the robot at (14.72, 6.91); main
    # (trusted) at (15.30, 4.49) — 2.48 m apart but only 0.20 m off the ray.
    per_cam = {
        "main": [(_det(15.30, 4.49, "blue", (100, 100, 200, 200)), b"m")],
        "corner_right": [(_det(14.72, 6.91, "blue", None), b"c")],
    }
    without = fuse_detections(per_cam, weights=W2)
    assert len(without) == 2  # default behavior unchanged
    fused = fuse_detections(per_cam, weights=W2, camera_centers=CENTERS)
    assert len(fused) == 1
    assert fused[0].detection.pos == (15.30, 4.49)  # main still owns geometry
    assert set(fused[0].members) == {("main", None), ("corner_right", None)}


def test_ray_gate_merges_close_range_clipped_corner_pair():
    # Measured corner_left pair (calibration verified good): a robot 3.3 m
    # from the low corner camera with its box clipped at the crop edge showed
    # 0.97 m cross-track — half-robot near-face/clip offset plus main's own
    # ~0.3 m, all bounded in meters. Same color, so the wide gate may merge.
    per_cam = {
        "main": [(_det(2.13, 1.72, "red", (10, 10, 20, 20)), b"m")],
        "corner_left": [(_det(1.43, 2.95, "red", None), b"c")],
    }
    fused = fuse_detections(per_cam, weights=W2, camera_centers=CENTERS)
    assert len(fused) == 1


def test_ray_gate_vetoes_contradicting_colors():
    # Same geometry as the merge above, but the bumper evidence disagrees:
    # known, contradicting colors must never marry through geometric slack.
    per_cam = {
        "main": [(_det(2.13, 1.72, "blue", (10, 10, 20, 20)), b"m")],
        "corner_left": [(_det(1.43, 2.95, "red", None), b"c")],
    }
    fused = fuse_detections(per_cam, weights=W2, camera_centers=CENTERS)
    assert len(fused) == 2


def test_ray_gate_rejects_large_cross_track():
    # ~1.5 m of bearing disagreement exceeds every bounded error source.
    per_cam = {
        "main": [(_det(2.60, 1.50, "red", (10, 10, 20, 20)), b"m")],
        "corner_left": [(_det(1.43, 2.95, "red", None), b"c")],
    }
    fused = fuse_detections(per_cam, weights=W2, camera_centers=CENTERS)
    assert len(fused) == 2


def test_ray_gate_never_chains_two_main_robots():
    # Two real robots along the same corner ray (the nearer occludes the
    # farther in that camera). The corner det may merge with only ONE main
    # det, and the two main dets must never collapse into one observation.
    per_cam = {
        "main": [
            (FieldDetection(pos=(15.30, 4.49), color="blue", box=(0, 0, 10, 10), track_id=1), b"m1"),
            (FieldDetection(pos=(15.05, 5.55), color="blue", box=(40, 0, 50, 10), track_id=2), b"m2"),
        ],
        "corner_right": [(_det(14.72, 6.91, "blue", None), b"c")],
    }
    fused = fuse_detections(per_cam, weights=W2, camera_centers=CENTERS)
    assert len(fused) == 2
    main_ids = [m for f in fused for m in f.members if m[0] == "main"]
    assert len(main_ids) == 2  # both main robots survive as members somewhere


def test_ray_gate_along_tolerance_scales_with_range():
    # At short range the along-ray slack shrinks: a 2.5 m range error on a
    # ~3 m-range detection is implausible even for a carpet-level camera.
    per_cam = {
        # 2.5 m nearer corner_right than its (14.5, 2.4) estimate, on the ray
        "main": [(_det(15.62, 0.16, "red", (10, 10, 20, 20)), b"m")],
        "corner_right": [(_det(14.5, 2.4, "red", None), b"c")],
    }
    fused = fuse_detections(per_cam, weights=W2, camera_centers=CENTERS)
    assert len(fused) == 2


def test_ray_gate_rejects_other_det_beyond_corner_estimate():
    # Projection error pushes the corner estimate AWAY from its camera, so a
    # main det even farther out along the ray cannot be the same robot.
    per_cam = {
        "main": [(_det(13.91, 9.95, "blue", (10, 10, 20, 20)), b"m")],
        "corner_right": [(_det(14.72, 6.91, "blue", None), b"c")],
    }
    fused = fuse_detections(per_cam, weights=W2, camera_centers=CENTERS)
    assert len(fused) == 2


def test_ray_gate_pairs_corner_only_views():
    # Main misses the robot (e.g. hub occlusion); the two corner views still
    # unify through either camera's ray.
    per_cam = {
        "corner_right": [(_det(14.0, 6.0, "red", None), b"cr")],
        # 2.0 m nearer corner_right along its ray toward (14.0, 6.0)
        "corner_left": [(_det(14.56, 4.08, "red", None), b"cl")],
    }
    fused = fuse_detections(per_cam, weights=W2, camera_centers=CENTERS)
    assert len(fused) == 1
    assert {m[0] for m in fused[0].members} == {"corner_left", "corner_right"}
