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
