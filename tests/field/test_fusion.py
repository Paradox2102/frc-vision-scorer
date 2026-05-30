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
    per_cam = {
        "main": [(_det(5.0, 3.0, "blue"), b"m")],
        "corner_left": [(_det(5.2, 3.1, "blue"), b"c")],
    }
    fused = fuse_detections(per_cam, weights={"main": 1.0, "corner_left": 2.5}, radius_m=0.6)
    assert len(fused) == 1
    obs = fused[0]
    assert 5.0 < obs.detection.pos[0] <= 5.2
    assert {e[0] for e in obs.evidence} == {"main", "corner_left"}


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
