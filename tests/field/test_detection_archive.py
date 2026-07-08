import json
from web.field.detection_archive import DetectionArchive


def test_record_and_corroborate_roundtrip(tmp_path):
    p = tmp_path / "detections.jsonl"
    a = DetectionArchive(str(p))
    a.record("main", 10, 300, [((100, 100, 200, 200), "red", False)])
    # overlapping box within window -> True
    assert a.corroborates("main", 310, (110, 110, 210, 210),
                          max_gap_frames=60) is True
    # non-overlapping box -> False
    assert a.corroborates("main", 310, (500, 500, 600, 600),
                          max_gap_frames=60) is False
    # outside the window -> None (no coverage; callers fail open)
    assert a.corroborates("main", 900, (110, 110, 210, 210),
                          max_gap_frames=60) is None
    # unknown view -> None
    assert a.corroborates("corner1", 310, (110, 110, 210, 210),
                          max_gap_frames=60) is None
    a.close()
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert rows[0]["view"] == "main" and rows[0]["source_frame"] == 300
    assert rows[0]["dets"][0]["color"] == "red"


def test_none_path_is_memory_only():
    a = DetectionArchive(None)
    a.record("main", 0, 0, [((0, 0, 10, 10), None, True)])
    assert a.corroborates("main", 5, (0, 0, 10, 10), max_gap_frames=10) is True
    a.close()
