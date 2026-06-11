import json

import numpy as np

from web.field.hard_negatives import HardNegativeLogger


def _frame():
    return np.zeros((120, 240, 3), dtype=np.uint8)


def _rejected():
    return [{"box": (10.0, 20.0, 40.0, 80.0), "reason": "off_field", "field_xy": None}]


def test_disabled_writes_nothing(tmp_path):
    log = HardNegativeLogger(tmp_path, enabled=False)
    out = log.maybe_log(_frame(), [], _rejected(), view_name="main", frame_num=0)
    assert out is None
    assert list(tmp_path.iterdir()) == []


def test_no_rejections_writes_nothing(tmp_path):
    log = HardNegativeLogger(tmp_path, enabled=True)
    assert log.maybe_log(_frame(), [(1, 2, 3, 4)], [], frame_num=0) is None
    assert list(tmp_path.iterdir()) == []


def test_enabled_writes_frame_manifest_and_preview(tmp_path):
    log = HardNegativeLogger(tmp_path, enabled=True, log_every=1)
    out = log.maybe_log(_frame(), [(5, 6, 30, 60)], _rejected(),
                        view_name="main", frame_num=7)
    assert out is not None and out.exists()
    files = {p.name for p in tmp_path.iterdir()}
    assert "main_00000007.jpg" in files
    assert "main_00000007_preview.jpg" in files
    assert "main_00000007.json" in files
    manifest = json.loads((tmp_path / "main_00000007.json").read_text())
    assert manifest["frame_num"] == 7
    assert len(manifest["rejected"]) == 1
    assert len(manifest["accepted"]) == 1


def test_capture_accepted_mode_logs_without_rejections(tmp_path):
    """With require_rejected=False (review/triage mode), a frame logs on accepted
    detections alone — used to surface on-field FPs (hub face, field bumps) that
    pass the geometry filters and so are never in `rejected`."""
    log = HardNegativeLogger(tmp_path, enabled=True, log_every=1)
    # default mode: accepted-only frame is NOT logged
    assert log.maybe_log(_frame(), [(1, 2, 3, 4)], [], frame_num=0) is None
    # review mode: it IS logged
    out = log.maybe_log(_frame(), [(1, 2, 3, 4)], [], frame_num=1,
                        require_rejected=False)
    assert out is not None and out.exists()


def test_rate_limit_skips_nearby_frames(tmp_path):
    log = HardNegativeLogger(tmp_path, enabled=True, log_every=30)
    assert log.maybe_log(_frame(), [], _rejected(), frame_num=0) is not None
    assert log.maybe_log(_frame(), [], _rejected(), frame_num=10) is None   # within stride
    assert log.maybe_log(_frame(), [], _rejected(), frame_num=40) is not None  # past stride


def test_max_files_cap(tmp_path):
    log = HardNegativeLogger(tmp_path, enabled=True, log_every=1, max_files=2)
    assert log.maybe_log(_frame(), [], _rejected(), frame_num=0) is not None
    assert log.maybe_log(_frame(), [], _rejected(), frame_num=1) is not None
    assert log.maybe_log(_frame(), [], _rejected(), frame_num=2) is None  # capped
