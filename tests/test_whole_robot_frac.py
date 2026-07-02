"""_detect_whole_robots close-range height tolerance.

The whole-robot gap-fill has its own >ROBOT_FILL_MAX_BOX_FRAC(0.5) frame-
fraction cap — an independent copy of the assumption that killed close corner
detections in detect_bridge (qm10: 4027 at 53-84% of the 377px corner panel
height was invisible to the fill path too). With det_frac_backstop=
"uncalibrated" the cap must relax to the close-range value so nearby robots
survive; legacy default keeps 0.5.
"""
import numpy as np
import supervision as sv
import web.app as app_mod
from web.job_config import JobConfig


class _StubModel:
    """YOLO stand-in returning one tall whole-robot box (cls 1)."""
    def __init__(self, box):
        self._box = box

    def __call__(self, frame, conf=0.0, verbose=False):
        det = sv.Detections(
            xyxy=np.array([self._box], np.float32),
            class_id=np.array([1], np.int32),
            confidence=np.array([0.9], np.float32),
        )
        return [det]


def _passthrough(results):
    return results if isinstance(results, sv.Detections) else sv.Detections.empty()


def _job(config, box):
    job = app_mod.InferenceJob.__new__(app_mod.InferenceJob)
    job.model = _StubModel(box)
    job.whole_robot_class_id = 1
    job.config = JobConfig(config)
    return job


FRAME = np.zeros((377, 960, 3), np.uint8)
TALL_BOX = (0.0, 70.0, 222.0, 359.0)   # 77% of frame height — 4027 at close range


def test_tall_whole_robot_kept_when_backstop_scoped(monkeypatch):
    monkeypatch.setattr(sv.Detections, "from_ultralytics", _passthrough)
    job = _job({"det_frac_backstop": "uncalibrated"}, TALL_BOX)
    assert job._detect_whole_robots(FRAME) == [TALL_BOX]


def test_tall_whole_robot_dropped_by_default(monkeypatch):
    monkeypatch.setattr(sv.Detections, "from_ultralytics", _passthrough)
    job = _job({}, TALL_BOX)
    assert job._detect_whole_robots(FRAME) == []


def test_near_full_frame_box_still_dropped_in_scoped_mode(monkeypatch):
    """The relaxed cap is a tolerance, not a removal: a box filling nearly the
    entire frame (banner/glare) is still rejected."""
    monkeypatch.setattr(sv.Detections, "from_ultralytics", _passthrough)
    job = _job({"det_frac_backstop": "uncalibrated"}, (0.0, 5.0, 950.0, 372.0))
    assert job._detect_whole_robots(FRAME) == []
