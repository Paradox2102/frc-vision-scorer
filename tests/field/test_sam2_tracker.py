from __future__ import annotations

import numpy as np
import pytest
from web.field.sam2_tracker import mask_to_box, bottom_center, match_new_detections


def test_mask_to_box_tight_bbox():
    m = np.zeros((10, 10), bool); m[2:5, 3:8] = True
    assert mask_to_box(m) == (3, 2, 8, 5)  # x1,y1,x2,y2 — exclusive max (max+1)


def test_mask_to_box_empty_returns_none():
    assert mask_to_box(np.zeros((4, 4), bool)) is None


def test_bottom_center():
    assert bottom_center((3, 2, 8, 5)) == (5.5, 5.0)


def test_match_new_detections_flags_unmatched():
    # existing mask covers left half; first new box overlaps it, second is a NEW robot
    existing = {1: _rect_mask(0, 0, 5, 10)}
    new_boxes = [(1, 1, 4, 9), (6, 1, 9, 9)]
    unmatched = match_new_detections(existing, new_boxes, (10, 10), ios_thresh=0.5)
    assert unmatched == [1]  # index of the second box only (the unmatched/new one)


def _rect_mask(x1, y1, x2, y2, wh=(10, 10)):
    m = np.zeros((wh[1], wh[0]), bool); m[y1:y2, x1:x2] = True; return m


# ---------------------------------------------------------------------------
# Task 1.2 — Sam2FieldTracker stub tests
# ---------------------------------------------------------------------------

class FakeTensor:
    """Minimal tensor stub: supports > comparison, .cpu(), and .numpy()."""

    def __init__(self, array: np.ndarray) -> None:
        self._a = array

    def __gt__(self, x):
        return FakeTensor(self._a > x)

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._a

    # allow numpy boolean operations on the result of __gt__
    def __bool__(self):
        return bool(self._a)


def _make_mask_array(H: int, W: int, rects: list[tuple[int, int, int, int]]) -> FakeTensor:
    """Return a FakeTensor of shape (num_obj, 1, H, W) with one nonzero rect per object."""
    num_obj = len(rects)
    arr = np.zeros((num_obj, 1, H, W), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(rects):
        arr[i, 0, y1:y2, x1:x2] = 1.0
    return FakeTensor(arr)


class StubPredictor:
    """GPU-free SAM2 predictor stub for unit testing Sam2FieldTracker."""

    H = 20
    W = 30

    # Per-frame rects for 2 objects; they shift by 1px each frame to simulate motion
    # Each entry: [(obj1_x1, obj1_y1, obj1_x2, obj1_y2), (obj2_x1, obj2_y1, obj2_x2, obj2_y2)]
    _rects_per_frame = [
        [(2, 2, 6, 6), (15, 3, 20, 8)],   # frame 0
        [(3, 3, 7, 7), (16, 4, 21, 9)],   # frame 1
        [(4, 4, 8, 8), (17, 5, 22, 10)],  # frame 2
    ]

    def __init__(self) -> None:
        self.prompted_obj_ids: list[int] = []

    def init_state(self, video_path: str, **kwargs) -> dict:
        return {}

    def add_new_points_or_box(
        self, state: dict, frame_idx: int, obj_id: int, box=None
    ) -> None:
        self.prompted_obj_ids.append(obj_id)

    def propagate_in_video(self, state: dict):
        obj_ids = [1, 2]
        for f_idx, rects in enumerate(self._rects_per_frame):
            mask_logits = _make_mask_array(self.H, self.W, rects)
            yield f_idx, obj_ids, mask_logits


def test_sam2_field_tracker_returns_correct_structure():
    """Sam2FieldTracker with a stub predictor produces the expected nested dict."""
    from web.field.sam2_tracker import Sam2FieldTracker

    stub = StubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)

    # prompt_boxes: two arbitrary boxes (values don't matter; stub ignores them)
    prompt_boxes = [(2, 2, 6, 6), (15, 3, 20, 8)]
    result = tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=prompt_boxes,
    )

    # ---- structure checks ----
    assert isinstance(result, dict), "result must be a dict keyed by frame index"
    assert set(result.keys()) == {0, 1, 2}, f"expected 3 frames, got {set(result.keys())}"

    for f_idx in range(3):
        frame_data = result[f_idx]
        assert isinstance(frame_data, dict), f"frame {f_idx} value must be a dict"
        assert set(frame_data.keys()) == {1, 2}, (
            f"expected obj_ids {{1, 2}} at frame {f_idx}, got {set(frame_data.keys())}"
        )
        for obj_id in (1, 2):
            entry = frame_data[obj_id]
            assert "box" in entry, f"missing 'box' for obj {obj_id} at frame {f_idx}"
            assert "mask" in entry, f"missing 'mask' for obj {obj_id} at frame {f_idx}"
            assert isinstance(entry["mask"], np.ndarray), (
                f"mask must be ndarray, got {type(entry['mask'])}"
            )
            assert entry["mask"].ndim == 2, (
                f"mask must be 2D, got shape {entry['mask'].shape}"
            )

    # ---- box content checks: should match the stub's moving rectangles via mask_to_box ----
    rects_per_frame = StubPredictor._rects_per_frame
    for f_idx in range(3):
        frame_data = result[f_idx]
        for obj_idx, obj_id in enumerate((1, 2)):
            x1, y1, x2, y2 = rects_per_frame[f_idx][obj_idx]
            # mask_to_box uses exclusive max (max+1), matching our rect definition
            expected_box = (x1, y1, x2, y2)
            assert frame_data[obj_id]["box"] == expected_box, (
                f"box mismatch at frame {f_idx} obj {obj_id}: "
                f"expected {expected_box}, got {frame_data[obj_id]['box']}"
            )

    # ---- verify obj_ids were prompted (both 1 and 2 should have been added) ----
    assert set(stub.prompted_obj_ids) == {1, 2}, (
        f"expected obj_ids {{1, 2}} to be prompted, got {stub.prompted_obj_ids}"
    )
