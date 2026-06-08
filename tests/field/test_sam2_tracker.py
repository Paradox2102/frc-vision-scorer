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


def test_mask_corroborated_true_when_box_overlaps():
    from web.field.sam2_tracker import mask_corroborated
    mask = np.zeros((20, 60), bool)
    mask[2:6, 6:10] = True
    assert mask_corroborated(mask, [(6, 2, 10, 6)], (60, 20), ios_thresh=0.3) is True


def test_mask_corroborated_false_when_no_box_overlaps():
    from web.field.sam2_tracker import mask_corroborated
    mask = np.zeros((20, 60), bool)
    mask[2:6, 6:10] = True
    assert mask_corroborated(mask, [(40, 2, 44, 6)], (60, 20), ios_thresh=0.3) is False


def test_mask_corroborated_false_when_no_boxes():
    from web.field.sam2_tracker import mask_corroborated
    mask = np.zeros((20, 60), bool)
    mask[2:6, 6:10] = True
    assert mask_corroborated(mask, [], (60, 20), ios_thresh=0.3) is False


def test_mask_corroborated_ignores_color_conflicting_box():
    from web.field.sam2_tracker import mask_corroborated
    mask = np.zeros((20, 60), bool)
    mask[2:6, 6:10] = True
    box = (6, 2, 10, 6)
    assert mask_corroborated(mask, [box], (60, 20), ios_thresh=0.3,
                             mask_color="red", new_colors=["blue"]) is False
    assert mask_corroborated(mask, [box], (60, 20), ios_thresh=0.3,
                             mask_color="red", new_colors=["red"]) is True


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


# ---------------------------------------------------------------------------
# Re-prompting (chunked propagation) stub + tests
# ---------------------------------------------------------------------------

class RepromptStubPredictor:
    """Stub modelling SAM2's mid-stream object addition.

    An object added via ``add_new_points_or_box(frame_idx=K)`` becomes active
    from frame K onward, exactly like the real predictor.  ``propagate_in_video``
    honours ``start_frame_idx``/``max_frame_num_to_track`` and yields, for each
    frame, the obj_ids active at that frame plus their (static, non-overlapping)
    masks.  Object ``oid`` lives at x in [6*oid, 6*oid+4).
    """

    H = 20
    W = 60
    NUM_FRAMES = 6

    def __init__(self) -> None:
        self.add_calls: list[tuple[int, int]] = []  # (frame_idx, obj_id)

    def init_state(self, video_path: str, **kwargs) -> dict:
        return {"objs": {}}  # obj_id -> first active frame_idx

    def add_new_points_or_box(self, state, frame_idx, obj_id, box=None) -> None:
        state["objs"][obj_id] = frame_idx
        self.add_calls.append((frame_idx, obj_id))

    def _rect_for(self, oid: int) -> tuple[int, int, int, int]:
        x0 = oid * 6
        return (x0, 2, x0 + 4, 6)

    def propagate_in_video(self, state, start_frame_idx=None, max_frame_num_to_track=None):
        start = 0 if start_frame_idx is None else start_frame_idx
        if max_frame_num_to_track is None:
            end = self.NUM_FRAMES - 1
        else:
            end = min(start + max_frame_num_to_track, self.NUM_FRAMES - 1)
        for f in range(start, end + 1):
            active = sorted(oid for oid, af in state["objs"].items() if af <= f)
            rects = [self._rect_for(oid) for oid in active]
            yield f, active, _make_mask_array(self.H, self.W, rects)

    def reset_state(self, state) -> None:
        pass


def test_reprompt_adds_new_robot_and_tracks_it_forward():
    """A robot detected at a chunk boundary becomes a new tracked object."""
    from web.field.sam2_tracker import Sam2FieldTracker

    stub = RepromptStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)

    # detect_fn always reports: one box overlapping obj 1 (already tracked) and
    # one box at obj 3's location (a genuinely new robot).
    overlap_box = stub._rect_for(1)   # (6,2,10,6) — matches tracked obj 1
    new_box = stub._rect_for(3)       # (18,2,22,6) — unmatched -> new robot
    calls: list[int] = []

    metadata = {1: "blue", 2: "red"}

    def detect_fn(local_idx, frame_path):
        calls.append(local_idx)
        return [(overlap_box, "blue"), (new_box, "red")]

    result = tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],  # objs 1, 2
        reprompt_every=2,
        detect_fn=detect_fn,
        object_metadata=metadata,
        max_objects=6,
    )

    # All 6 frames present.
    assert set(result.keys()) == {0, 1, 2, 3, 4, 5}
    # Frames before the first reprompt boundary: only the 2 initial robots.
    assert set(result[0].keys()) == {1, 2}
    assert set(result[2].keys()) == {1, 2}
    # After re-prompting at frame 2, the new robot (obj 3) is tracked forward.
    assert set(result[3].keys()) == {1, 2, 3}
    assert set(result[5].keys()) == {1, 2, 3}
    # The new robot was added exactly once; the overlapping box was deduped.
    assert (2, 3) in stub.add_calls
    assert stub.add_calls == [(0, 1), (0, 2), (2, 3)]
    assert metadata == {1: "blue", 2: "red", 3: "red"}


def test_reprompt_respects_max_objects_cap():
    """No new objects are added once the robot cap is reached."""
    from web.field.sam2_tracker import Sam2FieldTracker

    stub = RepromptStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)

    def detect_fn(local_idx, frame_path):
        return [stub._rect_for(3)]  # a new robot every boundary

    result = tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],  # cap already at 2
        reprompt_every=2,
        detect_fn=detect_fn,
        max_objects=2,
    )

    # Cap reached at construction; obj 3 is never added.
    assert all(oid != 3 for (_f, oid) in stub.add_calls)
    assert set(result[5].keys()) == {1, 2}


def test_empty_initial_prompt_bootstraps_from_later_detection(tmp_path):
    from web.field.sam2_tracker import Sam2FieldTracker

    for idx in range(RepromptStubPredictor.NUM_FRAMES):
        (tmp_path / f"{idx:05d}.jpg").touch()

    stub = RepromptStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    calls = []

    def detect_fn(local_idx, frame_path):
        calls.append(local_idx)
        if local_idx == 3:
            return [(stub._rect_for(1), "blue")]
        return []

    metadata = {}
    result = tracker.track(
        frames_dir=str(tmp_path),
        prompt_idx=0,
        prompt_boxes=[],
        reprompt_every=2,
        detect_fn=detect_fn,
        object_metadata=metadata,
        max_active_objects=6,
        max_lifetime_objects=20,
    )

    assert calls[:4] == [0, 1, 2, 3]
    assert stub.add_calls[0] == (3, 1)
    assert 1 in result[3]
    assert 1 in result[5]
    assert metadata == {1: "blue"}


def test_capped_boundary_does_not_reference_uninitialized_detections():
    from web.field.identity_diagnostics import IdentityDiagnosticRecorder
    from web.field.sam2_tracker import Sam2FieldTracker

    stub = RepromptStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diagnostics = IdentityDiagnosticRecorder()

    tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2,
        detect_fn=lambda *_args: [stub._rect_for(3)],
        max_active_objects=2,
        diagnostics=diagnostics,
    )


def test_no_reprompt_when_detect_fn_absent_is_single_pass():
    """Without reprompt_every/detect_fn, track() runs a single-pass propagation."""
    from web.field.sam2_tracker import Sam2FieldTracker

    stub = StubPredictor()  # original single-pass stub: propagate_in_video(state)
    tracker = Sam2FieldTracker(predictor=stub)

    result = tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[(2, 2, 6, 6), (15, 3, 20, 8)],
    )
    # Same behavior as before: 3 frames, 2 objects.
    assert set(result.keys()) == {0, 1, 2}
    assert set(result[0].keys()) == {1, 2}


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


class AbsentStubPredictor:
    H = 20
    W = 60
    
    def __init__(self, visibility_schedule: dict[int, list[int]], num_frames: int = 6) -> None:
        self.schedule = visibility_schedule
        self.num_frames = num_frames
        self.add_calls = []

    def init_state(self, video_path: str, **kwargs) -> dict:
        return {"objs": {}}

    def add_new_points_or_box(self, state, frame_idx, obj_id, box=None) -> None:
        state["objs"][obj_id] = frame_idx
        self.add_calls.append((frame_idx, obj_id))

    def propagate_in_video(self, state, start_frame_idx=None, max_frame_num_to_track=None):
        start = 0 if start_frame_idx is None else start_frame_idx
        if max_frame_num_to_track is None:
            end = self.num_frames - 1
        else:
            end = min(start + max_frame_num_to_track, self.num_frames - 1)
        for f in range(start, end + 1):
            active = sorted(oid for oid, af in state["objs"].items() if af <= f)
            rects = []
            for oid in active:
                if oid in self.schedule.get(f, []):
                    rects.append((oid * 6, 2, oid * 6 + 4, 6))
                else:
                    rects.append((0, 0, 0, 0))
            yield f, active, _make_mask_array(self.H, self.W, rects)

    def reset_state(self, state) -> None:
        pass


def test_sam2_tracker_active_vs_lifetime_limits():
    from web.field.sam2_tracker import Sam2FieldTracker
    
    # 6 objects prompted initially at frame 0.
    # schedule:
    # Frame 0: all 6 visible
    # Frame 1: all 6 visible
    # Frame 2: obj 6 becomes invisible (absent)
    # Frame 3: obj 6 remains invisible -> retirement window (2 frames) exceeded!
    # Frame 4: obj 6 retired, so active count is 5.
    schedule = {
        0: [1, 2, 3, 4, 5, 6],
        1: [1, 2, 3, 4, 5, 6],
        2: [1, 2, 3, 4, 5],
        3: [1, 2, 3, 4, 5],
        4: [1, 2, 3, 4, 5],
        5: [1, 2, 3, 4, 5, 7], # 7 becomes visible
        6: [1, 2, 3, 4, 5, 7],
    }
    
    stub = AbsentStubPredictor(schedule, num_frames=7)
    tracker = Sam2FieldTracker(predictor=stub)
    
    def detect_fn(local_idx, frame_path):
        return [((42, 2, 46, 6), "blue")]
        
    result = tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[(i*6, 2, i*6+4, 6) for i in range(1, 7)], # 6 boxes
        reprompt_every=2,
        detect_fn=detect_fn,
        max_active_objects=6,
        max_lifetime_objects=10,
        retirement_frames=2,
    )
    
    # Assert obj 7 was successfully prompted at frame 5 because obj 6 retired
    assert (5, 7) in stub.add_calls
    # Assert obj 7 is tracked in the final frame
    assert 7 in result[6]


def test_sam2_tracker_evidence_generation():
    from web.field.sam2_tracker import Sam2FieldTracker
    
    # schedule:
    # Frame 0: obj 1, 2 visible
    # Frame 1: obj 1, 2 visible
    # Frame 2: obj 1 absent, obj 2 visible
    # Frame 3: obj 1 visible, obj 2 visible
    # Frame 4: obj 1, 2, 3 visible (3 is added at frame 2)
    schedule = {
        0: [1, 2],
        1: [1, 2],
        2: [2],
        3: [1, 2],
        4: [1, 2, 3],
        5: [1, 2, 3],
    }
    
    stub = AbsentStubPredictor(schedule, num_frames=6)
    tracker = Sam2FieldTracker(predictor=stub)
    
    def detect_fn(local_idx, frame_path):
        # At frame 2, report a new box at position 3
        return [((18, 2, 22, 6), "red")]
        
    result = tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[(6, 2, 10, 6), (12, 2, 16, 6)],
        reprompt_every=2,
        detect_fn=detect_fn,
        max_active_objects=6,
        max_lifetime_objects=10,
        retirement_frames=2,
    )
    
    # Frame 0 initial check
    assert result[0][1]["evidence"]["type"] == "initial"
    assert result[0][1]["evidence"]["gap_len"] == 0
    
    # Frame 1 propagated check
    assert result[1][1]["evidence"]["type"] == "propagated"
    assert result[1][1]["evidence"]["gap_len"] == 0
    
    # Frame 2 check: obj 1 is absent (not in result[2])
    assert 1 not in result[2]
    
    # Frame 3 check: obj 1 reappears
    assert result[3][1]["evidence"]["type"] == "reappeared_same_id"
    assert result[3][1]["evidence"]["gap_len"] == 1  # absent on frame 2
    
    # Frame 4 check: obj 3 has been added at frame 2, and first appears on frame 4
    assert result[4][3]["evidence"]["type"] == "reprompted_new_id"
    assert result[4][3]["evidence"]["gap_len"] == 0
    assert result[5][3]["evidence"]["type"] == "propagated"


def test_sam2_diagnostics_logging():
    from web.field.sam2_tracker import Sam2FieldTracker
    from web.field.identity_diagnostics import IdentityDiagnosticRecorder
    
    schedule = {
        0: [1, 2],
        1: [1, 2],
        2: [2],  # obj 1 absent
        3: [1, 2],  # obj 1 reappears
        4: [1, 2, 3],  # obj 3 added
        5: [1, 2, 3],
    }
    
    stub = AbsentStubPredictor(schedule, num_frames=6)
    tracker = Sam2FieldTracker(predictor=stub)
    diag = IdentityDiagnosticRecorder()
    
    def detect_fn(local_idx, frame_path):
        return [((18, 2, 22, 6), "red")]
        
    tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[(6, 2, 10, 6), (12, 2, 16, 6)],
        reprompt_every=2,
        detect_fn=detect_fn,
        max_active_objects=6,
        max_lifetime_objects=10,
        retirement_frames=2,
        diagnostics=diag,
    )
    
    events = diag._events
    created_events = [e for e in events if e["event"] == "sam2.object_created"]
    gap_events = [e for e in events if e["event"] == "sam2.mask_gap"]
    candidate_events = [e for e in events if e["event"] == "sam2.reprompt_candidate"]
    chunk_events = [e for e in events if e["event"] == "sam2.chunk_complete"]
    
    assert len(created_events) == 3  # 2 initial + 1 reprompted
    assert len(chunk_events) == 2
    assert len(gap_events) == 1
    assert len(candidate_events) == 2
    
    rejected_events = [e for e in events if e["event"] == "sam2.reprompt_rejected"]
    assert len(rejected_events) == 1
    assert rejected_events[0]["reason"] == "below_margin"


class CullStubPredictor:
    """Reprompt stub with controllable per-object motion, for cull tests.

    Object ``oid`` lives at x in [10*oid, 10*oid+4). An oid in ``moving`` drifts
    +3 px in x per frame (its projected position changes across boundaries);
    others are static. Honours start_frame_idx / max_frame_num_to_track.
    """
    H = 20
    W = 200
    NUM_FRAMES = 12

    def __init__(self, moving=()):
        self.moving = set(moving)
        self.add_calls: list[tuple[int, int]] = []

    def init_state(self, video_path: str, **kwargs) -> dict:
        return {"objs": {}}

    def add_new_points_or_box(self, state, frame_idx, obj_id, box=None) -> None:
        state["objs"][obj_id] = frame_idx
        self.add_calls.append((frame_idx, obj_id))

    def _rect_for(self, oid: int, f: int = 0) -> tuple[int, int, int, int]:
        x0 = oid * 10 + (3 * f if oid in self.moving else 0)
        return (x0, 2, x0 + 4, 6)

    def propagate_in_video(self, state, start_frame_idx=None, max_frame_num_to_track=None):
        start = 0 if start_frame_idx is None else start_frame_idx
        end = (self.NUM_FRAMES - 1 if max_frame_num_to_track is None
               else min(start + max_frame_num_to_track, self.NUM_FRAMES - 1))
        for f in range(start, end + 1):
            active = sorted(oid for oid, af in state["objs"].items() if af <= f)
            rects = [self._rect_for(oid, f) for oid in active]
            yield f, active, _make_mask_array(self.H, self.W, rects)

    def reset_state(self, state) -> None:
        pass


def test_detector_runs_at_capped_boundary_when_cull_enabled():
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    calls = []

    def detect_fn(local_idx, frame_path):
        calls.append(local_idx)
        return []

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        max_active_objects=2, cull_enabled=True,
    )
    assert calls, "detect_fn must run at capped boundaries when cull enabled"


def test_detector_skipped_at_capped_boundary_when_cull_disabled():
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    calls = []

    def detect_fn(local_idx, frame_path):
        calls.append(local_idx)
        return []

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        max_active_objects=2, cull_enabled=False,
    )
    assert calls == [], "detect_fn must NOT run at capped boundaries when cull disabled"


# ---------------------------------------------------------------------------
# Mask-shape plausibility — reject field fixtures (radio-on-a-stand) whose
# SAM2 mask is geometrically impossible for a robot.
# ---------------------------------------------------------------------------


def _solid_robot_mask(H: int = 60, W: int = 30) -> np.ndarray:
    """A compact, bottom-heavy blob — what a robot's mask looks like."""
    m = np.zeros((H, W), bool)
    m[40:56, 6:24] = True  # 16 tall x 18 wide solid block sitting low
    return m


def _tadpole_mask(H: int = 60, W: int = 30) -> np.ndarray:
    """A blob-on-a-stick: wide blob at the TOP, thin tail to the bottom.

    This is the field-side radio/antenna shape — tall, top-heavy, and the
    contact point with the carpet is a thin stalk, not a wide robot base.
    """
    m = np.zeros((H, W), bool)
    m[2:10, 10:22] = True     # 8 tall x 12 wide blob at the top
    m[10:H - 4, 14:17] = True  # thin 3px-wide tail running down
    return m


def _climber_robot_mask(H: int = 60, W: int = 30) -> np.ndarray:
    """A robot with a tall thin climber hook raised: tall, but BOTTOM-heavy.

    Must NOT be rejected — distinguishes 'tall+top-heavy' (radio) from
    'tall+bottom-heavy' (legitimate climbing robot).
    """
    m = np.zeros((H, W), bool)
    m[36:56, 6:24] = True  # wide chassis sitting low (bottom-heavy)
    m[5:36, 14:17] = True  # thin hook extending up
    return m


def test_is_stationary_true_for_fixed_positions():
    from web.field.sam2_tracker import is_stationary
    assert is_stationary([(10.0, 5.0), (10.1, 5.0), (9.95, 5.05)], 0.5) is True


def test_is_stationary_false_for_moving_positions():
    from web.field.sam2_tracker import is_stationary
    assert is_stationary([(10.0, 5.0), (11.0, 5.0), (12.0, 5.0)], 0.5) is False


def test_is_stationary_false_when_insufficient_known_positions():
    from web.field.sam2_tracker import is_stationary
    assert is_stationary([(10.0, 5.0)], 0.5) is False
    assert is_stationary([None, (10.0, 5.0)], 0.5) is False
    assert is_stationary([None, None], 0.5) is False


def test_is_stationary_ignores_none_entries():
    from web.field.sam2_tracker import is_stationary
    assert is_stationary([(10.0, 5.0), None, (10.2, 5.1)], 0.5) is True


def test_absent_enough_true_for_persistent_absence():
    from web.field.sam2_tracker import absent_enough
    assert absent_enough([False, False, False], k=3) is True
    assert absent_enough([False, False, False, False], k=3) is True


def test_absent_enough_true_despite_one_intermittent_hit():
    from web.field.sam2_tracker import absent_enough
    assert absent_enough([True, False, False, False], k=3) is True


def test_absent_enough_false_when_mostly_corroborated():
    from web.field.sam2_tracker import absent_enough
    assert absent_enough([True, True, True], k=3) is False
    assert absent_enough([True, True, False, True], k=3) is False


def test_absent_enough_false_before_window_fills():
    from web.field.sam2_tracker import absent_enough
    assert absent_enough([False, False], k=3) is False


# ---------------------------------------------------------------------------
# Mask-shape plausibility — reject field fixtures (radio-on-a-stand) whose
# SAM2 mask is geometrically impossible for a robot.
# ---------------------------------------------------------------------------


def test_plausible_robot_mask_accepts_solid_blob():
    from web.field.sam2_tracker import plausible_robot_mask
    assert plausible_robot_mask(_solid_robot_mask()) is True


def test_plausible_robot_mask_accepts_small_distant_robot():
    from web.field.sam2_tracker import plausible_robot_mask
    m = np.zeros((20, 20), bool)
    m[8:12, 8:12] = True  # tiny solid square — far-away robot
    assert plausible_robot_mask(m) is True


def test_plausible_robot_mask_rejects_radio_tadpole():
    from web.field.sam2_tracker import plausible_robot_mask
    assert plausible_robot_mask(_tadpole_mask()) is False


def test_plausible_robot_mask_accepts_climbing_robot():
    from web.field.sam2_tracker import plausible_robot_mask
    # Tall and thin overall, but bottom-heavy — a real robot climbing.
    assert plausible_robot_mask(_climber_robot_mask()) is True


def test_plausible_robot_mask_accepts_solid_vertical_sliver():
    from web.field.sam2_tracker import plausible_robot_mask
    # A robot mostly occluded to a tall thin strip is still solid (well-filled)
    # and uniform-width, not a top-heavy stalk.
    m = np.zeros((60, 30), bool)
    m[5:55, 13:20] = True
    assert plausible_robot_mask(m) is True


def test_plausible_robot_mask_empty_is_implausible():
    from web.field.sam2_tracker import plausible_robot_mask
    assert plausible_robot_mask(np.zeros((10, 10), bool)) is False


class MaskStubPredictor:
    """Emits explicit per-object masks each frame (obj 1 tadpole, obj 2 solid)."""

    H = 60
    W = 30

    def __init__(self) -> None:
        self.prompted_obj_ids: list[int] = []

    def init_state(self, video_path: str, **kwargs) -> dict:
        return {}

    def add_new_points_or_box(self, state, frame_idx, obj_id, box=None) -> None:
        self.prompted_obj_ids.append(obj_id)

    def propagate_in_video(self, state, start_frame_idx=None, max_frame_num_to_track=None):
        obj_ids = [1, 2]
        for f_idx in range(3):
            masks = [_tadpole_mask(self.H, self.W), _solid_robot_mask(self.H, self.W)]
            arr = np.stack([m.astype(np.float32) for m in masks])[:, None, :, :]
            yield f_idx, obj_ids, FakeTensor(arr)

    def reset_state(self, state) -> None:
        pass


def test_tracker_drops_implausible_mask_object():
    from web.field.sam2_tracker import Sam2FieldTracker
    from web.field.identity_diagnostics import IdentityDiagnosticRecorder

    stub = MaskStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = IdentityDiagnosticRecorder()

    result = tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[(10, 2, 22, 56), (6, 40, 24, 56)],
        diagnostics=diag,
    )

    # obj 1 (radio tadpole) is rejected on every frame; obj 2 (robot) survives.
    for f_idx in range(3):
        assert 1 not in result[f_idx], f"radio tadpole should be dropped at frame {f_idx}"
        assert 2 in result[f_idx], f"solid robot should survive at frame {f_idx}"

    rejected = [e for e in diag._events if e["event"] == "sam2.mask_rejected"]
    assert rejected, "expected a sam2.mask_rejected diagnostic for the implausible mask"
    assert rejected[0]["reason"] == "mask_implausible"

