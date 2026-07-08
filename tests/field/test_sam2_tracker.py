from __future__ import annotations

import numpy as np
import pytest
from web.field.sam2_tracker import (
    mask_to_box,
    bottom_center,
    mask_bottom_center,
    match_new_detections,
    size_prior_ratio,
    size_prior_violation,
)


def test_mask_to_box_tight_bbox():
    m = np.zeros((10, 10), bool); m[2:5, 3:8] = True
    assert mask_to_box(m) == (3, 2, 8, 5)  # x1,y1,x2,y2 — exclusive max (max+1)


def test_mask_to_box_empty_returns_none():
    assert mask_to_box(np.zeros((4, 4), bool)) is None


def test_bottom_center():
    assert bottom_center((3, 2, 8, 5)) == (5.5, 5.0)


def test_mask_bottom_center_empty_returns_none():
    assert mask_bottom_center(np.zeros((4, 4), bool)) is None


def test_mask_bottom_center_uses_bottom_band_not_box_center():
    # Asymmetric mask: a wide extended intake up top (cols 0..10) plus a narrow
    # carpet-contact footprint at the bottom (cols 8..10). The box bottom-center
    # x would be pulled left by the wide top rows; the true footprint x is the
    # mean column of only the bottom-band contact pixels.
    m = np.zeros((10, 12), bool)
    m[2:7, 0:11] = True   # wide extended arm/intake up top
    m[7:9, 8:11] = True   # narrow base actually touching the carpet
    box = mask_to_box(m)                 # (0, 2, 11, 9)
    box_cx = (box[0] + box[2]) / 2.0     # 5.5 — pulled toward the wide top
    ymax = 8                             # lowest nonzero row
    fp = mask_bottom_center(m)           # default band of 3 rows: rows 6,7,8
    assert fp is not None
    fx, fy = fp
    assert fy == float(ymax)             # y is the true lowest contact row
    # Bottom 3 rows: row 6 spans cols 0..10, rows 7,8 span cols 8..10.
    # Mean column of those pixels sits well right of the box center.
    assert fx > box_cx
    # Restricting to just the bottom 2 rows (the narrow base) -> centered on 9.
    assert mask_bottom_center(m, rows=2) == (9.0, 8.0)


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

    def propagate_in_video(
        self,
        state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        start = 0 if start_frame_idx is None else start_frame_idx
        if reverse:
            end = max(0, start - (max_frame_num_to_track or start))
            frame_range = range(start, end - 1, -1)
        else:
            if max_frame_num_to_track is None:
                end = self.NUM_FRAMES - 1
            else:
                end = min(start + max_frame_num_to_track, self.NUM_FRAMES - 1)
            frame_range = range(start, end + 1)
        for f in frame_range:
            active = sorted(
                oid
                for oid, af in state["objs"].items()
                if af <= f or reverse
            )
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
    """No new objects are added once the cap is reached and the tracked
    objects are corroborated (healthy masks are never evicted)."""
    from web.field.sam2_tracker import Sam2FieldTracker

    stub = RepromptStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)

    def detect_fn(local_idx, frame_path):
        # Corroborate both tracked objects; report one genuinely new robot.
        return [stub._rect_for(1), stub._rect_for(2), stub._rect_for(3)]

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


# ---------------------------------------------------------------------------
# qm12 lever 2 — opening-window dense reprompt
# ---------------------------------------------------------------------------

class _LongRepromptStub(RepromptStubPredictor):
    """RepromptStubPredictor with more frames, for observing boundary spacing."""
    NUM_FRAMES = 12


def _new_obj_seed_frame(add_calls, obj_id):
    """Frame index at which obj_id was first seeded, or None."""
    for f_idx, oid in add_calls:
        if oid == obj_id:
            return f_idx
    return None


def test_opening_dense_reprompt_seeds_new_robot_sooner():
    """During the opening window a smaller reprompt step seeds a newly
    detected robot at an EARLIER frame than the normal reprompt interval."""
    from web.field.sam2_tracker import Sam2FieldTracker

    overlap_box = RepromptStubPredictor()._rect_for(1)  # matches tracked obj 1
    new_box = RepromptStubPredictor()._rect_for(3)      # unmatched -> new robot

    def make_detect_fn():
        # The new robot only becomes visible from frame 1 onward. With the
        # normal step (reprompt_every=2) the first boundary at/after frame 1
        # is frame 2; the dense step (1) has a boundary AT frame 1.
        def detect_fn(local_idx, frame_path):
            boxes = [(overlap_box, "blue")]
            if local_idx >= 1:
                boxes.append((new_box, "red"))
            return boxes
        return detect_fn

    def run(dense):
        stub = RepromptStubPredictor()
        tracker = Sam2FieldTracker(predictor=stub)
        kwargs = dict(
            frames_dir="ignored_by_stub",
            prompt_idx=0,
            prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
            reprompt_every=2,
            detect_fn=make_detect_fn(),
            object_metadata={1: "blue", 2: "red"},
            max_objects=6,
        )
        if dense:
            kwargs["opening_reprompt_every"] = 1
            kwargs["opening_window_frames"] = RepromptStubPredictor.NUM_FRAMES
        tracker.track(**kwargs)
        return _new_obj_seed_frame(stub.add_calls, 3)

    normal_seed = run(dense=False)
    dense_seed = run(dense=True)

    assert normal_seed == 2, f"expected normal seed at frame 2, got {normal_seed}"
    assert dense_seed == 1, f"expected dense seed at frame 1, got {dense_seed}"
    assert dense_seed < normal_seed


def test_opening_window_reverts_after_window():
    """The dense step applies only while (cursor-bootstrap_idx) <
    opening_window_frames; later chunks fall back to reprompt_every."""
    from web.field.identity_diagnostics import IdentityDiagnosticRecorder
    from web.field.sam2_tracker import Sam2FieldTracker

    stub = _LongRepromptStub()
    tracker = Sam2FieldTracker(predictor=stub)
    diagnostics = IdentityDiagnosticRecorder()

    tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=3,
        detect_fn=lambda *_a: [stub._rect_for(1), stub._rect_for(2)],
        object_metadata={1: "blue", 2: "red"},
        max_objects=6,
        opening_reprompt_every=1,
        opening_window_frames=2,   # only cursor 0,1 are inside the window
        diagnostics=diagnostics,
    )

    boundaries = [ev["details"]["end_frame"]
                  for ev in diagnostics._events
                  if ev["event"] == "sam2.chunk_complete"]

    # Dense while (cursor-bootstrap) < 2: cursor 0 -> end 1. Then cursor 2 is
    # outside the window, so step reverts to reprompt_every=3: 2 -> 5, 6 -> 9,
    # 10 -> 11. Early boundary spacing is small (1), later spacing is normal (3+).
    assert boundaries[0] == 1, f"first boundary should be dense (frame 1), got {boundaries}"
    assert boundaries == [1, 5, 9, 11], f"unexpected boundary sequence: {boundaries}"


def test_opening_dense_disabled_by_default():
    """With the feature off (default kwargs) the add_calls and chunk boundaries
    are byte-identical to a run that never knew about the feature."""
    from web.field.identity_diagnostics import IdentityDiagnosticRecorder
    from web.field.sam2_tracker import Sam2FieldTracker

    def run(**extra):
        stub = _LongRepromptStub()
        tracker = Sam2FieldTracker(predictor=stub)
        diagnostics = IdentityDiagnosticRecorder()
        tracker.track(
            frames_dir="ignored_by_stub",
            prompt_idx=0,
            prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
            reprompt_every=3,
            detect_fn=lambda *_a: [stub._rect_for(1), stub._rect_for(2),
                                   stub._rect_for(3)],
            object_metadata={1: "blue", 2: "red"},
            max_objects=6,
            diagnostics=diagnostics,
            **extra,
        )
        boundaries = [ev["details"]["end_frame"]
                      for ev in diagnostics._events
                      if ev["event"] == "sam2.chunk_complete"]
        return list(stub.add_calls), boundaries

    # Baseline: feature never mentioned.
    base_adds, base_boundaries = run()
    # Explicitly-off: both knobs 0 (== the resolved default).
    off_adds, off_boundaries = run(opening_reprompt_every=0, opening_window_frames=0)

    assert off_adds == base_adds
    assert off_boundaries == base_boundaries


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


def test_initial_prompt_propagates_backward_before_forward():
    from web.field.sam2_tracker import Sam2FieldTracker

    stub = RepromptStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)

    result = tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=3,
        prompt_boxes=[stub._rect_for(1)],
        reprompt_every=2,
        detect_fn=lambda *_args: [],
        propagate_backward=True,
    )

    assert set(result) == {0, 1, 2, 3, 4, 5}
    assert all(1 in result[frame_idx] for frame_idx in range(6))
    assert stub.add_calls == [(3, 1)]


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
    # Legacy 3-tuple predictors carry no tracker confidence.
    assert result[0][1]["evidence"]["tracker_prob"] is None
    
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


class ProbsFourTupleStub:
    """Predictor yielding SAM3-style 4-tuples (adds per-object tracker probs)."""

    H = 20
    W = 30
    _rects_per_frame = [
        [(2, 2, 6, 6), (15, 3, 20, 8)],
        [(3, 3, 7, 7), (16, 4, 21, 9)],
    ]
    _probs_per_frame = [
        [0.91, 0.44],
        [0.88, 0.40],
    ]

    def init_state(self, video_path, **kwargs):
        return {}

    def add_new_points_or_box(self, state, frame_idx, obj_id, box=None):
        pass

    def propagate_in_video(self, state):
        obj_ids = [1, 2]
        for f_idx, rects in enumerate(self._rects_per_frame):
            mask_logits = _make_mask_array(self.H, self.W, rects)
            probs = np.asarray(self._probs_per_frame[f_idx], dtype=np.float32)
            yield f_idx, obj_ids, mask_logits, probs

    def reset_state(self, state):
        pass


def test_sam2_tracker_carries_tracker_prob_from_four_tuples():
    from web.field.sam2_tracker import Sam2FieldTracker

    stub = ProbsFourTupleStub()
    tracker = Sam2FieldTracker(predictor=stub)
    result = tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[(2, 2, 6, 6), (15, 3, 20, 8)],
    )
    assert result[0][1]["evidence"]["tracker_prob"] == pytest.approx(0.91)
    assert result[0][2]["evidence"]["tracker_prob"] == pytest.approx(0.44)
    assert result[1][1]["evidence"]["tracker_prob"] == pytest.approx(0.88)


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
    assert rejected_events[0]["reason"] == "already_tracked"


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


def test_detector_runs_at_capped_boundary_even_when_cull_disabled():
    """The detector always runs at chunk boundaries. Skipping it while the
    roster was capped (the old behavior) made the tracker blind to new robots
    for as long as stale masks pinned the active set — with zero diagnostics."""
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
    assert calls, "detect_fn must run at capped boundaries even with cull disabled"


def test_capped_unmatched_detection_logs_roster_cap_rejection():
    """With the roster capped by corroborated (healthy) objects, an unmatched
    detection is rejected with a visible roster_cap diagnostic — not silently."""
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        # Corroborate both tracked objects; report one unmatched new robot.
        return [stub._rect_for(1), stub._rect_for(2), stub._rect_for(15)]

    from web.field.sam2_tracker import bottom_center
    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        max_active_objects=2, cull_enabled=False,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert all(oid != 3 for (_f, oid) in stub.add_calls)
    assert any(e.get("event") == "sam2.reprompt_rejected"
               and e.get("reason") == "roster_cap" for e in diag._events)
    assert _culled_ids(diag) == set(), "corroborated objects must never be evicted"


def test_eviction_frees_slot_for_uncorroborated_stationary_object():
    """An unmatched fresh detection evicts the least-corroborated stationary
    object when the roster is capped — even with the spontaneous cull disabled."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        # Obj 1 corroborated every boundary; obj 2 never; one new robot at 15.
        return [(stub._rect_for(1), "blue"), (stub._rect_for(15), "red")]

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "blue", 2: "red"},
        max_active_objects=2, max_lifetime_objects=20,
        cull_enabled=False,
        cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 in _culled_ids(diag)
    evicted = [e for e in diag._events if e.get("event") == "sam2.object_culled"]
    assert any(e.get("reason") == "evicted_for_new_detection"
               and e["details"]["object_id"] == 2 for e in evicted)
    assert any(oid == 3 for (_f, oid) in stub.add_calls)
    assert 3 in result[stub.NUM_FRAMES - 1]
    assert 2 not in result[stub.NUM_FRAMES - 1]


def test_moving_uncorroborated_object_evicted_after_min_absences():
    """The roster holds the objects most likely to be robots: a MOVING object
    the detector repeatedly fails to back (a referee hijack) loses its slot to
    a fresh detector-backed newcomer — but never on its first absence, and a
    higher min_absences bar restores the old keep-moving-objects behavior."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor(moving={2})
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [(stub._rect_for(1), "blue"), (stub._rect_for(15), "red")]

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1, 0), stub._rect_for(2, 0)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "blue", 2: "red"},
        max_active_objects=2, cull_enabled=False,
        cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5, evict_min_absences=2,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 in _culled_ids(diag)
    evictions = [e["details"]["frame_idx"] for e in diag._events
                 if e.get("event") == "sam2.object_culled"
                 and e.get("reason") == "evicted_for_new_detection"
                 and e["details"]["object_id"] == 2]
    first_boundary = min(e["details"]["frame_idx"] for e in diag._events
                         if e.get("event") == "sam2.reprompt_candidate")
    assert all(f > first_boundary for f in evictions), \
        "one absence is not enough evidence to evict"
    assert any(oid == 3 for (_f, oid) in stub.add_calls)

    # With the bar raised out of reach, the moving object keeps its slot.
    stub2 = CullStubPredictor(moving={2})
    tracker2 = Sam2FieldTracker(predictor=stub2)
    diag2 = _ident_diag()
    tracker2.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub2._rect_for(1, 0), stub2._rect_for(2, 0)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "blue", 2: "red"},
        max_active_objects=2, cull_enabled=False,
        cull_uncorroborated_reprompts=99, cull_evidence_window=5,
        cull_stationary_m=0.5, evict_min_absences=99, evict_dark_frames=999,
        project_fn=lambda box: bottom_center(box), diagnostics=diag2,
    )
    assert _culled_ids(diag2) == set()
    assert all(oid != 3 for (_f, oid) in stub2.add_calls)


def test_eviction_disabled_via_flag():
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [(stub._rect_for(1), "blue"), (stub._rect_for(15), "red")]

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "blue", 2: "red"},
        max_active_objects=2, cull_enabled=False, evict_when_capped=False,
        cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert _culled_ids(diag) == set()
    assert all(oid != 3 for (_f, oid) in stub.add_calls)


def test_pick_evictable_prefers_most_absent_stationary():
    from collections import deque
    from web.field.sam2_tracker import pick_evictable
    hist = {
        1: deque([True, True, True], maxlen=5),          # corroborated
        2: deque([False, False, False], maxlen=5),        # absent, stationary
        3: deque([True, False, False, False], maxlen=5),  # absent (3), stationary
    }
    wins = {
        1: [(1.0, 1.0), (1.0, 1.0)],
        2: [(2.0, 2.0), (2.0, 2.0)],
        3: [(3.0, 3.0), (3.0, 3.0)],
    }
    # Both 2 and 3 meet the bar with 3 absences; the earlier id wins the tie.
    assert pick_evictable([1, 2, 3], hist, wins, k=3, stationary_m=0.5) == 2
    # A moving object never qualifies.
    wins[2] = [(2.0, 2.0), (9.0, 9.0)]
    assert pick_evictable([1, 2, 3], hist, wins, k=3, stationary_m=0.5) == 3
    # No history yet -> not evictable.
    assert pick_evictable([1], hist, wins, k=3, stationary_m=0.5) is None


def test_match_new_detections_ignores_sliver_mask():
    """A degenerate sliver mask must not suppress seeding of a full-size
    detection covering it (IoS-over-smaller would score 1.0)."""
    sliver = np.zeros((40, 40), bool)
    sliver[10:12, 10:12] = True  # 4 px residue of a collapsed mask
    unmatched = match_new_detections({1: sliver}, [(0, 0, 30, 30)], (40, 40),
                                     ios_thresh=0.5)
    assert unmatched == [0], "sliver mask must not claim a full-size detection"


def test_match_new_detections_small_mask_still_claims_small_box():
    """A legitimately small (distant-robot) mask still claims a matching
    small detection — the sliver guard is relative to box area."""
    small = np.zeros((40, 40), bool)
    small[10:14, 10:14] = True  # 16 px solid mask
    unmatched = match_new_detections({1: small}, [(9, 9, 15, 15)], (40, 40),
                                     ios_thresh=0.5)
    assert unmatched == []


def test_seed_gate_fn_blocks_seeding_with_diagnostic():
    """A detection failing seed_gate_fn (e.g. inside a static-structure zone)
    is never seeded as a new object, with a visible static_structure rejection."""
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = RepromptStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()
    gated_box = stub._rect_for(3)

    def detect_fn(local_idx, frame_path):
        return [gated_box]

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1)],
        reprompt_every=2, detect_fn=detect_fn,
        max_active_objects=6,
        seed_gate_fn=lambda box: tuple(box) != tuple(gated_box),
        diagnostics=diag,
    )
    assert all(oid != 2 for (_f, oid) in stub.add_calls)
    assert 2 not in result[5]
    assert any(e.get("event") == "sam2.reprompt_rejected"
               and e.get("reason") == "static_structure" for e in diag._events)


def test_color_backfill_after_two_consecutive_corroborations():
    """An unknown-colored object inherits the color of the bumper detections
    that corroborate its mask on 2 consecutive evidence boundaries — so a robot
    seeded from a colorless whole-robot gap-fill stops being 'unknown' forever."""
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()
    metadata = {1: "blue", 2: "unknown"}

    def detect_fn(local_idx, frame_path):
        return [(stub._rect_for(1), "blue"), (stub._rect_for(2), "red")]

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata=metadata, max_active_objects=6,
        diagnostics=diag,
    )
    assert metadata[2] == "red"
    assert metadata[1] == "blue"
    assert any(e.get("event") == "sam2.color_backfilled"
               and e["details"]["object_id"] == 2
               and e["details"]["color"] == "red" for e in diag._events)


def test_color_backfill_requires_two_votes():
    """A single colored corroboration is not enough to commit a color."""
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    metadata = {1: "blue", 2: "unknown"}
    calls = {"n": 0}

    def detect_fn(local_idx, frame_path):
        calls["n"] += 1
        if calls["n"] == 1:
            return [(stub._rect_for(2), "red")]
        return []

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata=metadata, max_active_objects=6,
    )
    assert metadata[2] == "unknown"


def test_color_backfill_conflict_resets_votes():
    """Alternating colors (two robots overlapping the mask) never backfill."""
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    # Objects 1 and 2 prompted; obj 2 is the unknown one under test.
    metadata = {1: "blue", 2: "unknown"}
    calls = {"n": 0}

    def detect_fn(local_idx, frame_path):
        calls["n"] += 1
        color = "red" if calls["n"] % 2 else "blue"
        return [(stub._rect_for(2), color)]  # always corroborates obj 2's mask

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata=metadata, max_active_objects=6,
    )
    assert metadata[2] == "unknown"


def test_color_backfill_never_overrides_definite_color():
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    metadata = {1: "red", 2: "blue"}

    def detect_fn(local_idx, frame_path):
        # Persistent contradicting color corroborating obj 2's mask.
        return [(stub._rect_for(2), "red")]

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata=metadata, max_active_objects=6,
    )
    assert metadata[2] == "blue"


def test_seed_gated_box_still_corroborates_mask():
    """A gate-blocked detection must still corroborate an existing mask —
    a robot parked inside a zone is detector-PRESENT, never cull/evict bait."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        # Only obj 1's own box, but the gate blocks ALL seeding.
        return [stub._rect_for(1)]

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1)],
        reprompt_every=2, detect_fn=detect_fn,
        max_active_objects=6, cull_enabled=True,
        cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5, seed_gate_fn=lambda box: False,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    # Stationary + corroborated by the gated box -> never culled.
    assert _culled_ids(diag) == set()


def test_detector_exception_is_unavailable_no_candidates_no_seed():
    """A detect_fn that raises is treated as detector-unavailable: no reprompt
    candidates are logged and no new objects are seeded that boundary."""
    from web.field.identity_diagnostics import IdentityDiagnosticRecorder
    from web.field.sam2_tracker import Sam2FieldTracker

    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = IdentityDiagnosticRecorder()

    def detect_fn(local_idx, frame_path):
        raise RuntimeError("detector blew up")

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1)],          # one object, NOT capped
        reprompt_every=2, detect_fn=detect_fn,
        max_active_objects=6, cull_enabled=True,   # run_detector True even though not capped
        diagnostics=diag,
    )

    # No reprompt candidates were logged (detector was unavailable every boundary).
    assert not any(e.get("event") == "sam2.reprompt_candidate" for e in diag._events)
    # Only the initial prompt object was ever added; nothing seeded at re-prompt.
    assert stub.add_calls == [(0, 1)]
    # The failure is visible: a detector_error event per failed boundary.
    errors = [e for e in diag._events if e.get("event") == "sam2.detector_error"]
    assert errors and "detector blew up" in errors[0]["details"]["error"]


# ---------------------------------------------------------------------------
# Sliding-window hijack cull — stationary + detector-absent objects are culled.
# ---------------------------------------------------------------------------


def _ident_diag():
    from web.field.identity_diagnostics import IdentityDiagnosticRecorder
    return IdentityDiagnosticRecorder()


def _culled_ids(diagnostics):
    return {e["details"]["object_id"]
            for e in diagnostics._events
            if e.get("event") == "sam2.object_culled"}


def test_cull_removes_uncorroborated_stationary_object():
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [(stub._rect_for(1), "blue")]

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "blue", 2: "red"}, max_active_objects=6,
        cull_enabled=True, cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_corrob_ios=0.3, cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 in _culled_ids(diag)
    assert 1 in result[stub.NUM_FRAMES - 1]
    assert 2 not in result[stub.NUM_FRAMES - 1]
    culled = [e for e in diag._events if e.get("event") == "sam2.object_culled"]
    assert any(e["details"]["object_id"] == 2
               and e.get("reason") == "detector_absent_stationary"
               for e in culled)


def test_cull_spares_moving_object_even_if_uncorroborated():
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor(moving={2})
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()
    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1, 0), stub._rect_for(2, 0)],
        reprompt_every=2, detect_fn=lambda *_a: [], max_active_objects=6,
        cull_enabled=True, cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5, project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 not in _culled_ids(diag)
    assert 2 in result[stub.NUM_FRAMES - 1]


def test_cull_spares_corroborated_stationary_object():
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()
    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=lambda *_a: [stub._rect_for(2)],
        max_active_objects=6, cull_enabled=True, cull_uncorroborated_reprompts=3,
        cull_evidence_window=5, cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 not in _culled_ids(diag)
    assert 2 in result[stub.NUM_FRAMES - 1]


def test_cull_fires_despite_one_early_corroboration():
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()
    seen = {"n": 0}

    def detect_fn(local_idx, frame_path):
        seen["n"] += 1
        return [stub._rect_for(2)] if seen["n"] == 1 else []

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn, max_active_objects=6,
        cull_enabled=True, cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5, project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 in _culled_ids(diag)


def test_detector_outage_does_not_cull():
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()
    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=lambda *_a: None,
        max_active_objects=6, cull_enabled=True, cull_uncorroborated_reprompts=3,
        cull_evidence_window=5, cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert _culled_ids(diag) == set()
    assert 1 in result[stub.NUM_FRAMES - 1] and 2 in result[stub.NUM_FRAMES - 1]


def test_cull_frees_active_slot_and_real_robot_reseeds():
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor(moving={3})
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [(stub._rect_for(1), "blue"), (stub._rect_for(3, local_idx), "red")]

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn, object_metadata={1: "blue", 2: "red"},
        max_active_objects=2, max_lifetime_objects=20,
        cull_enabled=True, cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5, project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 in _culled_ids(diag)
    assert any(oid == 3 for (_f, oid) in stub.add_calls)
    assert 3 in result[stub.NUM_FRAMES - 1]
    assert 2 not in result[stub.NUM_FRAMES - 1]


def test_cull_does_not_reseed_when_lifetime_capped():
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor(moving={3})
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [(stub._rect_for(1), "blue"), (stub._rect_for(3, local_idx), "red")]

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn, object_metadata={1: "blue", 2: "red"},
        max_active_objects=2, max_lifetime_objects=2,
        cull_enabled=True, cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5, project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 in _culled_ids(diag)
    assert all(oid != 3 for (_f, oid) in stub.add_calls)
    assert any(e.get("event") == "sam2.reprompt_rejected"
               and e.get("reason") == "lifetime_cap" for e in diag._events)


def test_none_projection_is_conservative():
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()
    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=lambda *_a: [], max_active_objects=6,
        cull_enabled=True, cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5, project_fn=lambda box: None, diagnostics=diag,
    )
    assert _culled_ids(diag) == set()


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


# ---------------------------------------------------------------------------
# Re-anchoring — a degraded (sliver) mask is re-prompted under its SAME obj_id
# when a fresh detection corroborates it, instead of seeding a churned new id.
# ---------------------------------------------------------------------------

class ReanchorStubPredictor:
    """Reprompt stub whose object 1 collapses to a 1px sliver after a frame.

    Object ``oid`` lives at x in [10*oid, 10*oid+4). From ``collapse_after``
    (inclusive) onward, object ``collapse_oid``'s mask is a 1x1 sliver at its
    box's top-left UNLESS add_new_points_or_box has been called again for that
    obj_id at frame >= collapse_after, which restores the full box from that
    frame (modelling SAM2 re-conditioning on a fresh prompt).
    """
    H = 20
    W = 200
    NUM_FRAMES = 12

    def __init__(self, collapse_oid=1, collapse_after=2):
        self.collapse_oid = collapse_oid
        self.collapse_after = collapse_after
        self.add_calls: list[tuple[int, int]] = []
        self._restored_at: int | None = None

    def init_state(self, video_path: str, **kwargs) -> dict:
        return {"objs": {}}

    def add_new_points_or_box(self, state, frame_idx, obj_id, box=None) -> None:
        first = obj_id not in state["objs"]
        if first:
            state["objs"][obj_id] = frame_idx
        self.add_calls.append((frame_idx, obj_id))
        if (not first and obj_id == self.collapse_oid
                and frame_idx >= self.collapse_after):
            self._restored_at = frame_idx

    def _rect_for(self, oid: int, f: int = 0) -> tuple[int, int, int, int]:
        x0 = oid * 10
        if (oid == self.collapse_oid and f >= self.collapse_after
                and (self._restored_at is None or f < self._restored_at)):
            return (x0, 2, x0 + 1, 3)  # collapsed sliver
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


def _reanchored_ids(diagnostics):
    return {e["details"]["object_id"]
            for e in diagnostics._events
            if e.get("event") == "sam2.object_reanchored"}


def test_reanchor_restores_collapsed_mask_under_same_id():
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = ReanchorStubPredictor(collapse_oid=1, collapse_after=2)
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        # The detector keeps seeing the full robot at obj 1's spot.
        return [(stub._rect_for(1, 0), "red")]

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[(10, 2, 14, 6)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "red"},
        reanchor_enabled=True,
        diagnostics=diag,
    )

    # Re-anchored: same obj id prompted again, no churned replacement id.
    reanchors = [(f, oid) for (f, oid) in stub.add_calls[1:] if oid == 1]
    assert reanchors, "expected a second add_new_points_or_box for obj 1"
    assert 1 in _reanchored_ids(diag)
    created = [e for e in diag._events if e["event"] == "sam2.object_created"]
    assert len(created) == 1, "no new object id may be seeded for the sliver's box"
    # After the re-anchor the mask is full-size again under id 1.
    last_frame = max(result)
    assert 1 in result[last_frame]
    x1, y1, x2, y2 = result[last_frame][1]["box"]
    assert (x2 - x1) >= 4, "mask should be restored to full width"


def test_reanchor_disabled_seeds_replacement_id():
    """Without re-anchoring the sliver cannot claim the detection
    (SEED_MIN_MASK_AREA_FRAC), so the box churns a NEW object id."""
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = ReanchorStubPredictor(collapse_oid=1, collapse_after=2)
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [(stub._rect_for(1, 0), "red")]

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[(10, 2, 14, 6)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "red"},
        reanchor_enabled=False,
        diagnostics=diag,
    )
    created = [e for e in diag._events if e["event"] == "sam2.object_created"]
    assert len(created) >= 2, "churn path: sliver's box seeds replacement id(s)"
    assert _reanchored_ids(diag) == set()


def test_reanchor_refuses_color_conflicting_detection():
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = ReanchorStubPredictor(collapse_oid=1, collapse_after=2)
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        # A BLUE detection over the collapsed RED object's spot.
        return [(stub._rect_for(1, 0), "blue")]

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[(10, 2, 14, 6)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "red"},
        reanchor_enabled=True,
        diagnostics=diag,
    )
    assert _reanchored_ids(diag) == set(), \
        "a blue box must never re-anchor a red object"


def test_reanchor_leaves_healthy_masks_alone():
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = ReanchorStubPredictor(collapse_oid=99, collapse_after=99)  # nothing collapses
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [(stub._rect_for(1, 0), "red")]

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[(10, 2, 14, 6)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "red"},
        reanchor_enabled=True,
        diagnostics=diag,
    )
    assert _reanchored_ids(diag) == set()
    assert len(stub.add_calls) == 1, "healthy mask must not be re-prompted"


# ---------------------------------------------------------------------------
# Re-anchor v2 guards: recency (fresh collapse only) + locality (near the last
# healthy extent). Modelled on the qm72 v1 failure where retired objects'
# noise slivers captured live robots' detections across the frame.
# ---------------------------------------------------------------------------

class ZombieStubPredictor:
    """Obj 1 is healthy at x=10 for frames [0, gone_from), then its mask is
    EMPTY until ``sliver_from``, after which a 1px noise sliver appears at
    ``sliver_x`` (far from home). A second add_new_points_or_box for obj 1
    'restores' a full box at the sliver site (so a wrongful re-anchor is
    observable). Obj 2 is a healthy robot parked at ``sliver_x`` throughout.
    """
    H = 20
    W = 200
    NUM_FRAMES = 12

    def __init__(self, gone_from=3, sliver_from=9, sliver_x=80):
        self.gone_from = gone_from
        self.sliver_from = sliver_from
        self.sliver_x = sliver_x
        self.add_calls: list[tuple[int, int]] = []
        self._restored_at: int | None = None

    def init_state(self, video_path: str, **kwargs) -> dict:
        return {"objs": {}}

    def add_new_points_or_box(self, state, frame_idx, obj_id, box=None) -> None:
        first = obj_id not in state["objs"]
        if first:
            state["objs"][obj_id] = frame_idx
        self.add_calls.append((frame_idx, obj_id))
        if not first and obj_id == 1:
            self._restored_at = frame_idx

    def _rect_for(self, oid: int, f: int) -> tuple[int, int, int, int] | None:
        if oid == 1:
            if self._restored_at is not None and f >= self._restored_at:
                return (self.sliver_x, 2, self.sliver_x + 4, 6)
            if f < self.gone_from:
                return (10, 2, 14, 6)
            if f >= self.sliver_from:
                return (self.sliver_x, 2, self.sliver_x + 1, 3)
            return None  # mask empty
        return (self.sliver_x, 2, self.sliver_x + 4, 6)  # obj 2: parked robot

    def propagate_in_video(self, state, start_frame_idx=None, max_frame_num_to_track=None):
        start = 0 if start_frame_idx is None else start_frame_idx
        end = (self.NUM_FRAMES - 1 if max_frame_num_to_track is None
               else min(start + max_frame_num_to_track, self.NUM_FRAMES - 1))
        for f in range(start, end + 1):
            active = sorted(oid for oid, af in state["objs"].items() if af <= f)
            rects, oids = [], []
            for oid in active:
                r = self._rect_for(oid, f)
                if r is not None:
                    rects.append(r)
                    oids.append(oid)
                else:
                    rects.append((0, 0, 0, 0))  # empty mask row
                    oids.append(oid)
            yield f, oids, _make_mask_array(self.H, self.W, rects)

    def reset_state(self, state) -> None:
        pass


def test_reanchor_refuses_zombie_resurrection():
    """A long-dead object's noise sliver must NOT capture a live robot's
    detection — the v1 failure mode (qm72: retired obj re-anchored 900+
    frames later onto a robot across the frame)."""
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = ZombieStubPredictor(gone_from=3, sliver_from=9, sliver_x=80)
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        # The detector sees the real parked robot at the sliver site.
        return [((80, 2, 84, 6), "red")]

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[(10, 2, 14, 6)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "red"},
        reanchor_enabled=True,
        reanchor_max_jump_px=30.0,
        diagnostics=diag,
    )
    assert 1 not in _reanchored_ids(diag), \
        "zombie sliver must not be re-anchored onto another robot"


def test_reanchor_refuses_drifted_sliver():
    """A sliver that drifted far from the object's last healthy position must
    not capture a detection there, even if the collapse is recent."""
    from web.field.sam2_tracker import Sam2FieldTracker

    class DriftStub(ZombieStubPredictor):
        def _rect_for(self, oid, f):
            if oid == 1:
                if self._restored_at is not None and f >= self._restored_at:
                    return (self.sliver_x, 2, self.sliver_x + 4, 6)
                if f < 3:
                    return (10, 2, 14, 6)
                return (self.sliver_x, 2, self.sliver_x + 1, 3)  # drifted sliver
            return (self.sliver_x, 2, self.sliver_x + 4, 6)

    stub = DriftStub(sliver_x=80)
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [((80, 2, 84, 6), "red")]

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[(10, 2, 14, 6)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "red"},
        reanchor_enabled=True,
        reanchor_max_jump_px=30.0,
        diagnostics=diag,
    )
    assert 1 not in _reanchored_ids(diag), \
        "a far-drifted sliver must not be re-anchored (locality guard)"


def test_reanchor_in_place_collapse_still_rescued():
    """The guards must not break the intended case: a fresh in-place collapse
    (healthy one boundary ago, detection at the same spot) re-anchors."""
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = ReanchorStubPredictor(collapse_oid=1, collapse_after=4)
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [(stub._rect_for(1, 0), "red")]

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[(10, 2, 14, 6)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "red"},
        reanchor_enabled=True,
        reanchor_max_jump_px=30.0,
        diagnostics=diag,
    )
    assert 1 in _reanchored_ids(diag)


# ---------------------------------------------------------------------------
# Evidence-ranked roster: dark-mask re-anchor + tiered eviction
# ---------------------------------------------------------------------------

def test_match_new_detections_with_claims_returns_claimant():
    """with_claims=True also maps each matched box index to the object that
    claimed it, so the caller can act on WHO blocked a detection."""
    existing = {4: _rect_mask(0, 0, 5, 10)}
    new_boxes = [(1, 1, 4, 9), (6, 1, 9, 9)]
    unmatched, claims = match_new_detections(
        existing, new_boxes, (10, 10), ios_thresh=0.5, with_claims=True
    )
    assert unmatched == [1]
    assert claims == {0: 4}


def test_dark_mask_reanchor_reprompts_same_object():
    """A detection overlapping a dark object's stale mask re-prompts that
    object in place instead of being dropped as already_tracked."""
    from web.field.sam2_tracker import Sam2FieldTracker

    # obj 1 always visible; obj 2 visible only frames 0-1, dark afterwards.
    schedule = {
        0: [1, 2], 1: [1, 2],
        2: [1], 3: [1], 4: [1], 5: [1], 6: [1],
    }
    stub = AbsentStubPredictor(schedule, num_frames=7)
    tracker = Sam2FieldTracker(predictor=stub)

    def detect_fn(local_idx, frame_path):
        # The detector still sees both robots at their fixed spots.
        return [(6, 2, 10, 6), (12, 2, 16, 6)]

    tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[(6, 2, 10, 6), (12, 2, 16, 6)],
        reprompt_every=2,
        detect_fn=detect_fn,
        max_active_objects=6,
        retirement_frames=10,
        dark_reanchor_frames=2,
    )

    # obj 2 went dark at frame 2; once the gap reaches dark_reanchor_frames
    # a boundary re-prompts the SAME obj id at the detector box.
    later_calls = stub.add_calls[2:]
    assert any(oid == 2 for (_f, oid) in later_calls)
    # No new object id was invented for the still-dark robot.
    assert all(oid <= 2 for (_f, oid) in stub.add_calls)


def test_visible_object_detection_not_dark_reanchored():
    """A detection over a currently-visible object stays a plain dedup: no
    extra re-prompt is issued for it."""
    from web.field.sam2_tracker import Sam2FieldTracker

    schedule = {f: [1, 2] for f in range(7)}  # both always visible
    stub = AbsentStubPredictor(schedule, num_frames=7)
    tracker = Sam2FieldTracker(predictor=stub)

    def detect_fn(local_idx, frame_path):
        return [(6, 2, 10, 6), (12, 2, 16, 6)]

    tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[(6, 2, 10, 6), (12, 2, 16, 6)],
        reprompt_every=2,
        detect_fn=detect_fn,
        max_active_objects=6,
        dark_reanchor_frames=2,
    )

    assert stub.add_calls == [(0, 1), (0, 2)]


def test_pick_evictable_evicts_dark_uncorroborated_object():
    """Tier 2: an object whose mask has been absent for >= dark_frames and
    whose stale mask drew no detector support is evictable even when the
    stationary-absence bar (tier 1) is not met."""
    from collections import deque
    from web.field.sam2_tracker import pick_evictable

    hist = {7: deque([True, False], maxlen=6)}     # only 1 absence -> tier 1 (k=3) can't fire
    window = {7: [(1.0, 1.0), (4.0, 5.0)]}          # moving -> old rule can't fire either
    victim = pick_evictable(
        [7], hist, window, 3, 0.5,
        last_visible_frame={7: 10}, current_idx=14, dark_frames=2,
    )
    assert victim == 7


def test_pick_evictable_evicts_moving_never_corroborated():
    """Tier 3: a moving object the detector repeatedly fails to back (e.g. a
    referee hijack) is evictable despite its motion."""
    from collections import deque
    from web.field.sam2_tracker import pick_evictable

    hist = {3: deque([False, False, True, False], maxlen=6)}
    window = {3: [(1.0, 1.0), (6.0, 4.0)]}          # moving
    victim = pick_evictable(
        [3], hist, window, 99, 0.5,                  # tier 1 unreachable (k=99)
        last_visible_frame={3: 14}, current_idx=14, dark_frames=2,  # visible now -> tier 2 can't fire
    )
    assert victim == 3


def test_pick_evictable_never_evicts_just_corroborated():
    """An object the detector backed at the latest boundary is never evicted,
    whatever its history."""
    from collections import deque
    from web.field.sam2_tracker import pick_evictable

    hist = {5: deque([False, False, False, True], maxlen=6)}
    window = {5: [(1.0, 1.0), (1.0, 1.1)]}          # stationary
    victim = pick_evictable(
        [5], hist, window, 3, 0.5,
        last_visible_frame={5: 2}, current_idx=14, dark_frames=2,  # long dark
    )
    assert victim is None


def test_pick_evictable_prefers_stationary_absent_over_dark():
    """Tier order: the classic stationary+absent victim wins over a merely
    dark one."""
    from collections import deque
    from web.field.sam2_tracker import pick_evictable

    hist = {
        1: deque([False, False, False], maxlen=6),   # stationary + absent (tier 1)
        2: deque([True, False], maxlen=6),           # dark (tier 2)
    }
    window = {
        1: [(1.0, 1.0), (1.0, 1.05)],
        2: [(3.0, 3.0), (5.0, 6.0)],
    }
    victim = pick_evictable(
        [1, 2], hist, window, 3, 0.5,
        last_visible_frame={1: 14, 2: 9}, current_idx=14, dark_frames=2,
    )
    assert victim == 1


def test_capped_roster_evicts_dark_zombie_for_new_detection():
    """A capped roster holding a dark, detector-orphaned object gives up that
    slot to a fresh detection of a real robot."""
    from web.field.sam2_tracker import Sam2FieldTracker

    # obj 2 goes dark at frame 2 and the detector NEVER sees anything there.
    schedule = {
        0: [1, 2], 1: [1, 2],
        2: [1], 3: [1], 4: [1], 5: [1], 6: [1],
    }
    stub = AbsentStubPredictor(schedule, num_frames=7)
    tracker = Sam2FieldTracker(predictor=stub)

    def detect_fn(local_idx, frame_path):
        # obj 1 corroborated; a NEW robot at a fresh spot (x=30..34).
        return [(6, 2, 10, 6), (30, 2, 34, 6)]

    tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[(6, 2, 10, 6), (12, 2, 16, 6)],
        reprompt_every=2,
        detect_fn=detect_fn,
        max_active_objects=2,                  # capped at construction
        retirement_frames=10,                  # obj 2 stays in-roster
        cull_uncorroborated_reprompts=99,      # tier 1 unreachable
        evict_when_capped=True,
        evict_dark_frames=2,
        dark_reanchor_frames=2,
    )

    # The dark zombie was evicted and the new robot seeded under a new id.
    assert any(oid == 3 for (_f, oid) in stub.add_calls)


def test_dark_reanchor_attempts_capped_then_new_seed():
    """A dark object that never revives gets a bounded number of re-anchor
    attempts; afterwards the detection seeds a NEW object id instead of
    re-prompting forever (every prompt accumulates session state — unbounded
    re-prompting OOM'd full-card SAM3 runs)."""
    from web.field.sam2_tracker import Sam2FieldTracker

    # obj 2 visible only at frames 0-1, dark for the rest of a long video.
    schedule = {0: [1, 2], 1: [1, 2]}
    for f in range(2, 12):
        schedule[f] = [1]
    stub = AbsentStubPredictor(schedule, num_frames=12)
    tracker = Sam2FieldTracker(predictor=stub)

    def detect_fn(local_idx, frame_path):
        return [(6, 2, 10, 6), (12, 2, 16, 6)]

    tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[(6, 2, 10, 6), (12, 2, 16, 6)],
        reprompt_every=2,
        detect_fn=detect_fn,
        max_active_objects=6,
        retirement_frames=50,            # keep the zombie in-roster
        dark_reanchor_frames=2,
        dark_reanchor_max_attempts=2,
    )

    reanchor_calls = [(f, oid) for (f, oid) in stub.add_calls[2:] if oid == 2]
    assert len(reanchor_calls) == 2, f"expected 2 capped attempts, got {reanchor_calls}"
    # After the cap, the still-detected robot seeds a fresh object id.
    assert any(oid == 3 for (_f, oid) in stub.add_calls)


# ---------------------------------------------------------------------------
# Motion-override: pixel-motion evidence unblocks the seed rejection gates
# (qm11: a sprinting robot's detections died as already_tracked against a
# mask parked on static field equipment, then as roster_cap).
# ---------------------------------------------------------------------------


def test_box_motion_energy_high_for_moving_object_low_for_static():
    from web.field.sam2_tracker import box_motion_energy
    prev = np.zeros((20, 200), np.uint8)
    cur = np.zeros((20, 200), np.uint8)
    # A bright robot-blob that moves 6 px between frames inside the box.
    prev[4:12, 60:80] = 200
    cur[4:12, 66:86] = 200
    moving_box = (55, 2, 90, 14)
    static_box = (120, 2, 155, 14)
    assert box_motion_energy(prev, cur, moving_box) > 0.05
    assert box_motion_energy(prev, cur, static_box) == 0.0
    # Degenerate box never divides by zero.
    assert box_motion_energy(prev, cur, (5, 5, 5, 5)) == 0.0


def test_motion_override_seeds_new_object_over_static_claimant():
    """A detection claimed by a STATIONARY mask ("already_tracked") seeds a
    new object anyway when motion_fn proves the detection is moving — the
    static claimant is equipment/hijack, the mover is a robot."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [stub._rect_for(1)]  # always claimed by obj 1's mask

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        motion_fn=lambda local_idx, box: True,
        max_active_objects=6, cull_enabled=False,
        cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert any(oid == 3 for (_f, oid) in stub.add_calls), \
        "moving detection over a static claimant must seed a new object"
    assert any(e.get("event") == "sam2.motion_override" for e in diag._events)


def test_motion_override_spares_moving_claimant():
    """A claimant mask that is itself moving is a robot following its own
    detection — the override must NOT fire (no duplicate ids)."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor(moving={1})
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        # Track obj 1's drifting rect so the detection stays claimed by it.
        boundary_frame = local_idx
        return [stub._rect_for(1, boundary_frame)]

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1, 0), stub._rect_for(2, 0)],
        reprompt_every=2, detect_fn=detect_fn,
        motion_fn=lambda local_idx, box: True,
        max_active_objects=6, cull_enabled=False,
        cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert all(oid != 3 for (_f, oid) in stub.add_calls), \
        "override must not duplicate a claimant that is itself moving"
    assert not any(e.get("event") == "sam2.motion_override" for e in diag._events)


def test_motion_override_ignored_when_candidate_static():
    """motion_fn says the candidate shows no pixel motion: the claimed
    detection stays already_tracked (a parked robot is not equipment)."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [stub._rect_for(1)]

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        motion_fn=lambda local_idx, box: False,
        max_active_objects=6, cull_enabled=False,
        cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert all(oid != 3 for (_f, oid) in stub.add_calls)
    assert any(e.get("event") == "sam2.reprompt_rejected"
               and e.get("reason") == "already_tracked" for e in diag._events)


def test_motion_fn_exception_treated_as_static():
    """A crashing motion_fn must degrade to no-override, never kill tracking."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [stub._rect_for(1)]

    def motion_fn(local_idx, box):
        raise RuntimeError("boom")

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn, motion_fn=motion_fn,
        max_active_objects=6, cull_enabled=False,
        cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert all(oid != 3 for (_f, oid) in stub.add_calls)
    assert any(e.get("event") == "sam2.reprompt_rejected"
               and e.get("reason") == "already_tracked" for e in diag._events)


def test_pick_evictable_moving_candidate_relaxes_absence_bar():
    """A motion-proven candidate may evict a stationary object absent at
    BOTH of the last two boundaries (YOLO priority: anything the detector
    backed recently is immune), even when min_absences/k put every normal
    tier out of reach. Without the moving-candidate flag the same evidence
    evicts nobody."""
    from web.field.sam2_tracker import pick_evictable
    from collections import deque
    hist = {1: deque([True, False, False]), 2: deque([True, True])}
    window = {1: [(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)],
              2: [(8.0, 8.0), (8.0, 8.0)]}
    kwargs = dict(k=99, stationary_m=0.5, last_visible_frame={1: 10, 2: 10},
                  current_idx=10, dark_frames=999, min_absences=99)
    assert pick_evictable([1, 2], hist, window, **kwargs) is None
    assert pick_evictable([1, 2], hist, window,
                          moving_candidate=True, **kwargs) == 1


def test_moving_candidate_evicts_stationary_absentee_when_capped():
    """Roster capped, every normal eviction tier out of reach: an unmatched
    detection that PROVES motion still claims a seat from a stationary object
    the detector failed to back; without motion proof it stays roster_cap."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center

    def make_detect(stub):
        # Corroborate obj 1 every boundary; obj 2 never; one new mover at 15.
        def detect_fn(local_idx, frame_path):
            return [(stub._rect_for(1), "blue"), (stub._rect_for(15), "red")]
        return detect_fn

    strict = dict(
        reprompt_every=2, max_active_objects=2, cull_enabled=False,
        cull_uncorroborated_reprompts=99, cull_evidence_window=5,
        cull_stationary_m=0.5, evict_min_absences=99, evict_dark_frames=999,
        object_metadata={1: "blue", 2: "red"},
    )

    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()
    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        detect_fn=make_detect(stub),
        motion_fn=lambda local_idx, box: True,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
        **strict,
    )
    assert 2 in _culled_ids(diag), \
        "motion-proven candidate must evict the stationary absentee"
    assert any(oid == 3 for (_f, oid) in stub.add_calls)

    stub2 = CullStubPredictor()
    tracker2 = Sam2FieldTracker(predictor=stub2)
    diag2 = _ident_diag()
    tracker2.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub2._rect_for(1), stub2._rect_for(2)],
        detect_fn=make_detect(stub2),
        project_fn=lambda box: bottom_center(box), diagnostics=diag2,
        **strict,
    )
    assert _culled_ids(diag2) == set()
    assert all(oid != 3 for (_f, oid) in stub2.add_calls)
    assert any(e.get("event") == "sam2.reprompt_rejected"
               and e.get("reason") == "roster_cap" for e in diag2._events)


def test_make_frame_diff_motion_fn_reads_exported_frames(tmp_path):
    """The factory builds a motion_fn over the exported-frames directory:
    True where pixels moved between local_idx-1 and local_idx, False in
    static regions, False (never raising) when a frame is missing."""
    import cv2
    from web.field.sam2_tracker import make_frame_diff_motion_fn
    prev = np.zeros((20, 200, 3), np.uint8)
    cur = np.zeros((20, 200, 3), np.uint8)
    prev[4:12, 60:80] = 200
    cur[4:12, 66:86] = 200
    cv2.imwrite(str(tmp_path / "00004.jpg"), prev)
    cv2.imwrite(str(tmp_path / "00005.jpg"), cur)

    motion_fn = make_frame_diff_motion_fn(str(tmp_path), energy_thresh=0.05)
    assert motion_fn(5, (55, 2, 90, 14)) is True
    assert motion_fn(5, (120, 2, 155, 14)) is False
    # local_idx 0 has no predecessor; missing files are not an error.
    assert motion_fn(0, (55, 2, 90, 14)) is False
    assert motion_fn(9, (55, 2, 90, 14)) is False


def test_pick_evictable_motion_never_evicts_recently_corroborated():
    """YOLO priority: a roster member the detector backed at EITHER of the
    last two boundaries is immune to motion-driven (tier 2.5) eviction — a
    parked robot missing one boundary read must not lose its seat to a
    motion-only candidate (ball cluster / misread)."""
    from web.field.sam2_tracker import pick_evictable
    from collections import deque
    hist = {1: deque([False, True, False]),   # corroborated one boundary ago
            2: deque([True, False, False])}   # absent last TWO boundaries
    window = {1: [(5.0, 5.0)] * 3, 2: [(8.0, 8.0)] * 3}
    kwargs = dict(k=99, stationary_m=0.5, last_visible_frame={1: 10, 2: 10},
                  current_idx=10, dark_frames=999, min_absences=99)
    assert pick_evictable([1, 2], hist, window,
                          moving_candidate=True, **kwargs) == 2
    # With only the recently-corroborated object present: nobody is evicted.
    assert pick_evictable([1], hist, window,
                          moving_candidate=True, **kwargs) is None


# ---------------------------------------------------------------------------
# Person-overlap veto: a mask sitting mostly inside a person detection for
# consecutive boundaries is a hijack (a robot is never a person) — the
# positive signal the absence-based cull lacks (qm24: absence+stationarity
# alone killed real blind-spot robots; qm11: R3 spent all of auto on a ref).
# ---------------------------------------------------------------------------

def _track_with_person_fn(stub, person_fn, **overrides):
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()
    kwargs = dict(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=lambda *_a: [(stub._rect_for(1), "blue")],
        object_metadata={1: "blue", 2: "red"}, max_active_objects=6,
        cull_enabled=False,
        person_fn=person_fn,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    kwargs.update(overrides)
    result = tracker.track(**kwargs)
    return result, diag


def test_person_veto_culls_object_locked_on_person():
    """Object 2 (rect x[20,24) y[2,6)) fully inside a person box at every
    boundary -> culled with reason person_overlap, even with cull_enabled=False.
    Object 1 is untouched."""
    stub = CullStubPredictor()
    result, diag = _track_with_person_fn(
        stub, person_fn=lambda idx, path: [(18.0, 0.0, 26.0, 8.0)])
    culled = [e for e in diag._events if e.get("event") == "sam2.object_culled"]
    assert any(e["details"]["object_id"] == 2 and e.get("reason") == "person_overlap"
               for e in culled), f"expected person_overlap cull of obj 2, got {culled}"
    assert 1 in result[stub.NUM_FRAMES - 1]
    assert 2 not in result[stub.NUM_FRAMES - 1]


def test_person_veto_requires_consecutive_boundaries():
    """A single-boundary overlap (person walks past) must NOT cull."""
    stub = CullStubPredictor()
    calls = {"n": 0}

    def person_fn(idx, path):
        calls["n"] += 1
        return [(18.0, 0.0, 26.0, 8.0)] if calls["n"] == 1 else []

    result, diag = _track_with_person_fn(stub, person_fn)
    assert 2 not in _culled_ids(diag)
    assert 2 in result[stub.NUM_FRAMES - 1]


def test_person_veto_ignores_partial_overlap():
    """Coverage below the threshold (person box clips 1px of the 4px-wide
    mask box -> 0.25) is not a hijack signal."""
    stub = CullStubPredictor()
    result, diag = _track_with_person_fn(
        stub, person_fn=lambda idx, path: [(23.0, 0.0, 40.0, 8.0)])
    assert 2 not in _culled_ids(diag)
    assert 2 in result[stub.NUM_FRAMES - 1]


def test_person_fn_error_is_nonfatal():
    """A crashing person_fn degrades to no-veto; track() completes."""
    stub = CullStubPredictor()

    def person_fn(idx, path):
        raise RuntimeError("person model exploded")

    result, diag = _track_with_person_fn(stub, person_fn)
    assert 2 not in _culled_ids(diag)
    assert 2 in result[stub.NUM_FRAMES - 1]


def test_person_veto_spares_moving_object():
    """A MOVING mask overlapped by a person (robot driving past a ref at the
    wall) is not vetoed — the veto is for parked hijacks, and motion is the
    strongest robot-not-equipment signal we have."""
    stub = CullStubPredictor(moving={2})
    result, diag = _track_with_person_fn(
        stub, person_fn=lambda idx, path: [(0.0, 0.0, 200.0, 20.0)])
    assert 2 not in _culled_ids(diag)
    assert 2 in result[stub.NUM_FRAMES - 1]


def test_cull_enabled_for_view_scopes_to_main():
    """Main-view scoping: mid-field equipment is fully visible to main-view
    YOLO (absence there is strong evidence); corner-view objects are the
    qm24 false-cull victims and stay exempt."""
    from web.field.sam2_tracker import cull_enabled_for_view
    assert cull_enabled_for_view(True, "main", main_only=True) is True
    assert cull_enabled_for_view(True, "corner_left", main_only=True) is False
    assert cull_enabled_for_view(True, "corner_left", main_only=False) is True
    assert cull_enabled_for_view(False, "main", main_only=True) is False
    assert cull_enabled_for_view(None, "main", main_only=True) is None
    assert cull_enabled_for_view(None, "corner_left", main_only=True) is False


# ---------------------------------------------------------------------------
# Zone-blind corroboration: a detection whose box the corrob_gate_fn rejects
# (contact point inside a static-structure zone) must not count as mask
# corroboration — the OVERFLOW bin's bumper-lookalike false positive otherwise
# corroborates its own hijacked mask forever, so the absence cull never fires.
# Robots physically cannot park inside the bin footprint, so blinding
# corroboration there is safe.
# ---------------------------------------------------------------------------

def test_zone_gated_detection_does_not_corroborate():
    """Object 2 is stationary and 'detected' every boundary, but the gate
    rejects those detections (equipment zone) -> absence accumulates -> the
    cull fires exactly as if the detector had never seen it."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [(stub._rect_for(1), "blue"), (stub._rect_for(2), "red")]

    def corrob_gate_fn(box):
        # reject anything near object 2's spot (the "bin footprint")
        return not (18 <= (box[0] + box[2]) / 2 <= 26)

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "blue", 2: "red"}, max_active_objects=6,
        cull_enabled=True, cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_corrob_ios=0.3, cull_stationary_m=0.5,
        corrob_gate_fn=corrob_gate_fn,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 in _culled_ids(diag)
    assert 1 in result[stub.NUM_FRAMES - 1]        # obj 1 corroborated, kept
    assert 2 not in result[stub.NUM_FRAMES - 1]


def test_corrob_gate_error_fails_open():
    """A crashing corrob_gate_fn must degrade to normal corroboration (no
    cull) — never to mass blinding."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def corrob_gate_fn(box):
        raise RuntimeError("projector exploded")

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2,
        detect_fn=lambda *_a: [(stub._rect_for(2), "red")],
        object_metadata={1: "blue", 2: "red"}, max_active_objects=6,
        cull_enabled=True, cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5, corrob_gate_fn=corrob_gate_fn,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 not in _culled_ids(diag)
    assert 2 in result[stub.NUM_FRAMES - 1]


def test_corrob_only_detection_survives_gate_filter():
    """A corrob-only detection that PASSES the gate must still count as
    corroboration — the gate filter runs over the combined (seedable +
    corrob-only) evidence, so a gate-passing corrob-only box keeps its
    object alive instead of being silently dropped."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        # obj 1 backed by a seedable box; obj 2 backed ONLY by a corrob-only
        # box. Both sit outside the gated (equipment) zone.
        return [(stub._rect_for(1), "blue"),
                (stub._rect_for(2), "red", True)]

    def corrob_gate_fn(box):
        # gate rejects a far-off equipment footprint neither object touches
        return not (40 <= (box[0] + box[2]) / 2 <= 50)

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "blue", 2: "red"}, max_active_objects=6,
        cull_enabled=True, cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_corrob_ios=0.3, cull_stationary_m=0.5,
        corrob_gate_fn=corrob_gate_fn,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    # obj 2's corrob-only detection survived the gate -> no absence -> no cull.
    assert 2 not in _culled_ids(diag)
    assert 2 in result[stub.NUM_FRAMES - 1]


# ---------------------------------------------------------------------------
# Seed-box aspect gate + motion-required eviction (qm12 static-sliver fix).
#
# In match 86d6e154 qm12 a static vertical sliver on the blue hub face
# (box [320.8, 145.0, 334.5, 206.1] — aspect h/w = 4.47) recurred at every
# chunk boundary. Once unmatched on a capped roster it EVICTED a real robot
# object to seed itself, then emitted mask_implausible masks for 7 frames.
# The two gates below stop that: reject a geometrically implausible seed box
# before it can seed OR evict, and (separately) require motion evidence before
# a candidate may evict a tracked object.
# ---------------------------------------------------------------------------


def test_plausible_seed_box_rejects_tall_thin_sliver():
    from web.field.sam2_tracker import plausible_seed_box
    # The qm12 sliver: 13.7w x 61.1h, aspect 4.47 — rejected at max_aspect=3.0.
    assert plausible_seed_box((320.8, 145.0, 334.5, 206.1), 3.0) is False


def test_plausible_seed_box_keeps_square_box():
    from web.field.sam2_tracker import plausible_seed_box
    # A 4x4 robot box (aspect 1.0) is kept.
    assert plausible_seed_box((10, 2, 14, 6), 3.0) is True


def test_plausible_seed_box_keeps_wide_box():
    from web.field.sam2_tracker import plausible_seed_box
    # A wide box (aspect < 1) is kept.
    assert plausible_seed_box((10, 2, 40, 10), 3.0) is True


def test_plausible_seed_box_rejects_degenerate_box():
    from web.field.sam2_tracker import plausible_seed_box
    assert plausible_seed_box((10, 2, 10, 6), 3.0) is False   # zero width
    assert plausible_seed_box((10, 6, 14, 6), 3.0) is False   # zero height
    assert plausible_seed_box((14, 6, 10, 2), 3.0) is False   # inverted


def test_plausible_seed_box_keeps_tall_box_just_under_threshold():
    from web.field.sam2_tracker import plausible_seed_box
    # 4w x 11h -> aspect 2.75 < 3.0 -> kept.
    assert plausible_seed_box((10, 2, 14, 13), 3.0) is True


# A tall-thin candidate box for the tracker tests: 4w x 28h -> aspect 7.0, at
# x in [150, 154) which overlaps no tracked object (objs 1,2 live at x 10, 20).
_SLIVER_BOX = (150, 2, 154, 30)


def test_implausible_seed_box_never_seeds_or_evicts():
    """A tall-thin static detection must not seed a new object NOR evict a real
    track, even when the roster is capped and the sliver is unmatched."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        # Obj 1 corroborated every boundary; obj 2 never; plus the tall-thin
        # unmatched sliver (would evict obj 2 without the gate).
        return [(stub._rect_for(1), "blue"), (_SLIVER_BOX, "red")]

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "blue", 2: "red"},
        max_active_objects=2, max_lifetime_objects=20,
        cull_enabled=False,
        cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5, seed_box_max_aspect=3.0,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    # No eviction of any object.
    assert _culled_ids(diag) == set()
    # Obj 3 (the sliver's would-be id) is never added.
    assert all(oid != 3 for (_f, oid) in stub.add_calls)
    # A distinguishable rejection diagnostic was recorded.
    rejects = [e for e in diag._events
               if e.get("event") == "sam2.reprompt_rejected"
               and e.get("reason") == "implausible_seed_box"]
    assert rejects, "expected an implausible_seed_box rejection"
    # The gate does nothing else: obj 2 survives.
    assert 2 in result[stub.NUM_FRAMES - 1]


def test_seed_box_gate_disabled_by_default():
    """With the gate off (default kwargs) the identical scenario evicts obj 2
    and seeds obj 3 — proves default behavior is unchanged."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [(stub._rect_for(1), "blue"), (_SLIVER_BOX, "red")]

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "blue", 2: "red"},
        max_active_objects=2, max_lifetime_objects=20,
        cull_enabled=False,
        cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5,  # seed_box_max_aspect defaults OFF
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 in _culled_ids(diag)
    assert any(oid == 3 for (_f, oid) in stub.add_calls)
    assert 3 in result[stub.NUM_FRAMES - 1]
    assert 2 not in result[stub.NUM_FRAMES - 1]


def test_static_candidate_cannot_evict_when_motion_required():
    """With evict_requires_motion=True and a motion_fn reporting the candidate
    static, eviction is skipped (candidate falls through to roster_cap). With
    motion_fn reporting motion, eviction proceeds."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center

    # --- static candidate: no eviction ---
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        # Obj 1 corroborated; obj 2 never; a plausible unmatched new robot.
        return [(stub._rect_for(1), "blue"), (stub._rect_for(15), "red")]

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "blue", 2: "red"},
        max_active_objects=2, max_lifetime_objects=20,
        cull_enabled=False,
        cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5,
        evict_requires_motion=True,
        motion_fn=lambda local_idx, box: False,  # candidate is static
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert _culled_ids(diag) == set()
    assert all(oid != 3 for (_f, oid) in stub.add_calls)
    assert any(e.get("event") == "sam2.object_culled" for e in diag._events) is False
    assert any(e.get("event") == "sam2.eviction_skipped"
               and e.get("reason") == "static_candidate" for e in diag._events)
    assert any(e.get("event") == "sam2.reprompt_rejected"
               and e.get("reason") == "roster_cap" for e in diag._events)
    assert 2 in result[stub.NUM_FRAMES - 1]

    # --- moving candidate: eviction proceeds ---
    stub2 = CullStubPredictor()
    tracker2 = Sam2FieldTracker(predictor=stub2)
    diag2 = _ident_diag()
    tracker2.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub2._rect_for(1), stub2._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "blue", 2: "red"},
        max_active_objects=2, max_lifetime_objects=20,
        cull_enabled=False,
        cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5,
        evict_requires_motion=True,
        motion_fn=lambda local_idx, box: True,  # candidate is moving
        project_fn=lambda box: bottom_center(box), diagnostics=diag2,
    )
    assert 2 in _culled_ids(diag2)
    assert any(oid == 3 for (_f, oid) in stub2.add_calls)


def test_evict_requires_motion_noop_without_motion_fn():
    """evict_requires_motion=True with motion_fn=None must NOT disable eviction
    — the gate only blocks when motion evidence is actually available."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [(stub._rect_for(1), "blue"), (stub._rect_for(15), "red")]

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "blue", 2: "red"},
        max_active_objects=2, max_lifetime_objects=20,
        cull_enabled=False,
        cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5,
        evict_requires_motion=True, motion_fn=None,  # no motion probe
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    # Eviction proceeds exactly as today.
    assert 2 in _culled_ids(diag)
    assert any(oid == 3 for (_f, oid) in stub.add_calls)
    assert 3 in result[stub.NUM_FRAMES - 1]
    assert not any(e.get("event") == "sam2.eviction_skipped"
                   for e in diag._events)


def test_bootstrap_rejects_implausible_seed_box(tmp_path):
    """The bootstrap seeding path also filters implausible boxes: a tall-thin
    box in the first detection is skipped while a plausible box is seeded."""
    from web.field.sam2_tracker import Sam2FieldTracker

    for idx in range(CullStubPredictor.NUM_FRAMES):
        (tmp_path / f"{idx:05d}.jpg").touch()

    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)

    def detect_fn(local_idx, frame_path):
        if local_idx == 0:
            # A tall-thin sliver (rejected) and a plausible robot box (seeded).
            return [(_SLIVER_BOX, "red"), (stub._rect_for(1), "blue")]
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
        seed_box_max_aspect=3.0,
    )
    # Exactly one object bootstrapped, from the plausible box (obj 1), not the
    # sliver. It sits at obj 1's location, not the sliver's x.
    assert stub.add_calls == [(0, 1)]
    assert metadata == {1: "blue"}
    assert 1 in result[CullStubPredictor.NUM_FRAMES - 1]


# ---------------------------------------------------------------------------
# qm11/qm10/qm12 GATED leak cleanup — clean_disconnected_leaks helper + wiring
#
# A robot is one blob. When SAM leaks the mask onto an adjacent structure (a
# bump/ramp beside the robot) mask_to_box returns the union bbox spanning both,
# so the rendered box stretches and the projected field position (box
# bottom-center) lands between them. Unconditional largest-component drops that
# leak but ALSO drops legitimate secondary components — a robot arm, or a robot
# split by hub/ball occlusion into two blobs — which shifts box bottom-center
# and REGRESSED qm10/qm12 (red meanNN worsened). The gated cleaner drops a
# component only if it is BOTH small (area < min_frac of the largest) AND
# detached (bbox gap > gap_px). The knob defaults OFF pending re-validation;
# when off, behavior must be byte-identical.
# ---------------------------------------------------------------------------


def test_leak_cleanup_drops_small_far_blob():
    from web.field.sam2_tracker import clean_disconnected_leaks
    mask = np.zeros((40, 60), bool)
    mask[10:30, 10:30] = True   # big blob (400px), bbox (10,10,30,30)
    mask[2:5, 50:55] = True     # small (15px < 15% of 400) + far (gap > 10px)
    cleaned = clean_disconnected_leaks(mask)
    # Small AND detached -> dropped; only the big blob survives.
    assert mask_to_box(cleaned) == (10, 10, 30, 30)
    assert not cleaned[2:5, 50:55].any()
    assert cleaned[10:30, 10:30].all()


def test_leak_cleanup_keeps_large_far_blob():
    """A SECOND blob far away but LARGE (an occlusion-split robot half) is KEPT.

    This is the qm10/qm12 regression-fixer: unconditional largest-component
    wrongly dropped this component, shifting box bottom-center. area >= 15% of
    the largest keeps it despite the gap.
    """
    from web.field.sam2_tracker import clean_disconnected_leaks
    mask = np.zeros((40, 60), bool)
    mask[10:30, 10:30] = True   # big blob (400px), bbox (10,10,30,30)
    mask[10:26, 44:54] = True   # 16x10 = 160px (>= 15% of 400) but far (gap>10)
    cleaned = clean_disconnected_leaks(mask)
    # Large component kept -> union bbox spans both.
    assert cleaned[10:30, 10:30].all()
    assert cleaned[10:26, 44:54].all()
    assert mask_to_box(cleaned) == (10, 10, 54, 30)


def test_leak_cleanup_keeps_small_adjacent_blob():
    """A small blob touching/near the body (a robot arm) is KEPT — small but
    NOT detached (bbox gap <= gap_px)."""
    from web.field.sam2_tracker import clean_disconnected_leaks
    mask = np.zeros((40, 60), bool)
    mask[10:30, 10:30] = True   # big blob (400px), bbox (10,10,30,30)
    # Small 4x3 = 12px (< 15% of 400) but only ~2px from the body's right edge.
    mask[14:18, 32:35] = True
    cleaned = clean_disconnected_leaks(mask)
    assert cleaned[10:30, 10:30].all()
    assert cleaned[14:18, 32:35].all()   # arm kept (adjacent)


def test_leak_cleanup_single_blob_unchanged():
    from web.field.sam2_tracker import clean_disconnected_leaks
    mask = np.zeros((20, 20), bool)
    mask[3:9, 4:11] = True
    cleaned = clean_disconnected_leaks(mask)
    assert np.array_equal(cleaned, mask)


def test_leak_cleanup_empty_unchanged():
    from web.field.sam2_tracker import clean_disconnected_leaks
    mask = np.zeros((10, 10), bool)
    cleaned = clean_disconnected_leaks(mask)
    assert not cleaned.any()
    assert cleaned.shape == mask.shape


class _LeakStubPredictor(RepromptStubPredictor):
    """RepromptStubPredictor whose object 1 mask leaks onto a second blob.

    Object 1's per-frame mask is a robot blob at its normal location PLUS a
    small disconnected leak blob far to the right. Every other object keeps a
    single clean blob. Used to prove the gated leak cleanup tightens the
    emitted box (and drops the leak) at both mask sites.
    """

    # Leak blob well clear of every _rect_for(oid) (oid<=9 -> x<=58); place it
    # at the far right edge as a small, disconnected patch. 3x3 = 9px is under
    # 15% of obj1's 80px primary blob (12px) AND the gap to it exceeds 10px, so
    # the gated cleaner drops it.
    _LEAK = (57, 14, 60, 17)   # 3x3 = 9px, small AND far from obj1's blob

    def _rect_for(self, oid):
        # Object 1 gets a LARGER primary blob so it is unambiguously the largest
        # component once the leak is dropped; others keep the base geometry.
        if oid == 1:
            return (6, 2, 14, 12)   # 8x10 = 80px primary blob for obj 1
        x0 = oid * 6
        return (x0, 2, x0 + 4, 6)

    def propagate_in_video(self, state, start_frame_idx=None,
                           max_frame_num_to_track=None, reverse=False):
        start = 0 if start_frame_idx is None else start_frame_idx
        if reverse:
            end = max(0, start - (max_frame_num_to_track or start))
            frame_range = range(start, end - 1, -1)
        else:
            if max_frame_num_to_track is None:
                end = self.NUM_FRAMES - 1
            else:
                end = min(start + max_frame_num_to_track, self.NUM_FRAMES - 1)
            frame_range = range(start, end + 1)
        for f in frame_range:
            active = sorted(
                oid for oid, af in state["objs"].items()
                if af <= f or reverse
            )
            num_obj = len(active)
            arr = np.zeros((num_obj, 1, self.H, self.W), dtype=np.float32)
            for i, oid in enumerate(active):
                x1, y1, x2, y2 = self._rect_for(oid)
                arr[i, 0, y1:y2, x1:x2] = 1.0
                if oid == 1:
                    lx1, ly1, lx2, ly2 = self._LEAK
                    arr[i, 0, ly1:ly2, lx1:lx2] = 1.0   # disconnected leak blob
            yield f, active, FakeTensor(arr)

    def reset_state(self, state):
        pass


def test_mask_leak_cleanup_untangles_stretched_box():
    """With the knob on, object 1's emitted box tightens to its primary blob;
    the union box that spans the small, detached leak is gone."""
    from web.field.sam2_tracker import Sam2FieldTracker

    stub = _LeakStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)

    result = tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[stub._rect_for(1)],   # only obj 1 (the leaky one)
        reprompt_every=2,
        detect_fn=lambda *_a: [stub._rect_for(1)],
        object_metadata={1: "blue"},
        max_objects=6,
        mask_leak_cleanup=True,
    )
    # Union bbox (leak included) would be x2=60 (leak right edge); the primary
    # blob's tight box is (6,2,14,12). Assert the leak is dropped everywhere.
    for f_idx, entry in result.items():
        if 1 in entry:
            box = entry[1]["box"]
            assert box == (6, 2, 14, 12), f"frame {f_idx}: box not tightened: {box}"
            # Stored mask is the cleaned mask (leak dropped) for downstream IoS.
            assert not entry[1]["mask"][14:17, 57:60].any()


def test_mask_leak_cleanup_runs_without_error_on_plain_stub():
    """The knob on a normal single-blob stub runs without error and still
    produces the expected boxes (no-crash regression)."""
    from web.field.sam2_tracker import Sam2FieldTracker

    stub = RepromptStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    result = tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2,
        detect_fn=lambda *_a: [stub._rect_for(1), stub._rect_for(2)],
        object_metadata={1: "blue", 2: "red"},
        max_objects=6,
        mask_leak_cleanup=True,
    )
    assert set(result.keys()) == {0, 1, 2, 3, 4, 5}
    assert result[0][1]["box"] == stub._rect_for(1)
    assert result[5][2]["box"] == stub._rect_for(2)


def test_mask_leak_cleanup_default_off_byte_identical():
    """Default kwargs and explicit mask_leak_cleanup=False produce
    identical results and add_calls (back-compat)."""
    from web.field.sam2_tracker import Sam2FieldTracker

    def run(**extra):
        stub = _LeakStubPredictor()
        tracker = Sam2FieldTracker(predictor=stub)
        result = tracker.track(
            frames_dir="ignored_by_stub",
            prompt_idx=0,
            prompt_boxes=[stub._rect_for(1)],
            reprompt_every=2,
            detect_fn=lambda *_a: [stub._rect_for(1)],
            object_metadata={1: "blue"},
            max_objects=6,
            **extra,
        )
        boxes = {f: {o: e["box"] for o, e in entry.items()}
                 for f, entry in result.items()}
        return list(stub.add_calls), boxes

    default_adds, default_boxes = run()
    off_adds, off_boxes = run(mask_leak_cleanup=False)

    assert off_adds == default_adds
    assert off_boxes == default_boxes
    # And with the knob off, the leak IS included -> union box spans to x2=60.
    assert any(box[2] == 60 for entry in default_boxes.values()
               for box in entry.values()), \
        "expected the union box (leak included) when the knob is off"


def test_size_prior_ratio_basic():
    assert size_prior_ratio((10, 10, 110, 60), 50.0) == 2.0


def test_size_prior_ratio_unmeasurable():
    assert size_prior_ratio((10, 10, 110, 60), None) is None
    assert size_prior_ratio((10, 10, 110, 60), 0.0) is None
    assert size_prior_ratio((110, 10, 10, 60), 50.0) is None  # degenerate box


def test_size_prior_violation_band():
    assert size_prior_violation(3.5, hi=3.0, lo=0.25)
    assert size_prior_violation(0.1, hi=3.0, lo=0.25)
    assert not size_prior_violation(1.0, hi=3.0, lo=0.25)
    assert not size_prior_violation(None, hi=3.0, lo=0.25)


# ---------------------------------------------------------------------------
# Size-prior hijack cull: a tracked mask far wider than a robot could appear
# at its projected field position for sustained boundaries has latched onto
# structure/a pile. Geometry is the one channel a detector false-fire cannot
# corroborate, so this cull fires INDEPENDENT of the absence machinery
# (cull_enabled). CullStubPredictor masks are 4 px wide.
# ---------------------------------------------------------------------------

def _track_with_expected_px(stub, expected_px_fn, **overrides):
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()
    kwargs = dict(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2,
        # Corroborate both objects at every boundary: the size cull must be
        # able to evict a DETECTOR-CORROBORATED mask (YOLO false-fires on the
        # equipment itself), which the absence cull structurally cannot.
        detect_fn=lambda *_a: [stub._rect_for(1), stub._rect_for(2)],
        object_metadata={1: "blue", 2: "red"}, max_active_objects=6,
        cull_enabled=False,
        project_fn=lambda box: bottom_center(box),
        expected_px_fn=expected_px_fn,
        diagnostics=diag,
    )
    kwargs.update(overrides)
    result = tracker.track(**kwargs)
    return result, diag


def test_size_prior_cull_evicts_oversized_mask():
    """A 4-px mask whose expected width is 1 px (ratio 4.0 > hi 3.0) for 3
    consecutive boundaries is culled when size_prior_cull=True, even though the
    detector corroborates it at every boundary."""
    stub = CullStubPredictor()
    result, diag = _track_with_expected_px(
        stub, expected_px_fn=lambda xy: 1.0,
        size_prior_cull=True, size_prior_boundaries=3)
    culled = [e for e in diag._events if e.get("event") == "sam2.object_culled"]
    assert culled, f"expected a size-prior cull, got none: {diag._events}"
    assert all(e.get("reason") == "size_prior_violation" for e in culled)
    # Both objects are oversized; both should be gone by the final frame.
    assert 1 not in result[stub.NUM_FRAMES - 1]
    assert 2 not in result[stub.NUM_FRAMES - 1]


def test_size_prior_cull_disabled_by_default_keeps_object():
    """Same oversized scenario with the default size_prior_cull=False leaves
    the object tracked."""
    stub = CullStubPredictor()
    result, diag = _track_with_expected_px(
        stub, expected_px_fn=lambda xy: 1.0, size_prior_boundaries=3)
    assert not [e for e in diag._events
                if e.get("event") == "sam2.object_culled"]
    assert 1 in result[stub.NUM_FRAMES - 1]
    assert 2 in result[stub.NUM_FRAMES - 1]


def test_size_prior_cull_never_culls_unmeasurable():
    """expected_px_fn returning None (no calibration pose at that position) is
    unmeasurable, never a violation — the object survives with the cull on."""
    stub = CullStubPredictor()
    result, diag = _track_with_expected_px(
        stub, expected_px_fn=lambda xy: None,
        size_prior_cull=True, size_prior_boundaries=3)
    assert not [e for e in diag._events
                if e.get("event") == "sam2.object_culled"]
    assert 1 in result[stub.NUM_FRAMES - 1]
    assert 2 in result[stub.NUM_FRAMES - 1]


def test_size_prior_cull_requires_sustained_violation():
    """A single-boundary violation (expected width dips only at the first
    boundary) does not reach the size_prior_boundaries threshold, so the object
    is spared."""
    stub = CullStubPredictor()
    calls = {"n": 0}

    def expected_px_fn(xy):
        calls["n"] += 1
        # ratio 4.0 (violation) for the first boundary's two objects, then
        # ratio 1.0 (in-band) — one violating boundary can't reach 3-in-a-row.
        return 1.0 if calls["n"] <= 2 else 4.0

    result, diag = _track_with_expected_px(
        stub, expected_px_fn=expected_px_fn,
        size_prior_cull=True, size_prior_boundaries=3)
    assert not [e for e in diag._events
                if e.get("event") == "sam2.object_culled"]
    assert 1 in result[stub.NUM_FRAMES - 1]
    assert 2 in result[stub.NUM_FRAMES - 1]


# ---------------------------------------------------------------------------
# should_stop — cooperative abort of propagation (Task 1: reliable STOP)
# ---------------------------------------------------------------------------

def test_track_should_stop_aborts_forward_propagation():
    """When ``should_stop`` returns True the forward pass raises TrackStopped
    at the next per-frame poll instead of running to the end of the video."""
    from web.field.sam2_tracker import Sam2FieldTracker, TrackStopped

    stub = StubPredictor()  # 3 frames, 2 objects, forward-only (no reprompt)
    tracker = Sam2FieldTracker(predictor=stub)

    seen = {"n": 0}

    def should_stop():
        # Stop after the first propagated frame has been recorded.
        seen["n"] += 1
        return seen["n"] >= 1

    with pytest.raises(TrackStopped):
        tracker.track(
            frames_dir="ignored_by_stub",
            prompt_idx=0,
            prompt_boxes=[(2, 2, 6, 6), (15, 3, 20, 8)],
            should_stop=should_stop,
        )
    # It bailed early rather than polling once per frame for all 3 frames.
    assert seen["n"] == 1


def test_track_should_stop_none_runs_to_completion():
    """Default (should_stop=None) is a no-op — full propagation, no raise."""
    from web.field.sam2_tracker import Sam2FieldTracker

    stub = StubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    result = tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[(2, 2, 6, 6), (15, 3, 20, 8)],
        should_stop=None,
    )
    assert set(result.keys()) == {0, 1, 2}  # all frames present


def test_track_should_stop_resets_predictor_state_on_abort():
    """The ``finally`` cleanup (reset_state) still runs when should_stop fires."""
    from web.field.sam2_tracker import Sam2FieldTracker, TrackStopped

    stub = StubPredictor()
    reset = {"called": False}
    stub.reset_state = lambda state: reset.__setitem__("called", True)
    tracker = Sam2FieldTracker(predictor=stub)

    with pytest.raises(TrackStopped):
        tracker.track(
            frames_dir="ignored_by_stub",
            prompt_idx=0,
            prompt_boxes=[(2, 2, 6, 6), (15, 3, 20, 8)],
            should_stop=lambda: True,
        )
    assert reset["called"] is True


def test_normalize_detections_accepts_corrob_only_triples():
    from web.field.sam2_tracker import normalize_detections
    items = [((1, 2, 3, 4), "red"), ((5, 6, 7, 8), "blue", True), (9, 10, 11, 12)]
    boxes, meta, corrob_only = normalize_detections(items)
    assert boxes == [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)]
    assert meta == ["red", "blue", None]
    assert corrob_only == [False, True, False]


def test_corrob_only_detection_never_seeds_but_corroborates():
    """A corrob-only box at a new location must NOT create an object, but a
    corrob-only box over a tracked mask must count as corroboration (no
    absence accrues)."""
    from web.field.sam2_tracker import Sam2FieldTracker

    stub = RepromptStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    new_box = stub._rect_for(3)       # unmatched location
    overlap_box = stub._rect_for(1)   # over tracked obj 1

    def detect_fn(local_idx, frame_path):
        # obj 1 corroborated ONLY by a weak (corrob-only) detection; the new
        # location is also weak so it must not seed.
        return [(overlap_box, "blue", True), (new_box, "red", True)]

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn, max_objects=6,
    )
    # No third object was ever seeded.
    assert all(set(frame.keys()) <= {1, 2} for frame in result.values())
    assert all(oid in (1, 2) for _f, oid in stub.add_calls)


# --------------------------------------------------------------------------- #
# tracker_prob bookkeeping: low-prob cull gate + eviction immunity
# --------------------------------------------------------------------------- #
def test_mean_or_none():
    from web.field.sam2_tracker import mean_or_none
    assert mean_or_none([]) is None
    assert mean_or_none([0.5, None, 0.7]) == 0.6


def test_pick_evictable_prob_immunity_spares_high_prob_object():
    from web.field.sam2_tracker import pick_evictable
    from collections import deque
    # obj 1: 2 absences (tier-3 evictable). With high mean prob + immunity
    # threshold it must be spared; without immunity it is the victim.
    hist = {1: deque([False, False], maxlen=5)}
    assert pick_evictable([1], hist, {}, k=3, stationary_m=0.5,
                          min_absences=2) == 1
    assert pick_evictable([1], hist, {}, k=3, stationary_m=0.5,
                          min_absences=2,
                          prob_means={1: 0.9}, prob_immune=0.8) is None
    # Low prob stays evictable even with immunity enabled.
    assert pick_evictable([1], hist, {}, k=3, stationary_m=0.5,
                          min_absences=2,
                          prob_means={1: 0.3}, prob_immune=0.8) == 1


def test_pick_evictable_prob_immunity_spares_tier1():
    from web.field.sam2_tracker import pick_evictable
    from collections import deque
    # obj 1: stationary + 3 windowed absences with k=3 -> classic tier-1
    # victim (min_absences left at its permissive default so tier 3 would
    # also claim it; tier 1 fires first regardless). High mean prob must
    # spare it; low prob leaves it evictable.
    hist = {1: deque([False, False, False], maxlen=5)}
    window = {1: [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]}
    assert pick_evictable([1], hist, window, k=3, stationary_m=0.5) == 1
    assert pick_evictable([1], hist, window, k=3, stationary_m=0.5,
                          prob_means={1: 0.9}, prob_immune=0.8) is None
    assert pick_evictable([1], hist, window, k=3, stationary_m=0.5,
                          prob_means={1: 0.3}, prob_immune=0.8) == 1


def test_pick_evictable_prob_immunity_spares_tier25():
    from web.field.sam2_tracker import pick_evictable
    from collections import deque
    # obj 1: stationary, hist[-2] is False (unproven at the prior boundary)
    # and hist[-1] is False (absent now), but only 3 absences against an
    # unreachable k=99 and min_absences=99 -- tiers 1 and 3 are out of
    # reach, and dark_frames is omitted so tier 2 can't fire either. Only
    # the motion-proven tier 2.5 path can evict it.
    hist = {1: deque([False, False, False], maxlen=5)}
    window = {1: [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]}
    kwargs = dict(k=99, stationary_m=0.5, min_absences=99)
    assert pick_evictable([1], hist, window, moving_candidate=True,
                          **kwargs) == 1
    assert pick_evictable([1], hist, window, moving_candidate=True,
                          prob_means={1: 0.9}, prob_immune=0.8, **kwargs) is None
    assert pick_evictable([1], hist, window, moving_candidate=True,
                          prob_means={1: 0.3}, prob_immune=0.8, **kwargs) == 1


def test_negative_points_for_picks_nearby_neighbor_centers():
    from web.field.sam2_tracker import negative_points_for
    target = (10, 10, 20, 20)
    others = [
        (21, 10, 27, 20),    # adjacent -> center (24, 15) inside 1.5x inflation
        (100, 100, 110, 110) # far -> excluded
    ]
    pts = negative_points_for(target, others)
    assert pts == [(24.0, 15.0)]


def test_negative_points_for_caps_and_orders_by_distance():
    from web.field.sam2_tracker import negative_points_for
    target = (0, 0, 10, 10)
    others = [(10 + i, 0, 12 + i, 4) for i in range(6)]  # 6 near neighbors
    pts = negative_points_for(target, others, max_points=4)
    assert len(pts) == 4
    # nearest first
    d = [((x - 5) ** 2 + (y - 5) ** 2) for x, y in pts]
    assert d == sorted(d)


def test_prompt_with_negatives_falls_back_when_predictor_lacks_points():
    """Stub predictors (and legacy paths) accept only box=; the helper must
    degrade to a box-only prompt instead of crashing."""
    from web.field.sam2_tracker import _prompt_with_negatives

    class BoxOnly:
        def __init__(self): self.calls = []
        def add_new_points_or_box(self, state, frame_idx, obj_id, box=None):
            self.calls.append((frame_idx, obj_id, tuple(box)))

    p = BoxOnly()
    _prompt_with_negatives(p, {}, 3, 7, (1, 2, 3, 4), [(9.0, 9.0)])
    assert p.calls == [(3, 7, (1.0, 2.0, 3.0, 4.0))]


def test_prompt_with_negatives_propagates_real_error_from_supporting_predictor():
    """A predictor whose add_new_points_or_box accepts points/labels but
    raises for an internal reason must have that error propagate -- it must
    NOT be swallowed and silently retried as a box-only call."""
    from web.field.sam2_tracker import _prompt_with_negatives

    class BuggyPointsPredictor:
        def __init__(self): self.calls = []
        def add_new_points_or_box(self, state, *, frame_idx, obj_id, box=None,
                                   points=None, labels=None):
            raise TypeError("internal bug")

    p = BuggyPointsPredictor()
    try:
        _prompt_with_negatives(p, {}, 3, 7, (1, 2, 3, 4), [(9.0, 9.0)])
    except TypeError as exc:
        assert "internal bug" in str(exc)
    else:
        raise AssertionError("expected TypeError to propagate")

    class SupportingPredictor:
        def __init__(self): self.calls = []
        def add_new_points_or_box(self, state, *, frame_idx, obj_id, box=None,
                                   points=None, labels=None):
            self.calls.append((frame_idx, obj_id, tuple(box), tuple(map(tuple, points)),
                                tuple(labels)))

    p2 = SupportingPredictor()
    _prompt_with_negatives(p2, {}, 3, 7, (1, 2, 3, 4), [(9.0, 9.0)])
    assert p2.calls == [(3, 7, (1.0, 2.0, 3.0, 4.0), ((9.0, 9.0),), (0,))]


# ---------------------------------------------------------------------------
# Task 9 — misaligned-mask re-anchor
# ---------------------------------------------------------------------------

def _hw_rect_mask(H, W, r):
    m = np.zeros((H, W), bool)
    m[r[1]:r[3], r[0]:r[2]] = True
    return m


def test_best_partial_claimant_finds_drifted_healthy_mask():
    from web.field.sam2_tracker import best_partial_claimant
    H, W = 20, 60
    # healthy mask at x 0-10; detection box at x 6-16 -> partial overlap
    mask = _hw_rect_mask(H, W, (0, 2, 10, 12))
    box_mask = _hw_rect_mask(H, W, (6, 2, 16, 12))
    oid, ios = best_partial_claimant(
        {1: mask}, {1: "red"}, {1: int(mask.sum())},
        box_mask, int(box_mask.sum()), "red",
        ios_lo=0.2, ios_hi=0.5, min_area_frac=0.1)
    assert oid == 1 and 0.2 <= ios < 0.5


def test_best_partial_claimant_rejects_color_conflict_and_slivers():
    from web.field.sam2_tracker import best_partial_claimant
    H, W = 20, 60
    mask = _hw_rect_mask(H, W, (0, 2, 10, 12))
    box_mask = _hw_rect_mask(H, W, (6, 2, 16, 12))
    area = int(box_mask.sum())
    # color conflict
    oid, _ = best_partial_claimant({1: mask}, {1: "blue"}, {1: int(mask.sum())},
                                   box_mask, area, "red", 0.2, 0.5, 0.1)
    assert oid is None
    # sliver (too small to be a "healthy drifted" mask)
    sliver = _hw_rect_mask(H, W, (6, 2, 8, 4))
    oid, _ = best_partial_claimant({1: sliver}, {1: "red"}, {1: int(sliver.sum())},
                                   box_mask, area, "red", 0.2, 0.5, 0.1)
    assert oid is None


def test_misalign_drift_box_ios_lands_in_band():
    """Sanity-check the RepromptStubPredictor geometry used below: the drift
    box's IoS over obj 1's static rect (6,2,10,6) must land in [0.2, 0.5) —
    inside the misalign band and below new_obj_ios_thresh (0.5)."""
    from web.field.sam2_tracker import _ios
    H, W = 20, 60
    obj1 = _hw_rect_mask(H, W, (6, 2, 10, 6))
    drift_box = _hw_rect_mask(H, W, (9, 2, 15, 6))
    ios = _ios(obj1, drift_box)
    assert 0.2 <= ios < 0.5, f"drift box IoS {ios} not in expected band"


def test_misalign_reanchor_reprompts_same_id_after_streak():
    """A detection persistently half-overlapping a healthy tracked mask must
    re-prompt that SAME object id (no new id churn) once the streak is met."""
    from web.field.sam2_tracker import Sam2FieldTracker

    stub = RepromptStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    # obj 1's stub mask sits at x in [6,10); a box at x 9-15 overlaps it
    # at IoS 0.25 -- inside [0.2, 0.5) per the sanity check above.
    drift_box = (9, 2, 15, 6)

    def detect_fn(local_idx, frame_path):
        return [(drift_box, "blue")]

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1)], reprompt_every=2,
        detect_fn=detect_fn, object_metadata={1: "blue"}, max_objects=6,
        misalign_reanchor=True, misalign_boundaries=2,
    )
    # obj 1 re-prompted at a later frame; NO new object id was created.
    later_prompts = [(f, o) for f, o in stub.add_calls if f > 0]
    assert later_prompts and all(o == 1 for _f, o in later_prompts)


def test_misalign_reanchor_disabled_by_default_is_inert():
    """misalign_reanchor defaults OFF: the same drifting-detection scenario
    produces byte-identical add_calls whether or not the kwarg is mentioned."""
    from web.field.sam2_tracker import Sam2FieldTracker

    drift_box = (9, 2, 15, 6)

    def run(**extra):
        stub = RepromptStubPredictor()
        tracker = Sam2FieldTracker(predictor=stub)

        def detect_fn(local_idx, frame_path):
            return [(drift_box, "blue")]

        tracker.track(
            frames_dir="ignored_by_stub", prompt_idx=0,
            prompt_boxes=[stub._rect_for(1)], reprompt_every=2,
            detect_fn=detect_fn, object_metadata={1: "blue"}, max_objects=6,
            **extra,
        )
        return list(stub.add_calls)

    baseline = run()
    explicit_off = run(misalign_reanchor=False)
    assert baseline == explicit_off
