"""End-to-end tests for ``Sam2FieldTracker.track_selective`` (Task 4).

The orchestrator glues the pure Task 2/3 logic (``TbdTracker`` -> ``find_windows``
-> ``ExitMachine`` -> ``merge_selective_results``) to the real SAM predictor
protocol. These tests drive it with a scripted ``detect_frame_fn`` (which builds
the TBD history) and a stub predictor (which scripts the propagated masks), so no
GPU / torch / real model is ever touched.

Scenarios: (a) clean settle, (b) swap, (c) predictor raises mid-window, plus
early-break, progress/stats, and cooperative-stop assertions.
"""
import os

import cv2
import numpy as np

from web.field.sam2_tracker import Sam2FieldTracker
from web.field.selective_prop import SelectiveConfig

# Two 20x20 boxes far enough apart that neither their raw boxes (no witness
# expansion) nor their 25%-inflated boxes (no finite margin competitor) overlap
# -> both stay unambiguous (margin inf) whenever both are present.
L = (10, 40, 30, 60)
R = (70, 40, 90, 60)


def _cfg():
    # Short streaks so the scenarios stay a handful of frames long.
    return SelectiveConfig(min_streak=2, gap_trigger=3, margin_entry=0.05,
                           margin_exit=0.10, seed_clean_iou=0.10, ioma_exit=0.80,
                           area_degradation=0.25, border_margin_px=5,
                           window_max_frames=120, max_missed=30)


def _write_frames(dir_path, n, width=100, height=100):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(n):
        cv2.imwrite(os.path.join(str(dir_path), f"{i:05d}.jpg"), img)


class StubSelectivePredictor:
    """Scripts propagated masks per (frame, oid) and mirrors the predictor
    protocol ``track()`` uses: ``init_state`` -> ``add_new_points_or_box`` ->
    ``propagate_in_video`` (a generator) -> ``reset_state``.

    ``mask_fn(frame_idx, oid) -> box | None`` decides each object's mask that
    frame (None -> an all-False / dark mask). ``emit_probs`` toggles 3-tuple vs
    4-tuple yields so both collection paths are exercised. ``raise_after`` makes
    the propagate generator blow up after that many yields (mid-window fault).
    """

    def __init__(self, mask_fn, hw=(100, 100), emit_probs=False, prob=0.8,
                 raise_after=None):
        self.mask_fn = mask_fn
        self.hw = hw
        self.emit_probs = emit_probs
        self.prob = prob
        self.raise_after = raise_after
        self.init_calls = 0
        self.reset_calls = 0
        self.prompts = []          # (frame_idx, oid, box)
        self.solo_at_prompt = []   # solo_ids snapshot at each prompt
        self.solo_at_reset = "unset"
        self.yielded = []          # frame indices actually produced (cost metric)
        self._prompted = set()

    def init_state(self, video_path, offload_video_to_cpu=None,
                   offload_state_to_cpu=None):
        self.init_calls += 1
        self._prompted = set()     # fresh per seed-frame group
        return {"solo_ids": None, "video_path": video_path}

    def add_new_points_or_box(self, state, frame_idx, obj_id, box):
        oid = int(obj_id)
        self._prompted.add(oid)
        self.prompts.append((frame_idx, oid, tuple(int(v) for v in box)))
        self.solo_at_prompt.append(set(state.get("solo_ids") or []))

    def reset_state(self, state):
        self.reset_calls += 1
        self.solo_at_reset = state.get("solo_ids")

    def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track):
        prompted = sorted(self._prompted)
        h, w = self.hw
        produced = 0
        f = start_frame_idx
        end = start_frame_idx + max_frame_num_to_track
        while f <= end:
            if self.raise_after is not None and produced >= self.raise_after:
                raise RuntimeError("predictor blew up mid-window")
            oids, masks, probs = [], [], []
            for oid in prompted:
                box = self.mask_fn(f, oid)
                m = np.zeros((1, h, w), dtype=np.float32)
                if box is not None:
                    x1, y1, x2, y2 = box
                    m[0, y1:y2, x1:x2] = 1.0
                oids.append(oid)
                masks.append(m)
                probs.append(self.prob)
            obj_ids = np.asarray(oids, dtype=np.int64)
            mask_logits = np.stack(masks, axis=0)       # (N, 1, H, W) float
            self.yielded.append(f)
            produced += 1
            if self.emit_probs:
                yield (f, obj_ids, mask_logits,
                       np.asarray(probs, dtype=np.float64))
            else:
                yield (f, obj_ids, mask_logits)
            f += 1


# --------------------------------------------------------------------------
# Detection scripts (fed to detect_frame_fn -> TbdTracker builds the history).
# --------------------------------------------------------------------------
# One robot at L that gaps out (frames 4-6) then returns -> exactly one window.
_CLEAN_DETS = {0: [(L, None)], 1: [(L, None)], 2: [(L, None)], 3: [(L, None)],
               4: [], 5: [], 6: [],
               7: [(L, None)], 8: [(L, None)], 9: [(L, None)], 10: [(L, None)]}

# Two robots: A(->oid1) at L gaps out and returns; B(->oid2) sits at R the whole
# time (isolated). A's gap opens one window on oid1; B never overlaps A.
_SWAP_DETS = {0: [(L, None), (R, None)], 1: [(L, None), (R, None)],
              2: [(L, None), (R, None)], 3: [(L, None), (R, None)],
              4: [(R, None)], 5: [(R, None)], 6: [(R, None)],
              7: [(L, None), (R, None)], 8: [(L, None), (R, None)],
              9: [(L, None), (R, None)], 10: [(L, None), (R, None)]}


def _detect_fn(script):
    return lambda local_idx, path: list(script.get(local_idx, []))


# --------------------------------------------------------------------------
# (a) clean settle
# --------------------------------------------------------------------------
def test_track_selective_clean_settle(tmp_path):
    _write_frames(tmp_path, 11)
    pred = StubSelectivePredictor(lambda f, oid: L, emit_probs=True, prob=0.8)
    tracker = Sam2FieldTracker(predictor=pred)

    progress = []
    results = tracker.track_selective(
        str(tmp_path), _detect_fn(_CLEAN_DETS), _cfg(),
        progress_cb=lambda done, total: progress.append((done, total)),
    )

    stats = tracker.selective_stats
    assert stats["windows"] == 1
    assert stats["outcomes"] == {"clean": 1}
    assert stats["renames"] == 0
    assert stats["frames"] == 11

    # Progress reaches (total, total) at the end of the enumeration phase.
    assert progress[-1] == (11, 11)

    # Outside the window [entry=3, close=8] the stream is pure TBD box L.
    for f in (0, 1, 2, 9, 10):
        assert results[f][1]["box"] == L
        assert results[f][1]["mask"] is None
        assert results[f][1]["evidence"]["type"] == "tbd"

    # Inside the window the SAM mask is present, incl. the TBD gap frames (4-6)
    # that had no detection at all -- exactly what selective propagation buys.
    for f in (3, 4, 5, 6, 7, 8):
        assert results[f][1]["evidence"]["type"] == "selective_sam"
        assert results[f][1]["mask"] is not None
        assert results[f][1]["evidence"]["tracker_prob"] == 0.8

    # Early break: settle closes at frame 8, well before the full span (the
    # generator could have run frames 1..10 = 10 yields).
    assert pred.yielded == [1, 2, 3, 4, 5, 6, 7, 8]
    assert stats["sam_frames"] == 8
    assert len(pred.yielded) < 10

    # solo_ids fenced the group, then reset in the finally.
    assert pred.solo_at_prompt == [{1}]
    assert pred.solo_at_reset is None
    assert pred.reset_calls == 1

    # obj-color side channel is populated (color was None here).
    assert tracker.selective_obj_colors == {1: None}


# --------------------------------------------------------------------------
# (b) swap
# --------------------------------------------------------------------------
def test_track_selective_swap_renames_from_entry(tmp_path):
    _write_frames(tmp_path, 11)
    # SAM seeded on oid1 stays on L through the entry frame, then follows the
    # physical robot to R -- where TBD (wrongly) calls it oid2 -> swap.
    pred = StubSelectivePredictor(lambda f, oid: L if f <= 3 else R,
                                  emit_probs=False)
    tracker = Sam2FieldTracker(predictor=pred)

    results = tracker.track_selective(
        str(tmp_path), _detect_fn(_SWAP_DETS), _cfg())

    stats = tracker.selective_stats
    assert stats["windows"] == 1
    assert stats["outcomes"] == {"swap": 1}
    assert stats["renames"] == 1

    # The 1<->2 relabel takes effect from the entry frame (3) onward: after the
    # window closes, label 1 carries R's box and label 2 carries L's box.
    for f in (9, 10):
        assert results[f][1]["box"] == R
        assert results[f][1]["evidence"]["type"] == "tbd"
        assert results[f][2]["box"] == L
    # Before the entry the original labels stand.
    assert results[0][1]["box"] == L
    assert results[0][2]["box"] == R

    # 3-tuple (no-probs) path still yields masks inside the window.
    assert results[4][1]["evidence"]["type"] == "selective_sam"
    assert results[4][1]["mask"] is not None

    assert pred.solo_at_prompt == [{1}]
    assert pred.solo_at_reset is None
    assert pred.reset_calls == 1


# --------------------------------------------------------------------------
# (c) predictor raises mid-window -> propagates (fallback lives in app.py)
# --------------------------------------------------------------------------
def test_track_selective_predictor_raises_propagates(tmp_path):
    _write_frames(tmp_path, 11)
    pred = StubSelectivePredictor(lambda f, oid: L, raise_after=2)
    tracker = Sam2FieldTracker(predictor=pred)

    import pytest
    with pytest.raises(RuntimeError):
        tracker.track_selective(str(tmp_path), _detect_fn(_CLEAN_DETS), _cfg())

    # State was still torn down despite the fault (solo_ids reset + reset_state).
    assert pred.reset_calls == 1
    assert pred.solo_at_reset is None


# --------------------------------------------------------------------------
# cooperative stop -> pure-TBD partial, no SAM touched
# --------------------------------------------------------------------------
def test_track_selective_should_stop_returns_tbd_partial(tmp_path):
    _write_frames(tmp_path, 11)
    pred = StubSelectivePredictor(lambda f, oid: L, emit_probs=True)
    tracker = Sam2FieldTracker(predictor=pred)

    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] >= 3        # stop after frame index 2 (3 frames)

    results = tracker.track_selective(
        str(tmp_path), _detect_fn(_CLEAN_DETS), _cfg(),
        should_stop=should_stop)

    stats = tracker.selective_stats
    assert stats["windows"] == 0
    assert stats["sam_frames"] == 0
    assert stats["outcomes"] == {}
    assert stats["frames"] == 3       # frames 0,1,2 stepped before the stop

    # Only pure-TBD entries; the predictor was never initialised.
    assert set(results) == {0, 1, 2}
    for f in (0, 1, 2):
        assert results[f][1]["evidence"]["type"] == "tbd"
        assert results[f][1]["mask"] is None
    assert pred.init_calls == 0
    assert pred.reset_calls == 0


# --------------------------------------------------------------------------
# no windows -> pure-TBD stream, predictor untouched
# --------------------------------------------------------------------------
def test_track_selective_no_windows_pure_tbd(tmp_path):
    _write_frames(tmp_path, 4)
    # A single robot sitting still: unambiguous every frame, never gaps -> no
    # window is ever opened.
    dets = {f: [(L, None)] for f in range(4)}
    pred = StubSelectivePredictor(lambda f, oid: L)
    tracker = Sam2FieldTracker(predictor=pred)

    results = tracker.track_selective(str(tmp_path), _detect_fn(dets), _cfg())

    stats = tracker.selective_stats
    assert stats["windows"] == 0
    assert stats["sam_frames"] == 0
    assert stats["outcomes"] == {}
    assert pred.init_calls == 0
    for f in range(4):
        assert results[f][1]["evidence"]["type"] == "tbd"
