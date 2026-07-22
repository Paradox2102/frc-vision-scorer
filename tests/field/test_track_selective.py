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


# ==========================================================================
# Task B: episode execution (interval-merged windows) + per-view SAM budget.
# ==========================================================================
def _cfg_budget(frac):
    """``_cfg()`` with a SAM-frame budget fraction (Design 4)."""
    c = _cfg()
    c.sam_budget_frac = frac
    return c


# Two robots that each gap out at slightly offset times: their padded
# [seed-min_streak, entry+min_streak] intervals overlap so the two windows land
# on DIFFERENT seed frames yet cluster into one episode. L->oid1 (present 0-4,
# gap 5-7, back 8-11) yields (seed=2, entry=4); R->oid2 (present 0-3, gap 4-6,
# back 7-11) yields (seed=1, entry=3). Disjoint boxes -> no witness, no margin.
_L_PRESENT = set(range(0, 5)) | set(range(8, 12))
_R_PRESENT = set(range(0, 4)) | set(range(7, 12))
_TWO_WIN_DETS = {
    f: ([(L, None)] if f in _L_PRESENT else [])
    + ([(R, None)] if f in _R_PRESENT else [])
    for f in range(12)
}

# One robot with TWO gaps far enough apart to stay TWO episodes: present 0-3,
# gap 4-6, back 7-12, gap 13-15, back 16-18 -> windows (seed1, entry3) and
# (seed10, entry12), padded intervals [-1,5] and [8,14] (disjoint).
_TWO_GAP_PRESENT = set(range(0, 4)) | set(range(7, 13)) | set(range(16, 19))
_TWO_GAP_DETS = {
    f: ([(L, None)] if f in _TWO_GAP_PRESENT else []) for f in range(19)
}

# One robot with a single gap and a long tail: present 0-3, gap 4-6, back 7-15
# -> one window (seed1, entry3) that would naturally settle CLEAN at frame 8.
_ONE_GAP_PRESENT = set(range(0, 4)) | set(range(7, 16))
_ONE_GAP_DETS = {
    f: ([(L, None)] if f in _ONE_GAP_PRESENT else []) for f in range(16)
}


# --------------------------------------------------------------------------
# (a) two overlapping windows -> ONE init_state, correct merged results.
# --------------------------------------------------------------------------
def test_track_selective_two_windows_one_episode(tmp_path):
    _write_frames(tmp_path, 12)
    # Each object's mask sits on its own home box the whole pass -> both settle
    # clean on their own oid (no swap).
    pred = StubSelectivePredictor(lambda f, oid: L if oid == 1 else R,
                                  emit_probs=True, prob=0.8)
    tracker = Sam2FieldTracker(predictor=pred)

    results = tracker.track_selective(
        str(tmp_path), _detect_fn(_TWO_WIN_DETS), _cfg())

    stats = tracker.selective_stats
    # The whole point: two windows on two different seed frames run under ONE
    # SAM session, not two.
    assert pred.init_calls == 1
    assert pred.reset_calls == 1
    assert stats["windows"] == 2
    assert stats["episodes"] == 1
    assert stats["dropped_windows"] == 0
    assert stats["outcomes"] == {"clean": 2}
    assert stats["renames"] == 0

    # solo roster spans the episode's oids at every prompt (one per oid).
    assert pred.solo_at_prompt == [{1, 2}, {1, 2}]
    assert pred.solo_at_reset is None

    # Outside every window the stream is pure TBD.
    assert results[0][1]["box"] == L
    assert results[0][1]["evidence"]["type"] == "tbd"
    assert results[0][2]["box"] == R
    assert results[0][2]["evidence"]["type"] == "tbd"

    # The episode fills each oid's TBD gap with SAM masks -- including frame 6,
    # where BOTH robots had gapped out and TBD saw nothing at all.
    assert results[6][1]["evidence"]["type"] == "selective_sam"
    assert results[6][1]["mask"] is not None
    assert results[6][2]["evidence"]["type"] == "selective_sam"
    assert results[6][2]["mask"] is not None


# --------------------------------------------------------------------------
# (b) budget pre-check drops a far-apart second episode.
# --------------------------------------------------------------------------
def test_track_selective_budget_precheck_drops_episode(tmp_path):
    _write_frames(tmp_path, 19)
    pred = StubSelectivePredictor(lambda f, oid: L)
    tracker = Sam2FieldTracker(predictor=pred)

    # budget = round(0.55 * 19) = 10. Episode 1 (windows (1,3)) settles clean at
    # frame 8 -> sam_used=8. Episode 2's conservative min cost (12-10)+2=4 makes
    # 8+4=12 > 10, so the pre-check drops it whole.
    results = tracker.track_selective(
        str(tmp_path), _detect_fn(_TWO_GAP_DETS), _cfg_budget(0.55))

    stats = tracker.selective_stats
    assert stats["windows"] == 2
    assert stats["episodes"] == 1
    assert stats["dropped_windows"] == 1
    assert stats["outcomes"] == {"clean": 1}
    assert pred.init_calls == 1        # only the first episode ever inited

    # Episode 1 ran: its close frame carries a SAM mask.
    assert results[8][1]["evidence"]["type"] == "selective_sam"
    # The dropped episode's span is left as pure TBD (no masks overlaid).
    assert results[12][1]["evidence"]["type"] == "tbd"
    assert results[17][1]["evidence"]["type"] == "tbd"


# --------------------------------------------------------------------------
# (c) mid-episode budget stop -> window closes END, masks so far retained.
# --------------------------------------------------------------------------
def test_track_selective_budget_stops_mid_episode(tmp_path):
    _write_frames(tmp_path, 16)
    pred = StubSelectivePredictor(lambda f, oid: L)
    tracker = Sam2FieldTracker(predictor=pred)

    # budget = round(0.3125 * 16) = 5. The lone window (seed1, entry3) would
    # settle clean at frame 8, but the budget is hit after feeding frame 5, so
    # the still-open window is force-closed as healthy END at frame 5.
    results = tracker.track_selective(
        str(tmp_path), _detect_fn(_ONE_GAP_DETS), _cfg_budget(0.3125))

    stats = tracker.selective_stats
    assert stats["outcomes"] == {"end": 1}
    assert stats["episodes"] == 1
    assert stats["dropped_windows"] == 0
    assert stats["sam_frames"] == 5

    # Generator stopped early: only frames 1..5 were ever produced (a natural
    # clean settle would have needed 8).
    assert pred.yielded == [1, 2, 3, 4, 5]

    # Masks up to the stop are retained (END is healthy).
    assert results[3][1]["evidence"]["type"] == "selective_sam"
    assert results[5][1]["evidence"]["type"] == "selective_sam"
    assert results[5][1]["mask"] is not None
    # Past the stop the stream falls back to pure TBD (SAM never reached it).
    assert results[8][1]["evidence"]["type"] == "tbd"


# --------------------------------------------------------------------------
# (d) episodes stat counts only episodes actually run.
# --------------------------------------------------------------------------
def test_track_selective_episodes_stat_counts_runs(tmp_path):
    _write_frames(tmp_path, 19)
    pred = StubSelectivePredictor(lambda f, oid: L)
    tracker = Sam2FieldTracker(predictor=pred)

    # Default budget (frac 1.0 -> 19 frames) fits both episodes: both settle
    # clean, so episodes counts 2 and nothing is dropped.
    tracker.track_selective(str(tmp_path), _detect_fn(_TWO_GAP_DETS), _cfg())

    stats = tracker.selective_stats
    assert stats["windows"] == 2
    assert stats["episodes"] == 2
    assert stats["dropped_windows"] == 0
    assert stats["outcomes"] == {"clean": 2}
    assert pred.init_calls == 2
    assert pred.reset_calls == 2


# --------------------------------------------------------------------------
# Phase A: per-frame appearance embedding integration (embed_frame_fn).
# --------------------------------------------------------------------------
def test_track_selective_phase_a_embed_stats(tmp_path):
    """embed_frame_fn is consulted once per frame (when dets are non-empty and
    app_weight > 0) and selective_stats["app"] tallies frames touched and how
    many of the returned per-det embeddings were non-None."""
    _write_frames(tmp_path, 4)
    # Two isolated, always-present robots -> unambiguous every frame (margin
    # inf, per the module comment on L/R), no gap ever opens -> no window,
    # predictor untouched. Keeps the scenario to the Phase A loop itself
    # rather than the window machinery.
    pred = StubSelectivePredictor(lambda f, oid: L if oid == 1 else R)
    tracker = Sam2FieldTracker(predictor=pred)
    cfg = _cfg()
    cfg.app_weight = 1.0

    dets = {i: [(L, None), (R, None)] for i in range(4)}
    unit_vec = np.zeros(384, dtype=np.float32)
    unit_vec[0] = 1.0                  # np.float32[384], L2-normalized

    def embed_fn(local_idx, path, boxes):
        assert len(boxes) == 2
        return [unit_vec, None]        # matching-length: mix of vector + None

    results = tracker.track_selective(
        str(tmp_path), _detect_fn(dets), cfg, embed_frame_fn=embed_fn)

    stats = tracker.selective_stats
    assert stats["app"] == {"frames_embedded": 4, "dets_embedded": 4}
    assert stats["windows"] == 0
    assert pred.init_calls == 0

    # dets still flow through to ordinary TBD output.
    for f in range(4):
        assert results[f][1]["box"] == L
        assert results[f][2]["box"] == R


def test_track_selective_phase_a_embed_wrong_length_discarded(tmp_path):
    """A per-frame embedding list whose length doesn't match dets is discarded
    outright: dets stay 2-tuples for that frame (no crash) and neither counter
    in selective_stats["app"] advances."""
    _write_frames(tmp_path, 4)
    pred = StubSelectivePredictor(lambda f, oid: L)
    tracker = Sam2FieldTracker(predictor=pred)
    cfg = _cfg()
    cfg.app_weight = 1.0

    dets = {i: [(L, None)] for i in range(4)}
    calls = []

    def bad_embed_fn(local_idx, path, boxes):
        calls.append(local_idx)
        return []                      # wrong length vs. the single det

    results = tracker.track_selective(
        str(tmp_path), _detect_fn(dets), cfg, embed_frame_fn=bad_embed_fn)

    assert calls == [0, 1, 2, 3]        # embed_frame_fn was in fact consulted
    stats = tracker.selective_stats
    assert stats["app"] == {"frames_embedded": 0, "dets_embedded": 0}

    # Pipeline proceeds normally on the untouched 2-tuple dets -- no crash, no
    # window (single always-present robot).
    assert stats["windows"] == 0
    assert pred.init_calls == 0
    for f in range(4):
        assert results[f][1]["box"] == L
        assert results[f][1]["evidence"]["type"] == "tbd"


# ==========================================================================
# Task 3: starvation-triggered windows + per-trigger stats.
# ==========================================================================
# One robot at L, detected frames 0-4, then gone for the rest of the sequence
# (no reappearance -> no gap trigger, no witness). Its last box L is well off-
# border and the dark tail (frames 5-10) is >= the starve trigger, so with
# starve_windows ON exactly one "starve" window opens on oid 1.
_STARVE_DETS = {0: [(L, None)], 1: [(L, None)], 2: [(L, None)],
                3: [(L, None)], 4: [(L, None)],
                5: [], 6: [], 7: [], 8: [], 9: [], 10: []}


def _cfg_starve():
    return SelectiveConfig(min_streak=2, gap_trigger=3, margin_entry=0.05,
                           margin_exit=0.10, seed_clean_iou=0.10, ioma_exit=0.80,
                           area_degradation=0.25, border_margin_px=5,
                           window_max_frames=120, max_missed=30,
                           starve_windows=True, starve_trigger=3,
                           starve_max_frames=0)


def test_track_selective_passes_frame_hw_and_reports_triggers(tmp_path):
    _write_frames(tmp_path, 11)
    # SAM keeps the robot latched at L even after the detector loses it. The
    # starve scan only fires because track_selective passes frame_hw into
    # find_windows (the off-border check); without it, no window -> no trigger.
    pred = StubSelectivePredictor(lambda f, oid: L, emit_probs=True, prob=0.8)
    tracker = Sam2FieldTracker(predictor=pred)

    results = tracker.track_selective(
        str(tmp_path), _detect_fn(_STARVE_DETS), _cfg_starve())

    stats = tracker.selective_stats
    # The per-trigger breakdown is reported and counts the starve window.
    assert stats["triggers"]["starve"] >= 1
    assert stats["windows"] == 1

    # oid 1's last TBD detection is frame 4; the starve window carries SAM masks
    # PAST it (frames 5+) -- exactly the detector blind-spot persistence pure
    # TBD could never do (history is empty for those frames).
    for f in (5, 6, 7):
        assert results[f][1]["evidence"]["type"] == "selective_sam"
        assert results[f][1]["mask"] is not None
