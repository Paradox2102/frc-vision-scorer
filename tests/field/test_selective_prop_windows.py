"""Tests for the selective-mask-propagation window finder, exit state machine,
and sam2_raw merge (Task 3).

Pure numpy / synthetic boxes and masks; no GPU, no torch. Masks are small
``np.zeros((H, W), bool)`` rectangles built to match / mismatch the TBD boxes
that live in a hand-constructed ``history`` dict.
"""
import random

import numpy as np

from web.field.selective_prop import (
    ExitMachine,
    FrameObs,
    RenameEvent,
    SelectiveConfig,
    WindowOutcome,
    WindowSpec,
    cluster_windows,
    find_windows,
    merge_selective_results,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _obs(box, margin=float("inf"), color=None):
    return FrameObs(box=box, margin=margin, color=color)


def _rect_mask(hw, box):
    """Bool mask that is True over ``box`` = (x1, y1, x2, y2) [x2/y2 exclusive]."""
    h, w = hw
    m = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = box
    m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
    return m


# --------------------------------------------------------------------------
# find_windows
# --------------------------------------------------------------------------
def test_margin_trigger_backward_seed():
    cfg = SelectiveConfig(min_streak=3, margin_entry=0.05, margin_exit=0.10)
    box = (10, 10, 30, 30)
    history = {f: {1: _obs(box, margin=float("inf"))} for f in range(5)}
    history[5] = {1: _obs(box, margin=0.02)}          # low margin -> entry 5
    wins = find_windows(history, [], 6, cfg)
    assert len(wins) == 1
    w = wins[0]
    assert w.oid == 1
    assert w.entry_frame == 5
    assert w.trigger == "margin"
    # streak of min_streak clean frames ends walking back from entry-1=4:
    # frames 4,3,2 clean -> seed = 2 (entry - min_streak).
    assert w.seed_frame == 2


def test_gap_trigger_window():
    cfg = SelectiveConfig(min_streak=2, margin_exit=0.10)
    box = (10, 10, 30, 30)
    history = {f: {1: _obs(box)} for f in range(3)}    # frames 0,1,2 clean
    wins = find_windows(history, [(1, 2, 7)], 8, cfg)
    assert len(wins) == 1
    w = wins[0]
    assert w.oid == 1
    assert w.trigger == "gap"
    assert w.entry_frame == 2
    # walk back from entry-1=1: frames 1,0 clean -> seed 0.
    assert w.seed_frame == 0


def test_witness_expansion():
    cfg = SelectiveConfig(min_streak=2, margin_entry=0.05, margin_exit=0.10,
                          seed_clean_iou=0.10)
    left = (10, 40, 30, 60)
    right = (70, 40, 90, 60)
    history = {f: {1: _obs(left), 2: _obs(right)} for f in range(3)}
    # entry frame 3: the two boxes overlap (crossing). oid1 is the low-margin
    # trigger; oid2 keeps a high margin so it is ONLY a witness, not a trigger.
    history[3] = {1: _obs((35, 40, 55, 60), margin=0.02),
                  2: _obs((45, 40, 65, 60), margin=0.20)}
    wins = find_windows(history, [], 4, cfg)
    by_oid = {w.oid: w for w in wins}
    assert set(by_oid) == {1, 2}
    assert by_oid[1].trigger == "margin"
    assert by_oid[2].trigger == "witness"
    assert by_oid[1].entry_frame == 3 and by_oid[2].entry_frame == 3
    assert by_oid[1].seed_frame == 1 and by_oid[2].seed_frame == 1


def test_dedup_same_oid_entry():
    cfg = SelectiveConfig(min_streak=2, margin_entry=0.05, margin_exit=0.10)
    box = (10, 10, 30, 30)
    history = {f: {1: _obs(box)} for f in range(3)}
    history[2] = {1: _obs(box, margin=0.02)}          # margin trigger at entry 2
    # gap event ALSO produces (oid1, entry 2): must dedup to a single window.
    wins = find_windows(history, [(1, 2, 6)], 7, cfg)
    sel = [w for w in wins if w.oid == 1 and w.entry_frame == 2]
    assert len(sel) == 1
    assert sel[0].trigger == "margin"                 # margin processed first


def test_no_clean_seed_dropped():
    cfg = SelectiveConfig(min_streak=2, margin_entry=0.05, margin_exit=0.10)
    box = (10, 10, 30, 30)
    # every frame is ambiguous (margin below the exit floor) -> no clean seed.
    history = {f: {1: _obs(box, margin=0.02)} for f in range(4)}
    assert find_windows(history, [], 4, cfg) == []


def test_window_at_sequence_start_dropped():
    cfg = SelectiveConfig(min_streak=2)
    box = (10, 10, 30, 30)
    history = {0: {1: _obs(box, margin=0.02)}}        # entry 0 -> nothing behind
    assert find_windows(history, [], 1, cfg) == []


def test_find_windows_sorted_order():
    cfg = SelectiveConfig(min_streak=2, margin_entry=0.05, margin_exit=0.10)
    a = (10, 10, 30, 30)
    b = (200, 200, 230, 230)
    history = {}
    for f in range(6):
        history[f] = {1: _obs(a), 2: _obs(b)}
    history[3] = {1: _obs(a, margin=0.02), 2: _obs(b)}   # oid1 entry 3 (seed 1)
    history[5] = {1: _obs(a), 2: _obs(b, margin=0.02)}   # oid2 entry 5 (seed 3)
    wins = find_windows(history, [], 6, cfg)
    keys = [(w.seed_frame, w.entry_frame, w.oid) for w in wins]
    assert keys == sorted(keys)


# --------------------------------------------------------------------------
# ExitMachine
# --------------------------------------------------------------------------
def test_exit_clean_settle():
    cfg = SelectiveConfig(min_streak=2, margin_exit=0.10, ioma_exit=0.80,
                          area_degradation=0.25, border_margin_px=5)
    hw = (100, 100)
    box = (10, 40, 30, 60)
    history = {f: {1: _obs(box, margin=float("inf"))} for f in range(7)}
    spec = WindowSpec(oid=1, seed_frame=0, entry_frame=2, trigger="margin")
    em = ExitMachine([spec], {1: box}, hw, history, cfg)
    mask = _rect_mask(hw, box)
    done = False
    for f in range(7):
        done = em.step(f, {1: mask}, {1: 0.9})
        if done:
            break
    assert done
    o = em.outcomes()[0]
    assert o.state == "clean"
    assert o.swap_to is None
    assert o.close_frame == 3
    assert set(o.masks) == {2, 3}                    # only [entry, close]


def test_exit_stale_abort():
    cfg = SelectiveConfig(min_streak=2, border_margin_px=5)
    hw = (100, 100)
    box = (10, 10, 30, 30)
    history = {f: {1: _obs(box)} for f in range(4)}
    spec = WindowSpec(oid=1, seed_frame=0, entry_frame=2, trigger="margin")
    em = ExitMachine([spec], {1: box}, hw, history, cfg)
    seed = _rect_mask(hw, box)
    drift = _rect_mask(hw, (60, 60, 80, 80))         # no overlap with box
    em.step(0, {1: seed}, {1: None})
    em.step(1, {1: seed}, {1: None})
    done = em.step(2, {1: drift}, {1: None})          # entry: no overlap -> stale
    assert done
    o = em.outcomes()[0]
    assert o.state == "stale"
    assert o.masks == {}                              # aborts contribute no masks


def test_exit_degraded_area():
    cfg = SelectiveConfig(min_streak=2, area_degradation=0.25, border_margin_px=5)
    hw = (100, 100)
    box = (10, 10, 40, 40)
    history = {f: {1: _obs(box, margin=0.0)} for f in range(4)}   # margin 0 -> no settle
    spec = WindowSpec(oid=1, seed_frame=0, entry_frame=1, trigger="margin")
    em = ExitMachine([spec], {1: box}, hw, history, cfg)
    seed = _rect_mask(hw, box)                        # area 900
    tiny = _rect_mask(hw, (10, 10, 14, 14))           # area 16 (< 0.25*900)
    em.step(0, {1: seed}, {1: None})
    done = em.step(1, {1: tiny}, {1: None})
    assert done
    o = em.outcomes()[0]
    assert o.state == "degraded"
    assert o.masks == {}


def test_exit_degraded_dark():
    cfg = SelectiveConfig(min_streak=2, border_margin_px=5)
    hw = (100, 100)
    box = (10, 10, 40, 40)
    history = {f: {1: _obs(box, margin=0.0)} for f in range(6)}
    spec = WindowSpec(oid=1, seed_frame=0, entry_frame=1, trigger="margin")
    em = ExitMachine([spec], {1: box}, hw, history, cfg)
    m = _rect_mask(hw, box)
    em.step(0, {1: m}, {1: None})
    em.step(1, {1: m}, {1: None})
    em.step(2, {1: None}, {1: None})                  # dark streak 1
    done = em.step(3, {1: None}, {1: None})           # dark streak 2 -> degraded
    assert done
    o = em.outcomes()[0]
    assert o.state == "degraded"
    assert o.close_frame == 3


def test_exit_edge_abort():
    cfg = SelectiveConfig(min_streak=2, border_margin_px=20)
    hw = (100, 100)
    box = (85, 40, 99, 60)                            # hugs the right edge
    history = {f: {1: _obs(box, margin=0.0)} for f in range(4)}
    spec = WindowSpec(oid=1, seed_frame=0, entry_frame=1, trigger="margin")
    em = ExitMachine([spec], {1: box}, hw, history, cfg)
    m = _rect_mask(hw, box)
    em.step(0, {1: m}, {1: None})
    em.step(1, {1: m}, {1: None})                     # live near border
    done = em.step(2, {1: None}, {1: None})           # vanishes -> edge
    assert done
    o = em.outcomes()[0]
    assert o.state == "edge"
    assert o.close_frame == 2


def test_exit_killed():
    cfg = SelectiveConfig(min_streak=2, mask_overlap_exit=0.90, border_margin_px=5)
    hw = (100, 100)
    b1 = (10, 10, 50, 50)
    b2 = (15, 10, 55, 50)                             # overlaps b1 -> never isolated
    history = {f: {1: _obs(b1, margin=0.0), 2: _obs(b2, margin=0.0)}
               for f in range(4)}
    s1 = WindowSpec(oid=1, seed_frame=0, entry_frame=1, trigger="margin")
    s2 = WindowSpec(oid=2, seed_frame=0, entry_frame=1, trigger="witness")
    em = ExitMachine([s1, s2], {1: b1, 2: b2}, hw, history, cfg)
    m = _rect_mask(hw, (20, 20, 40, 40))              # identical mask -> IoU 1.0
    em.step(0, {1: m, 2: m}, {1: None, 2: None})
    em.step(1, {1: m, 2: m}, {1: None, 2: None})      # overlap streak 1
    done = em.step(2, {1: m, 2: m}, {1: None, 2: None})   # streak 2 -> killed
    assert done
    outs = {o.spec.oid: o for o in em.outcomes()}
    assert outs[1].state == "killed" and outs[2].state == "killed"
    assert outs[1].masks == {} and outs[2].masks == {}


def test_exit_killed_respects_mask_overlap_exit_knob():
    """selective_mask_overlap_exit (Task 1 knob) drives the kill threshold, not
    a hardcoded constant: the SAME 0.923 pixel-IoU mask pair kills at the
    knob's 0.90 default but survives once the knob is raised to 0.95 (the
    qm11 blue-gap fix candidate, docs/selective-mask-prop-blue-gap.md)."""
    hw = (200, 200)
    b1 = (10, 10, 50, 50)
    b2 = (15, 10, 55, 50)                             # overlaps b1 -> never isolated
    history = {f: {1: _obs(b1, margin=0.0), 2: _obs(b2, margin=0.0)}
               for f in range(4)}
    s1 = WindowSpec(oid=1, seed_frame=0, entry_frame=1, trigger="margin")
    s2 = WindowSpec(oid=2, seed_frame=0, entry_frame=1, trigger="witness")
    mA = _rect_mask(hw, (20, 20, 120, 120))
    mB = _rect_mask(hw, (24, 20, 124, 120))           # pixel-IoU 9600/10400 = 0.9231

    cfg_default = SelectiveConfig(min_streak=2, mask_overlap_exit=0.90, border_margin_px=5)
    em_default = ExitMachine([s1, s2], {1: b1, 2: b2}, hw, history, cfg_default)
    done = False
    for f in range(3):
        done = em_default.step(f, {1: mA, 2: mB}, {1: None, 2: None})
    assert done
    outs = {o.spec.oid: o for o in em_default.outcomes()}
    assert outs[1].state == "killed" and outs[2].state == "killed"

    cfg_raised = SelectiveConfig(min_streak=2, mask_overlap_exit=0.95, border_margin_px=5)
    em_raised = ExitMachine([s1, s2], {1: b1, 2: b2}, hw, history, cfg_raised)
    done = False
    for f in range(3):
        done = em_raised.step(f, {1: mA, 2: mB}, {1: None, 2: None})
    assert not done                                   # never killed -> never both closed
    outs_raised = {o.spec.oid: o for o in em_raised.outcomes()}
    assert outs_raised[1].state == "end" and outs_raised[2].state == "end"


def _kill_coast_scenario():
    """Two same-alliance objects that track distinctly (oid1 drifts right, oid2
    left) for two post-entry frames, then their masks collapse onto one patch
    for ``min_streak`` frames -> both killed. Returns everything an ExitMachine
    run needs; the last confident (pre-collapse) SAM boxes are frames 1 and 2.
    """
    hw = (200, 200)
    # oid1 masks: distinct+moving right at f1,f2, then merged at f3,f4.
    m1 = {
        0: _rect_mask(hw, (10, 80, 30, 100)),
        1: _rect_mask(hw, (20, 80, 40, 100)),
        2: _rect_mask(hw, (30, 80, 50, 100)),
        3: _rect_mask(hw, (60, 80, 80, 100)),
        4: _rect_mask(hw, (60, 80, 80, 100)),
    }
    # oid2 masks: distinct+moving left at f1,f2, then merged onto oid1 at f3,f4.
    m2 = {
        0: _rect_mask(hw, (160, 80, 180, 100)),
        1: _rect_mask(hw, (150, 80, 170, 100)),
        2: _rect_mask(hw, (140, 80, 160, 100)),
        3: _rect_mask(hw, (60, 80, 80, 100)),
        4: _rect_mask(hw, (60, 80, 80, 100)),
    }
    # History boxes sit inside each mask (entry STALE check passes); margin 0.0
    # keeps every frame ambiguous so no CLEAN/SWAP settle pre-empts the kill.
    history = {
        0: {1: _obs((12, 82, 28, 98), margin=0.0), 2: _obs((162, 82, 178, 98), margin=0.0)},
        1: {1: _obs((22, 82, 38, 98), margin=0.0), 2: _obs((152, 82, 168, 98), margin=0.0)},
        2: {1: _obs((32, 82, 48, 98), margin=0.0), 2: _obs((142, 82, 158, 98), margin=0.0)},
        3: {1: _obs((62, 82, 78, 98), margin=0.0), 2: _obs((62, 82, 78, 98), margin=0.0)},
        4: {1: _obs((62, 82, 78, 98), margin=0.0), 2: _obs((62, 82, 78, 98), margin=0.0)},
    }
    s1 = WindowSpec(oid=1, seed_frame=0, entry_frame=1, trigger="margin")
    s2 = WindowSpec(oid=2, seed_frame=0, entry_frame=1, trigger="witness")
    return hw, history, s1, s2, m1, m2


def test_kill_coast_off_captures_no_seed():
    """Regression guard: knob OFF -> a kill closes exactly as today (masks
    wiped, no coast metadata on the outcome)."""
    hw, history, s1, s2, m1, m2 = _kill_coast_scenario()
    cfg = SelectiveConfig(min_streak=2, mask_overlap_exit=0.90, margin_exit=0.10,
                          border_margin_px=5)               # kill_coast default OFF
    em = ExitMachine([s1, s2], {1: history[0][1].box, 2: history[0][2].box},
                     hw, history, cfg)
    done = False
    for f in range(5):
        done = em.step(f, {1: m1[f], 2: m2[f]}, {1: None, 2: None})
    assert done
    outs = {o.spec.oid: o for o in em.outcomes()}
    assert outs[1].state == "killed" and outs[2].state == "killed"
    assert outs[1].masks == {} and outs[2].masks == {}
    assert outs[1].coast_seed_frame is None and outs[1].coast_boxes == []
    assert outs[2].coast_seed_frame is None and outs[2].coast_boxes == []


def test_kill_coast_on_captures_last_confident_boxes():
    """Knob ON -> each killed window records its last confident (pre-collapse)
    SAM boxes: cutoff = close_frame - min_streak = 2, so the seed frame is 2 and
    the coast trajectory is the boxes at frames 1 and 2."""
    hw, history, s1, s2, m1, m2 = _kill_coast_scenario()
    cfg = SelectiveConfig(min_streak=2, mask_overlap_exit=0.90, margin_exit=0.10,
                          border_margin_px=5, kill_coast=True)
    em = ExitMachine([s1, s2], {1: history[0][1].box, 2: history[0][2].box},
                     hw, history, cfg)
    done = False
    for f in range(5):
        done = em.step(f, {1: m1[f], 2: m2[f]}, {1: None, 2: None})
    assert done
    outs = {o.spec.oid: o for o in em.outcomes()}
    assert outs[1].state == "killed" and outs[2].state == "killed"
    assert outs[1].masks == {} and outs[2].masks == {}     # masks still wiped
    assert outs[1].coast_seed_frame == 2
    assert outs[1].coast_boxes == [(20, 80, 40, 100), (30, 80, 50, 100)]
    assert outs[2].coast_seed_frame == 2
    assert outs[2].coast_boxes == [(150, 80, 170, 100), (140, 80, 160, 100)]


def test_exit_end_window_max():
    cfg = SelectiveConfig(min_streak=10, window_max_frames=3, margin_exit=0.10,
                          border_margin_px=5)
    hw = (100, 100)
    box = (10, 40, 30, 60)
    history = {f: {1: _obs(box)} for f in range(10)}
    spec = WindowSpec(oid=1, seed_frame=0, entry_frame=1, trigger="margin")
    em = ExitMachine([spec], {1: box}, hw, history, cfg)
    m = _rect_mask(hw, box)
    done = False
    for f in range(10):
        done = em.step(f, {1: m}, {1: 0.5})
        if done:
            break
    assert done
    o = em.outcomes()[0]
    assert o.state == "end"
    assert o.close_frame == 3                          # seed_frame + window_max_frames


def test_exit_end_sequence_end():
    cfg = SelectiveConfig(min_streak=10, window_max_frames=120, border_margin_px=5)
    hw = (100, 100)
    box = (10, 40, 30, 60)
    history = {f: {1: _obs(box)} for f in range(3)}
    spec = WindowSpec(oid=1, seed_frame=0, entry_frame=1, trigger="margin")
    em = ExitMachine([spec], {1: box}, hw, history, cfg)
    m = _rect_mask(hw, box)
    for f in range(3):
        em.step(f, {1: m}, {1: 0.5})                   # never closes on its own
    o = em.outcomes()[0]                               # outcomes() finalizes to "end"
    assert o.state == "end"
    assert o.close_frame == 2


# --------------------------------------------------------------------------
# ExitMachine SWAP + merge_selective_results identity correction
# --------------------------------------------------------------------------
def test_exit_swap_and_merge_renames_stream():
    cfg = SelectiveConfig(min_streak=2, margin_exit=0.10, ioma_exit=0.80,
                          area_degradation=0.25, border_margin_px=5)
    hw = (100, 100)
    left = (10, 40, 30, 60)
    right = (70, 40, 90, 60)
    history = {}
    for f in range(3):                                # pre-crossing: 1=left, 2=right
        history[f] = {1: _obs(left), 2: _obs(right)}
    history[3] = {1: _obs((35, 40, 55, 60), margin=0.02),   # crossing (entry)
                  2: _obs((45, 40, 65, 60), margin=0.02)}
    for f in range(4, 7):                             # TBD swapped: 1=left(B), 2=right(A)
        history[f] = {1: _obs(left), 2: _obs(right)}
    spec = WindowSpec(oid=1, seed_frame=0, entry_frame=3, trigger="margin")
    em = ExitMachine([spec], {1: left}, hw, history, cfg)
    # SAM seeded on oid1 (physical A on the left) and correctly follows A across
    # the crossing to the right, where TBD now (wrongly) labels it oid2.
    mask_by_f = {
        0: _rect_mask(hw, left), 1: _rect_mask(hw, left), 2: _rect_mask(hw, left),
        3: _rect_mask(hw, (40, 40, 60, 60)),
        4: _rect_mask(hw, right), 5: _rect_mask(hw, right), 6: _rect_mask(hw, right),
    }
    done = False
    for f in range(7):
        done = em.step(f, {1: mask_by_f[f]}, {1: 0.8})
        if done:
            break
    assert done
    o = em.outcomes()[0]
    assert o.state == "swap"
    assert o.swap_to == 2
    assert o.close_frame == 5

    results, renames = merge_selective_results(history, [o], cfg)
    assert len(renames) == 1
    assert renames[0].frame == 3
    assert {renames[0].a, renames[0].b} == {1, 2}

    # Post-crossing identity is corrected: oid1 is the RIGHT box (physical A),
    # oid2 the LEFT box (physical B), both plain TBD beyond the window close.
    assert results[6][1]["box"] == right
    assert results[6][1]["evidence"]["type"] == "tbd"
    assert results[6][2]["box"] == left
    assert results[6][2]["evidence"]["type"] == "tbd"
    # Inside the window oid1 carries the SAM mask (on the right) w/ its prob.
    assert results[5][1]["evidence"]["type"] == "selective_sam"
    assert results[5][1]["box"] == right
    assert results[5][1]["evidence"]["tracker_prob"] == 0.8
    # ExitMachine defensively copies collected masks; compare content.
    assert np.array_equal(results[5][1]["mask"], mask_by_f[5])


# --------------------------------------------------------------------------
# merge_selective_results schema / abort invariants
# --------------------------------------------------------------------------
def test_merge_schema_and_types():
    cfg = SelectiveConfig()
    box = (10, 10, 30, 30)
    history = {0: {1: _obs(box)}, 1: {1: _obs(box)}}
    hw = (50, 50)
    mask = _rect_mask(hw, box)
    spec = WindowSpec(oid=1, seed_frame=0, entry_frame=1, trigger="margin")
    outcome = WindowOutcome(spec=spec, state="clean", swap_to=None,
                            close_frame=1, masks={1: mask}, probs={1: 0.7})
    results, renames = merge_selective_results(history, [outcome], cfg)
    for frame in results.values():
        for entry in frame.values():
            assert set(entry) == {"box", "mask", "evidence"}
            assert set(entry["evidence"]) == {
                "type", "gap_len", "detector_overlap", "tracker_prob"}
    assert results[0][1]["evidence"]["type"] == "tbd"
    assert results[0][1]["mask"] is None
    assert results[0][1]["evidence"]["tracker_prob"] is None
    assert results[1][1]["evidence"]["type"] == "selective_sam"
    assert results[1][1]["mask"] is mask
    assert results[1][1]["evidence"]["tracker_prob"] == 0.7
    assert renames == []


def test_degraded_abort_leaves_tbd_unchanged():
    cfg = SelectiveConfig()
    box = (10, 10, 30, 30)
    history = {0: {1: _obs(box)},
               1: {1: _obs(box, margin=0.02)},
               2: {1: _obs(box)}}
    spec = WindowSpec(oid=1, seed_frame=0, entry_frame=1, trigger="margin")
    degraded = WindowOutcome(spec=spec, state="degraded", swap_to=None,
                             close_frame=1, masks={}, probs={})
    with_abort, ren_a = merge_selective_results(history, [degraded], cfg)
    baseline, ren_b = merge_selective_results(history, [], cfg)
    assert with_abort == baseline                     # byte-identical to pure TBD
    assert ren_a == [] and ren_b == []


def test_merge_two_disjoint_swap_windows_compose():
    # Two independent SWAP windows: 1<->2 effective frame 3, 3<->4 effective
    # frame 7. Frames >= 7 must carry BOTH corrections (renames compose; the
    # later event must not drop the earlier one).
    cfg = SelectiveConfig()
    b1 = (10, 10, 30, 30)
    b2 = (50, 10, 70, 30)
    b3 = (110, 10, 130, 30)
    b4 = (150, 10, 170, 30)
    history = {f: {1: _obs(b1), 2: _obs(b2), 3: _obs(b3), 4: _obs(b4)}
               for f in range(10)}
    o12 = WindowOutcome(
        spec=WindowSpec(oid=1, seed_frame=0, entry_frame=3, trigger="margin"),
        state="swap", swap_to=2, close_frame=5, masks={}, probs={})
    o34 = WindowOutcome(
        spec=WindowSpec(oid=3, seed_frame=4, entry_frame=7, trigger="margin"),
        state="swap", swap_to=4, close_frame=9, masks={}, probs={})
    results, renames = merge_selective_results(history, [o12, o34], cfg)
    assert [(e.frame, {e.a, e.b}) for e in renames] == [(3, {1, 2}), (7, {3, 4})]
    # before any event: original labels
    assert results[2][1]["box"] == b1 and results[2][3]["box"] == b3
    # frames 3-6: only 1<->2 applied
    assert results[4][1]["box"] == b2 and results[4][2]["box"] == b1
    assert results[4][3]["box"] == b3 and results[4][4]["box"] == b4
    # frames >= 7: BOTH corrections in effect
    for f in (7, 8, 9):
        assert results[f][1]["box"] == b2 and results[f][2]["box"] == b1
        assert results[f][3]["box"] == b4 and results[f][4]["box"] == b3


def test_merge_same_pair_reswap_is_identity():
    # 1<->2 at frame 3, then 1<->2 again at frame 9 (SAM detected a swap back):
    # frames 3-8 swapped, frames >= 9 back to the original labels.
    cfg = SelectiveConfig()
    b1 = (10, 10, 30, 30)
    b2 = (50, 10, 70, 30)
    history = {f: {1: _obs(b1), 2: _obs(b2)} for f in range(12)}
    first = WindowOutcome(
        spec=WindowSpec(oid=1, seed_frame=0, entry_frame=3, trigger="margin"),
        state="swap", swap_to=2, close_frame=5, masks={}, probs={})
    back = WindowOutcome(
        spec=WindowSpec(oid=1, seed_frame=6, entry_frame=9, trigger="margin"),
        state="swap", swap_to=2, close_frame=11, masks={}, probs={})
    results, renames = merge_selective_results(history, [first, back], cfg)
    assert len(renames) == 2                       # distinct frames: no dedup
    assert results[2][1]["box"] == b1              # pre-event
    for f in range(3, 9):                          # swapped span
        assert results[f][1]["box"] == b2 and results[f][2]["box"] == b1
    for f in range(9, 12):                         # composed back to identity
        assert results[f][1]["box"] == b1 and results[f][2]["box"] == b2


def test_merge_chained_swaps_share_oid():
    # Chained renames sharing an oid: physical robots A,B,C on raw TBD streams
    # 1,2,3. Crossing (1,2) effective f3 puts A on raw2 / B on raw1; crossing
    # (2,3) effective f7 then puts A on raw3 / C on raw2. Events are in RAW
    # history-label space (ExitMachine settles against never-renamed history),
    # so the composed correction from f7 on must be raw1->2, raw2->3, raw3->1
    # (output label i always carries physical robot i's box).
    cfg = SelectiveConfig()
    b1 = (10, 10, 30, 30)      # raw stream 1's box
    b2 = (50, 10, 70, 30)      # raw stream 2's box
    b3 = (110, 10, 130, 30)    # raw stream 3's box
    history = {f: {1: _obs(b1), 2: _obs(b2), 3: _obs(b3)} for f in range(10)}
    o12 = WindowOutcome(
        spec=WindowSpec(oid=1, seed_frame=0, entry_frame=3, trigger="margin"),
        state="swap", swap_to=2, close_frame=5, masks={}, probs={})
    o23 = WindowOutcome(
        spec=WindowSpec(oid=2, seed_frame=4, entry_frame=7, trigger="margin"),
        state="swap", swap_to=3, close_frame=9, masks={}, probs={})
    results, renames = merge_selective_results(history, [o12, o23], cfg)
    assert [(e.frame, {e.a, e.b}) for e in renames] == [(3, {1, 2}), (7, {2, 3})]
    # pre-f3: identity
    for f in (0, 1, 2):
        assert results[f][1]["box"] == b1
        assert results[f][2]["box"] == b2
        assert results[f][3]["box"] == b3
    # f3-f6: only (1,2) -> label1=raw2, label2=raw1, label3=raw3
    for f in (3, 4, 5, 6):
        assert results[f][1]["box"] == b2
        assert results[f][2]["box"] == b1
        assert results[f][3]["box"] == b3
    # >= f7: composed cycle raw1->2, raw2->3, raw3->1 ->
    # label1=raw3's entry, label2=raw1's, label3=raw2's.
    for f in (7, 8, 9):
        assert results[f][1]["box"] == b3
        assert results[f][2]["box"] == b1
        assert results[f][3]["box"] == b2


def test_merge_rename_dedup_identical_pairs():
    cfg = SelectiveConfig()
    box = (10, 10, 30, 30)
    history = {f: {1: _obs(box), 2: _obs(box)} for f in range(3)}
    s1 = WindowSpec(oid=1, seed_frame=0, entry_frame=1, trigger="margin")
    s2 = WindowSpec(oid=2, seed_frame=0, entry_frame=1, trigger="witness")
    # Two swap outcomes describing the SAME (frame, {a, b}) transposition.
    o1 = WindowOutcome(spec=s1, state="swap", swap_to=2, close_frame=1,
                       masks={}, probs={})
    o2 = WindowOutcome(spec=s2, state="swap", swap_to=1, close_frame=1,
                       masks={}, probs={})
    _, renames = merge_selective_results(history, [o1, o2], cfg)
    assert len(renames) == 1
    assert isinstance(renames[0], RenameEvent)


# --------------------------------------------------------------------------
# merge_selective_results kill-coast fallback (Task 2, selective_kill_coast)
# --------------------------------------------------------------------------
def _killed_outcome(oid, seed_frame, coast_boxes, entry_frame=1, close_frame=4):
    """A killed WindowOutcome carrying the coast seed the ExitMachine would have
    captured (masks wiped, as a real kill leaves them)."""
    return WindowOutcome(
        spec=WindowSpec(oid=oid, seed_frame=0, entry_frame=entry_frame,
                        trigger="margin"),
        state="killed", swap_to=None, close_frame=close_frame,
        masks={}, probs={}, coast_seed_frame=seed_frame, coast_boxes=coast_boxes)


def test_kill_coast_off_leaves_tbd_unchanged():
    """Regression guard (merge level): with the knob OFF a killed outcome is
    byte-identical to pure TBD, even when coast metadata rides along."""
    cfg = SelectiveConfig(min_streak=2, margin_exit=0.10)   # kill_coast default OFF
    b_flip = (120, 80, 140, 100)
    history = {f: {1: _obs(b_flip, margin=0.0)} for f in range(6)}
    killed = _killed_outcome(1, seed_frame=1,
                             coast_boxes=[(60, 80, 80, 100), (50, 80, 70, 100)])
    with_kill, ren_a = merge_selective_results(history, [killed], cfg)
    baseline, ren_b = merge_selective_results(history, [], cfg)
    assert with_kill == baseline                            # nothing coasted
    assert ren_a == [] and ren_b == []
    for f in range(6):
        assert with_kill[f][1]["evidence"]["type"] == "tbd"


def test_kill_coast_on_pins_identity_through_ambiguous_flip():
    """Knob ON: raw TBD has flipped oid1 onto the other robot's box (far right),
    but the coast pins oid1 to a constant-velocity extrapolation of its last
    confident SAM box (drifting left) with ``kill_coast`` evidence, for the
    whole coast window."""
    cfg = SelectiveConfig(min_streak=2, margin_exit=0.10, iou_expand=0.25,
                          kill_coast=True, kill_coast_frames=24)
    b_flip = (120, 80, 140, 100)                            # TBD's (wrong) box
    history = {f: {1: _obs(b_flip, margin=0.0)} for f in range(6)}
    # seed frame 1, box (50,80,70,100), prior box (60,80,80,100) -> velocity (-10,0).
    killed = _killed_outcome(1, seed_frame=1,
                             coast_boxes=[(60, 80, 80, 100), (50, 80, 70, 100)])
    results, renames = merge_selective_results(history, [killed], cfg)
    assert renames == []
    # frames <= seed are untouched raw TBD.
    for f in (0, 1):
        assert results[f][1]["box"] == b_flip
        assert results[f][1]["evidence"]["type"] == "tbd"
    # frames seed+1..end: coasted box (identity pinned), NOT the flipped TBD box.
    expected = {2: (40, 80, 60, 100), 3: (30, 80, 50, 100),
                4: (20, 80, 40, 100), 5: (10, 80, 30, 100)}
    for f, box in expected.items():
        assert results[f][1]["box"] == box
        assert results[f][1]["box"] != b_flip
        assert results[f][1]["mask"] is None
        assert results[f][1]["evidence"]["type"] == "kill_coast"


def test_kill_coast_two_close_boxes_keep_coasting():
    """The ambiguity gate: two same-alliance detections straddle the coasted box
    with near-equal expanded-IoU, so the re-claim margin stays below margin_exit
    and the coast HOLDS -- exactly the two-close-boxes case where raw TBD would
    flip. The coast does not hand identity to either near-tied detection."""
    cfg = SelectiveConfig(min_streak=2, margin_exit=0.10, iou_expand=0.25,
                          kill_coast=True, kill_coast_frames=1)
    # coasted box at f2 is (40,80,60,100); two detections straddle it equally.
    history = {
        0: {1: _obs((60, 80, 80, 100), margin=0.0)},
        1: {1: _obs((50, 80, 70, 100), margin=0.0)},
        2: {1: _obs((36, 80, 56, 100), margin=0.0),
            2: _obs((44, 80, 64, 100), margin=0.0)},
    }
    killed = _killed_outcome(1, seed_frame=1,
                             coast_boxes=[(60, 80, 80, 100), (50, 80, 70, 100)])
    results, _ = merge_selective_results(history, [killed], cfg)
    assert results[2][1]["evidence"]["type"] == "kill_coast"    # held (ambiguous)
    assert results[2][1]["box"] == (40, 80, 60, 100)


def test_kill_coast_unambiguous_reclaim_ends_coast():
    """A frame where the coasted box matches exactly one TBD detection (margin
    inf >= margin_exit) ends the coast; that frame and all later ones stand as
    raw TBD."""
    cfg = SelectiveConfig(min_streak=2, margin_exit=0.10, iou_expand=0.25,
                          kill_coast=True, kill_coast_frames=24)
    far = (120, 80, 140, 100)
    history = {
        0: {1: _obs(far, margin=0.0)},
        1: {1: _obs(far, margin=0.0)},
        2: {1: _obs(far, margin=0.0)},                      # nothing near coast -> hold
        # f3 coasted box is (30,80,50,100); one detection sits right on it and
        # no other detection overlaps -> unambiguous re-claim.
        3: {1: _obs((32, 80, 52, 100), margin=0.5)},
        4: {1: _obs((20, 80, 40, 100), margin=0.5)},
        5: {1: _obs((10, 80, 30, 100), margin=0.5)},
    }
    killed = _killed_outcome(1, seed_frame=1,
                             coast_boxes=[(60, 80, 80, 100), (50, 80, 70, 100)])
    results, _ = merge_selective_results(history, [killed], cfg)
    assert results[2][1]["evidence"]["type"] == "kill_coast"   # held (no match)
    assert results[2][1]["box"] == (40, 80, 60, 100)
    # f3 re-claims -> raw TBD stands from here on.
    assert results[3][1]["evidence"]["type"] == "tbd"
    assert results[3][1]["box"] == (32, 80, 52, 100)
    assert results[4][1]["evidence"]["type"] == "tbd"
    assert results[5][1]["evidence"]["type"] == "tbd"


def test_kill_coast_budget_expiry_reverts_to_tbd():
    """The coast lasts at most ``kill_coast_frames`` frames; past the budget the
    raw TBD stream stands even when the geometry would still coast."""
    cfg = SelectiveConfig(min_streak=2, margin_exit=0.10, iou_expand=0.25,
                          kill_coast=True, kill_coast_frames=2)
    far = (120, 80, 140, 100)                               # never near the coast
    history = {f: {1: _obs(far, margin=0.0)} for f in range(6)}
    killed = _killed_outcome(1, seed_frame=1,
                             coast_boxes=[(60, 80, 80, 100), (50, 80, 70, 100)])
    results, _ = merge_selective_results(history, [killed], cfg)
    # budget = 2 -> only seed+1, seed+2 coast.
    assert results[2][1]["evidence"]["type"] == "kill_coast"
    assert results[3][1]["evidence"]["type"] == "kill_coast"
    # seed+3 onward: budget spent, back to raw TBD.
    assert results[4][1]["evidence"]["type"] == "tbd"
    assert results[4][1]["box"] == far
    assert results[5][1]["evidence"]["type"] == "tbd"


def test_merge_swap_with_clean_witness_same_group():
    # Same seed-frame group: physical robots A, B cross so the margin trigger
    # (oid1) settles SWAP -> oid2, while robot C rides along as a WITNESS (its
    # box overlapped the crossing at the entry frame) and settles CLEAN under
    # its own oid3. The two corrections must not corrupt each other: the 1<->2
    # relabel applies to the whole stream from the entry frame, AND both healthy
    # SAM masks overlay under their RAW spec.oid in the SAME [entry, close]
    # frames -- oid1 the swap-follow mask, oid3 the witness's own mask, with
    # oid3 never relabeled. (merge applies renames BEFORE mask overlay, so a
    # mask keyed by raw spec.oid lands on the post-rename slot; this exercises
    # that ordering for a swap + healthy-witness pair, which the swap-only and
    # mask-less-swap tests above never combine.)
    cfg = SelectiveConfig()
    hw = (60, 200)
    bA = (10, 10, 30, 30)     # raw stream 1 (physical A)
    bB = (50, 10, 70, 30)     # raw stream 2 (physical B)
    bC = (110, 10, 130, 30)   # raw stream 3 (physical C, the witness)
    history = {f: {1: _obs(bA), 2: _obs(bB), 3: _obs(bC)} for f in range(9)}

    # oid1 -> oid2 at entry frame 3; SAM followed physical A onto B's spatial
    # slot (bB) across [entry, close] = [3, 5].
    swap_masks = {f: _rect_mask(hw, bB) for f in (3, 4, 5)}
    o_swap = WindowOutcome(
        spec=WindowSpec(oid=1, seed_frame=0, entry_frame=3, trigger="margin"),
        state="swap", swap_to=2, close_frame=5,
        masks=swap_masks, probs={3: 0.8, 4: 0.8, 5: 0.8})

    # oid3 witness in the SAME group (shared seed/entry frame); SAM held C on bC
    # and it settles clean over the same [3, 5] frames.
    witness_masks = {f: _rect_mask(hw, bC) for f in (3, 4, 5)}
    o_witness = WindowOutcome(
        spec=WindowSpec(oid=3, seed_frame=0, entry_frame=3, trigger="witness"),
        state="clean", swap_to=None, close_frame=5,
        masks=witness_masks, probs={3: 0.7, 4: 0.7, 5: 0.7})

    results, renames = merge_selective_results(
        history, [o_swap, o_witness], cfg)

    # Exactly one rename -- the clean witness contributes none.
    assert [(e.frame, {e.a, e.b}) for e in renames] == [(3, {1, 2})]

    # Pre-entry: identity, all pure TBD.
    assert results[2][1]["box"] == bA
    assert results[2][1]["evidence"]["type"] == "tbd"
    assert results[2][2]["box"] == bB
    assert results[2][3]["box"] == bC and results[2][3]["mask"] is None

    # In-window [3, 5]: 1<->2 relabel in force AND both healthy masks landed.
    for f in (3, 4, 5):
        # Swapped oid1 now carries A's SAM mask on bB (the relabel already put
        # raw-oid2's box there; the SAM overlay agrees and adds the mask/prob).
        assert results[f][1]["box"] == bB
        assert results[f][1]["evidence"]["type"] == "selective_sam"
        assert results[f][1]["evidence"]["tracker_prob"] == 0.8
        assert np.array_equal(results[f][1]["mask"], swap_masks[f])
        # oid2 is the relabeled raw-oid1 box (bA), still pure TBD (no overlay).
        assert results[f][2]["box"] == bA
        assert results[f][2]["evidence"]["type"] == "tbd"
        assert results[f][2]["mask"] is None
        # Witness landed correctly: oid3 never relabeled, carries its own mask.
        assert results[f][3]["box"] == bC
        assert results[f][3]["evidence"]["type"] == "selective_sam"
        assert results[f][3]["evidence"]["tracker_prob"] == 0.7
        assert np.array_equal(results[f][3]["mask"], witness_masks[f])

    # Post-close: 1<->2 relabel persists; masks only inside the window.
    for f in (6, 7, 8):
        assert results[f][1]["box"] == bB
        assert results[f][1]["evidence"]["type"] == "tbd"
        assert results[f][2]["box"] == bA
        assert results[f][3]["box"] == bC and results[f][3]["mask"] is None


# --------------------------------------------------------------------------
# Task A: cluster_windows (episode merging) + SelectiveConfig.sam_budget_frac
# --------------------------------------------------------------------------
def _spec(oid, seed, entry, trigger="margin"):
    return WindowSpec(oid=oid, seed_frame=seed, entry_frame=entry,
                      trigger=trigger)


def test_selective_config_sam_budget_frac_default():
    # New field consumed by Task B's budget accounting; default is a no-op cap.
    assert SelectiveConfig().sam_budget_frac == 1.0
    assert SelectiveConfig(sam_budget_frac=0.5).sam_budget_frac == 0.5


def test_cluster_overlapping_intervals_one_episode():
    # Design 1 (a): padded intervals overlap -> a single episode.
    cfg = SelectiveConfig(min_streak=5)
    # A padded = [0-5, 10+5] = [-5, 15]; B padded = [8-5, 12+5] = [3, 17].
    eps = cluster_windows([_spec(1, 0, 10), _spec(2, 8, 12)], cfg)
    assert len(eps) == 1
    assert [w.oid for w in eps[0]] == [1, 2]


def test_cluster_far_apart_two_episodes():
    # Design 1 (b): non-overlapping padded intervals -> two episodes, sorted
    # by episode start (min seed).
    cfg = SelectiveConfig(min_streak=5)
    a = _spec(1, 0, 5)       # padded [-5, 10]
    b = _spec(2, 100, 105)   # padded [95, 110]
    eps = cluster_windows([a, b], cfg)
    assert len(eps) == 2
    assert [w.oid for w in eps[0]] == [1]
    assert [w.oid for w in eps[1]] == [2]


def test_cluster_padding_boundary():
    # Design 1 (c): pin the exact merge boundary and assert BOTH sides.
    # Point windows (seed == entry) at positions p and p + gap. Padded
    # intervals touch when gap == 2*min_streak (merge) and separate when
    # gap == 2*min_streak + 1.
    cfg = SelectiveConfig(min_streak=5)
    k = cfg.min_streak
    base = 20
    a = _spec(1, base, base)
    # gap == 2k: A padded hi = base + k; B padded lo = base + 2k - k = base + k
    #            -> lo == hi -> touch -> merged.
    merged = cluster_windows([a, _spec(2, base + 2 * k, base + 2 * k)], cfg)
    assert len(merged) == 1
    assert [w.oid for w in merged[0]] == [1, 2]
    # gap == 2k + 1: B padded lo = base + k + 1 > hi -> separate.
    split = cluster_windows(
        [a, _spec(2, base + 2 * k + 1, base + 2 * k + 1)], cfg)
    assert len(split) == 2


def test_cluster_witness_seed_before_trigger_same_episode():
    # Design 1 (d): a witness whose seed lands 2 frames before the trigger's
    # seed still shares the episode; episode start is the min seed.
    cfg = SelectiveConfig(min_streak=5)
    trigger = _spec(1, 10, 14, "margin")
    witness = _spec(2, 8, 14, "witness")     # seed 2 frames earlier
    eps = cluster_windows([trigger, witness], cfg)
    assert len(eps) == 1
    # sorted within episode by (seed, entry, oid): witness(seed 8) first.
    assert [w.oid for w in eps[0]] == [2, 1]
    assert min(w.seed_frame for w in eps[0]) == 8


def test_cluster_determinism_shuffled_input():
    # Design 1 (e): the function sorts internally, so a shuffled input yields
    # a byte-identical episode partition.
    cfg = SelectiveConfig(min_streak=5)
    specs = [
        _spec(3, 100, 104),
        _spec(1, 0, 4),
        _spec(2, 6, 9),        # padded [1, 14] touches spec1's [-5, 9]
        _spec(4, 200, 205),
    ]
    ref = [[w.oid for w in ep] for ep in cluster_windows(list(specs), cfg)]
    assert ref == [[1, 2], [3], [4]]
    for s in range(6):
        shuffled = list(specs)
        random.Random(s).shuffle(shuffled)
        got = [[w.oid for w in ep] for ep in cluster_windows(shuffled, cfg)]
        assert got == ref


def test_cluster_empty_input():
    assert cluster_windows([], SelectiveConfig(min_streak=5)) == []


# --------------------------------------------------------------------------
# Task A: ExitMachine per-window seed gating (Design 3)
# --------------------------------------------------------------------------
def test_exitmachine_pre_seed_frames_inert():
    # Design 3: a window is COMPLETELY inert for any frame before its seed.
    # Here entry_frame (3) is deliberately placed inside the pre-seed span so
    # the gate is observable: WITHOUT it the machine reaches the STALE check at
    # frame 3 (no mask seeded yet -> dark) and aborts; WITH it frames 0-4 are
    # skipped entirely and the window settles clean once the real seed mask
    # arrives at frame 5.
    cfg = SelectiveConfig(min_streak=2, margin_exit=0.10, ioma_exit=0.80,
                          area_degradation=0.25, border_margin_px=5)
    hw = (100, 100)
    box = (10, 40, 30, 60)
    history = {f: {2: _obs(box, margin=float("inf"))} for f in range(10)}
    spec = WindowSpec(oid=2, seed_frame=5, entry_frame=3, trigger="witness")
    em = ExitMachine([spec], {2: box}, hw, history, cfg)
    mask = _rect_mask(hw, box)
    for f in range(9):
        # oid2 has no SAM object until it is prompted at its own seed (frame 5).
        em.step(f, {2: (mask if f >= 5 else None)}, {2: 0.9})
    o = em.outcomes()[0]
    assert o.state == "clean"
    assert min(o.masks) >= spec.seed_frame       # nothing collected pre-seed


def test_exitmachine_episode_earlier_start_later_window_clean():
    # Design 2 + 3: one episode session hosting two windows whose seeds are 5
    # frames apart, fed from the earlier seed (episode start). The later window
    # must stay inert until its own seed and then settle clean; the pre-seed
    # frames add no dark / degraded pressure and trigger no kill against it.
    cfg = SelectiveConfig(min_streak=2, margin_exit=0.10, ioma_exit=0.80,
                          area_degradation=0.25, border_margin_px=5,
                          mask_overlap_exit=0.90, window_max_frames=120)
    hw = (100, 200)
    boxA = (10, 40, 30, 60)
    boxB = (150, 40, 170, 60)        # disjoint from A -> never kills it
    history = {f: {1: _obs(boxA, margin=float("inf")),
                   2: _obs(boxB, margin=float("inf"))} for f in range(14)}
    a = WindowSpec(oid=1, seed_frame=0, entry_frame=2, trigger="margin")
    b = WindowSpec(oid=2, seed_frame=5, entry_frame=7, trigger="margin")
    em = ExitMachine([a, b], {1: boxA, 2: boxB}, hw, history, cfg)
    maskA = _rect_mask(hw, boxA)
    maskB = _rect_mask(hw, boxB)
    for f in range(14):
        masks = {1: maskA}
        if f >= 5:                    # oid2 seeded only from its own seed frame
            masks[2] = maskB
        em.step(f, masks, {1: 0.9, 2: 0.9})
    outs = {o.spec.oid: o for o in em.outcomes()}
    assert outs[1].state == "clean"
    assert outs[2].state == "clean"
    assert min(outs[2].masks) >= b.entry_frame
