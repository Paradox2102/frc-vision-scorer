"""Tests for the selective-mask-propagation window finder, exit state machine,
and sam2_raw merge (Task 3).

Pure numpy / synthetic boxes and masks; no GPU, no torch. Masks are small
``np.zeros((H, W), bool)`` rectangles built to match / mismatch the TBD boxes
that live in a hand-constructed ``history`` dict.
"""
import numpy as np

from web.field.selective_prop import (
    ExitMachine,
    FrameObs,
    RenameEvent,
    SelectiveConfig,
    WindowOutcome,
    WindowSpec,
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
