"""Starvation-triggered selective windows (blue-gap v2 fix #1).

Trigger: a TBD track never observed again, with a dark tail of at least
``starve_trigger`` frames to sequence end and an off-border last box, opens a
window at ``entry = last_seen`` via the normal backward seed walk. Pure
numpy/synthetic history; no GPU, no torch. Spec:
docs/superpowers/specs/2026-07-21-selective-starvation-windows-design.md
"""
import numpy as np

from web.field.selective_prop import (
    ExitMachine,
    FrameObs,
    SelectiveConfig,
    WindowSpec,
    find_windows,
    merge_selective_results,
)


def _obs(box, margin=float("inf"), color=None):
    return FrameObs(box=box, margin=margin, color=color)


def _rect_mask(hw, box):
    h, w = hw
    m = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = box
    m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
    return m


HW = (200, 300)                       # (H, W) for border checks
MID = (100, 80, 130, 110)             # comfortably off-border box


def _cfg(**kw):
    kw.setdefault("starve_windows", True)
    kw.setdefault("min_streak", 2)
    kw.setdefault("starve_trigger", 3)
    return SelectiveConfig(**kw)


# ---------------------------------------------------------------- trigger --
def test_starve_trigger_fires_on_vanished_track():
    cfg = _cfg()
    # oid 1 observed frames 0-4, then dark through frame 9 (tail 5 >= 3).
    history = {f: {1: _obs(MID)} for f in range(5)}
    wins = find_windows(history, [], 10, cfg, frame_hw=HW)
    assert len(wins) == 1
    w = wins[0]
    assert (w.oid, w.entry_frame, w.trigger) == (1, 4, "starve")
    # backward seed walk from entry-1=3: frames 3,2 clean -> seed 2.
    assert w.seed_frame == 2


def test_no_trigger_when_tail_too_short():
    cfg = _cfg(starve_trigger=6)
    history = {f: {1: _obs(MID)} for f in range(5)}    # tail = 9-4 = 5 < 6
    assert find_windows(history, [], 10, cfg, frame_hw=HW) == []


def test_no_trigger_when_last_box_near_border():
    cfg = _cfg()
    edge = (2, 80, 30, 110)                            # x1=2 < border_margin_px
    history = {f: {1: _obs(edge)} for f in range(5)}
    assert find_windows(history, [], 10, cfg, frame_hw=HW) == []


def test_no_trigger_when_knob_off():
    cfg = _cfg(starve_windows=False)
    history = {f: {1: _obs(MID)} for f in range(5)}
    assert find_windows(history, [], 10, cfg, frame_hw=HW) == []


def test_no_trigger_without_frame_hw():
    # frame_hw is required for the border check; without it the starve scan
    # must not fire at all (defensive: orchestrator always passes it).
    cfg = _cfg()
    history = {f: {1: _obs(MID)} for f in range(5)}
    assert find_windows(history, [], 10, cfg) == []


def test_mid_gap_and_terminal_tail_are_distinct_windows():
    cfg = _cfg(gap_trigger=3)
    # oid 1: frames 0-2, dark 3-6, back 7-8, dark 9-14 (never returns).
    history = {f: {1: _obs(MID)} for f in range(3)}
    history[7] = {1: _obs(MID)}
    history[8] = {1: _obs(MID)}
    wins = find_windows(history, [(1, 2, 7)], 15, cfg, frame_hw=HW)
    trig = {(w.trigger, w.entry_frame) for w in wins}
    assert ("gap", 2) in trig                          # mid-video gap
    assert ("starve", 8) in trig                       # terminal tail
    assert len(wins) == 2


def test_margin_beats_starve_on_same_entry():
    cfg = _cfg(margin_entry=0.05, margin_exit=0.10)
    history = {f: {1: _obs(MID)} for f in range(4)}
    history[4] = {1: _obs(MID, margin=0.02)}           # margin trigger, entry 4
    # frame 4 is also oid 1's last observation with a long tail.
    wins = find_windows(history, [], 12, cfg, frame_hw=HW)
    sel = [w for w in wins if w.oid == 1 and w.entry_frame == 4]
    assert len(sel) == 1
    assert sel[0].trigger == "margin"


def test_starve_trigger_spawns_no_witnesses():
    cfg = _cfg()
    far = (200, 150, 230, 180)                         # clear of MID
    near = (105, 85, 135, 115)                         # overlaps MID
    # Frames 0-3: separated (so BOTH oids have clean seed runs). Frame 4:
    # oid 2 overlaps oid 1's box at oid 1's LAST frame -- if starve triggers
    # (wrongly) spawned witnesses, oid 2 would get a witness window here
    # (its own seed walk succeeds off frames 3, 2). Then oid 1 vanishes;
    # oid 2 survives to the end (tail 0 -> no starve for it).
    history = {f: {1: _obs(MID), 2: _obs(far)} for f in range(4)}
    history[4] = {1: _obs(MID), 2: _obs(near)}
    history.update({f: {2: _obs(far)} for f in range(5, 10)})
    wins = find_windows(history, [], 10, cfg, frame_hw=HW)
    assert {(w.oid, w.trigger) for w in wins} == {(1, "starve")}


# ------------------------------------------------------------ ExitMachine --
def _run_machine(spec, cfg, history, mask_fn, n_frames, hw=HW):
    """Drive one window through ExitMachine; ``mask_fn(f) -> mask|None``."""
    machine = ExitMachine([spec], {spec.oid: history[spec.seed_frame][spec.oid].box},
                          hw, history, cfg)
    for f in range(spec.seed_frame, n_frames):
        m = mask_fn(f)
        machine.step(f, {} if m is None else {spec.oid: m}, {})
    return machine.outcomes()[0]


def test_starve_window_survives_area_shrink():
    """A mask shrinking to a sliver aborts a margin window as degraded but
    must NOT abort a starve window (that sliver is the whole point)."""
    cfg = _cfg(area_degradation=0.25, window_max_frames=50)
    history = {f: {1: _obs(MID)} for f in range(5)}    # vanishes after 4
    big = _rect_mask(HW, MID)                          # 900 px at seed
    sliver = _rect_mask(HW, (100, 80, 104, 110))       # 120 px < 25% of seed

    def masks(f):
        return big if f <= 4 else sliver

    starve = WindowSpec(oid=1, seed_frame=2, entry_frame=4, trigger="starve")
    oc = _run_machine(starve, cfg, history, masks, 20)
    assert oc.state == "end"                           # ran to sequence end
    assert set(oc.masks) == set(range(4, 20))          # sliver frames kept

    margin = WindowSpec(oid=1, seed_frame=2, entry_frame=4, trigger="margin")
    oc = _run_machine(margin, cfg, history, masks, 20)
    assert oc.state == "degraded"                      # existing behavior


def test_starve_window_uncapped_by_default():
    cfg = _cfg(window_max_frames=8, starve_max_frames=0)
    history = {f: {1: _obs(MID)} for f in range(5)}
    mask = _rect_mask(HW, MID)
    spec = WindowSpec(oid=1, seed_frame=2, entry_frame=4, trigger="starve")
    oc = _run_machine(spec, cfg, history, lambda f: mask, 30)
    assert oc.state == "end"
    assert oc.close_frame == 29                        # past seed+8, to the end


def test_starve_max_frames_caps_window():
    cfg = _cfg(window_max_frames=50, starve_max_frames=6)
    history = {f: {1: _obs(MID)} for f in range(5)}
    mask = _rect_mask(HW, MID)
    spec = WindowSpec(oid=1, seed_frame=2, entry_frame=4, trigger="starve")
    oc = _run_machine(spec, cfg, history, lambda f: mask, 30)
    assert oc.state == "end"
    assert oc.close_frame == 8                         # seed 2 + cap 6


def test_starve_window_still_degrades_dark():
    cfg = _cfg(min_streak=2)
    history = {f: {1: _obs(MID)} for f in range(5)}
    mask = _rect_mask(HW, MID)
    spec = WindowSpec(oid=1, seed_frame=2, entry_frame=4, trigger="starve")
    oc = _run_machine(spec, cfg, history, lambda f: mask if f <= 6 else None, 20)
    assert oc.state == "degraded"                      # true mask loss closes


def test_fresh_oid_settle_swaps_and_renames():
    """The vanished robot reappears as a NEW oid; the starve window's mask
    settles onto it -> swap -> rename merges the streams (the 2992 case)."""
    cfg = _cfg(min_streak=2, ioma_exit=0.8, margin_exit=0.10)
    history = {f: {1: _obs(MID)} for f in range(5)}    # oid 1, frames 0-4
    for f in range(10, 14):                            # fresh oid 7 reappears
        history[f] = {7: _obs(MID)}
    mask = _rect_mask(HW, (102, 82, 128, 108))         # inside MID box
    spec = WindowSpec(oid=1, seed_frame=2, entry_frame=4, trigger="starve")
    oc = _run_machine(spec, cfg, history, lambda f: mask, 14)
    assert oc.state == "swap" and oc.swap_to == 7
    results, renames = merge_selective_results(history, [oc], cfg)
    assert len(renames) == 1
    # The rename event fires from the fresh oid's FIRST appearance (frame 10),
    # not the vanished track's last-observation entry frame (4). Renaming from
    # entry would transpose raw oid 1's genuine last TBD box into a one-frame
    # phantom output oid 7 at the disappearance location.
    assert renames[0].frame == 10
    # No phantom at the entry frame: output oid 7 must NOT exist there, and
    # oid 1's entry is the mask overlay that carries it through the blind gap.
    assert 7 not in results[4]
    assert results[4][1]["evidence"]["type"] == "selective_sam"
    # post-rename, oid 1 owns the reappeared stream at frame 12.
    assert 1 in results[12] and 7 not in results[12]
