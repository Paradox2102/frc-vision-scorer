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
