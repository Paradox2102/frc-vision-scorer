"""web.field.identity_swap: swap-aware, time-varying slot->team resolve.

Synthetic matches: 6 slots on separate lanes (3 m apart, smooth motion, no
gaps -> the swap gate is closed unless a test makes two slots touch or lose
their robots), one identity per slot, reads of weight 1.0 unless stated.
"""
from __future__ import annotations

import math
import random

import pytest

from web.field.identity_resolve import (
    assign_teams_global,
    build_slot_evidence,
    resolve_slot_labels,
)
from web.field.identity_swap import (
    SwapParams,
    discontinuity_intervals,
    format_swap_summary,
    pair_open_masks,
    resolve_slot_segments,
    segment_at,
    slot_position_samples,
)

BLUE = ["1111", "2222", "3333"]
RED = ["4444", "5555", "6666"]
STEP = 8
N = 200                       # frames 0 .. 1592
FPS = 60.0
P = SwapParams(fps=FPS)


def _alliance(s):
    return "blue" if s < 3 else "red"


def _lane(s, f):
    """Default lanes: x drifts slowly, y = 3 m apart per slot; R2 sits apart."""
    return [2.0 + 0.001 * f + (6.0 if s == 4 else 0.0), 1.0 + 3.0 * s]


def _log(pos_fn=_lane, n=N, iid_fn=None):
    rows = []
    for k in range(n):
        f = k * STEP
        for s in range(6):
            p = pos_fn(s, f)
            rows.append({"frame_num": f, "identity_id": iid_fn(s, f) if iid_fn else 100 + s,
                         "slot_id": s, "alliance": _alliance(s),
                         "pos": None if p is None else list(p)})
    return rows


def _read(s, f, team, w=1.0, iid=None):
    assert iid is not None or f % STEP == 0, f"read frame {f} is off the log grid"
    return {"type": "vlm_read", "frame": f, "robot_id": 100 + s if iid is None else iid,
            "matched_team": team, "vote_weight": w}


def _blank_meta():
    return {s: {"team_number": None, "confirmed": False, "alliance": _alliance(s)}
            for s in range(6)}


def _run(reads, log, params=P, blue=BLUE, red=RED, meta=None, votes=None, gallery=None):
    return resolve_slot_segments(reads, log, votes or {}, gallery or {}, _alliance,
                                 blue, red, meta if meta is not None else _blank_meta(),
                                 params=params)


def _gir(reads, log, blue=BLUE, red=RED, meta=None, votes=None, gallery=None):
    return resolve_slot_labels(reads, log, votes or {}, gallery or {}, _alliance,
                               blue, red, meta if meta is not None else _blank_meta())


def _labels(res, s):
    return [g["team_number"] for g in res["segments"][s]]


def _crossing(a, b, fc, half=6.0):
    """Lanes where slots a and b converge to the same point around frame fc
    (smoothly, never a jump > 3 m between frames) and part again."""
    def pos(s, f):
        x, y = _lane(s, f)
        if s in (a, b):
            mid = (1.0 + 3.0 * a + 1.0 + 3.0 * b) / 2.0
            t = min(abs(f - fc) / (half * FPS), 1.0)
            y = mid + (y - mid) * t
        return [x, y]
    return pos


def _swap_reads(a, b, ta, tb, n_each=4, split=N * STEP // 2):
    """Slot a reads ta in the first half and tb in the second; slot b the
    opposite (the slots exchanged robots around ``split``)."""
    lo = [split // 4 + 40 * i for i in range(n_each)]
    hi = [split + split // 2 + 40 * i for i in range(n_each)]
    return ([_read(a, f, ta) for f in lo] + [_read(b, f, tb) for f in lo]
            + [_read(a, f, tb) for f in hi] + [_read(b, f, ta) for f in hi])


# --------------------------------------------------------------------------- #
# Regression guarantee: no swap -> exactly today's gir output
# --------------------------------------------------------------------------- #
def test_no_swap_equals_gir():
    log = _log()
    reads = ([_read(0, 80 + 16 * i, "1111") for i in range(5)]
             + [_read(1, 400, "2222", 0.5), _read(1, 408, "1111", 0.5)]
             + [_read(3, 64 * i, "4444") for i in range(1, 6)]
             + [_read(4, 800, "5555", 1.25), _read(4, 1200, "6666", 2.5)]
             + [_read(5, 304, "4444"), _read(5, 904, "6666")]
             + [_read(5, 300, "6666", 2.5, iid=105)]            # off the log: dropped
             + [_read(2, 504, "4444")])                       # cross-alliance: ignored
    meta = _blank_meta()
    meta[2]["team_number"] = "3333"                           # greedy label kept on None
    gir = _gir(reads, log, meta=meta, votes={1: {"2222": 1.0}}, gallery={2: {"3333": 0.9}})
    res = _run(reads, log, meta=meta, votes={1: {"2222": 1.0}}, gallery={2: {"3333": 0.9}})
    assert res["swaps"] == []
    assert res["slot_meta"] == gir["slot_meta"]
    assert res["virtual"]["assignment"] == gir["assignment"]
    assert res["virtual"]["contested"] == gir["contested"]
    assert res["virtual"]["scores"] == gir["scores"]
    for s in range(6):
        segs = res["segments"][s]
        assert len(segs) == 1
        assert (segs[0]["start_frame"], segs[0]["end_frame"]) == (0, (N - 1) * STEP)
        assert segs[0]["team_number"] == gir["slot_meta"][s]["team_number"]
        assert segs[0]["confirmed"] == bool(gir["slot_meta"][s]["confirmed"])


@pytest.mark.parametrize("seed", range(12))
def test_infinite_lambda_equals_assign_teams_global(seed):
    """lambda = inf (or no swap possible) must reproduce assign_teams_global
    exactly, even on evidence that WOULD swap with a finite lambda and an
    open gate (no positions)."""
    rng = random.Random(seed)
    log = _log(pos_fn=lambda s, f: None)                      # gate fully open
    reads = []
    for _ in range(40):
        s = rng.randrange(6)
        roster = BLUE if s < 3 else RED
        reads.append(_read(s, STEP * rng.randrange(N), rng.choice(roster),
                           rng.choice([0.5, 1.0, 1.25, 2.5])))
    reads += _swap_reads(3, 5, "4444", "6666", n_each=6)
    ev, _ = build_slot_evidence(reads, log, {}, {}, _alliance)
    contested: set = set()
    want = assign_teams_global(ev, BLUE, RED, contested_out=contested)
    res = _run(reads, log, params=SwapParams(swap_lambda=math.inf, fps=FPS))
    assert res["swaps"] == []
    assert res["virtual"]["assignment"] == want
    assert res["virtual"]["contested"] == contested
    finite = _run(reads, log, params=SwapParams(swap_lambda=3.0, fps=FPS))
    assert finite["swaps"], "the planted R1/R3 swap is found once lambda is finite"


# --------------------------------------------------------------------------- #
# Swaps
# --------------------------------------------------------------------------- #
def test_clean_swap_gives_two_segments_per_slot():
    split = N * STEP // 2                                      # 800
    log = _log(pos_fn=_crossing(3, 5, split))
    reads = _swap_reads(3, 5, "4444", "6666") + [_read(4, f, "5555") for f in (104, 904)]
    res = _run(reads, log)
    assert [(s["alliance"], s["slots"]) for s in res["swaps"]] == [("red", (3, 5))]
    sw = res["swaps"][0]
    assert sw["support"] == pytest.approx(8.0)                # 8 reads each side
    # last old-label read at 320, first new-label read at 1200; the gate is
    # open only around the crossing at 800 (+- 6 s convergence ramp is the
    # proximity window), and the swap is centred in the read-free window.
    assert 320 < sw["frame"] <= 1200
    assert abs(sw["frame"] - split) <= 2 * FPS
    assert _labels(res, 3) == ["4444", "6666"]
    assert _labels(res, 5) == ["6666", "4444"]
    assert _labels(res, 4) == ["5555"]
    r1 = res["segments"][3]
    assert r1[0]["end_frame"] + STEP == r1[1]["start_frame"] == sw["frame"]
    assert r1[0]["score"] == pytest.approx(4.0) and r1[1]["score"] == pytest.approx(4.0)
    # gir alone can only give each slot one whole-match label: a tie here.
    gir = _gir(reads, log)
    assert gir["slot_meta"][3]["team_number"] is None and 3 in gir["contested"]
    line = format_swap_summary(res)
    assert "red: R1<->R3 @f" in line and "blue: none" in line
    assert "R1=4444[0-" in line and ">6666[" in line


def test_single_stray_read_cannot_flip_a_segment():
    log = _log(pos_fn=lambda s, f: None)                      # gate fully open
    reads = ([_read(3, 80 * i, "4444") for i in range(1, 20)]
             + [_read(5, 80 * i + 8, "6666") for i in range(1, 20)]
             + [_read(3, 1560, "6666", 2.5)]                   # heaviest read, at the edge
             + [_read(5, 808, "4444", 2.5)])                   # and one mid-match
    res = _run(reads, log)
    assert res["swaps"] == []
    assert _labels(res, 3) == ["4444"] and _labels(res, 5) == ["6666"]
    # ... while three consistent reads at the edge (> lambda = 3) do flip it.
    more = reads + [_read(3, 1568, "6666", 1.0), _read(5, 1576, "4444", 1.0)]
    res2 = _run(more, log)
    assert [s["slots"] for s in res2["swaps"]] == [(3, 5)]


def test_swap_without_plausibility_is_disallowed():
    reads = _swap_reads(3, 5, "4444", "6666")
    far = _run(reads, _log())                                  # lanes never meet, no gaps
    assert far["swaps"] == []
    assert len(far["segments"][3]) == 1 and len(far["segments"][5]) == 1
    assert far["stats"]["gate_open_frac"]["R1-R3"] == 0.0
    # Same reads, the two slots touch around f800 -> swap allowed.
    near = _run(reads, _log(pos_fn=_crossing(3, 5, 800)))
    assert [s["slots"] for s in near["swaps"]] == [(3, 5)]
    # One slot losing its robot is not enough: both must be disrupted.
    def one_lost(s, f):
        return None if (s == 3 and 760 <= f <= 840) else _lane(s, f)
    assert _run(reads, _log(pos_fn=one_lost))["swaps"] == []

    def both_lost(s, f):
        return None if (s in (3, 5) and 760 <= f <= 840) else _lane(s, f)
    lost = _run(reads, _log(pos_fn=both_lost))
    assert [s["slots"] for s in lost["swaps"]] == [(3, 5)]
    assert 760 - P.window_s * FPS <= lost["swaps"][0]["frame"] <= 840 + P.window_s * FPS


def test_teleport_counts_as_losing_the_robot():
    reads = _swap_reads(3, 5, "4444", "6666")

    def teleport(s, f):                                        # R1/R3 exchange lanes at f800
        x, y = _lane(s, f)
        if f >= 800 and s in (3, 5):
            y = _lane(8 - s, f)[1]
        return [x, y]
    res = _run(reads, _log(pos_fn=teleport))
    assert [s["slots"] for s in res["swaps"]] == [(3, 5)]


def test_three_slots_with_missing_reads_and_short_roster():
    """B2 is never read; B1 reads 1111 early, B3 reads it late (B1 lost the
    robot, B3 picked it up). Red roster has only 2 teams (padding)."""
    def pos(s, f):
        return _crossing(0, 2, 800)(s, f)
    log = _log(pos_fn=pos)
    reads = ([_read(0, 104 + 40 * i, "1111") for i in range(4)]
             + [_read(2, 1104 + 40 * i, "1111") for i in range(4)]
             + [_read(3, 200, "4444"), _read(3, 1304, "4444"), _read(4, 600, "5555")])
    res = _run(reads, log, red=["4444", "5555"])
    assert [(s["alliance"], s["slots"]) for s in res["swaps"]] == [("blue", (0, 2))]
    assert _labels(res, 0) == ["1111", None]
    assert _labels(res, 2) == [None, "1111"]
    assert _labels(res, 1) == [None]
    assert _labels(res, 3) == ["4444"] and _labels(res, 4) == ["5555"] and _labels(res, 5) == [None]
    # gir has to pick one slot for 1111 (tie -> neither); segments get both halves.
    gir = _gir(reads, log, red=["4444", "5555"])
    assert gir["slot_meta"][0]["team_number"] is None and gir["slot_meta"][2]["team_number"] is None


def test_three_cycle_is_two_transpositions():
    split = 800

    def pos(s, f):                                             # all three red slots meet
        x, y = _lane(s, f)
        if s >= 3:
            t = min(abs(f - split) / (6.0 * FPS), 1.0)
            x, y = 2.0 + (x - 2.0) * t, 13.0 + (y - 13.0) * t
        return [x, y]
    lo = [104 + 40 * i for i in range(4)]
    hi = [1104 + 40 * i for i in range(4)]
    reads = ([_read(3, f, "4444") for f in lo] + [_read(4, f, "5555") for f in lo]
             + [_read(5, f, "6666") for f in lo]
             + [_read(3, f, "5555") for f in hi] + [_read(4, f, "6666") for f in hi]
             + [_read(5, f, "4444") for f in hi])
    res = _run(reads, _log(pos_fn=pos))
    # ONE move of two transpositions (the permutation between them has no
    # reads) applied atomically on one frame, centred in the read-free
    # window after the last old read (224) and up to the first new one (1104).
    assert len(res["swaps"]) == 1
    sw = res["swaps"][0]
    assert sw["alliance"] == "red" and sw["slots"] == (3, 4, 5) and len(sw["moves"]) == 2
    assert sw["support"] == pytest.approx(12.0)
    assert abs(sw["frame"] - (232 + 1104) / 2) <= 2 * STEP
    for s, ab in ((3, ["4444", "5555"]), (4, ["5555", "6666"]), (5, ["6666", "4444"])):
        assert _labels(res, s) == ab
        assert res["segments"][s][1]["start_frame"] == sw["frame"]
    assert "R1<->" in format_swap_summary(res) and "+R" in format_swap_summary(res)


def test_unattributed_and_off_roster_reads_are_ignored_like_gir():
    log = _log(pos_fn=lambda s, f: None)
    reads = (_swap_reads(3, 5, "4444", "6666")
             + [_read(3, 1504, "9999", 2.5), _read(5, 1504, "1111", 2.5),
                {"type": "vlm_read", "frame": 7, "robot_id": 103, "matched_team": "6666",
                 "vote_weight": 5.0}])                         # frame 7 not on the log
    res = _run(reads, log)
    assert [s["slots"] for s in res["swaps"]] == [(3, 5)]
    assert res["virtual"]["stats"]["reads_dropped_misattributed"] == 1


# --------------------------------------------------------------------------- #
# Gate helpers
# --------------------------------------------------------------------------- #
def test_discontinuity_intervals():
    p = SwapParams(fps=10.0, gap_s=1.0, jump_m=3.0)
    samples = [(0, 0.0, 0.0), (5, 0.5, 0.0), (20, 1.0, 0.0), (21, 5.0, 0.0), (22, 5.1, 0.0)]
    # gap 5->20 (1.5 s), jump 20->21 (4 m), trailing gap 22->40
    assert discontinuity_intervals(samples, 0, 40, p) == [(5, 20), (20, 21), (22, 40)]
    assert discontinuity_intervals([], 0, 40, p) == [(0, 40)]


def test_pair_open_masks_proximity_window():
    p = SwapParams(fps=10.0, window_s=1.0, prox_m=1.5)
    grid = list(range(0, 100, 2))
    pos = {0: [(f, 0.0, 0.0) for f in grid],
           1: [(f, 0.0, 1.0 if f == 50 else 5.0) for f in grid]}  # touches at f50 only
    # f50's 4 m excursion is a "jump" for slot 1 but slot 0 never jumps
    m = pair_open_masks(grid, (0, 1), pos, p)[(0, 1)]
    open_frames = [f for f, o in zip(grid, m) if o]
    assert open_frames == list(range(40, 61, 2))


def test_slot_position_samples_first_finite_wins():
    log = [{"frame_num": 8, "identity_id": 1, "slot_id": 0, "pos": [1, 2]},
           {"frame_num": 8, "identity_id": 2, "slot_id": 0, "pos": [9, 9]},
           {"frame_num": 16, "identity_id": 1, "slot_id": 0, "pos": [float("nan"), 0]},
           {"frame_num": 16, "identity_id": 3, "slot_id": 1, "pos": None}]
    assert slot_position_samples(log) == {0: [(8, 1.0, 2.0)]}


def test_segment_at():
    segs = [{"start_frame": 0, "end_frame": 96, "team_number": "a"},
            {"start_frame": 104, "end_frame": 200, "team_number": "b"}]
    assert segment_at(segs, -5)["team_number"] == "a"
    assert segment_at(segs, 96)["team_number"] == "a"
    assert segment_at(segs, 100)["team_number"] == "a"
    assert segment_at(segs, 104)["team_number"] == "b"
    assert segment_at(segs, 999)["team_number"] == "b"
    assert segment_at([], 5) is None and segment_at(None, 5) is None


def test_empty_log_is_harmless():
    res = _run([_read(0, 8, "1111")], [])
    assert res["swaps"] == [] and all(v == [] for v in res["segments"].values())
