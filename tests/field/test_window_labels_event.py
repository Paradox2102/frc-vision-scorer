"""Event lineup (docs/superpowers/specs/2026-11-10-event-lineup-design.md) in
the window-label stage: knob-off golden snapshot, store_anchors, borrowed
rows, elimination anchors, cards, anchors out, and the digit-call cache.
Fakes only: no network, no video, no GPU."""
import hashlib
import json
import random

import pytest
from pathlib import Path

from web.field import event_anchors as ea
from web.field import window_labels as wl
from tests.field.test_window_labels import (
    FPS, FakeSession, _frames, _lineup_frames, _lineup_session, _run, _still)

GOLDEN = Path(__file__).parent / "data" / "window_labels_golden.json"


def _snap(frames, session, **kw):
    """What the knob-off guarantee covers (labels, segments, stats, card
    JPEG) plus every lineup payload, as JSON-ready values."""
    sent = []
    real_ask = session.ask

    def ask(items, question):
        if question == "who":
            sent.extend(json.dumps(p, sort_keys=True) for p, _sizes in items)
        return real_ask(items, question)

    session.ask = ask
    res = _run(frames, session, **kw)
    labels = sorted([i, slot] + (list(e) if e else [None])
                    for (i, slot), e in res["labels"].items())
    return {"labels": labels,
            "segments": {str(k): v for k, v in res["segments"].items()},
            "stats": res["stats"],
            "card_sha256": (hashlib.sha256(res["lineup_jpeg"]).hexdigest()
                            if res["lineup_jpeg"] else None),
            "who_sha256": [hashlib.sha256(p.encode()).hexdigest() for p in sent]}


def _golden(**kw):
    lineup = _lineup_frames(second_unread=True)
    digits = _frames({"B1": _still(range(20)), "B2": _still(range(20), (9.0, 6.0))}, 20)
    return json.loads(json.dumps({
        "lineup": _snap(lineup, _lineup_session(lineup, who_p={"A": 0.1, "B": 0.85,
                                                               "unsure": 0.05}),
                        seed=1, **kw),
        "digits_only": _snap(digits, FakeSession(team_p={"3128": 0.9, "unreadable": 0.1}),
                             **kw)}))


def test_knob_off_matches_the_golden_snapshot():
    assert _golden() == json.loads(GOLDEN.read_text())


def test_store_anchors_keeps_single_good_reads_and_spreads_in_time():
    labels = {i: ("2724", 0.9) for i in range(10)}
    labels.update({20: ("6497", 0.95), 21: ("3128", 0.9), 22: ("3128", 0.86), 23: ("2724", 0.84)})
    t0 = {i: float(i) for i in labels}
    s = wl.store_anchors(labels, t0)
    assert s["6497"] == [20]                       # one good read is kept for the store
    assert s["3128"] == [21, 22] and s["2724"] == [0, 2, 4, 5, 7, 9]
    assert 23 not in s["2724"]                     # 0.84 is below the anchor floor
    # pick_anchors (the knob-off card) is unchanged: it still needs two
    assert wl.pick_anchors(labels, t0) == {t: w for t, w in s.items() if len(w) >= wl.MIN_ANCHORS}


def test_best_tile_k_is_the_sample_whose_wide_call_rated_the_team_highest():
    tiles = {7: {1: "t1", 2: "t2", 3: "t3"}}
    wide_p = {7: {1: {"3128": 0.6}, 2: {"3128": 0.9}, 3: {"3128": 0.9}}}
    assert wl.best_tile_k(wide_p, tiles, 7, "3128") == 2      # first of the tied best
    assert wl.best_tile_k({}, {7: {3: "a", 2: "b"}}, 7, "3128") == 2   # no wide call: lowest k


def _b(p=0.9, mk="2026cur_qm3", fn=100, colour="blue", kind="digits", team="6773"):
    return ea.Anchor(team=team, kind=kind, p=p, frame_num=fn, alliance_color=colour, tile=b"",
                     match_key=mk, match_number=ea.schedule_rank(mk))


def _keys(picked):
    return [(a.kind, a.alliance_color, a.p, a.match_key, a.frame_num) for a in picked]


def test_borrowed_anchors_rank_digits_colour_p_recency_then_frame():
    cands = [_b(p=0.9, mk="2026cur_qm3", colour="red"), _b(p=0.86, mk="2026cur_qm5"),
             _b(p=0.95, mk="2026cur_qm2"), _b(p=0.9, mk="2026cur_qm7"),
             _b(p=0.9, mk="2026cur_qm7", fn=50), _b(p=0.99, mk="2026cur_qm9", kind="elimination")]
    assert _keys(wl.fill_row(0, cands, "blue")) == [
        ("digits", "blue", 0.95, "2026cur_qm2", 100),
        ("digits", "blue", 0.9, "2026cur_qm7", 50),
        ("digits", "blue", 0.9, "2026cur_qm7", 100),
        ("digits", "blue", 0.86, "2026cur_qm5", 100),
        ("digits", "red", 0.9, "2026cur_qm3", 100)]


def test_more_recent_match_wins_a_tie_on_p():
    got = wl.fill_row(4, [_b(mk="2026cur_qm3"), _b(mk="2026cur_qm7")], "blue")
    assert [a.match_key for a in got] == ["2026cur_qm7", "2026cur_qm3"]


def test_at_most_two_anchors_come_from_one_source_match():
    got = wl.fill_row(0, [_b(fn=n) for n in range(5)], "blue")
    assert [a.frame_num for a in got] == [0, 1]


def test_the_row_stops_at_six_counting_own_anchors():
    cands = [_b(mk=f"2026cur_qm{n}") for n in range(1, 9)]
    assert len(wl.fill_row(0, cands, "blue")) == 6
    assert len(wl.fill_row(4, cands, "blue")) == 2
    assert wl.fill_row(6, cands, "blue") == []


def test_elimination_tiles_only_when_fewer_than_two_digit_anchors():
    elim = [_b(kind="elimination", mk=f"2026cur_qm{n}") for n in (1, 2, 3)]
    assert [a.kind for a in wl.fill_row(0, elim, "blue")] == ["elimination"] * 3
    assert len(wl.fill_row(1, elim, "blue")) == 3          # one own digit anchor: still < 2
    got = wl.fill_row(1, elim + [_b(mk="2026cur_qm9")], "blue")
    assert [a.kind for a in got] == ["digits"]             # own + borrowed digits = 2
    assert wl.fill_row(2, elim, "blue") == []
    assert wl.fill_row(0, elim, "blue", use_elimination=False) == []


def test_ranking_is_deterministic_whatever_the_input_order():
    cands = [_b(p=p, mk=f"2026cur_qm{m}", fn=f, colour=c)
             for p in (0.86, 0.9) for m in (3, 4) for f in (10, 20) for c in ("red", "blue")]
    want = _keys(wl.fill_row(0, cands, "red"))
    for seed in range(5):
        random.Random(seed).shuffle(cands)
        assert _keys(wl.fill_row(0, cands, "red")) == want


RED, BLUE = ["2637", "2724", "6497"], ["3128", "6773", "9066"]
PERIOD = 0.2                                     # 12 source frames at 60 fps


def _quiet(n=9, **extra):
    """W3's calls: unreadable, no roster team near 0.4."""
    p = {"unreadable": 0.85, "not_robot": 0.05, "6497": 0.1}
    p.update(extra)
    return [dict(p) for _ in range(n)]


def _wides(not_robot=(0.05, 0.05, 0.05, 0.05)):
    return {k: {"unreadable": 0.85, "not_robot": nr} for k, nr in enumerate(not_robot, 1)}


def _base(r3_xy=(10.0, 2.0), r2_frames=range(20), b1=None):
    """R1 reads 2637 and R2 reads 2724 (p 0.95 / 0.9) for 4 s; R3 is unread.
    Returns frames, windows, {slot: window}, digit, calls, wide_p."""
    tracks = {"R1": _still(range(20), (2.0, 2.0)), "R2": _still(r2_frames, (6.0, 2.0)),
              "R3": _still(range(20), r3_xy)}
    if b1 is not None:
        tracks["B1"] = _still(b1, (4.0, 6.0))
    frames = _frames(tracks, 20)
    ws = wl.build_windows(frames, FPS)
    w = {x.slot: x for x in ws}
    digit = {w["R1"].wid: ("2637", 0.95), w["R2"].wid: ("2724", 0.9)}
    return frames, ws, w, digit, {w["R3"].wid: _quiet()}, {w["R3"].wid: _wides()}


def _elim(frames, ws, digit, calls, wide_p, red=RED):
    return wl.elimination_anchors(ws, frames, digit, calls, wide_p, red, BLUE, PERIOD)


def test_both_partners_named_gives_three_tiles_for_the_third_team():
    frames, ws, w, digit, calls, wide_p = _base()
    r3 = w["R3"].wid
    assert _elim(frames, ws, digit, calls, wide_p) == {
        "6497": [(r3, 1, 0.9), (r3, 3, 0.9), (r3, 4, 0.9)]}


def test_guard_1_partner_overlap_under_one_second():
    frames, ws, w, digit, calls, wide_p = _base(r2_frames=range(16, 20))   # 4 shared = 0.8 s
    assert _elim(frames, ws, digit, calls, wide_p) == {}


@pytest.mark.parametrize("change", ["partner_p", "same_team", "other_alliance", "w3_named"])
def test_guard_1_and_2_partner_labels(change):
    frames, ws, w, digit, calls, wide_p = _base()
    if change == "partner_p":
        digit[w["R2"].wid] = ("2724", 0.84)
    elif change == "same_team":
        digit[w["R2"].wid] = ("2637", 0.95)
    elif change == "other_alliance":
        digit[w["R2"].wid] = ("3128", 0.95)
    else:
        digit[w["R3"].wid] = ("6497", 0.6)
    assert _elim(frames, ws, digit, calls, wide_p) == {}


def test_guard_2_a_roster_team_at_0_4_in_any_call():
    frames, ws, w, digit, calls, wide_p = _base()
    calls[w["R3"].wid][4] = {"unreadable": 0.55, "6497": 0.4, "not_robot": 0.05}
    assert _elim(frames, ws, digit, calls, wide_p) == {}


def test_guard_2_cross_alliance_mass():
    frames, ws, w, digit, calls, wide_p = _base()
    r3 = w["R3"].wid
    calls[r3] = _quiet(**{"6497": 0.05, "3128": 0.3})        # blue mass 2.7 > red 0.45
    assert _elim(frames, ws, digit, calls, wide_p) == {}
    calls[r3] = _quiet(**{"6497": 0.2, "3128": 0.15})        # red 1.8 >= blue 1.35
    assert set(_elim(frames, ws, digit, calls, wide_p)) == {"6497"}


def test_guard_3_the_third_team_read_on_another_slot_meanwhile():
    frames, ws, w, digit, calls, wide_p = _base(b1=range(5, 15))
    digit[w["B1"].wid] = ("6497", 0.7)
    assert _elim(frames, ws, digit, calls, wide_p) == {}
    del digit[w["B1"].wid]
    assert set(_elim(frames, ws, digit, calls, wide_p)) == {"6497"}


def test_guard_4_w3_too_close_to_a_partner():
    frames, ws, w, digit, calls, wide_p = _base(r3_xy=(2.5, 2.4))          # 0.64 m from R1
    assert _elim(frames, ws, digit, calls, wide_p) == {}


@pytest.mark.parametrize("wides", [
    {1: {"not_robot": 0.05}, 2: {"not_robot": 0.05}},                     # 2 returned calls
    _wides((0.05, 0.31, 0.0, 0.0))])                                       # one says not_robot
def test_guard_5_returned_wide_calls(wides):
    frames, ws, w, digit, calls, _wide_p = _base()
    assert _elim(frames, ws, digit, calls, {w["R3"].wid: wides}) == {}


def test_an_alliance_without_three_roster_teams_gives_nothing():
    frames, ws, w, digit, calls, wide_p = _base()
    assert _elim(frames, ws, digit, calls, wide_p, red=RED[:2]) == {}


def test_at_most_three_elimination_tiles_per_team_spread_in_time():
    tracks = {"R1": _still(range(40), (2.0, 2.0)), "R2": _still(range(40), (6.0, 2.0)),
              "R3": _still(range(40), (10.0, 2.0))}
    frames = _frames(tracks, 40)                     # 8 s: two 4 s windows per slot
    ws = wl.build_windows(frames, FPS)
    digit, calls, wide_p = {}, {}, {}
    for x in ws:
        if x.slot == "R1":
            digit[x.wid] = ("2637", 0.95)
        elif x.slot == "R2":
            digit[x.wid] = ("2724", 0.9)
        else:
            calls[x.wid], wide_p[x.wid] = _quiet(), _wides()
    a, b = sorted(x.wid for x in ws if x.slot == "R3")
    assert _elim(frames, ws, digit, calls, wide_p) == {
        "6497": [(a, 1, 0.9), (a, 4, 0.9), (b, 4, 0.9)]}
