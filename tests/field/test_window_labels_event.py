"""Event lineup (docs/superpowers/specs/2026-11-10-event-lineup-design.md) in
the window-label stage: knob-off golden snapshot, store_anchors, borrowed
rows, elimination anchors, cards, anchors out, and the digit-call cache.
Fakes only: no network, no video, no GPU."""
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from web import jev_reader

from web.field import event_anchors as ea
from web.field import window_crops as wc
from web.field import window_labels as wl
from tests.field.test_window_labels import (
    BySlot, FPS, FakeSession, _expected_letters, _frames, _lineup_frames, _lineup_session, _run, _still)

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


def test_identical_anchors_rank_the_same_whatever_the_input_order():
    a = ea.Anchor(team="6773", kind="digits", p=0.9, frame_num=100, alliance_color="blue",
                  tile=b"", match_key="2026cur_qm3", match_number=3, id=11)
    b = ea.Anchor(team="6773", kind="digits", p=0.9, frame_num=100, alliance_color="blue",
                  tile=b"", match_key="2026cur_qm3", match_number=3, id=12)
    assert wl.fill_row(5, [a, b], "blue") == wl.fill_row(5, [b, a], "blue") == [a]


def test_guard_3_third_team_read_on_another_slot_after_w3_is_ignored():
    tracks = {"R1": _still(range(20), (2.0, 2.0)), "R2": _still(range(20), (6.0, 2.0)),
              "R3": _still(range(20), (10.0, 2.0)), "B1": _still(range(20, 30), (4.0, 6.0))}
    frames = _frames(tracks, 30)
    ws = wl.build_windows(frames, FPS)
    w = {x.slot: x for x in ws}
    assert not set(w["B1"].idx) & set(w["R3"].idx)
    digit = {w["R1"].wid: ("2637", 0.95), w["R2"].wid: ("2724", 0.9),
             w["B1"].wid: ("6497", 0.7)}
    out = _elim(frames, ws, digit, {w["R3"].wid: _quiet()}, {w["R3"].wid: _wides()})
    assert set(out) == {"6497"}


@pytest.mark.parametrize("case, passes", [
    ("mass_equal", True), ("not_robot_at_0.3", False), ("overlap_1s", True),
    ("partner_p_0.85", True)])
def test_guard_boundaries(case, passes):
    kw = {}
    if case == "overlap_1s":
        kw["r2_frames"] = range(15, 20)                      # 5 shared frames x 0.2 s
    frames, ws, w, digit, calls, wide_p = _base(**kw)
    r3 = w["R3"].wid
    if case == "mass_equal":
        calls[r3] = _quiet(**{"6497": 0.1, "3128": 0.1})     # red == blue
    elif case == "not_robot_at_0.3":
        wide_p[r3] = _wides((0.05, 0.3, 0.05, 0.05))
    elif case == "partner_p_0.85":
        digit[w["R2"].wid] = ("2724", 0.85)
    assert bool(_elim(frames, ws, digit, calls, wide_p)) is passes


def _reading_session(frames, read, who_p=None):
    """Digit answers per window, one per request in request order: ``read``
    is slot -> probs (other slots unreadable) or a function of the window.
    Lineup answers are ``who_p``."""
    s = BySlot(who_p=who_p or {})
    plan = []
    for w in wl.build_windows(frames, FPS):
        probs = read(w) if callable(read) else read.get(w.slot, {"unreadable": 0.9})
        plan += [probs] * (2 * len(w.samples) + 1)
    s.plan = plan
    return s


def _borrow(team, n, mk="2026cur_qm3", colour="blue", kind="digits", shade=90):
    tile = wc.jpeg(np.full((240, 240, 3), shade, np.uint8))
    return [ea.Anchor(team=team, kind=kind, p=0.9, frame_num=100 + j, alliance_color=colour,
                      tile=tile, match_key=mk, match_number=ea.schedule_rank(mk), replay_id="r")
            for j in range(n)]


def test_knob_off_ignores_event_inputs_and_adds_no_stats_keys():
    kw = dict(borrowed={"2637": _borrow("2637", 2, colour="red")}, this_colors={"2637": "red"},
              use_elimination=False)
    assert _golden(**kw) == json.loads(GOLDEN.read_text())


def test_knob_off_returns_no_anchors_and_no_row_sources_but_the_digit_calls():
    frames = _lineup_frames()
    res = _run(frames, _lineup_session(frames, who_p={"B": 0.9}), seed=1)
    assert res["anchors_out"] == [] and res["row_sources"] == {}
    assert res["lineup_jpeg_blue"] is None
    assert {"calls", "wide_p"} <= set(res["digit_calls"])


def _letter_of(team, seed=1):
    return next(l for l, t in _expected_letters(seed).items() if t == team)


def test_a_team_never_read_borrows_a_row_and_the_lineup_pass_runs():
    # 3128 is read on B1 in four runs; 6773 (B2) never. Knob off: one row and
    # no lineup pass. Knob on with two borrowed 6773 anchors: two rows.
    frames = _lineup_frames()
    who = {_letter_of("6773"): 0.9}
    off = _run(frames, _reading_session(frames, {"B1": {"3128": 0.9}}, who), seed=1)
    assert off["stats"]["lineup_rows"] == 1 and off["stats"]["lineup_named"] == 0
    s = _reading_session(frames, {"B1": {"3128": 0.9}}, who)
    on = _run(frames, s, seed=1, event_lineup=True, borrowed={"6773": _borrow("6773", 2)})
    st = on["stats"]
    assert st["lineup_rows"] == 2 and (st["rows_own"], st["rows_borrowed"], st["rows_elim"]) == (1, 1, 0)
    assert on["row_sources"]["6773"] == {"own": 0, "borrowed": 2, "elim": 0,
                                         "from": ["2026cur_qm3"]}
    assert on["row_sources"]["3128"]["own"] == 4
    assert ("who", 5) in s.asked                         # four B2 windows + R1
    b2 = [w for w in wl.build_windows(frames, FPS) if w.slot == "B2"]
    assert {on["labels"][(w.idx[0], "B2")][:2] for w in b2} == {("6773", wl.RANK_LINEUP)}


def test_a_borrowed_row_needs_two_anchors_counting_own():
    frames = _lineup_frames()
    res = _run(frames, _reading_session(frames, {"B1": {"3128": 0.9}}), seed=1,
               event_lineup=True, borrowed={"6773": _borrow("6773", 1)})
    assert res["stats"]["lineup_rows"] == 1 and "6773" not in res["row_sources"]


def test_one_own_read_plus_one_borrowed_anchor_makes_a_row():
    frames = _lineup_frames()
    first_b2 = min(w.wid for w in wl.build_windows(frames, FPS) if w.slot == "B2")

    def read(w):
        if w.slot == "B1":
            return {"3128": 0.9}
        return {"6773": 0.9} if w.wid == first_b2 else {"unreadable": 0.9}

    res = _run(frames, _reading_session(frames, read), seed=1, event_lineup=True,
               borrowed={"6773": _borrow("6773", 1)})
    assert res["row_sources"]["6773"] == {"own": 1, "borrowed": 1, "elim": 0,
                                          "from": ["2026cur_qm3"]}
    assert res["stats"]["lineup_rows"] == 2


def test_anchors_out_are_this_matchs_confident_digit_windows():
    frames = _lineup_frames()
    res = _run(frames, _lineup_session(frames, who_p={"B": 0.9}), seed=1, event_lineup=True)
    out = res["anchors_out"]
    ws = wl.build_windows(frames, FPS)
    assert sorted(a.team for a in out) == ["3128"] * 4 + ["6773"] * 4
    assert {(a.kind, a.alliance_color, a.p) for a in out} == {("digits", "blue", 0.9)}
    b1_starts = sorted(frames[w.samples[0]]["frame_num"] for w in ws if w.slot == "B1")
    assert sorted(a.frame_num for a in out if a.team == "3128") == b1_starts
    assert all(a.tile[:2] == b"\xff\xd8" and a.match_key is None for a in out)
    st = res["stats"]
    assert (st["rows_own"], st["rows_borrowed"], st["rows_elim"]) == (2, 0, 0)


def test_elimination_anchors_reach_anchors_out_and_obey_the_ablation():
    frames = _frames({"R1": _still(range(20), (2.0, 2.0)), "R2": _still(range(20), (6.0, 2.0)),
                      "R3": _still(range(20), (10.0, 2.0))}, 20)
    read = {"R1": {"2637": 0.95}, "R2": {"2724": 0.9},
            "R3": {"unreadable": 0.85, "not_robot": 0.05, "6497": 0.1}}
    res = _run(frames, _reading_session(frames, read), event_lineup=True)
    elim = [a for a in res["anchors_out"] if a.kind == "elimination"]
    assert [a.team for a in elim] == ["6497"] * 3
    assert all(a.p == 0.9 and a.alliance_color == "red" for a in elim)
    off = _run(frames, _reading_session(frames, read), event_lineup=True, use_elimination=False)
    assert [a.kind for a in off["anchors_out"]] == ["digits", "digits"]


def test_alliance_rows_build_one_card_per_alliance_and_ask_only_the_slots_card():
    frames = _lineup_frames()                       # 3128 / 6773 read; R1 unread
    s = _lineup_session(frames, who_p={"A": 0.9})
    red = {"2637": _borrow("2637", 2, colour="red"),
           "2724": _borrow("2724", 2, mk="2026cur_qm4", colour="red")}
    res = _run(frames, s, seed=1, event_lineup=True, borrowed=red, rows_alliance_only=True)
    assert res["lineup_jpeg"][:2] == b"\xff\xd8" and res["lineup_jpeg_blue"][:2] == b"\xff\xd8"
    assert res["stats"]["lineup_rows"] == 4
    assert [n for q, n in s.asked if q == "who"] == [1]          # R1 only, on the red card
    order = ["2637", "2724"]
    random.Random(1).shuffle(order)
    r1 = next(w for w in wl.build_windows(frames, FPS) if w.slot == "R1")
    assert res["labels"][(r1.idx[0], "R1")][0] == order[0]       # letter A of the red card


def test_default_card_has_every_teams_row_a_colour_bar_and_the_colour_sentence():
    frames = _lineup_frames()
    s = _lineup_session(frames, who_p={"A": 0.9})
    sent = []
    real_ask = s.ask

    def ask(items, question):
        if question == "who":
            sent.extend(p for p, _sizes in items)
        return real_ask(items, question)

    s.ask = ask
    res = _run(frames, s, seed=1, event_lineup=True,
               borrowed={"2637": _borrow("2637", 2, colour="red")})
    assert res["stats"]["lineup_rows"] == 3 and res["lineup_jpeg_blue"] is None
    assert jev_reader.LINEUP_COLOUR_NOTE in sent[0]["state"]["context"]
    assert list(sent[0]["questions"]["who"]["criteria"])[:3] == ["A", "B", "C"]
    card = cv2.imdecode(np.frombuffer(res["lineup_jpeg"], np.uint8), cv2.IMREAD_COLOR)
    order = ["2637", "3128", "6773"]
    random.Random(1).shuffle(order)
    y = (wc.BAR_ROWS[0] + wc.BAR_ROWS[1]) // 2
    for j, team in enumerate(order):
        b, _g, r = (int(v) for v in card[240 * j + y, wc.LABEL_W // 2])
        assert (r > 180 and b < 80) if team == "2637" else (b > 180 and r < 80)


def test_an_abort_returns_no_anchors_no_calls_and_no_cards():
    res = _run(_frames({"B1": _still(range(20))}, 20), FakeSession(available=False),
               event_lineup=True)
    assert res["aborted"] and res["anchors_out"] == [] and res["digit_calls"] is None
    assert res["lineup_jpeg"] is None and res["lineup_jpeg_blue"] is None
    assert res["row_sources"] == {}
    assert {"rows_own", "rows_borrowed", "rows_elim"} <= set(res["stats"])
