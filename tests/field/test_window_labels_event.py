"""Event lineup (docs/superpowers/specs/2026-11-10-event-lineup-design.md) in
the window-label stage: knob-off golden snapshot, store_anchors, borrowed
rows, elimination anchors, cards, anchors out, and the digit-call cache.
Fakes only: no network, no video, no GPU."""
import hashlib
import json
import random
from pathlib import Path

from web.field import event_anchors as ea
from web.field import window_labels as wl
from tests.field.test_window_labels import (
    FakeSession, _frames, _lineup_frames, _lineup_session, _run, _still)

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
