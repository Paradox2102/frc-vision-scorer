"""Window labels: pure logic (windows, decisions, assembly) and the
orchestration with a fake reader and a fake frame source. No network."""
import pytest

from web.field import window_labels as wl

FPS = 60.0
STEP = 12          # source frames per processed frame -> 5 processed fps


def _frames(tracks, n, step=STEP):
    """tracks: {slot: {index: (x, y)}} -> replay frames (frame_num = 420 + index * step)."""
    out = []
    for i in range(n):
        robots = [{"slot": s, "alliance": "blue" if s.startswith("B") else "red",
                   "team_number": "9999", "confirmed": False, "field_pos": list(p[i])}
                  for s, p in tracks.items() if i in p]
        out.append({"frame_num": 420 + i * step, "robots": robots})
    return out


def _still(indices, xy=(3.0, 2.0)):
    return {i: xy for i in indices}


def test_frame_period_from_median_step():
    assert wl.frame_period_s(_frames({"B1": _still(range(5))}, 5), FPS) == pytest.approx(0.2)
    assert wl.frame_period_s([], FPS) is None
    assert wl.frame_period_s(_frames({"B1": _still(range(5))}, 5), 0) is None


def test_long_run_splits_into_even_windows_of_at_most_4s():
    ws = wl.build_windows(_frames({"B1": _still(range(50))}, 50), FPS)     # 10 s
    assert [len(w.idx) for w in ws] == [17, 17, 16]
    assert ws[0].idx[0] == 0 and ws[-1].idx[-1] == 49
    assert all(len(w.samples) == 4 for w in ws)
    assert ws[0].samples[0] == ws[0].idx[0] and ws[0].samples[-1] == ws[0].idx[-1]
    assert len({w.run for w in ws}) == 1 and [w.wid for w in ws] == [0, 1, 2]


def test_exactly_4s_is_one_window_and_one_more_sample_makes_two():
    assert len(wl.build_windows(_frames({"B1": _still(range(20))}, 20), FPS)) == 1
    assert len(wl.build_windows(_frames({"B1": _still(range(21))}, 21), FPS)) == 2


def test_gap_longer_than_two_and_a_half_periods_starts_a_new_run():
    present = list(range(10)) + list(range(13, 23))          # 3 frames missing
    ws = wl.build_windows(_frames({"B1": _still(present)}, 23), FPS)
    assert len(ws) == 2 and ws[0].run != ws[1].run
    assert ws[0].idx == list(range(10)) and ws[1].idx == list(range(13, 23))


def test_one_missing_frame_does_not_split():
    present = [i for i in range(12) if i != 5]
    ws = wl.build_windows(_frames({"B1": _still(present)}, 12), FPS)
    assert len(ws) == 1 and ws[0].idx == present


def test_jump_faster_than_7_5_mps_starts_a_new_run():
    track = {i: (3.0, 2.0) for i in range(6)}
    track.update({i: (8.0, 2.0) for i in range(6, 12)})       # 5 m in 0.2 s
    ws = wl.build_windows(_frames({"R1": track}, 12), FPS)
    assert [w.idx for w in ws] == [list(range(6)), list(range(6, 12))]


def test_single_sample_runs_are_dropped_and_slots_are_independent():
    tracks = {"B1": _still([5] + list(range(20, 30))), "R2": _still(range(4))}
    ws = wl.build_windows(_frames(tracks, 30), FPS)
    assert [(w.slot, w.idx) for w in ws] == [("B1", list(range(20, 30))), ("R2", [0, 1, 2, 3])]


def test_two_sample_window_uses_both_as_samples_and_no_frames_gives_no_windows():
    ws = wl.build_windows(_frames({"B1": _still([0, 1])}, 2), FPS)
    assert ws[0].samples == [0, 1]
    assert wl.build_windows([], FPS) == []


ROSTER = ["2637", "2724", "6497", "3128", "6773", "9066"]


def _p(**kw):
    """A reader answer: team/escape probabilities."""
    base = {"not_robot": 0.0, "unreadable": 0.0}
    base.update({k.lstrip("t"): v for k, v in kw.items()})
    return base


def test_digit_label_needs_p_half_no_conflict_and_two_top_calls():
    good = [_p(t2724=0.9, unreadable=0.05), _p(t2724=0.6, unreadable=0.3), _p(unreadable=0.9)]
    assert wl.digit_decision(good, ROSTER) == ("2724", 0.9)
    # best team below 0.5
    assert wl.digit_decision([_p(t2724=0.45, unreadable=0.5)] * 3, ROSTER) is None
    # a second team at 0.4 or more blocks the label
    conflict = [_p(t2724=0.9), _p(t2724=0.8), _p(t6497=0.4, unreadable=0.5)]
    assert wl.digit_decision(conflict, ROSTER) is None
    # team is the top option in only one call
    once = [_p(t2724=0.9), _p(t2724=0.3, unreadable=0.6), _p(unreadable=0.9)]
    assert wl.digit_decision(once, ROSTER) is None


def test_digit_decision_ignores_failed_calls_and_empty_input():
    calls = [None, {}, _p(t3128=0.7), _p(t3128=0.55, unreadable=0.4)]
    assert wl.digit_decision(calls, ROSTER) == ("3128", 0.7)
    assert wl.digit_decision([], ROSTER) is None
    assert wl.digit_decision([None], ROSTER) is None


def test_anchors_need_085_two_per_team_and_spread_in_time():
    labels = {i: ("2724", 0.9) for i in range(10)}
    labels.update({20: ("6497", 0.95), 21: ("3128", 0.9), 22: ("3128", 0.86), 23: ("2724", 0.84)})
    t0 = {i: float(i) for i in labels}
    a = wl.pick_anchors(labels, t0)
    assert a["2724"] == [0, 2, 4, 5, 7, 9]          # 6 of 10, first and last kept
    assert a["3128"] == [21, 22]
    assert "6497" not in a                          # only one anchor
    assert 23 not in a["2724"]                      # 0.84 is below the anchor floor


def test_lineup_decision_takes_best_letter_at_or_above_the_floor():
    letters = {"A": "2724", "B": "6497"}
    assert wl.lineup_decision({"A": 0.82, "B": 0.1, "unsure": 0.08}, letters, 0.8) == ("2724", 0.82)
    assert wl.lineup_decision({"A": 0.7, "B": 0.1, "unsure": 0.2}, letters, 0.8) is None
    assert wl.lineup_decision({"A": 0.7, "B": 0.1, "unsure": 0.2}, letters, 0.65) == ("2724", 0.7)
    assert wl.lineup_decision(None, letters, 0.8) is None
    assert wl.lineup_decision({"none": 1.0}, letters, 0.8) is None
