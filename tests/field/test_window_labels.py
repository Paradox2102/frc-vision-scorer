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


def _two_slot_frames():
    # B1 and B2 both present for 20 frames; R1 for the first 10
    return _frames({"B1": _still(range(20)), "B2": _still(range(20), (9.0, 6.0)),
                    "R1": _still(range(10), (12.0, 3.0))}, 20)


def test_window_label_goes_to_every_frame_and_the_rest_is_unlabelled():
    frames = _two_slot_frames()
    ws = wl.build_windows(frames, FPS)
    b1 = next(w for w in ws if w.slot == "B1")
    lab, cleared = wl.assemble_labels(ws, {b1.wid: ("3128", wl.RANK_DIGITS, 0.9)})
    assert cleared == 0
    wl.apply_labels(frames, lab)
    for f in frames:
        rows = {r["slot"]: r for r in f["robots"]}
        assert rows["B1"]["team_number"] == "3128" and rows["B1"]["confirmed"] is True
        assert rows["B2"]["team_number"] is None and rows["B2"]["confirmed"] is False
        if "R1" in rows:
            assert rows["R1"]["team_number"] is None


def test_one_team_per_frame_digits_beat_lineup_then_higher_p_then_lower_slot():
    frames = _two_slot_frames()
    ws = wl.build_windows(frames, FPS)
    b1, b2 = (next(w for w in ws if w.slot == s) for s in ("B1", "B2"))
    # lineup with higher p loses to digits
    lab, cleared = wl.assemble_labels(ws, {b1.wid: ("3128", wl.RANK_LINEUP, 0.99),
                                           b2.wid: ("3128", wl.RANK_DIGITS, 0.6)})
    assert cleared == 20 and lab[(0, "B1")] is None and lab[(0, "B2")][0] == "3128"
    # same rank: higher p wins
    lab, _ = wl.assemble_labels(ws, {b1.wid: ("3128", wl.RANK_DIGITS, 0.7),
                                     b2.wid: ("3128", wl.RANK_DIGITS, 0.9)})
    assert lab[(5, "B1")] is None and lab[(5, "B2")][0] == "3128"
    # full tie: lower slot id keeps it
    lab, _ = wl.assemble_labels(ws, {b1.wid: ("3128", wl.RANK_DIGITS, 0.9),
                                     b2.wid: ("3128", wl.RANK_DIGITS, 0.9)})
    assert lab[(5, "B1")][0] == "3128" and lab[(5, "B2")] is None
    # different teams never collide
    lab, cleared = wl.assemble_labels(ws, {b1.wid: ("3128", wl.RANK_DIGITS, 0.9),
                                           b2.wid: ("6773", wl.RANK_LINEUP, 0.85)})
    assert cleared == 0 and lab[(5, "B2")][0] == "6773"


def test_segments_are_contiguous_with_explicit_unlabelled_rows():
    frames = _frames({"B1": _still(range(50))}, 50)          # 3 windows: 17, 17, 16 frames
    ws = wl.build_windows(frames, FPS)
    lab, _ = wl.assemble_labels(ws, {ws[0].wid: ("3128", wl.RANK_DIGITS, 0.9),
                                     ws[2].wid: ("3128", wl.RANK_LINEUP, 0.82)})
    segs = wl.window_segments(frames, lab)
    rows = segs[0]                                            # B1 -> slot id 0
    assert [(r["team_number"], r["source"]) for r in rows] == [
        ("3128", "window_digits"), (None, "window_none"), ("3128", "window_lineup")]
    assert rows[0]["start_frame"] == 420 and rows[0]["end_frame"] == 420 + 16 * STEP
    assert rows[1]["start_frame"] == 420 + 17 * STEP and rows[1]["score"] is None
    assert rows[2]["end_frame"] == 420 + 49 * STEP and rows[2]["confirmed"] is False
    assert rows[0]["confirmed"] is True and rows[0]["score"] == 0.9


def test_consecutive_windows_with_the_same_team_and_source_merge():
    frames = _frames({"R2": _still(range(50))}, 50)
    ws = wl.build_windows(frames, FPS)
    lab, _ = wl.assemble_labels(ws, {w.wid: ("2724", wl.RANK_DIGITS, 0.6 + 0.1 * k)
                                     for k, w in enumerate(ws)})
    rows = wl.window_segments(frames, lab)[4]                 # R2 -> slot id 4
    assert len(rows) == 1 and rows[0]["team_number"] == "2724" and rows[0]["score"] == 0.8


def test_events_are_restamped_from_frame_labels_and_gaps_become_none():
    frames = _frames({"R1": _still(range(20))}, 20)
    ws = wl.build_windows(frames, FPS)
    lab, _ = wl.assemble_labels(ws, {})                       # nothing named
    wl.apply_labels(frames, lab)
    frames[3]["robots"][0]["team_number"] = "2637"
    events = [{"frame_num": 3, "slot": "R1", "team_number": "9999", "robot_label": "9999"},
              {"frame_num": 7, "slot": "R1", "team_number": "9999", "robot_label": "9999"},
              {"frame_num": 99, "slot": "R1", "team_number": "9999", "robot_label": "9999"},
              {"frame_num": 4, "slot": "B3", "team_number": "9999", "robot_label": "9999"}]
    assert wl.restamp_events(frames, events) == 3
    assert events[0]["team_number"] == "2637" and events[0]["robot_label"] == "2637"
    assert events[1]["team_number"] is None and events[1]["robot_label"] == "R1"
    assert events[2]["team_number"] == "9999"                 # index outside the recording
    assert events[3]["team_number"] is None                   # slot absent in that frame


def test_longer_window_wins_a_full_tie_on_rank_and_p():
    # B2 is on screen for 20 frames (one long window), B1 only for the first 10
    frames = _frames({"B1": _still(range(10)), "B2": _still(range(20), (9.0, 6.0))}, 20)
    ws = wl.build_windows(frames, FPS)
    b1, b2 = (next(w for w in ws if w.slot == s) for s in ("B1", "B2"))
    lab, cleared = wl.assemble_labels(ws, {b1.wid: ("3128", wl.RANK_DIGITS, 0.9),
                                           b2.wid: ("3128", wl.RANK_DIGITS, 0.9)})
    assert cleared == 10
    assert lab[(3, "B1")] is None and lab[(3, "B2")][0] == "3128"   # longer beats lower slot id


def test_absent_stretches_and_cleared_frames_get_window_none_rows():
    present = list(range(10)) + list(range(20, 30))               # absent in frames 10..19
    frames = _frames({"B1": _still(present)}, 30)
    ws = wl.build_windows(frames, FPS)
    lab, _ = wl.assemble_labels(ws, {w.wid: ("3128", wl.RANK_DIGITS, 0.9) for w in ws})
    rows = wl.window_segments(frames, lab)[0]
    assert [(r["team_number"], r["source"], r["start_frame"], r["end_frame"]) for r in rows] == [
        ("3128", "window_digits", 420, 420 + 9 * STEP),
        (None, "window_none", 420 + 10 * STEP, 420 + 19 * STEP),
        ("3128", "window_digits", 420 + 20 * STEP, 420 + 29 * STEP)]
    # a slot whose label was cleared by the one-team-per-frame rule is an unlabelled row
    frames2 = _two_slot_frames()
    ws2 = wl.build_windows(frames2, FPS)
    b1, b2 = (next(w for w in ws2 if w.slot == s) for s in ("B1", "B2"))
    lab2, _ = wl.assemble_labels(ws2, {b1.wid: ("3128", wl.RANK_LINEUP, 0.9),
                                       b2.wid: ("3128", wl.RANK_DIGITS, 0.9)})
    segs2 = wl.window_segments(frames2, lab2)
    assert [r["source"] for r in segs2[0]] == ["window_none"]
    assert [r["source"] for r in segs2[1]] == ["window_digits"]


def test_unknown_slot_labels_and_missing_frame_numbers_are_skipped():
    frames = _frames({"B1": _still(range(6)), "X9": _still(range(6), (5.0, 5.0))}, 6)
    frames.append({"frame_num": None, "robots": [{"slot": "B1", "field_pos": [3.0, 2.0]}]})
    ws = wl.build_windows(frames, FPS)
    assert {w.slot for w in ws} == {"B1", "X9"} and all(6 not in w.idx for w in ws)
    lab, _ = wl.assemble_labels(ws, {w.wid: ("3128", wl.RANK_DIGITS, 0.9) for w in ws})
    assert lab[(0, "B1")][0] == "3128" and lab[(0, "X9")] is None      # the known slot keeps it
    assert list(wl.window_segments(frames, lab)) == [0]
