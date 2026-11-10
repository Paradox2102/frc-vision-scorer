"""Window labels: pure logic (windows, decisions, assembly) and the
orchestration with a fake reader and a fake frame source. No network."""
import copy
import random

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


# --------------------------------------------------------------------------- #
# run_window_labels with a fake reader and a fake frame source
# --------------------------------------------------------------------------- #
import numpy as np


class _Cal:
    camera_center = np.array([8.0, -5.0, 3.0])

    def field_to_image(self, p):
        return (100.0 * p[0], 700.0 - 60.0 * p[1])


def _source(video_path, frame_nums, view):
    for n in sorted({int(x) for x in frame_nums}):
        yield n, np.zeros((view[3], view[2], 3), np.uint8)


class FakeSession:
    """Answers by slot position: the test maps a frame's target x to a team."""

    def __init__(self, team_p=None, who_p=None, fail=0, available=True):
        self.team_p, self.who_p = team_p or {}, who_p or {}
        self.fail, self.available, self.disabled_reason = fail, available, "auth"
        self.asked = []

    def ask(self, items, question):
        self.asked.append((question, len(items)))
        out = []
        for payload, _sizes in items:
            if self.fail:
                self.fail -= 1
                out.append({"error": "boom"})
            elif question == "team":
                out.append({"probs": dict(self.team_p)})
            else:
                out.append({"probs": dict(self.who_p)})
        return out

    def stats(self):
        n = sum(k for _q, k in self.asked)
        return {"requests": n, "input_tokens": 1000 * n, "cost_usd": 0.0001 * n, "errors": 0}


def _params(**kw):
    kw.setdefault("fps", FPS)
    kw.setdefault("red", ["2637", "2724", "6497"])
    kw.setdefault("blue", ["3128", "6773", "9066"])
    return wl.Params(**kw)


def _run(frames, session, **kw):
    return wl.run_window_labels(frames, "video.mp4", _Cal(), session, _params(**kw),
                                frame_source=_source)


def test_run_names_windows_from_digits_and_reports_stats():
    frames = _frames({"B1": _still(range(20), (3.0, 2.0))}, 20)
    snapshot = copy.deepcopy(frames)
    s = FakeSession(team_p={"3128": 0.9, "unreadable": 0.1})
    res = _run(frames, s)
    assert res["aborted"] is None
    assert res["labels"][(0, "B1")][:2] == ("3128", wl.RANK_DIGITS)
    assert res["segments"][0][0]["source"] == "window_digits"
    st = res["stats"]
    assert st["windows"] == 1 and st["digit_named"] == 1 and st["lineup_named"] == 0
    assert st["unlabelled"] == 0 and st["requests"] == 9 and st["input_tokens"] == 9000
    assert st["cost_cents"] == pytest.approx(0.09) and st["aborted"] is None
    assert s.asked == [("team", 9)]                      # 4 wide + 4 zoom + 1 grid
    assert frames == snapshot                            # input not mutated
    assert st["samples"] == 4 and st["tiles"] == 4


class BySlot(FakeSession):
    """Digit answers from ``plan`` (one per request, in request order); lineup
    answers from FakeSession."""

    def ask(self, items, question):
        if question == "team":
            self.asked.append((question, len(items)))
            return [{"probs": dict(self.plan.pop(0))} for _item in items]
        return super().ask(items, question)


def _lineup_frames(second_unread=False):
    # 3128 is read on B1 and 6773 on B2 in four runs each (anchors). The R1 window is unread
    # and the lineup names it. R1 is on screen only while B1/B2 are absent (frames 10-14), so
    # the one-team-per-frame rule has nothing to clear. ``second_unread`` adds an unread R2
    # window in the next gap (frames 25-29).
    b1 = _still([i for i in range(60) if i % 15 < 10], (3.0, 2.0))     # four 10-frame runs
    b2 = _still([i for i in range(60) if i % 15 < 10], (5.0, 6.0))
    tracks = {"B1": b1, "B2": b2, "R1": _still(range(10, 15), (12.0, 3.0))}
    if second_unread:
        tracks["R2"] = _still(range(25, 30), (10.0, 4.0))
    return _frames(tracks, 60)


def _lineup_session(frames, **kw):
    s = BySlot(**kw)
    plan = []
    for w in wl.build_windows(frames, FPS):             # request order = window order
        probs = ({"3128": 0.9} if w.slot == "B1" else {"6773": 0.9} if w.slot == "B2"
                 else {"unreadable": 0.9})
        plan += [probs] * (2 * len(w.samples) + 1)
    s.plan = plan
    return s


def _expected_letters(seed):
    """The letter -> team map the stage builds: anchored teams in roster order,
    shuffled by Random(seed)."""
    roster = ["2637", "2724", "6497", "3128", "6773", "9066"]
    order = [t for t in roster if t in ("3128", "6773")]
    random.Random(seed).shuffle(order)
    return {wl.LETTERS[j]: t for j, t in enumerate(order)}


def test_run_lineup_pass_names_unread_windows():
    frames = _lineup_frames()
    s = _lineup_session(frames, who_p={"A": 0.1, "B": 0.85, "unsure": 0.05})
    ws = wl.build_windows(frames, FPS)
    res = _run(frames, s, seed=1)
    assert res["aborted"] is None
    st = res["stats"]
    assert st["digit_named"] == 8 and st["lineup_rows"] == 2 and st["lineup_named"] == 1
    r1 = next(w for w in ws if w.slot == "R1")
    team = res["labels"][(r1.idx[0], "R1")][0]
    assert team == _expected_letters(1)["B"]
    assert res["labels"][(r1.idx[0], "R1")][1] == wl.RANK_LINEUP
    assert ("who", 1) in s.asked
    assert res["lineup_jpeg"][:2] == b"\xff\xd8"


def test_lineup_disabled_sends_no_who_request_and_keeps_digit_labels():
    frames = _lineup_frames()
    s = _lineup_session(frames, who_p={"A": 0.1, "B": 0.85, "unsure": 0.05})
    res = _run(frames, s, seed=1, lineup_min_p=1.01)
    assert res["aborted"] is None
    assert all(q == "team" for q, _n in s.asked)
    st = res["stats"]
    assert st["digit_named"] == 8 and st["lineup_named"] == 0 and st["unlabelled"] == 1
    b1 = next(w for w in wl.build_windows(frames, FPS) if w.slot == "B1")
    assert res["labels"][(b1.idx[0], "B1")][:2] == ("3128", wl.RANK_DIGITS)
    assert res["lineup_jpeg"] is None


def test_lineup_pass_errors_abort_with_no_labels():
    frames = _lineup_frames()
    s = _lineup_session(frames, who_p={"B": 0.9}, fail=1)       # the one "who" request fails
    res = _run(frames, s, seed=1)
    assert res["aborted"] and "request errors" in res["aborted"]
    assert "lineup pass" in res["aborted"]
    assert res["labels"] is None and res["segments"] is None


def test_lineup_pass_is_skipped_below_two_rows_or_when_disabled():
    frames = _frames({"B1": _still(range(20)), "R1": _still(range(10), (12.0, 3.0))}, 20)
    s = FakeSession(team_p={"unreadable": 0.9})
    res = _run(frames, s)
    assert res["aborted"] is None and res["stats"]["lineup_rows"] == 0
    assert all(q == "team" for q, _n in s.asked) and res["stats"]["unlabelled"] == 2


@pytest.mark.parametrize("kw,reason", [
    (dict(session=FakeSession(available=False)), "reader unavailable"),
    (dict(session=FakeSession(fail=5)), "request errors"),
    (dict(deadline=-1.0), "deadline"),
])
def test_run_aborts_cleanly(kw, reason):
    frames = _frames({"B1": _still(range(20))}, 20)
    session = kw.pop("session", None) or FakeSession(team_p={"3128": 0.9})
    res = _run(frames, session, **kw)
    assert res["aborted"] and reason in res["aborted"]
    assert res["labels"] is None and res["segments"] is None
    assert res["stats"]["aborted"] == res["aborted"]


def test_a_passed_deadline_sends_no_requests():
    s = FakeSession(team_p={"3128": 0.9})
    _run(_frames({"B1": _still(range(20))}, 20), s, deadline=-1.0)
    assert s.asked == []


def test_run_aborts_when_there_is_nothing_to_do_or_no_video():
    assert "no windows" in _run([], FakeSession())["aborted"]

    def empty(video_path, frame_nums, view):
        return iter(())

    res = wl.run_window_labels(_frames({"B1": _still(range(20))}, 20), "v.mp4", _Cal(),
                               FakeSession(), _params(), frame_source=empty)
    assert "no frames decoded" in res["aborted"]

    def broken(video_path, frame_nums, view):
        raise OSError("cannot open video")

    res = wl.run_window_labels(_frames({"B1": _still(range(20))}, 20), "v.mp4", _Cal(),
                               FakeSession(), _params(), frame_source=broken)
    assert res["aborted"].startswith("error: OSError")


# --------------------------------------------------------------------------- #
# Deadlines between stages, lockouts, thresholds, guards
# --------------------------------------------------------------------------- #

class _Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def _run_clocked(frames, session, clock, frame_source=_source, **kw):
    return wl.run_window_labels(frames, "video.mp4", _Cal(), session, _params(**kw),
                                frame_source=frame_source, clock=clock)


def test_deadline_passing_during_the_video_pass_sends_no_requests():
    clock = _Clock()

    def source(video_path, frame_nums, view):
        yield from _source(video_path, frame_nums, view)
        clock.now = 100.0                                # the decode took too long

    s = FakeSession(team_p={"3128": 0.9})
    res = _run_clocked(_frames({"B1": _still(range(20))}, 20), s, clock, source, deadline=10.0)
    assert res["aborted"] and "deadline" in res["aborted"] and "video pass" in res["aborted"]
    assert s.asked == [] and res["labels"] is None


def test_deadline_passing_after_the_first_digit_batch_stops_the_digit_pass(monkeypatch):
    monkeypatch.setattr(wl, "REQUEST_BATCH", 4)
    clock = _Clock()

    class Slow(FakeSession):
        def ask(self, items, question):
            clock.now = 100.0
            return super().ask(items, question)

    s = Slow(team_p={"3128": 0.9})
    res = _run_clocked(_frames({"B1": _still(range(20))}, 20), s, clock, deadline=10.0)
    assert res["aborted"] and "deadline" in res["aborted"] and "digit pass" in res["aborted"]
    assert s.asked == [("team", 4)] and res["labels"] is None


def test_deadline_passing_after_the_first_lineup_batch_stops_the_lineup_pass(monkeypatch):
    monkeypatch.setattr(wl, "REQUEST_BATCH", 1)
    clock = _Clock()
    frames = _lineup_frames(second_unread=True)
    s = _lineup_session(frames, who_p={"B": 0.9})
    real_ask = s.ask

    def ask(items, question):
        out = real_ask(items, question)
        if question == "who":
            clock.now = 100.0
        return out

    s.ask = ask
    res = _run_clocked(frames, s, clock, seed=1, deadline=10.0)
    assert res["aborted"] and "deadline" in res["aborted"] and "lineup pass" in res["aborted"]
    assert [q for q, _n in s.asked].count("who") == 1 and res["labels"] is None


def test_a_mid_batch_lockout_aborts_even_under_the_error_threshold():
    class Lockout(FakeSession):
        def ask(self, items, question):
            out = super().ask(items, question)
            self.available = False                       # 401 latched during this batch
            out[-1] = {"error": "auth"}
            return out

    frames = _frames({"B1": _still(range(20)), "B2": _still(range(20), (9.0, 6.0))}, 20)
    s = Lockout(team_p={"3128": 0.9})                    # 1 error of 18 requests
    res = _run(frames, s)
    assert res["aborted"] and "reader unavailable" in res["aborted"]
    assert res["labels"] is None and res["stats"]["request_errors"] == 1


def test_one_failure_in_nine_aborts_and_one_in_eighteen_does_not():
    res = _run(_frames({"B1": _still(range(20))}, 20), FakeSession(team_p={"3128": 0.9}, fail=1))
    assert res["aborted"] and "request errors: 1 of 9" in res["aborted"]
    frames = _frames({"B1": _still(range(20)), "B2": _still(range(20), (9.0, 6.0))}, 20)
    res = _run(frames, FakeSession(team_p={"3128": 0.9}, fail=1))
    assert res["aborted"] is None and res["stats"]["request_errors"] == 1


def test_batches_are_small_so_checks_run_often():
    assert wl.REQUEST_BATCH == 32
    assert wl.MIN_ERRORS_TO_ABORT_EARLY == 20


def test_a_pass_where_every_request_fails_stops_after_the_first_batch():
    frames = _frames({"B1": _still(range(300))}, 300)    # 15 windows -> 135 digit requests
    s = FakeSession(team_p={"3128": 0.9}, fail=10_000)
    res = _run(frames, s)
    assert res["aborted"] and "request errors: 32 of 32 sent so far" in res["aborted"]
    assert "digit pass" in res["aborted"]
    assert s.asked == [("team", 32)]                     # not all 5 batches
    assert res["labels"] is None and res["stats"]["request_errors"] == 32


def test_a_few_early_errors_do_not_abort_before_the_pass_ends():
    # 3 errors in the first batch of 32 is 9% of what was sent but under the floor of 20
    frames = _frames({"B1": _still(range(300))}, 300)    # 135 requests, 14 allowed
    res = _run(frames, FakeSession(team_p={"3128": 0.9}, fail=3))
    assert res["aborted"] is None and res["stats"]["request_errors"] == 3


def test_no_roster_aborts_before_decoding_video_or_sending_requests():
    calls = []

    def source(video_path, frame_nums, view):
        calls.append(1)
        yield from _source(video_path, frame_nums, view)

    s = FakeSession(team_p={"3128": 0.9})
    res = _run_clocked(_frames({"B1": _still(range(20))}, 20), s, _Clock(), source,
                       red=[], blue=[])
    assert res["aborted"] and "no roster" in res["aborted"]
    assert calls == [] and s.asked == [] and res["labels"] is None


def test_an_unavailable_reader_aborts_before_decoding_video():
    calls = []

    def source(video_path, frame_nums, view):
        calls.append(1)
        yield from _source(video_path, frame_nums, view)

    res = _run_clocked(_frames({"B1": _still(range(20))}, 20), FakeSession(available=False),
                       _Clock(), source)
    assert res["aborted"] == "reader unavailable: auth" and calls == []


def test_no_sample_in_view_aborts_with_that_reason_before_decoding_video():
    calls = []

    def source(video_path, frame_nums, view):
        calls.append(1)
        yield from _source(video_path, frame_nums, view)

    frames = _frames({"B1": _still(range(20), (-5.0, 2.0))}, 20)        # u = -500 px
    res = _run_clocked(frames, FakeSession(team_p={"3128": 0.9}), _Clock(), source)
    assert "no window sample is in the main view" in res["aborted"]
    assert calls == [] and res["stats"]["samples"] == 0


def test_a_short_answer_list_aborts_instead_of_misaligning():
    class Short(FakeSession):
        def ask(self, items, question):
            return super().ask(items, question)[:-1]

    res = _run(_frames({"B1": _still(range(20))}, 20), Short(team_p={"3128": 0.9}))
    assert res["aborted"] and "reader returned" in res["aborted"]
    assert res["labels"] is None


def test_the_frame_source_is_closed_when_the_video_pass_aborts():
    clock = _Clock()

    class Source:
        closed = False

        def __init__(self, video_path, frame_nums, view):
            self.it = _source(video_path, frame_nums, view)

        def __iter__(self):
            return self

        def __next__(self):
            clock.now = 100.0                            # past the deadline from frame 1
            return next(self.it)

        def close(self):
            Source.closed = True

    frames = _frames({"B1": _still(range(300))}, 300)    # 15 windows -> 60 sample frames
    res = _run_clocked(frames, FakeSession(team_p={"3128": 0.9}), clock, Source, deadline=10.0)
    assert res["aborted"] == "deadline passed during the video pass"
    assert Source.closed


def test_stats_report_samples_and_tiles_so_a_truncated_video_shows():
    assert wl._stats()["samples"] == 0 and wl._stats()["tiles"] == 0

    def truncated(video_path, frame_nums, view):
        yield from list(_source(video_path, frame_nums, view))[:2]

    res = _run_clocked(_frames({"B1": _still(range(20))}, 20), FakeSession(team_p={"3128": 0.9}),
                       _Clock(), truncated)
    st = res["stats"]
    assert st["samples"] == 4 and st["tiles"] == 2


def test_lineup_payloads_share_one_encoded_card_asset():
    frames = _lineup_frames(second_unread=True)
    s = _lineup_session(frames, who_p={"B": 0.9})
    sent = []
    real_ask = s.ask

    def ask(items, question):
        if question == "who":
            sent.extend(items)
        return real_ask(items, question)

    s.ask = ask
    res = _run(frames, s, seed=1)
    assert res["aborted"] is None and len(sent) == 2
    assert sent[0][0]["assets"]["lineup_image"] is sent[1][0]["assets"]["lineup_image"]
    assert sent[0][0]["assets"]["crops_image"] is not sent[1][0]["assets"]["crops_image"]
