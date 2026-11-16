"""Scoreboard score OCR misread guard (web/score_ocr_filter.py)."""
from web.score_ocr_filter import ScoreOcrFilter


def feed(f, reads):
    return [f.accept(v) for v in reads]


def test_small_steps_accepted_like_old_guard():
    f = ScoreOcrFilter()
    assert feed(f, [0, 1, 2, 7, 4, 20]) == [0, 1, 2, 7, 4, 20]
    assert f.last == 20 and f.resyncs == 0


def test_lone_garbage_read_rejected():
    f = ScoreOcrFilter()
    assert feed(f, [2, 117, 3, 5]) == [2, None, 3, 5]
    assert f.last == 5


def test_real_jump_resyncs_instead_of_freezing():
    # qm5 red: last accepted 4, then the real score climbs past 24 between
    # reads. The old guard froze at 4 for the rest of the match.
    f = ScoreOcrFilter()
    feed(f, [0, 1, 2, 7, 4])
    out = feed(f, [40, 45, 52, 60, 75, 90, 110])
    assert out[:2] == [None, None]
    assert out[2] == 52                     # third consistent read re-syncs
    assert out[3:] == [60, 75, 90, 110]     # normal tracking afterwards
    assert f.resyncs == 1


def test_inconsistent_run_does_not_resync():
    f = ScoreOcrFilter()
    feed(f, [10])
    # Rejected reads that jump around (or go down) never form a run.
    assert feed(f, [117, 60, 300, 55, 400]) == [None] * 5
    assert f.last == 10


def test_none_reads_do_not_break_or_build_a_run():
    f = ScoreOcrFilter()
    feed(f, [4])
    assert feed(f, [40, None, 44, None, 50]) == [None, None, None, None, 50]


def test_run_far_below_score_never_resyncs_down():
    # 2026cur_qm76 blue: broadcast cut away from the scoreboard; a run of
    # garbage 8s pulled the score down from 435. Scores never decrease.
    f = ScoreOcrFilter()
    f.last = 435
    assert feed(f, [8, 8, 8, 8, 8]) == [None] * 5
    assert f.last == 435 and f.resyncs == 0
    # Small dips (a 7 misread as 4) are still accepted like before.
    assert f.accept(431) == 431


def test_low_garbage_does_not_break_an_upward_run():
    f = ScoreOcrFilter()
    f.last = 100
    assert feed(f, [140, 3, 150, 160]) == [None, None, None, 160]


def test_fast_end_of_match_climb_resyncs():
    # 2026cur_qm85 red: 311 then 345, 368, 371 (steps 34, 23, 3). The run used
    # to reset on the +23 step and the final score stayed at 311 (TBA 375).
    f = ScoreOcrFilter()
    f.last = 311
    assert feed(f, [345, 368, 371]) == [None, None, 371]


def test_score_starts_at_zero():
    f = ScoreOcrFilter()
    assert f.accept(None) is None
    assert f.accept(7) == 7
    g = ScoreOcrFilter()
    # A large first read is a jump from 0 and needs confirming.
    assert feed(g, [250, 0, 1]) == [None, 0, 1]
    h = ScoreOcrFilter()
    # Joining mid-match: consistent reads re-sync upward from 0.
    assert feed(h, [250, 252, 255]) == [None, None, 255]


def test_unreachable_garbage_burst_cannot_lock_score_high():
    # 2026cur_qm85 blue: 75 at t=37, then 661, 689, 689... one second later.
    # Confirming that run would lock blue at 689 (no downward re-sync).
    f = ScoreOcrFilter()
    f.last, f.last_t = 75, 37.0
    reads = [(661, 38), (689, 39), (689, 40), (689, 41), (88, 42), (95, 43)]
    assert [f.accept(v, t) for v, t in reads] == [None, None, None, None, 88, 95]
    assert f.resyncs == 0


def test_real_jump_after_a_gap_is_reachable():
    # qm76-style: score 157 at t=61, scoreboard back at t=190 reading 435.
    f = ScoreOcrFilter()
    f.accept(150, 60.0); f.accept(157, 61.0)
    assert [f.accept(v, t) for v, t in [(435, 190.0), (435, 191.0), (435, 192.0)]] \
        == [None, None, 435]
