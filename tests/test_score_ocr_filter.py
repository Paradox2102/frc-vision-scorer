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


def test_repeated_misread_is_corrected_once_true_value_is_read():
    f = ScoreOcrFilter()
    feed(f, [180])
    # Two identical misreads (185 read as 105) and one more: re-sync to 105...
    assert feed(f, [105, 105, 106]) == [None, None, 106]
    # ...then consistent true reads pull it back.
    assert feed(f, [186, 187, 188]) == [None, None, 188]


def test_first_read_accepted():
    f = ScoreOcrFilter()
    assert f.accept(None) is None
    assert f.accept(250) == 250
