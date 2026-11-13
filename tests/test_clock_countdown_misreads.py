"""Countdown finder vs single low OCR misreads.

Tesseract occasionally misreads one clock digit LOW mid-period (69->60,
28->22, 57->52, ...). The finder kept a running minimum, so the bogus low
value stuck, every true read after it looked like a high outlier, and the
whole match fell back to full-video processing. With fps, a read that falls
faster than real time is a bad sample instead.

Samples are modelled on the real 1080p60 scans: one read every 0.5 s (30
frames), a 20 s auto countdown with a hold before it, an immediate jump to a
140 hold at the end of auto (no 0 read), then the 140 s teleop countdown.
"""
import pytest

from web.app import _find_clock_period, compute_active_windows

FPS = 60.0
STEP = 30  # 0.5 s
AUTO_T0 = 450      # auto clock shows 20 here, 19 one second later
TELEOP_T0 = 1950   # teleop clock shows 140 here, 139 one second later
TOTAL = 12870
# The finder's start_frame is the last read still showing the start value.
AUTO_START_FRAME = int(AUTO_T0 + FPS - STEP)
TELEOP_START_FRAME = int(TELEOP_T0 + FPS - STEP)


def _clock(frame):
    """True clock value at ``frame`` (None before the match)."""
    if frame < 210:
        return None
    if frame < AUTO_T0 + 20 * FPS:
        return 20 - max(0, int((frame - AUTO_T0) // FPS))
    if frame < TELEOP_T0:
        return 140
    if frame <= TELEOP_T0 + 140 * FPS:
        return 140 - int((frame - TELEOP_T0) // FPS)
    return 0


def _match(misreads=None, step=STEP, drop=()):
    misreads = misreads or {}
    return [
        (frame, misreads.get(frame, _clock(frame)))
        for frame in range(0, TOTAL, step)
        if frame not in drop
    ]


def _teleop_frames(value, samples=None):
    return [f for f, v in (samples or _match()) if f > TELEOP_T0 and v == value]


def _windows(samples):
    return compute_active_windows(samples, FPS, TOTAL)


@pytest.fixture(scope="module")
def clean():
    ranges, periods, fell_back, reason = _windows(_match())
    assert not fell_back, reason
    return ranges, periods


def test_clean_synthetic_match_locks_both_periods(clean):
    _, (auto, teleop) = clean
    assert (auto["start_frame"], auto["start_seconds"]) == (AUTO_START_FRAME, 20)
    assert (teleop["start_frame"], teleop["start_seconds"]) == (TELEOP_START_FRAME, 140)
    assert teleop["end_frame"] == TELEOP_T0 + 140 * FPS


# One low misread mid-teleop, at the values seen in the five failing caches.
@pytest.mark.parametrize("true_value,misread", [
    (69, 60),   # 32a98a80
    (28, 22),   # 5eeafb7a / d47403ec
    (57, 52),   # 196d621d
    (27, 22),   # d47403ec
    (68, 62),   # 677415b8
])
def test_single_low_misread_mid_teleop_still_locks(clean, true_value, misread):
    samples = _match({_teleop_frames(true_value)[0]: misread})
    ranges, periods, fell_back, reason = _windows(samples)
    assert not fell_back, reason
    assert (ranges, periods) == clean


def test_low_misread_was_fatal_without_fps():
    # Pins the bug: fps=None is the legacy finder, which loses teleop.
    samples = _match({_teleop_frames(69)[0]: 60})
    auto = _find_clock_period(samples, 20)
    assert _find_clock_period(samples, 140, after_frame=auto["end_frame"]) is None
    assert _find_clock_period(
        samples, 140, after_frame=auto["end_frame"], fps=FPS
    ) is not None


def test_alternating_low_misreads_each_followed_by_a_good_read(clean):
    # 196d621d: 58, 52*, 57, 52*, 56 -- two misreads, never adjacent.
    misreads = {_teleop_frames(58)[1]: 52, _teleop_frames(57)[1]: 52}
    assert _windows(_match(misreads))[:2] == clean


def test_low_misread_to_zero_does_not_end_teleop_early(clean):
    # The legacy finder took a mid-teleop "0" as the end of the period.
    samples = _match({_teleop_frames(69)[0]: 0})
    auto = _find_clock_period(samples, 20)
    legacy = _find_clock_period(samples, 140, after_frame=auto["end_frame"])
    assert legacy["end_frame"] == _teleop_frames(69)[0]
    assert _windows(samples)[:2] == clean


def test_low_misread_mid_auto_still_locks(clean):
    frame = next(f for f, v in _match() if v == 12 and f < TELEOP_T0)
    assert _windows(_match({frame: 2}))[:2] == clean


def test_legit_one_to_zero_end_of_auto_is_kept():
    # A clock that DOES read 0 at the end of auto ends the period there, and
    # the teleop search still starts fresh from the 140 hold after it.
    end = AUTO_T0 + 20 * FPS
    samples = [
        (f, 0 if end - STEP <= f < end + 4 * STEP else v) for f, v in _match()
    ]
    auto = _find_clock_period(samples, 20, fps=FPS)
    assert auto["end_frame"] == end - STEP and auto["end_seconds"] == 0
    teleop = _find_clock_period(samples, 140, after_frame=auto["end_frame"], fps=FPS)
    assert teleop["start_frame"] == TELEOP_START_FRAME
    assert teleop["end_frame"] == TELEOP_T0 + 140 * FPS


def test_long_sampling_gap_is_real_time_not_a_misread(clean):
    # 12 s of missing samples mid-teleop: the clock is ~12 lower at the next
    # read, which the allowance (elapsed seconds + slack) must accept.
    gap = set(range(4020, 4740, STEP))
    ranges, periods, fell_back, _ = _windows(_match(drop=gap))
    assert not fell_back
    assert periods[1]["end_frame"] == clean[1][1]["end_frame"]


@pytest.mark.parametrize("bad", [None, 60, 140])
def test_drop_allowance_is_measured_from_last_good_read(bad):
    # 3 s sampling: good 113, bad sample, good 107. The 6 s drop is only
    # legal measured from the last GOOD read (6 s + 2), not from the bad
    # sample (3 s + 2) -- a bad sample must not move the reference.
    step = 180
    samples = _match(step=step)
    idx = next(i for i, (f, v) in enumerate(samples) if f > TELEOP_T0 and v == 113)
    assert (samples[idx + 2][1]) == 107
    samples[idx + 1] = (samples[idx + 1][0], bad)
    auto = _find_clock_period(samples, 20, fps=FPS)
    teleop = _find_clock_period(samples, 140, after_frame=auto["end_frame"], fps=FPS)
    assert teleop is not None and teleop["end_seconds"] == 0


@pytest.mark.parametrize("second", [None, 60, 99])
def test_two_bad_samples_in_a_row_are_tolerated_with_fps(clean, second):
    # 2026cc (Chezy Champs HUD): "1:11" unreadable twice, then "1:09" read
    # as 109 twice (colon dropped). Two bad reads in a row (1 s since the
    # last good read) used to break the walk and fall back to the full video.
    first, then = _teleop_frames(69)
    samples = _match({first: 60, then: second})
    assert _windows(samples)[:2] == clean


def test_legacy_finder_still_breaks_on_two_bad_samples():
    first, then = _teleop_frames(69)
    samples = _match({first: None, then: None})
    auto = _find_clock_period(samples, 20)
    assert _find_clock_period(samples, 140, after_frame=auto["end_frame"]) is None


def test_2026cc_qm1_pattern_locks_teleop(clean):
    # Real qm1 reads: 72, None, None, 70, 70, 109, 109, 68 ...
    f71 = _teleop_frames(71)
    f69 = _teleop_frames(69)
    samples = _match({f71[0]: None, f71[1]: None, f69[0]: 109, f69[1]: 109})
    ranges, periods, fell_back, reason = _windows(samples)
    assert not fell_back, reason
    assert (ranges, periods) == clean


def test_bad_run_longer_than_gap_budget_still_breaks():
    # 3 s of garbage mid-teleop (6 reads) is past the 2 s budget: the walk
    # gives up and teleop is not found, as before.
    bad = {f: None for f in _teleop_frames(69) + _teleop_frames(68) + _teleop_frames(67)}
    _, _, fell_back, reason = _windows(_match(bad))
    assert fell_back and "teleop" in reason
