"""Auto true start: scoreboard start-indicator onset, clock-lag fallback, and
the longer auto-only processing lead (clamped to the clock run's first sample).
"""
from types import SimpleNamespace

import numpy as np

import web.app as app_mod
from web.app import (
    _find_clock_period,
    compute_active_windows,
    find_indicator_onset,
    start_indicator_fractions,
)
from web.job_config import KNOBS, JobConfig


def _countdown(start_frame, start_seconds, fps=10):
    return [
        (start_frame + elapsed * fps, start_seconds - elapsed)
        for elapsed in range(start_seconds + 1)
    ]


def _static(first_frame, value, count, step=10):
    """`count` samples of a clock held at `value` (pre-start display)."""
    return [(first_frame + i * step, value) for i in range(count)]


def _fracs(states, first_frame=100):
    """states: string of '0' (both off), '1' (both on), 'L'/'R' (one side)."""
    table = {"0": (0.0, 0.0), "1": (0.32, 0.31), "L": (0.32, 0.0), "R": (0.0, 0.32)}
    return [(first_frame + i, *table[c]) for i, c in enumerate(states)]


# --- find_indicator_onset -------------------------------------------------

def test_indicator_onset_clean():
    assert find_indicator_onset(_fracs("00000111111")) == 105


def test_indicator_already_on_at_window_start_is_ambiguous():
    assert find_indicator_onset(_fracs("1111111111")) is None
    # Not enough dark entries before the onset either.
    assert find_indicator_onset(_fracs("00111111")) is None


def test_indicator_flicker_shorter_than_min_run_is_ignored():
    assert find_indicator_onset(_fracs("0000110001111")) == 109
    assert find_indicator_onset(_fracs("000011000")) is None


def test_indicator_one_side_only_is_not_an_onset():
    assert find_indicator_onset(_fracs("000LLLLLLL")) is None
    assert find_indicator_onset(_fracs("000RRRRRRR")) is None


def test_indicator_threshold_respected():
    fr = [(i, 0.0, 0.0) for i in range(5)] + [(5 + i, 0.14, 0.2) for i in range(5)]
    assert find_indicator_onset(fr) is None
    assert find_indicator_onset(fr, threshold=0.1) == 5


def test_start_indicator_fractions_scales_with_frame_size():
    # 1280x720 frame: the 1080p boxes scale by 2/3.
    frame = np.zeros((720, 1280, 3), np.uint8)
    yellow_bgr = (0, 220, 230)  # OpenCV HSV hue ~28
    frame[30:63, 193:233] = yellow_bgr     # left box (290-350, 45-95) * 2/3
    left, right = start_indicator_fractions(frame)
    assert left > 0.9
    assert right == 0.0
    frame[30:63, 1046:1086] = yellow_bgr   # right box (1570-1630, 45-95) * 2/3
    assert start_indicator_fractions(frame)[1] > 0.9


# --- _find_clock_period / compute_active_windows ---------------------------

def test_find_clock_period_records_clock_first_frame():
    samples = _static(60, 20, 4) + _countdown(100, 20)
    auto = _find_clock_period(samples, 20, fps=10)
    assert auto["start_frame"] == 100
    assert auto["clock_first_frame"] == 60


def test_auto_lead_pads_only_auto():
    fps = 10
    samples = _static(20, 20, 8) + _countdown(100, 20, fps) + _countdown(500, 140, fps)
    ranges, periods, fell_back, _ = compute_active_windows(
        samples, fps=fps, total_frames=2200, auto_lead_seconds=2.5,
    )
    assert not fell_back
    auto, teleop = periods
    assert auto["start_frame"] == 100           # clock meaning unchanged
    assert auto["processing_start_frame"] == 75
    assert teleop["processing_start_frame"] == 495
    assert ranges == [(75, 330), (495, 1930)]


def test_auto_lead_clamps_at_clock_first_frame():
    fps = 10
    samples = _static(80, 20, 2) + _countdown(100, 20, fps)
    _, periods, _, _ = compute_active_windows(
        samples, fps=fps, total_frames=2200, auto_only=True, auto_lead_seconds=5.0,
    )
    assert periods[0]["clock_first_frame"] == 80
    assert periods[0]["processing_start_frame"] == 80


def test_auto_lead_clamp_never_shortens_the_legacy_lead():
    # The countdown is caught already ticking: the run's first sample IS the
    # start, so the clamp floor must not cut the 0.5 s legacy lead.
    fps = 10
    samples = _countdown(100, 20, fps)
    _, periods, _, _ = compute_active_windows(
        samples, fps=fps, total_frames=2200, auto_only=True, auto_lead_seconds=2.5,
    )
    assert periods[0]["clock_first_frame"] == 100
    assert periods[0]["processing_start_frame"] == 95


def test_compute_active_windows_unchanged_without_auto_lead():
    fps = 10
    samples = _static(20, 20, 8) + _countdown(100, 20, fps) + _countdown(500, 140, fps)
    legacy = compute_active_windows(samples, fps=fps, total_frames=2200)
    explicit = compute_active_windows(
        samples, fps=fps, total_frames=2200, auto_lead_seconds=0.5)
    assert legacy == explicit
    assert legacy[1][0]["processing_start_frame"] == 95
    assert legacy[0] == [(95, 330), (495, 1930)]


# --- _detect_auto_start_indicator (onset acceptance + fallback) -------------

def _job(fps=60.0, total=10_000):
    return SimpleNamespace(video_fps=fps, total_frames=total, video_path="unused.mp4")


def _detect(monkeypatch, onset, period):
    calls = []

    def fake(video_path, lo, hi, **_):
        calls.append((lo, hi))
        return onset

    monkeypatch.setattr(app_mod, "detect_start_indicator_onset", fake)
    out = app_mod.InferenceJob._detect_auto_start_indicator(_job(), period)
    return out, calls


def test_detect_accepts_indicator_onset_in_window(monkeypatch):
    period = {"start_frame": 420, "clock_first_frame": 100}
    out, calls = _detect(monkeypatch, 339, period)
    assert out == (339, "indicator")
    assert calls == [(240, 480)]  # start - 3 s .. start + 1 s


def test_detect_window_starts_at_clock_first_frame(monkeypatch):
    period = {"start_frame": 420, "clock_first_frame": 300}
    _, calls = _detect(monkeypatch, None, period)
    assert calls[0][0] == 300


def test_detect_falls_back_to_clock_offset(monkeypatch):
    period = {"start_frame": 420, "clock_first_frame": 100}
    assert _detect(monkeypatch, None, period)[0] == (360, "clock_offset")
    # An onset after start_frame + 0.5 s is not the start arrow.
    assert _detect(monkeypatch, 460, period)[0] == (360, "clock_offset")
    # Fallback never precedes the clock run.
    period = {"start_frame": 420, "clock_first_frame": 400}
    assert _detect(monkeypatch, None, period)[0] == (400, "clock_offset")


def test_detect_failure_falls_back(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(app_mod, "detect_start_indicator_onset", boom)
    period = {"start_frame": 420, "clock_first_frame": 100}
    assert app_mod.InferenceJob._detect_auto_start_indicator(_job(), period) == (
        360, "clock_offset")


# --- knob ------------------------------------------------------------------

def test_auto_start_lead_knob(monkeypatch):
    knob = next(k for k in KNOBS if k.name == "auto_start_lead_seconds")
    assert knob.type is float
    assert knob.default == 2.5 == app_mod.MATCH_CLOCK_AUTO_LEAD_SECONDS
    assert knob.min == 0.0
    assert knob.section == "Seeding & re-seeding"
    monkeypatch.delenv(knob.env, raising=False)
    assert JobConfig().auto_start_lead_seconds == 2.5


# --- clock_first_frame = start of the contiguous readable clock run --------

def test_clock_first_frame_spans_2025_style_zero_display():
    # 2025 broadcasts show 0:00 before the match, then jump into the 0:14
    # countdown: the run (and the lead floor) starts at the first 0:00 sample,
    # not the first countdown sample.
    fps = 10
    samples = [(f, None) for f in (0, 10)] + _static(20, 0, 6) + _countdown(80, 14, fps)
    _, periods, _, _ = compute_active_windows(
        samples, fps=fps, total_frames=2200, auto_only=True, auto_s=15,
        auto_lead_seconds=2.5,
    )
    assert periods[0]["clock_first_frame"] == 20
    assert periods[0]["processing_start_frame"] == periods[0]["start_frame"] - 25


def test_clock_first_frame_tolerates_one_unreadable_sample_but_not_two():
    one_gap = _static(20, 20, 2) + [(40, None)] + _static(50, 20, 5) + _countdown(100, 20)
    assert _find_clock_period(one_gap, 20, fps=10)["clock_first_frame"] == 20
    two_gap = _static(10, 20, 2) + [(30, None), (40, None)] + _static(50, 20, 5) + _countdown(100, 20)
    assert _find_clock_period(two_gap, 20, fps=10)["clock_first_frame"] == 50
