"""frame_knobs_fps_relative: frame-denominated tracking thresholds are tuned
"in frames at 5 fps" and rescaled to the job's tracking fps at runtime.

Hard requirement: at the 5 fps reference the runtime objects are IDENTICAL to
the knob-OFF (raw frame count) build; at other fps each threshold keeps its
duration in seconds. The stored knob values never change.
"""
import asyncio
import dataclasses
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from web.app import InferenceJob
from web.field import tracker as field_tracker_mod
from web.field.mask_tracker import (
    BACKFILL_DARK_STOP, BACKFILL_MIN_LIVED_FRAMES, DARK_REANCHOR_FRAMES,
    RETIREMENT_FRAMES, MaskFieldTracker)
from web.field.sam3_tracker import SAM3_RESEED_MAX_AGE, Sam3PredictorAdapter
from web.field.selective_prop import SelectiveConfig
from web.field.tracker import FieldTracker
from web.field.view_stride import scale_frame_knob
from web.job_config import FRAME_KNOB_REF_FPS, JobConfig, scale_frames


# ---------------------------------------------------------------------------
# scale_frames
# ---------------------------------------------------------------------------
def test_ref_fps_is_five():
    assert FRAME_KNOB_REF_FPS == 5.0


@pytest.mark.parametrize("n", list(range(0, 301)) + [1000, 3000, 10000])
def test_scale_frames_identity_at_reference_fps(n):
    assert scale_frames(n, 5.0) == n
    assert scale_frames(n, 5) == n


def test_scale_frames_zero_means_disabled_at_any_fps():
    for fps in (0.5, 2.5, 5.0, 8.0, 10.0, 30.0):
        assert scale_frames(0, fps) == 0
        assert scale_frames(0, fps, min_value=3) == 0


def test_scale_frames_doubles_at_10_fps():
    for n in (1, 2, 5, 7, 8, 24, 30, 90, 120):
        assert scale_frames(n, 10.0) == 2 * n


def test_scale_frames_rounds_at_8_fps():
    # n * 8 / 5 = n * 1.6
    assert scale_frames(7, 8.0) == 11      # 11.2
    assert scale_frames(5, 8.0) == 8       # 8.0
    assert scale_frames(24, 8.0) == 38     # 38.4
    assert scale_frames(8, 8.0) == 13      # 12.8
    assert scale_frames(120, 8.0) == 192   # 192.0
    assert scale_frames(2, 8.0) == 3       # 3.2


def test_scale_frames_min_clamp():
    assert scale_frames(1, 1.0) == 1       # 0.2 -> 0 -> clamp 1
    assert scale_frames(2, 0.5) == 1       # 0.2 -> clamp 1
    assert scale_frames(1, 1.0, min_value=2) == 2
    assert scale_frames(7, 2.5) == 4       # 3.5 -> round-half-even 4
    assert scale_frames(5, 2.5) == 2       # 2.5 -> round-half-even 2


def test_scale_frames_custom_ref():
    assert scale_frames(10, 10.0, ref_fps=10.0) == 10
    assert scale_frames(10, 5.0, ref_fps=10.0) == 5


# ---------------------------------------------------------------------------
# knob
# ---------------------------------------------------------------------------
def test_knob_default_on_and_env_override(monkeypatch):
    assert JobConfig().frame_knobs_fps_relative is True
    monkeypatch.setenv("FRAME_KNOBS_FPS_RELATIVE", "0")
    assert JobConfig().frame_knobs_fps_relative is False
    assert JobConfig({"frame_knobs_fps_relative": True}).frame_knobs_fps_relative is True


# ---------------------------------------------------------------------------
# runtime builders on InferenceJob
# ---------------------------------------------------------------------------
def _job(fps=5.0, on=True, **overrides):
    cfg = JobConfig({"sam3_track_fps": fps, "frame_knobs_fps_relative": on,
                     **overrides})
    with patch("web.app.YOLO"):
        return InferenceJob("test_job", "/mock/video.mp4", config=cfg)


# Non-default reference values so an unscaled field can't pass by accident.
_SEL_OVERRIDES = dict(
    selective_gap_trigger=7, selective_min_streak=5,
    selective_window_max_frames=120, selective_kill_coast_frames=24,
    selective_starve_trigger=9, selective_starve_max_frames=240,
    selective_starve_min_obs=30, dup_reap_updates=8, blind_zone_max_coast=40,
    evict_dark_frames=8)


@pytest.mark.parametrize("overrides", [{}, _SEL_OVERRIDES,
                                       {"selective_starve_max_frames": 0,
                                        "selective_starve_min_obs": 0,
                                        "blind_zone_max_coast": 0}])
def test_identity_at_5_fps_on_equals_off(overrides):
    on, off = _job(5.0, True, **overrides), _job(5.0, False, **overrides)
    assert on._selective_config() == off._selective_config()
    assert on._field_tracker_frame_kwargs() == off._field_tracker_frame_kwargs()
    for stride in (1, 2, 3, 1.37, 2.5):
        assert (on._mask_tracker_frame_kwargs(stride)
                == off._mask_tracker_frame_kwargs(stride))
    assert on._scale_frames(SAM3_RESEED_MAX_AGE) == SAM3_RESEED_MAX_AGE


def test_off_matches_historical_raw_values():
    """Knob OFF (any fps) == the pre-change wiring: raw knob values, tracker
    defaults, evict_dark / backfill_min_lived divided by stride only."""
    job = _job(10.0, False, **_SEL_OVERRIDES)
    c = job.config
    sel = job._selective_config()
    assert (sel.gap_trigger, sel.min_streak, sel.window_max_frames,
            sel.kill_coast_frames, sel.starve_trigger, sel.starve_max_frames,
            sel.starve_min_obs, sel.max_missed) == (
        c.selective_gap_trigger, c.selective_min_streak,
        c.selective_window_max_frames, c.selective_kill_coast_frames,
        c.selective_starve_trigger, c.selective_starve_max_frames,
        c.selective_starve_min_obs, SelectiveConfig().max_missed)
    assert job._mask_tracker_frame_kwargs(3) == dict(
        evict_dark_frames=scale_frame_knob(c.evict_dark_frames, 3),
        backfill_min_lived_frames=scale_frame_knob(BACKFILL_MIN_LIVED_FRAMES, 3),
        retirement_frames=RETIREMENT_FRAMES,
        dark_reanchor_frames=DARK_REANCHOR_FRAMES,
        backfill_dark_stop=BACKFILL_DARK_STOP)
    assert job._field_tracker_frame_kwargs() == dict(
        max_coast_open=8, max_coast_occluded=90, guard_motion_window=5,
        dup_reap_updates=8, blind_zone_max_coast=40)


def test_selective_config_scaled_at_10_fps():
    job = _job(10.0, True, **_SEL_OVERRIDES)
    sel = job._selective_config()
    assert sel.gap_trigger == 14
    assert sel.min_streak == 10
    assert sel.window_max_frames == 240
    assert sel.kill_coast_frames == 48
    assert sel.starve_trigger == 18
    assert sel.starve_max_frames == 480
    assert sel.starve_min_obs == 60
    assert sel.max_missed == 60
    # non-frame fields untouched
    ref = _job(10.0, False, **_SEL_OVERRIDES)._selective_config()
    frame_fields = {"gap_trigger", "min_streak", "window_max_frames",
                    "kill_coast_frames", "starve_trigger", "starve_max_frames",
                    "starve_min_obs", "max_missed"}
    for f in dataclasses.fields(SelectiveConfig):
        if f.name not in frame_fields:
            assert getattr(sel, f.name) == getattr(ref, f.name), f.name
    # stored knob values unchanged
    assert job.config.selective_gap_trigger == 7
    assert job.config.selective_window_max_frames == 120


def test_selective_zero_stays_zero_at_10_fps():
    sel = _job(10.0, True, selective_starve_max_frames=0,
               selective_starve_min_obs=0)._selective_config()
    assert sel.starve_max_frames == 0
    assert sel.starve_min_obs == 0


def test_selective_config_rounds_at_8_fps():
    sel = _job(8.0, True)._selective_config()
    assert sel.gap_trigger == 11       # 7 * 1.6 = 11.2
    assert sel.min_streak == 8
    assert sel.window_max_frames == 192
    assert sel.max_missed == 48


def test_mask_tracker_kwargs_scaled_at_10_fps():
    job = _job(10.0, True, evict_dark_frames=8)
    assert job._mask_tracker_frame_kwargs(1) == dict(
        evict_dark_frames=16,
        backfill_min_lived_frames=2 * BACKFILL_MIN_LIVED_FRAMES,
        retirement_frames=2 * RETIREMENT_FRAMES,
        dark_reanchor_frames=2 * DARK_REANCHOR_FRAMES,
        backfill_dark_stop=2 * BACKFILL_DARK_STOP)
    # stride still divides the two historically stride-scaled knobs
    kw3 = job._mask_tracker_frame_kwargs(3)
    assert kw3["evict_dark_frames"] == scale_frame_knob(16, 3)
    assert kw3["backfill_min_lived_frames"] == scale_frame_knob(
        2 * BACKFILL_MIN_LIVED_FRAMES, 3)
    assert kw3["retirement_frames"] == 2 * RETIREMENT_FRAMES


def test_field_tracker_kwargs_scaled_at_10_fps():
    job = _job(10.0, True, dup_reap_updates=8, blind_zone_max_coast=40)
    assert job._field_tracker_frame_kwargs() == dict(
        max_coast_open=16, max_coast_occluded=180, guard_motion_window=10,
        dup_reap_updates=16, blind_zone_max_coast=80)
    assert _job(10.0, True, blind_zone_max_coast=0)._field_tracker_frame_kwargs()[
        "blind_zone_max_coast"] == 0


def test_field_tracker_kwargs_are_valid_constructor_args():
    kw = _job(10.0, True)._field_tracker_frame_kwargs()
    t = FieldTracker(16.54, **kw)
    d = t.config_dict()
    assert d["max_coast_open"] == 16 and d["max_coast_occluded"] == 180
    assert d["guard_motion_window"] == 10 and d["dup_reap_updates"] == 16


def test_per_job_target_tracking_fps_override_wins():
    job = _job(5.0, True)
    job.target_tracking_fps = 10.0
    assert job._frame_knob_fps() == 10.0
    assert job._selective_config().gap_trigger == 14


def test_run_view_tracking_uses_scaled_selective_config():
    job = _job(10.0, True)
    job.config.selective_mask_prop = True
    job.timings = {}
    tracker = MagicMock()
    tracker.track_selective.side_effect = RuntimeError("boom")
    tracker.track.return_value = {}

    async def run():
        return await job._run_view_tracking(
            tracker, SimpleNamespace(name="main"), {"frames_dir": "/frames"},
            lambda i, p: [], lambda a, b: None)

    asyncio.run(run())
    cfg = tracker.track_selective.call_args.kwargs["cfg"]
    assert cfg.gap_trigger == 2 * job.config.selective_gap_trigger


# ---------------------------------------------------------------------------
# tracker-side plumbing (defaults unchanged)
# ---------------------------------------------------------------------------
def test_field_tracker_defaults_are_the_named_constants():
    d = FieldTracker(16.54).config_dict()
    assert d["max_coast_open"] == field_tracker_mod.MAX_COAST_OPEN == 8
    assert d["max_coast_occluded"] == field_tracker_mod.MAX_COAST_OCCLUDED == 90
    assert d["guard_motion_window"] == field_tracker_mod.GUARD_MOTION_WINDOW == 5


def test_mask_tracker_track_defaults_unchanged():
    params = inspect.signature(MaskFieldTracker.track).parameters
    assert params["retirement_frames"].default == RETIREMENT_FRAMES == 15
    assert params["backfill_dark_stop"].default == BACKFILL_DARK_STOP
    assert params["dark_reanchor_frames"].default == DARK_REANCHOR_FRAMES


def test_sam3_adapter_reseed_max_age():
    assert Sam3PredictorAdapter(MagicMock()).reseed_max_age == SAM3_RESEED_MAX_AGE
    assert Sam3PredictorAdapter(MagicMock(), reseed_max_age=48).reseed_max_age == 48
