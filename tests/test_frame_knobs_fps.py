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
                    "starve_min_obs", "max_missed", "velocity_deltas"}
    assert sel.velocity_deltas == 6
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


# ===========================================================================
# Second pass: look-back windows, per-update rates, identity-layer values.
# Every new value must be IDENTICAL to the pre-change constant at 5 fps (knob
# ON) and with the knob OFF at any fps.
# ===========================================================================
import math

import numpy as np

from web.field import identity_confidence as ic_mod
from web.field import selective_prop as sp
from web.field import track_evidence as te_mod
from web.field import window_labels as wl
from web.field.identity_confidence import ConfidenceConfig
from web.field.tracker import GATE_WIDEN_PER_MISS, KF_PROCESS_Q, _KF
from web.job_config import scale_accel_noise, scale_alpha, scale_decay, scale_step

# The pre-change hardcoded values (5 fps reference) of every new parameter.
_OLD = dict(coast_vel_decay=0.85, gate_widen_per_miss=0.5, kf_q=2.0,
            decay_rate=0.005, observation_rise=0.02, low_margin_drop=0.05,
            roster_conflict_drop=0.2, velocity_deltas=3, fill_stop_slack=2,
            displacement_window=5, min_overlap_frames=2, identity_interval=15,
            identity_warmup=5, positions_history=30, ident_pos_history=600)


def _second_pass_values(job):
    from web.app import IDENTITY_INTERVAL
    from web.field.identity_segments import SegmentParams
    ft = job._field_tracker_rate_kwargs()
    cc = job._confidence_rate_kwargs()
    return dict(
        coast_vel_decay=ft["coast_vel_decay"],
        gate_widen_per_miss=ft["gate_widen_per_miss"], kf_q=ft["kf_q"],
        **cc,
        velocity_deltas=job._selective_config().velocity_deltas,
        fill_stop_slack=job._scale_frames(wl.FILL_STOP_SLACK_FRAMES),
        displacement_window=job._scale_frames(te_mod.DISPLACEMENT_WINDOW),
        min_overlap_frames=job._scale_frames(SegmentParams.min_overlap_frames),
        identity_interval=job._scale_frames(IDENTITY_INTERVAL),
        identity_warmup=job._identity_warmup_frames(),
        positions_history=job._scale_frames(30),
        ident_pos_history=job._scale_frames(600),
    )


def test_constants_are_the_old_values():
    assert GATE_WIDEN_PER_MISS == 0.5 and KF_PROCESS_Q == 2.0
    assert wl.FILL_STOP_SLACK_FRAMES == 2
    assert te_mod.DISPLACEMENT_WINDOW == 5
    assert SelectiveConfig().velocity_deltas == 3
    d = ConfidenceConfig()
    assert (d.decay_rate, d.observation_rise, d.low_margin_drop,
            d.roster_conflict_drop) == (0.005, 0.02, 0.05, 0.2)
    assert JobConfig().coast_vel_decay == 0.85
    assert JobConfig().identity_warmup_frames == 5


def test_second_pass_identity_at_5_fps_on_equals_off_equals_old():
    on, off = _job(5.0, True), _job(5.0, False)
    v_on, v_off = _second_pass_values(on), _second_pass_values(off)
    assert v_on == v_off == _OLD
    # exact float identity, not just ==
    for k in ("coast_vel_decay", "gate_widen_per_miss", "kf_q", "decay_rate",
              "observation_rise", "low_margin_drop", "roster_conflict_drop"):
        assert repr(v_on[k]) == repr(_OLD[k]), k


def test_second_pass_off_is_raw_at_10_fps():
    assert _second_pass_values(_job(10.0, False)) == _OLD


def test_second_pass_scaled_at_10_fps():
    v = _second_pass_values(_job(10.0, True))
    assert v["coast_vel_decay"] == pytest.approx(0.85 ** 0.5)
    assert v["gate_widen_per_miss"] == 0.25
    assert v["kf_q"] == 4.0
    assert v["decay_rate"] == pytest.approx(0.0025)
    assert v["observation_rise"] == pytest.approx(0.01)
    assert v["low_margin_drop"] == pytest.approx(0.025)
    assert v["roster_conflict_drop"] == pytest.approx(0.1)
    assert v["velocity_deltas"] == 6
    assert v["fill_stop_slack"] == 4
    assert v["displacement_window"] == 10
    assert v["min_overlap_frames"] == 4
    assert v["identity_interval"] == 30
    assert v["identity_warmup"] == 10
    assert v["positions_history"] == 60
    assert v["ident_pos_history"] == 1200


def test_custom_coast_vel_decay_knob_identity_and_scaling():
    for d in (0.0, 0.5, 0.7, 1.0):
        on = _job(5.0, True, coast_vel_decay=d)._field_tracker_rate_kwargs()
        off = _job(5.0, False, coast_vel_decay=d)._field_tracker_rate_kwargs()
        assert on == off and on["coast_vel_decay"] == d
    k10 = _job(10.0, True, coast_vel_decay=0.7)._field_tracker_rate_kwargs()
    assert k10["coast_vel_decay"] == pytest.approx(0.7 ** 0.5)
    # 0 (stop at once) and 1 (no decay) keep their meaning at any fps
    assert _job(10.0, True, coast_vel_decay=0.0)._field_tracker_rate_kwargs()[
        "coast_vel_decay"] == 0.0
    assert _job(10.0, True, coast_vel_decay=1.0)._field_tracker_rate_kwargs()[
        "coast_vel_decay"] == 1.0


def test_warmup_zero_stays_zero():
    assert _job(10.0, True, identity_warmup_frames=0)._identity_warmup_frames() == 0


# ---------------------------------------------------------------------------
# rate helper math
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("v", [0.0, 1e-9, 0.005, 0.02, 0.05, 0.2, 0.3, 0.5,
                               0.85, 0.9, 0.999, 1.0, 2.0, 7.25])
def test_rate_helpers_identity_at_reference(v):
    for fn in (scale_decay, scale_alpha, scale_step, scale_accel_noise):
        out = fn(v, 5.0)
        assert out == v and repr(out) == repr(float(v)), fn.__name__
        assert fn(v, 5) == v


def test_scale_decay_keeps_per_second_retention():
    for d in (0.5, 0.85, 0.95):
        for fps in (2.5, 8.0, 10.0, 15.0, 30.0):
            per_update = scale_decay(d, fps)
            # fps/5 updates == one 5-fps update
            assert per_update ** (fps / 5.0) == pytest.approx(d, rel=1e-12)
    assert scale_decay(0.85, 10.0) == pytest.approx(math.sqrt(0.85))
    assert scale_decay(0.0, 10.0) == 0.0 and scale_decay(1.0, 10.0) == 1.0


def test_scale_alpha_keeps_old_state_retention():
    for a in (0.1, 0.3, 0.9):
        for fps in (2.5, 8.0, 10.0, 30.0):
            a2 = scale_alpha(a, fps)
            assert (1.0 - a2) ** (fps / 5.0) == pytest.approx(1.0 - a, rel=1e-12)
    assert scale_alpha(0.3, 10.0) == pytest.approx(1.0 - math.sqrt(0.7))
    assert scale_alpha(0.0, 10.0) == 0.0 and scale_alpha(1.0, 10.0) == 1.0


def test_scale_step_and_accel_noise():
    assert scale_step(0.5, 10.0) == 0.25
    assert scale_step(0.02, 8.0) == pytest.approx(0.0125)
    assert scale_accel_noise(2.0, 10.0) == 4.0
    assert scale_accel_noise(2.0, 2.5) == 1.0


def test_job_rate_helpers_gated_by_knob():
    on, off = _job(10.0, True), _job(10.0, False)
    assert off._scale_decay(0.85) == 0.85 and off._scale_alpha(0.3) == 0.3
    assert off._scale_step(0.5) == 0.5 and off._scale_accel_noise(2.0) == 2.0
    assert on._scale_decay(0.85) == pytest.approx(0.85 ** 0.5)
    assert on._scale_alpha(0.3) == pytest.approx(1 - 0.7 ** 0.5)
    assert on._scale_step(0.5) == 0.25 and on._scale_accel_noise(2.0) == 4.0
    five = _job(5.0, True)
    for x in (0.85, 0.3, 0.5, 2.0):
        assert (five._scale_decay(x), five._scale_alpha(x), five._scale_step(x),
                five._scale_accel_noise(x)) == (x, x, x, x)


def test_bare_job_without_config_keeps_reference_values():
    bare = InferenceJob.__new__(InferenceJob)
    assert bare._scale_frames(5) == 5
    assert bare._scale_step(0.5) == 0.5


# ---------------------------------------------------------------------------
# FieldTracker: gate widening, KF q, coast decay
# ---------------------------------------------------------------------------
def test_gate_mult_matches_legacy_formula_at_default():
    t = FieldTracker(16.54)
    for m in range(0, 20):
        assert t._gate_mult(m) == min(4.0, 1.0 + 0.5 * m)


def test_gate_widening_same_per_second_at_10_fps():
    t5 = FieldTracker(16.54)
    t10 = FieldTracker(16.54, **_job(10.0, True)._field_tracker_rate_kwargs())
    for s in (0.2, 0.4, 0.6, 1.0, 1.2, 2.0):
        assert t10._gate_mult(round(s * 10)) == pytest.approx(t5._gate_mult(round(s * 5)))


def test_config_dict_new_keys_only_when_scaled():
    d5 = FieldTracker(16.54, **_job(5.0, True)._field_tracker_rate_kwargs()).config_dict()
    assert "gate_widen_per_miss" not in d5 and "kf_q" not in d5
    assert d5 == FieldTracker(16.54).config_dict()
    kw = {**_job(10.0, True)._field_tracker_frame_kwargs(),
          **_job(10.0, True)._field_tracker_rate_kwargs()}
    d = FieldTracker(16.54, **kw).config_dict()
    assert d["gate_widen_per_miss"] == 0.25 and d["kf_q"] == 4.0
    assert d["coast_vel_decay"] == pytest.approx(0.85 ** 0.5)
    # trace replay round-trip
    assert FieldTracker(**d).config_dict() == d


def test_seeded_slot_kf_uses_tracker_q():
    from web.field.tracker import FieldDetection
    t = FieldTracker(16.54, kf_q=4.0)
    t.update([FieldDetection(pos=(3.0, 4.0), color="blue", box=(0, 0, 1, 1))], {}, 0.1)
    assert next(s for s in t.tracks if s.active).kf.q == 4.0
    t5 = FieldTracker(16.54)
    t5.update([FieldDetection(pos=(3.0, 4.0), color="blue", box=(0, 0, 1, 1))], {}, 0.2)
    assert next(s for s in t5.tracks if s.active).kf.q == 2.0


def test_kf_velocity_variance_growth_per_second_matches():
    k5, k10 = _KF((0.0, 0.0), q=2.0), _KF((0.0, 0.0), q=scale_accel_noise(2.0, 10.0))
    for _ in range(5):
        k5.predict(0.2)
    for _ in range(10):
        k10.predict(0.1)
    assert k10.P[2, 2] == pytest.approx(k5.P[2, 2], rel=1e-12)
    assert k10.P[3, 3] == pytest.approx(k5.P[3, 3], rel=1e-12)


def test_coast_velocity_after_one_second_matches():
    v5, v10 = 1.0, 1.0
    for _ in range(5):
        v5 *= 0.85
    d10 = _job(10.0, True)._field_tracker_rate_kwargs()["coast_vel_decay"]
    for _ in range(10):
        v10 *= d10
    assert v10 == pytest.approx(v5, rel=1e-12)


def _drive(tracker, n, dt):
    """A deterministic two-robot sequence with misses (coast + re-acquire)."""
    from web.field.tracker import FieldDetection
    out = []
    for k in range(n):
        dets = []
        if k % 7 not in (3, 4):
            dets.append(FieldDetection(pos=(2.0 + 0.3 * k, 3.0), color="blue",
                                       box=(0, 0, 1, 1)))
        if k % 5 != 2:
            dets.append(FieldDetection(pos=(12.0 - 0.2 * k, 5.0 + 0.05 * k),
                                       color="red", box=(0, 0, 1, 1)))
        outs = tracker.update(dets, {}, dt)
        out.append([(o.slot_id, o.pos, o.vel, o.state) for o in outs])
    return out


def test_field_tracker_behaviour_byte_identical_at_5_fps():
    job = _job(5.0, True)
    kw = {**job._field_tracker_frame_kwargs(), **job._field_tracker_rate_kwargs()}
    assert _drive(FieldTracker(16.54, **kw), 40, 0.2) == _drive(FieldTracker(16.54), 40, 0.2)


# ---------------------------------------------------------------------------
# Selective TBD velocity look-back
# ---------------------------------------------------------------------------
def _legacy_velocity(centers):
    c = centers[-4:]
    if len(c) < 2:
        return (0.0, 0.0)
    dxs = [c[i][0] - c[i - 1][0] for i in range(1, len(c))]
    dys = [c[i][1] - c[i - 1][1] for i in range(1, len(c))]
    return (sum(dxs) / len(dxs), sum(dys) / len(dys))


def test_tbd_velocity_default_matches_legacy():
    trk = sp._Track(1, (0, 0, 10, 10), "red", 0)
    for f in range(1, 9):
        trk.observe((3 * f + f * f % 4, 2 * f, 3 * f + 10, 2 * f + 10), "red", f)
        assert trk.velocity() == _legacy_velocity(trk._centers)
        assert trk.velocity(3) == _legacy_velocity(trk._centers)


def test_tbd_velocity_lookback_scales():
    trk = sp._Track(1, (0, 0, 10, 10), None, 0)
    for f in range(1, 10):
        trk.observe((f * 2, 0, f * 2 + 10, 10), None, f)
    # constant motion: any look-back gives the same 2 px/frame
    assert trk.velocity(3) == trk.velocity(6) == (2.0, 0.0)
    trk.observe((100, 0, 110, 10), None, 10)      # one jump
    assert trk.velocity(6)[0] < trk.velocity(3)[0]  # longer window, less jump weight


def _tbd_run(cfg):
    tbd = sp.TbdTracker(cfg)
    for f in range(30):
        dets = [((5 * f, 10, 5 * f + 40, 50), "red"),
                ((300 - 4 * f, 12, 340 - 4 * f, 52), "blue")]
        if f % 6 == 3:
            dets = dets[:1]
        tbd.step(f, dets)
    return {f: {o: (v.box, v.margin, v.color) for o, v in obs.items()}
            for f, obs in tbd.history.items()}


def test_tbd_history_identical_at_5_fps():
    assert _tbd_run(_job(5.0, True)._selective_config()) == _tbd_run(SelectiveConfig())


def test_kill_coast_capture_uses_velocity_deltas_plus_one_boxes():
    def capture(cfg):
        spec = sp.WindowSpec(oid=1, seed_frame=0, entry_frame=0, trigger="margin")
        m = sp.ExitMachine([spec], {1: (0, 0, 4, 4)}, (40, 200), {}, cfg)
        w = m._wins[0]
        for f in range(12):
            mask = np.zeros((40, 200), bool)
            mask[5:15, 10 * f:10 * f + 8] = True
            w.masks[f] = mask
        m._capture_coast(w, 11 + cfg.min_streak)
        return w.coast_seed_frame, w.coast_boxes
    base = SelectiveConfig(kill_coast=True)
    seed, boxes = capture(base)
    assert seed == 11 and len(boxes) == 4                      # legacy frames[-4:]
    seed6, boxes6 = capture(dataclasses.replace(base, velocity_deltas=6))
    assert seed6 == 11 and len(boxes6) == 7
    assert sp._coast_velocity(boxes) == sp._coast_velocity(boxes6) == (10.0, 0.0)


# ---------------------------------------------------------------------------
# window-label fill slack (live Params + offline script)
# ---------------------------------------------------------------------------
def test_fill_runs_slack_default_is_reference():
    assert inspect.signature(wl.fill_runs).parameters[
        "stop_slack_frames"].default == wl.FILL_STOP_SLACK_FRAMES == 2
    assert wl.Params(fps=30.0, red=[], blue=[]).fill_stop_slack_frames == 2


def test_offline_fill_slack_from_replay_config():
    import scripts.fill_replay_labels as fr
    assert fr.fill_stop_slack({}) == 2
    assert fr.fill_stop_slack({"config": {}}) == 2
    assert fr.fill_stop_slack({"config": {"sam3_track_fps": 5.0}}) == 2
    assert fr.fill_stop_slack({"config": {"sam3_track_fps": 10.0}}) == 4
    assert fr.fill_stop_slack({"config": {"sam3_track_fps": 8.0}}) == 3
    assert fr.fill_stop_slack({"config": {"sam3_track_fps": 10.0,
                                          "frame_knobs_fps_relative": False}}) == 2
    assert fr.fill_stop_slack({"config": {"sam3_track_fps": "junk"}}) == 2


def _fill_fixture():
    """Slot B1 in two runs (a gap), and an R1 chain starting right beside
    B1's second run: B1's named first run fills across, then stops at the
    hazard."""
    frames = []
    for i in range(40):
        robots = []
        if i < 15 or i >= 18:
            robots.append({"slot": "B1", "field_pos": [2.0 + 0.01 * i, 3.0]})
        if i >= 25:
            robots.append({"slot": "R1", "field_pos": [2.5, 3.2]})
        frames.append({"frame_num": i * 6, "robots": robots})
    return frames


def test_fill_runs_slack_param_effect():
    frames = _fill_fixture()
    ws = wl.build_windows(frames, 30.0)

    def run(**kw):
        lab = {}
        first = min((w for w in ws if w.slot == "B1"), key=lambda w: w.idx[0])
        for i in first.idx:
            lab[(i, "B1")] = ("1234", wl.RANK_DIGITS, 0.9, len(first.idx))
        st = wl.fill_runs(frames, ws, lab, 30.0, **kw)
        return st, lab
    assert run() == run(stop_slack_frames=2)
    st2, lab2 = run(stop_slack_frames=2)
    st4, lab4 = run(stop_slack_frames=4)
    # R1 starts at frame 25: the B1 edge fill (from 18 on) stops 2 frames
    # earlier with the 4-frame (10 fps, same 0.4 s) slack
    assert st2["fill_frames"] == 5 and st4["fill_frames"] == 3


# ---------------------------------------------------------------------------
# track verdict displacement window
# ---------------------------------------------------------------------------
def test_classify_track_displacement_window_default_and_scaled():
    ev = te_mod.TrackEvidence(track_id=1)
    for f in range(10):            # 2 windows of 5 at 5 fps: moved 2 m
        ev.add_position(f, 0.0 if f < 5 else 2.0, 0.0)
    assert te_mod.classify_track(ev) == "robot"
    assert te_mod.classify_track(ev, displacement_window=5) == "robot"
    # the same 10 samples are only 1 s at 10 fps: < 2 windows of 10 -> 0 m
    assert te_mod.classify_track(ev, displacement_window=10) == "static"
    assert te_mod.track_verdicts(SimpleNamespace(by_track={1: ev}),
                                 displacement_window=10) == {1: "static"}


def test_slot_dropped_ranges_window_default():
    from web.field.track_segments import classify_segment, slot_dropped_ranges
    for fn in (slot_dropped_ranges, classify_segment):
        assert inspect.signature(fn).parameters[
            "displacement_window"].default == te_mod.DISPLACEMENT_WINDOW


# ---------------------------------------------------------------------------
# identity confidence ramps + warm-up + buffers
# ---------------------------------------------------------------------------
def test_confidence_config_built_at_init_is_reference_at_5_fps():
    job = _job(5.0, True)
    ref = ConfidenceConfig(confident_margin=job._emb_thr().confident_margin)
    assert job.confidence_config == ref
    assert _job(5.0, False).confidence_config == ref
    assert _job(10.0, False).confidence_config == ref


def test_confidence_update_roster_drop_uses_config():
    st = ic_mod.ConfidenceState(confidence=0.7, is_confirmed=False)
    sig = ic_mod.ConfidenceSignals(is_observed=False, roster_uniqueness_ok=False)
    out = ic_mod.calculate_confidence_update(st, sig, ConfidenceConfig())
    assert out.new_confidence == pytest.approx(0.7 - 0.005 - 0.2)
    out2 = ic_mod.calculate_confidence_update(
        st, sig, ConfidenceConfig(decay_rate=0.0025, roster_conflict_drop=0.1))
    assert out2.new_confidence == pytest.approx(0.7 - 0.0025 - 0.1)


def test_confidence_ramp_same_per_second_at_10_fps():
    cc10 = _job(10.0, True).confidence_config
    sig = ic_mod.ConfidenceSignals(is_observed=True, relink_margin=1.0)
    c5 = c10 = 0.5
    for _ in range(5):
        c5 = ic_mod.calculate_confidence_update(
            ic_mod.ConfidenceState(c5, False), sig, ConfidenceConfig()).new_confidence
    for _ in range(10):
        c10 = ic_mod.calculate_confidence_update(
            ic_mod.ConfidenceState(c10, False), sig, cc10).new_confidence
    assert c10 == pytest.approx(c5)


def test_identity_rates_reapplied_for_per_job_override():
    job = _job(5.0, True, identity_continuity=True)
    assert job.identity_registry.warmup_frames == 5
    job.target_tracking_fps = 10.0
    job._apply_identity_frame_rates()
    assert job.identity_registry.warmup_frames == 10
    assert job.confidence_config.observation_rise == pytest.approx(0.01)
    assert job.confidence_config.vlm_reinforce == ConfidenceConfig().vlm_reinforce


def test_positions_history_buffer_scales():
    def maxlen(job):
        return job._get_or_create_identity_track(7, "red").positions_history.maxlen
    assert maxlen(_job(5.0, True)) == 30
    assert maxlen(_job(10.0, True)) == 60
    assert maxlen(_job(10.0, False)) == 30
