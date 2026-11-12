"""The processing job on a season whose broadcast re-tiles the camera row at the
endgame (Season.endgame_layout_switch; 2025): the per-match corner check runs on the
pre-endgame framing only, and the job records the cutoff. 2026 is untouched. No real
footage."""
import importlib

import numpy as np
import pytest

from web.field import auto_calibrate as ac
from web.field import camera_motion as cm
from web.field import corner_per_match as cpm
from web.field import store as field_store
from web.field.camera_view import CameraView
from tests.field.test_corner_per_match import BASE, CROP, FM, _fake_outcome, _video
from tests.field.test_endgame_cutoff import FPS, S25, S26, _clock



@pytest.fixture
def app_mod(tmp_path, monkeypatch):
    mod = importlib.import_module("web.app")
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(mod, "_auto_calib_judge", lambda: None)
    return mod


def _windows(app_mod, samples, season):
    return app_mod.compute_active_windows(samples, FPS, int(200 * FPS),
                                          auto_s=season.auto_s, teleop_s=season.teleop_s)


def _job(app_mod, tmp_path, season=S25, samples=None, fallback=False):
    """A job after its clock scan: processing ranges / periods from ``samples`` (the
    synthetic 2025 clock by default) or the legacy fallback range."""
    from web.job_config import JobConfig
    job = app_mod.InferenceJob.__new__(app_mod.InferenceJob)
    job.config = JobConfig({})
    job.event_key = f"{season.year}tst"
    job.match_key = f"{season.year}tst_qm5"
    job.video_fps = FPS
    job.total_frames = int(200 * FPS)
    job.video_path = str(_video(tmp_path))
    job.field_model = FM
    job.timings = {}
    if samples is None and not fallback:
        samples, _ = _clock()
    job._clock_samples = samples or []
    job._match_season = season
    if fallback:
        job.processing_ranges = [(90, job.total_frames - 1)]
        job.match_periods, job.timer_gate_fallback = [], True
        job.timer_gate_fallback_reason = "autonomous countdown was not found"
    else:
        (job.processing_ranges, job.match_periods, job.timer_gate_fallback,
         job.timer_gate_fallback_reason) = _windows(app_mod, job._clock_samples, season)
    v0 = field_store.save_version(job.event_key, "corner_left", BASE)
    job._view_selection = {"corner_left": v0}
    job.views = [CameraView("main", (0, 0, 1920, 723), (), BASE),
                 CameraView("corner_left", CROP, (), BASE)]
    return job


# ------------------------------------------------------------ the job's cutoff

def test_cutoff_from_the_jobs_clock_scan(app_mod, tmp_path):
    samples, truth = _clock()
    job = _job(app_mod, tmp_path, samples=samples)
    eg = app_mod.InferenceJob._compute_endgame_cutoff(job)
    assert eg["method"] == "match_clock"
    assert eg["endgame_t_s"] == pytest.approx(truth, abs=1.0)
    assert eg["cutoff_s"] == pytest.approx(truth - ac.ENDGAME_CLOCK_MARGIN_S, abs=1.0)
    assert eg["cutoff_frame"] == int(round(eg["cutoff_s"] * FPS))
    assert eg["cutoff_frame"] < int(truth * FPS)           # before the layout switch


def test_no_cutoff_for_a_season_without_the_switch(app_mod, tmp_path):
    samples, _ = _clock()
    job = _job(app_mod, tmp_path, season=S26, samples=samples)
    assert app_mod.InferenceJob._compute_endgame_cutoff(job) is None


def test_cutoff_falls_back_to_the_estimate_without_a_clock(app_mod, tmp_path):
    job = _job(app_mod, tmp_path, fallback=True)
    eg = app_mod.InferenceJob._compute_endgame_cutoff(job)
    assert eg["method"] == "estimate"
    est = S25.auto_s + S25.teleop_s - S25.endgame_s - ac.ENDGAME_CLOCK_MARGIN_S
    assert eg["cutoff_s"] == pytest.approx(est)
    assert eg["cutoff_frame"] == int(round(est * FPS))


def test_season_comes_from_the_event_when_not_stored(app_mod, tmp_path):
    job = _job(app_mod, tmp_path, fallback=True)
    del job._match_season
    job.event_key = "2025hop"
    assert app_mod.InferenceJob._compute_endgame_cutoff(job)["method"] == "estimate"
    job.event_key = "2026hop"
    assert app_mod.InferenceJob._compute_endgame_cutoff(job) is None


# ------------------------------------------------ Part A: per-match corner check

def _capture_frange(monkeypatch, status="kept"):
    seen = {}

    def fake(video, mk, cam, crop, masks, fm, epoch, frange, fps, *a, **k):
        seen["frange"], seen["fps"] = frange, fps
        return _fake_outcome(status)()
    monkeypatch.setattr(cpm, "corner_step", fake)
    return seen


def test_2025_corner_check_stops_at_the_cutoff(app_mod, tmp_path, monkeypatch):
    job = _job(app_mod, tmp_path)
    job.endgame_cutoff = app_mod.InferenceJob._compute_endgame_cutoff(job)
    seen = _capture_frange(monkeypatch)
    out = app_mod.InferenceJob._corner_per_match_step(job)
    cut = job.endgame_cutoff["cutoff_frame"]
    assert seen["frange"] == (job.processing_ranges[0][0], cut - 1)
    assert seen["frange"][1] < job.processing_ranges[-1][1]
    # recorded where the calibrate page reads it
    e = field_store.load_permatch_log(job.event_key, "corner_left")[job.match_key]
    assert e["endgame_cutoff"]["cutoff_s"] == job.endgame_cutoff["cutoff_s"]
    assert e["endgame_cutoff"]["method"] == "match_clock"
    assert e["metrics"]["endgame_cutoff_s"] == job.endgame_cutoff["cutoff_s"]
    assert e["metrics"]["endgame_cutoff_method"] == "match_clock"
    assert out["corner_left"]["metrics"]["endgame_cutoff_s"] == job.endgame_cutoff["cutoff_s"]


def test_2025_fallback_corner_check_uses_the_estimate(app_mod, tmp_path, monkeypatch):
    job = _job(app_mod, tmp_path, fallback=True)
    job.endgame_cutoff = app_mod.InferenceJob._compute_endgame_cutoff(job)
    seen = _capture_frange(monkeypatch)
    app_mod.InferenceJob._corner_per_match_step(job)
    assert seen["frange"] == (90, job.endgame_cutoff["cutoff_frame"] - 1)
    e = field_store.load_permatch_log(job.event_key, "corner_left")[job.match_key]
    assert e["endgame_cutoff"]["method"] == "estimate"


def test_2026_corner_check_covers_the_whole_match(app_mod, tmp_path, monkeypatch):
    samples, _ = _clock()
    job = _job(app_mod, tmp_path, season=S26, samples=samples)
    job.endgame_cutoff = app_mod.InferenceJob._compute_endgame_cutoff(job)
    seen = _capture_frange(monkeypatch)
    out = app_mod.InferenceJob._corner_per_match_step(job)
    assert seen["frange"] == (job.processing_ranges[0][0], job.processing_ranges[-1][1])
    e = field_store.load_permatch_log(job.event_key, "corner_left")[job.match_key]
    assert "endgame_cutoff" not in e and "endgame_cutoff_s" not in e["metrics"]
    assert "endgame_cutoff_s" not in out["corner_left"]["metrics"]


def test_job_without_a_cutoff_attribute_is_unchanged(app_mod, tmp_path, monkeypatch):
    job = _job(app_mod, tmp_path)          # 2025 ranges, but no cutoff computed
    seen = _capture_frange(monkeypatch)
    app_mod.InferenceJob._corner_per_match_step(job)
    assert seen["frange"] == (job.processing_ranges[0][0], job.processing_ranges[-1][1])


def test_2025_motion_check_never_sees_the_layout_switch(app_mod, tmp_path, monkeypatch):
    """Real corner_step: the motion check reports a 'move' whenever its window reaches
    the endgame re-tile. With the clamp the corner is kept, not dropped."""
    samples, truth = _clock()
    job = _job(app_mod, tmp_path, samples=samples)
    job.endgame_cutoff = app_mod.InferenceJob._compute_endgame_cutoff(job)
    windows, checks = [], []

    def motion(video, crop, window, detect):
        windows.append(window)
        moved = window[1] >= truth
        return {"moved": moved, "t_s": truth + 1.0, "shift_px": 80.0, "stepping": [1, 2],
                "two_sided": [1, 2], "per_tag": {}, "samples": 40}
    monkeypatch.setattr(cm, "detect_camera_move", motion)
    monkeypatch.setattr(cpm, "check_epoch", lambda cal, video, crop, fr, *a, **k: (
        checks.append(fr) or {"best_m": 0.05, "n_tags": 5, "min_tags_used": 3, "frames": 6,
                              "frames_used": 6}))
    monkeypatch.setattr(cpm, "_overlay_on_median", lambda *a, **k: None)
    monkeypatch.setattr(ac, "make_detect_fn", lambda: (lambda f: {}))
    import web.field.tag_locator as tl
    monkeypatch.setattr(tl, "make_roi_detect_fn", lambda: (lambda *a, **k: {}))
    out = app_mod.InferenceJob._corner_per_match_step(job)
    assert out["corner_left"]["status"] == "kept"
    assert [v.name for v in job.views] == ["main", "corner_left"]
    assert windows and windows[0][1] < truth
    assert windows[0][1] <= job.endgame_cutoff["cutoff_s"]
    assert checks and checks[0][1] < job.endgame_cutoff["cutoff_frame"]


def test_2025_without_the_clamp_the_switch_drops_the_corner(app_mod, tmp_path, monkeypatch):
    """The regression the clamp fixes: the same motion check over the full range."""
    samples, truth = _clock()
    job = _job(app_mod, tmp_path, samples=samples)
    job.endgame_cutoff = None
    monkeypatch.setattr(cm, "detect_camera_move", lambda video, crop, window, detect: {
        "moved": window[1] >= truth, "t_s": truth + 1.0, "shift_px": 80.0,
        "stepping": [1, 2], "two_sided": [1, 2], "per_tag": {}, "samples": 40})
    monkeypatch.setattr(cpm, "_overlay_on_median", lambda *a, **k: None)
    monkeypatch.setattr(ac, "make_detect_fn", lambda: (lambda f: {}))
    import web.field.tag_locator as tl
    monkeypatch.setattr(tl, "make_roi_detect_fn", lambda: (lambda *a, **k: {}))
    out = app_mod.InferenceJob._corner_per_match_step(job)
    assert out["corner_left"]["status"] == "dropped"


def test_provenance_records_the_cutoff(app_mod, tmp_path):
    job = _job(app_mod, tmp_path)
    job.endgame_cutoff = app_mod.InferenceJob._compute_endgame_cutoff(job)
    prov = app_mod.InferenceJob._replay_provenance(job)
    assert prov["endgame_cutoff"]["method"] == "match_clock"
    job26 = _job(app_mod, tmp_path, season=S26, samples=_clock()[0])
    job26.endgame_cutoff = app_mod.InferenceJob._compute_endgame_cutoff(job26)
    assert "endgame_cutoff" not in app_mod.InferenceJob._replay_provenance(job26)
