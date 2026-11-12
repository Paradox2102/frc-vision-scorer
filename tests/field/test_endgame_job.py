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


# ------------------------------------- Part B: corner views stop feeding tracking

def _job_with_cutoff(app_mod, tmp_path, **kw):
    job = _job(app_mod, tmp_path, **kw)
    job.endgame_cutoff = app_mod.InferenceJob._compute_endgame_cutoff(job)
    app_mod.InferenceJob._apply_endgame_corner_cutoff(job)
    return job


def test_camera_view_is_active_until_its_cutoff():
    v = CameraView("corner_left", CROP, (), BASE)
    assert v.active_until_frame is None and v.active_at(10 ** 9)
    v.active_until_frame = 100
    assert v.active_at(99) and not v.active_at(100) and not v.active_at(101)


def test_2025_corners_end_at_the_cutoff_and_main_never(app_mod, tmp_path):
    job = _job_with_cutoff(app_mod, tmp_path)
    cut = job.endgame_cutoff["cutoff_frame"]
    by = {v.name: v for v in job.views}
    assert by["main"].active_until_frame is None
    assert by["corner_left"].active_until_frame == cut
    assert [v.name for v in app_mod.InferenceJob._frame_views(job, cut - 1)] == [
        "main", "corner_left"]
    assert [v.name for v in app_mod.InferenceJob._frame_views(job, cut)] == ["main"]
    assert [v.name for v in app_mod.InferenceJob._frame_views(job, cut + 900)] == ["main"]


def test_2025_fallback_corners_end_at_the_estimate(app_mod, tmp_path):
    job = _job_with_cutoff(app_mod, tmp_path, fallback=True)
    corner = next(v for v in job.views if v.name == "corner_left")
    assert job.endgame_cutoff["method"] == "estimate"
    assert corner.active_until_frame == job.endgame_cutoff["cutoff_frame"]


def test_2026_corners_stay_active_for_the_whole_match(app_mod, tmp_path):
    job = _job_with_cutoff(app_mod, tmp_path, season=S26, samples=_clock()[0])
    assert job.endgame_cutoff is None
    assert all(v.active_until_frame is None for v in job.views)
    for f in (0, job.processing_ranges[-1][1], job.total_frames + 10):
        assert app_mod.InferenceJob._frame_views(job, f) == job.views


def test_2025_corner_detections_after_the_cutoff_are_excluded(app_mod, tmp_path):
    """What the analysis loop gathers per frame: every view in _frame_views feeds
    per_camera (detections, masks, crops for reads); main is kept throughout."""
    job = _job_with_cutoff(app_mod, tmp_path)
    cut = job.endgame_cutoff["cutoff_frame"]
    dets = {"main": ["m"], "corner_left": ["c"]}

    def gather(f):
        return {v.name: dets[v.name] for v in app_mod.InferenceJob._frame_views(job, f)}
    assert gather(cut - 3) == {"main": ["m"], "corner_left": ["c"]}
    assert gather(cut) == {"main": ["m"]}
    assert gather(job.processing_ranges[-1][1]) == {"main": ["m"]}


def test_analysis_loop_iterates_the_frame_views():
    """The per-frame evidence / fusion / occlusion loops respect the cutoff: they
    walk ``frame_views`` (InferenceJob._frame_views), not every loaded view."""
    import inspect
    import web.app as app
    src = inspect.getsource(app.InferenceJob.process_video)
    assert "frame_views = self._frame_views(frame_num)" in src
    start = src.index("frame_views = self._frame_views(frame_num)")
    body = src[start:src.index("self._occ = _merge_occlusion", start)]
    assert "self.views" not in body
    assert body.count("for view in frame_views") == 2


def test_view_plan_ends_at_the_cutoff_and_keeps_the_prompt():
    from web.field.view_stride import build_view_frame_plan
    frames = [100 + 3 * i for i in range(40)]
    full = build_view_frame_plan(frames, 4, 2)
    assert build_view_frame_plan(frames, 4, 2, end_frame=None) == full
    cut = build_view_frame_plan(frames, 4, 2, end_frame=frames[20])
    assert cut.source_frames == [f for f in full.source_frames if f < frames[20]]
    assert cut.source_frames[cut.prompt_idx] == frames[4]
    assert all(cut.source_to_local[f] == i for i, f in enumerate(cut.source_frames))
    # a cutoff at/before the prompt frame never drops the prompt
    early = build_view_frame_plan(frames, 4, 1, end_frame=frames[2])
    assert early.source_frames[early.prompt_idx] == frames[4]


def test_prefetch_skips_inactive_views():
    from web.analysis_prefetch import MISS, AnalysisPrefetcher

    def read():
        for n in (10, 20, 30):
            yield n, np.full((8, 4, 3), n, dtype=np.uint8)
    calls = []

    def setup():
        return (lambda o: None), (lambda vf: calls.append(int(vf[0, 0, 0])) or [])
    pf = AnalysisPrefetcher(read, overhead_height=4, ocr_frames=set(), setup_gpu=setup,
                            views=[("main", lambda f: f, None),
                                   ("corner_left", lambda f: f + 1, 20)])
    pf.run()
    assert calls == [10, 11, 20, 30]
    assert pf.take_person(20, "corner_left") is MISS          # never computed
    assert pf.take_person(10, "corner_left") == []
