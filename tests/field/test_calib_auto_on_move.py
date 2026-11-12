"""calib_auto_on_move: auto-calibrate a camera that moved between matches.

  * auto_calibrate: prepare_inputs / run_camera(prep, exclude_tags) / low_support_tags /
    resolve_moved_camera (attempt order, stop at first ready, low-support retry);
  * the job's start-of-match drift check on main (adopt + re-init / abort with the
    manual prompt / operator-rejected match / knob off = staged autofix);
  * the per-match corner step's fresh full-solve fallback.

Synthetic footage like test_auto_calibrate: a tiny mp4 supplies frame timing and an
injected detect_fn returns tag corners projected from a known camera.
"""
import importlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from web.field import auto_calibrate as ac
from web.field import corner_per_match as cpm
from web.field import store as field_store
from web.field.camera_view import CameraView
from web.field.field_model import FieldModel
from web.field.match_order import select_epoch
from tests._field_synth import truth_pinhole_camera, project_tag_corners

FM = FieldModel.load("2026-rebuilt-welded")
CROP = (0, 0, 1920, 1080)
LENS_D = np.array([-0.12, 0.02, 0.0, 0.0, 0.0])        # realistic, monotonic lens
BASE = truth_pinhole_camera(D=LENS_D)
RELOCATED = truth_pinhole_camera(eye=(4.0, -3.0, 6.0), D=LENS_D)   # re-placed 4 m


def _video(tmp_path, n=30, name="evt_qm5.mp4"):
    p = tmp_path / name
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (64, 36))
    for _ in range(n):
        w.write(np.full((36, 64, 3), 10, np.uint8))
    w.release()
    return p


def _det(cam, only=None):
    corr = project_tag_corners(cam, FM)
    ids = [t for t in sorted(corr) if only is None or t in only]
    return lambda frame: {t: corr[t][0].copy() for t in ids}


def _version(cal, vid="v1", start_match=None):
    return {"calibration": cal, "start_match": start_match, "version_id": vid,
            "created_at": "2026-11-01T00:00:00", "label": None, "source": None}


def _strip_times(rec):
    return {k: v for k, v in rec.items() if k not in ("created_at",)}


# ------------------------------------------------------------ low_support_tags

def test_low_support_tags_thresholds():
    diag = {6: (150, 166, 0.5), 8: (110, 112, 0.6), 1: (3, 3, 0.2), 2: (3, 3, 0.1),
            11: (2, 3, 0.4), 27: (8, 9, 0.3)}
    # 5% of 166 = 8.3 > 5 -> 1, 2, 11 (3 frames) flagged; 27 (9 frames) kept
    assert ac.low_support_tags(diag) == [1, 2, 11]
    # small sample: the min_seen floor rules (5 > 5% of 20)
    small = {1: (20, 20, 0), 2: (4, 4, 0), 3: (5, 5, 0)}
    assert ac.low_support_tags(small) == [2]
    assert ac.low_support_tags(small, min_seen=3) == []
    assert ac.low_support_tags(diag, frac=0.06) == [1, 2, 11, 27]   # 9 < 9.96
    assert ac.low_support_tags({}) == []


# ------------------------------------------------- run_camera: prep / exclude

def test_run_camera_with_prep_is_identical(tmp_path):
    video = _video(tmp_path)
    det = _det(BASE)
    cfg = ac.AutoCalibConfig(locate_undecoded=False)
    a = ac.run_camera(video, "evt_qm1", "main", CROP, FM, [], cfg=cfg, detect_fn=det,
                      roi_detect_fn=lambda f: {})
    prep = ac.prepare_inputs(video, CROP, FM, cfg, det, lambda f: {})
    b = ac.run_camera(video, "evt_qm1", "main", CROP, FM, [], cfg=cfg, detect_fn=det,
                      roi_detect_fn=lambda f: {}, prep=prep)
    assert a.record["status"] == "ready"
    assert _strip_times(a.record) == _strip_times(b.record)
    assert "excluded_tags" not in a.record
    with pytest.raises(ValueError):
        ac.run_camera(video, "evt_qm1", "main", (0, 0, 10, 10), FM, [], cfg=cfg,
                      detect_fn=det, roi_detect_fn=lambda f: {}, prep=prep)


def test_exclude_tags_never_readded_from_median(tmp_path, monkeypatch):
    corr = project_tag_corners(BASE, FM)
    ids = sorted(corr)
    median_only, agg_excl = ids[-1], ids[0]
    marker = np.full((36, 64, 3), 77, np.uint8)
    monkeypatch.setattr(ac, "median_frame", lambda *a, **k: marker.copy())

    def detect(frame):
        out = {t: corr[t][0].copy() for t in ids[:-1]}
        if frame[0, 0, 0] == 77:          # the median shows every tag
            out[median_only] = corr[median_only][0].copy()
        return out
    cfg = ac.AutoCalibConfig(locate_undecoded=False)
    res = ac.run_camera(_video(tmp_path), "evt_qm1", "main", CROP, FM, [], cfg=cfg,
                        detect_fn=detect, roi_detect_fn=lambda f: {},
                        exclude_tags=(agg_excl, median_only))
    rec = res.record
    assert str(agg_excl) not in rec["auto_tags"] and str(median_only) not in rec["auto_tags"]
    assert rec["metrics"]["n_tags"] == len(ids) - 2
    assert rec["per_tag"][str(agg_excl)]["excluded"] is True
    assert str(median_only) not in rec["per_tag"]        # never joined from the median
    assert rec["excluded_tags"] == sorted([agg_excl, median_only])


# --------------------------------------------------- resolve_moved_camera (logic)

def _fake_prep(diag=None, med=()):
    diag = diag if diag is not None else {6: (160, 166, .5), 8: (110, 112, .5),
                                          17: (150, 160, .5), 27: (120, 130, .5),
                                          1: (3, 3, .2), 2: (3, 3, .2), 11: (3, 3, .2)}
    return ac.Prep(agg={"corners": {t: np.zeros((4, 2)) for t in diag}, "diag": diag,
                        "fps": 30.0, "n_frames": 100, "best_frame": None, "best_t": None,
                        "best_n": 0},
                   median=None, stack=None, med_dec={t: np.zeros((4, 2)) for t in med},
                   crop=CROP, window=(18.0, 1e9))


def _scripted_run(outcomes, calls):
    """run_camera stand-in: outcome per call in order ("ready" / "failed" /
    "failed_cand" = failed with a candidate calibration)."""
    def run(video, mk, cam, crop, fm, versions, **k):
        method = "pose_resolve" if versions else "full_solve"
        calls.append((method, tuple(k.get("exclude_tags") or ()), k.get("prep")))
        st = outcomes[len(calls) - 1]
        cal = BASE if st in ("ready", "failed_cand") else None
        rec = ac.new_record("ready" if st == "ready" else "failed", f"{method} {st}",
                            match_key=mk, method=method, metrics={"n_tags": 4})
        return ac.AutoCalibResult(rec, cal, None)
    return run


@pytest.mark.parametrize("outcomes,expect_calls,expect_method", [
    (["ready"], [("pose_resolve", ())], "pose_resolve"),
    (["failed", "ready"], [("pose_resolve", ()), ("full_solve", ())], "full_solve"),
    (["failed", "failed", "failed", "ready"],
     [("pose_resolve", ()), ("full_solve", ()), ("pose_resolve", (1, 2, 11)),
      ("full_solve", (1, 2, 11))], "full_solve"),
])
def test_resolve_attempt_order_stops_at_first_ready(monkeypatch, outcomes, expect_calls,
                                                    expect_method):
    calls = []
    monkeypatch.setattr(ac, "run_camera", _scripted_run(outcomes, calls))
    prep = _fake_prep()
    res = ac.resolve_moved_camera("v.mp4", "evt_qm45", "main", CROP, FM,
                                  [_version(BASE)], ac.AutoCalibConfig(), prep=prep,
                                  detect_fn=lambda f: {}, roi_detect_fn=lambda f: {})
    assert [c[:2] for c in calls] == expect_calls
    assert all(c[2] is prep for c in calls)               # inputs prepared once
    rec = res.record
    assert rec["status"] == "ready" and rec["method"] == expect_method
    assert [a["method"] for a in rec["attempts"]] == [c[0] for c in expect_calls]
    assert rec["attempts"][-1]["status"] == "ready"
    if expect_calls[-1][1]:
        assert rec["attempts"][-1]["excluded_tags"] == [1, 2, 11]
        assert "ignored low-support decodes: tag 1 (3/166 frames)" in rec["reason"]
    else:
        assert "low-support" not in rec["reason"]


def test_resolve_all_fail_returns_last_full_solve_candidate(monkeypatch):
    calls = []
    monkeypatch.setattr(ac, "run_camera",
                        _scripted_run(["failed", "failed_cand", "failed", "failed"], calls))
    res = ac.resolve_moved_camera("v.mp4", "evt_qm45", "main", CROP, FM,
                                  [_version(BASE)], ac.AutoCalibConfig(), prep=_fake_prep(),
                                  detect_fn=lambda f: {}, roi_detect_fn=lambda f: {})
    assert len(calls) == 4
    rec = res.record
    assert rec["status"] == "failed" and rec["method"] == "full_solve"
    assert res.calibration is BASE                     # the one with a candidate
    assert rec["moved_attempt"] == "full_solve" and "tried:" in rec["reason"]
    assert len(rec["attempts"]) == 4


def test_resolve_without_versions_skips_pose(monkeypatch):
    calls = []
    monkeypatch.setattr(ac, "run_camera", _scripted_run(["failed", "ready"], calls))
    res = ac.resolve_moved_camera("v.mp4", "evt_qm45", "corner_left", CROP, FM, [],
                                  ac.AutoCalibConfig(), prep=_fake_prep(),
                                  detect_fn=lambda f: {}, roi_detect_fn=lambda f: {})
    assert [c[:2] for c in calls] == [("full_solve", ()), ("full_solve", (1, 2, 11))]
    assert res.record["status"] == "ready"


def test_resolve_no_retry_when_too_few_tags_would_remain(monkeypatch):
    calls = []
    monkeypatch.setattr(ac, "run_camera", _scripted_run(["failed", "failed"], calls))
    diag = {6: (160, 166, .5), 8: (110, 112, .5), 1: (3, 3, .2)}   # 2 strong tags
    res = ac.resolve_moved_camera("v.mp4", "evt_qm45", "main", CROP, FM,
                                  [_version(BASE)], ac.AutoCalibConfig(min_tags=3),
                                  prep=_fake_prep(diag), detect_fn=lambda f: {},
                                  roi_detect_fn=lambda f: {})
    assert [c[:2] for c in calls] == [("pose_resolve", ()), ("full_solve", ())]
    assert res.record["status"] == "failed"
    # a median-only tag counts toward what remains
    calls.clear()
    monkeypatch.setattr(ac, "run_camera", _scripted_run(["failed"] * 4, calls))
    ac.resolve_moved_camera("v.mp4", "evt_qm45", "main", CROP, FM, [_version(BASE)],
                            ac.AutoCalibConfig(min_tags=3), prep=_fake_prep(diag, med=(17,)),
                            detect_fn=lambda f: {}, roi_detect_fn=lambda f: {})
    assert len(calls) == 4


def test_resolve_attempt_exception_is_recorded_not_raised(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("solver blew up")
    monkeypatch.setattr(ac, "run_camera", boom)
    res = ac.resolve_moved_camera("v.mp4", "evt_qm45", "main", CROP, FM, [_version(BASE)],
                                  ac.AutoCalibConfig(), prep=_fake_prep({6: (5, 5, 0)}),
                                  detect_fn=lambda f: {}, roi_detect_fn=lambda f: {})
    assert res.record["status"] == "failed"
    assert all("solver blew up" in a["reason"] for a in res.record["attempts"])


def test_resolve_relocated_camera_end_to_end(tmp_path):
    """Real pipeline: a camera re-placed 4 m fails the pose re-solve (centre-shift
    bound) and comes out ready from the fresh full solve."""
    res = ac.resolve_moved_camera(_video(tmp_path), "evt_qm9", "main", CROP, FM,
                                  [_version(BASE, "v1")],
                                  ac.AutoCalibConfig(locate_undecoded=False),
                                  detect_fn=_det(RELOCATED), roi_detect_fn=lambda f: {})
    rec = res.record
    assert [a["method"] for a in rec["attempts"]] == ["pose_resolve", "full_solve"]
    assert rec["attempts"][0]["status"] == "failed"
    assert "camera center moved" in rec["attempts"][0]["reason"]
    assert rec["status"] == "ready" and rec["method"] == "full_solve"
    c = np.asarray(res.calibration.camera_center)
    assert np.linalg.norm(c - np.asarray(RELOCATED.camera_center)) < 0.2


# ------------------------------------------------------------- job: main camera

@pytest.fixture
def app_mod(tmp_path, monkeypatch):
    mod = importlib.import_module("web.app")
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(mod, "_auto_calib_judge", lambda: None)
    monkeypatch.setattr(mod, "CALIB_CHECK_ENABLED", True)
    return mod


def _main_job(app_mod, tmp_path, overrides=None, drift_seq=(1.4, 0.05)):
    """An InferenceJob shell at the drift check: main epoch v0 (BASE), the alignment
    measurement scripted per call (``drift_seq``), _init_field_tracking faked to
    re-select the covering epoch like the real one."""
    from web.job_config import JobConfig
    job = app_mod.InferenceJob.__new__(app_mod.InferenceJob)
    job.config = JobConfig(overrides or {})
    job.event_key, job.match_key, job.camera = "evt", "evt_qm45", "main"
    job.processing_ranges = [(30, 600)]
    job.video_fps = 30.0
    job.video_path = str(_video(tmp_path))
    job.field_model = FM
    job.endgame_cutoff = None
    job.status = job.error = job.needs_calibration = None
    from web.field.calibration import CameraCalibration
    base = CameraCalibration.from_json(BASE.to_json())
    base.masks = [(0, 0, 10, 10)]
    v0 = field_store.save_version("evt", "main", base)
    job.calib, job._calib_version_id = base, v0
    job.views = [CameraView("main", CROP, ((0, 0, 10, 10),), base)]
    seq = list(drift_seq)
    job._align_calls = []

    def fake_align(video, view, rng, calib, fm, n):
        job._align_calls.append(calib)
        e = seq.pop(0) if seq else 0.05
        return e, 4
    return job, v0, fake_align


def _patch_init(app_mod, monkeypatch, calls):
    def fake_init(self):
        calls.append(1)
        ep = select_epoch(field_store.load_versions(self.event_key, self.camera),
                          self.match_key)
        self.calib, self._calib_version_id = ep["calibration"], ep["version_id"]
        self.views = [CameraView("main", CROP, tuple(ep["calibration"].masks or ()),
                                 ep["calibration"])]
        return True
    monkeypatch.setattr(app_mod.InferenceJob, "_init_field_tracking", fake_init)


def _ready_result(cal=RELOCATED):
    rec = ac.new_record("ready", "full solve from 4 tags (rms 1.01 px)", match_key="evt_qm45",
                        method="full_solve",
                        metrics={"n_tags": 4, "rms_px": 1.01, "val_gate_best_m": 0.016},
                        auto_tags={"6": [[0, 0]] * 4}, excluded_tags=[1, 2, 11],
                        attempts=[{"method": "pose_resolve", "status": "failed",
                                   "reason": "x", "excluded_tags": []},
                                  {"method": "full_solve", "status": "ready",
                                   "reason": "y", "excluded_tags": [1, 2, 11]}])
    return ac.AutoCalibResult(rec, cal, np.zeros((8, 8, 3), np.uint8))


def _failed_result():
    rec = ac.new_record("failed", "pp_k1: implausible pose (8 of 28 tag points are "
                        "behind the camera)", match_key="evt_qm45", method="full_solve",
                        metrics={"n_tags": 7})
    return ac.AutoCalibResult(rec, None, np.zeros((8, 8, 3), np.uint8))


def test_job_drift_ready_adopts_and_continues(app_mod, tmp_path, monkeypatch):
    job, v0, fake_align = _main_job(app_mod, tmp_path)
    monkeypatch.setattr(app_mod, "_main_tag_alignment", fake_align)
    seen = {}

    def fake_resolve(video, mk, cam, crop, fm, versions, cfg, **k):
        seen.update(video=video, mk=mk, cam=cam, versions=[v["version_id"] for v in versions],
                    cfg=cfg, masks=k.get("masks"))
        return _ready_result(truth_pinhole_camera(eye=(4.0, -3.0, 6.0), D=LENS_D))
    monkeypatch.setattr(ac, "resolve_moved_camera", fake_resolve)
    monkeypatch.setattr(app_mod.InferenceJob, "_attempt_autofix",
                        lambda *a, **k: pytest.fail("staged autofix must not run"))
    field_store.record_drift("evt", "main", "evt_qm45", 1.4, v0)
    inits = []
    _patch_init(app_mod, monkeypatch, inits)

    assert app_mod.InferenceJob._verify_calibration_alignment(job) is True
    assert job.error is None and job.status is None
    assert inits == [1] and len(job._align_calls) == 2       # re-checked once
    assert seen["mk"] == "evt_qm45" and seen["cam"] == "main" and seen["versions"] == [v0]
    assert seen["cfg"].window == (1.0, 20.0) and seen["cfg"].min_tags == 3
    assert seen["cfg"].drift_max_m == app_mod.CALIB_DRIFT_MAX_M
    new = select_epoch(field_store.load_versions("evt", "main"), "evt_qm45")
    assert new["version_id"] != v0 and new["start_match"] == "evt_qm45"
    assert new["label"] == "auto: camera moved"
    src = new["source"]
    assert src["method"] == "auto_on_move" and src["base_epoch"] == v0
    assert src["match_key"] == "evt_qm45" and len(src["attempts"]) == 2
    assert [list(m) for m in new["calibration"].masks] == [[0, 0, 10, 10]]   # carried over
    assert job._calib_version_id == new["version_id"]       # tracking re-selected it
    assert select_epoch(field_store.load_versions("evt", "main"), "evt_qm44")["version_id"] == v0
    e = field_store.load_permatch_log("evt", "main")["evt_qm45"]
    assert e["status"] == "adopted" and e["job_outcome"] == "adopted"
    assert e["version_id"] == new["version_id"] and e["epoch_checked"] == v0
    assert e["metrics"]["val_best_m"] == 0.016 and e["metrics"]["check_best_m"] == 1.4
    assert e["preview"] and field_store.permatch_preview_path("evt", "main", "evt_qm45").is_file()
    assert field_store.load_drift("evt").get("main", {}).get("evt_qm45") is None
    assert "auto-recalibrated" in job._calib_move_warning
    assert "/calibrate?match=evt_qm45&camera=main" in job._calib_move_warning


def test_job_drift_failed_aborts_with_auto_reason(app_mod, tmp_path, monkeypatch):
    job, v0, fake_align = _main_job(app_mod, tmp_path)
    monkeypatch.setattr(app_mod, "_main_tag_alignment", fake_align)
    monkeypatch.setattr(ac, "resolve_moved_camera", lambda *a, **k: _failed_result())
    monkeypatch.setattr(app_mod.InferenceJob, "_attempt_autofix",
                        lambda *a, **k: pytest.fail("staged autofix must not run"))
    inits = []
    _patch_init(app_mod, monkeypatch, inits)
    assert app_mod.InferenceJob._verify_calibration_alignment(job) is False
    assert job.status == "error" and inits == []
    nc = job.needs_calibration
    assert nc["auto_attempted"] is True and "implausible pose" in nc["auto_reason"]
    assert not nc.get("autofix_pending")
    assert "auto-calibration on this match failed" in job.error
    assert "calibrate manually at /calibrate?match=evt_qm45&camera=main" in job.error
    assert [v["version_id"] for v in field_store.load_versions("evt", "main")] == [v0]
    e = field_store.load_permatch_log("evt", "main")["evt_qm45"]
    assert e["status"] == "failed" and e["version_id"] is None and e["preview"]
    assert field_store.load_drift("evt")["main"]["evt_qm45"]["epoch"] == v0
    assert getattr(job, "_calib_move_warning", None) is None


def test_job_rejected_match_never_auto_solves(app_mod, tmp_path, monkeypatch):
    job, v0, fake_align = _main_job(app_mod, tmp_path)
    monkeypatch.setattr(app_mod, "_main_tag_alignment", fake_align)
    field_store.save_permatch_entry("evt", "main", "evt_qm45",
                                    {"status": "rejected", "rejected_version": "vX",
                                     "prev_status": "adopted"})
    monkeypatch.setattr(ac, "resolve_moved_camera",
                        lambda *a, **k: pytest.fail("must not auto-solve"))
    monkeypatch.setattr(app_mod.InferenceJob, "_attempt_autofix",
                        lambda *a, **k: pytest.fail("staged autofix must not run"))
    assert app_mod.InferenceJob._verify_calibration_alignment(job) is False
    assert job.needs_calibration["auto_rejected"] is True
    assert "rejected" in job.error and "calibrate manually" in job.error
    e = field_store.load_permatch_log("evt", "main")["evt_qm45"]
    assert e["status"] == "rejected" and e["rejected_version"] == "vX"   # untouched


def test_job_knob_off_uses_staged_autofix(app_mod, tmp_path, monkeypatch):
    job, v0, fake_align = _main_job(app_mod, tmp_path, {"calib_auto_on_move": False})
    monkeypatch.setattr(app_mod, "_main_tag_alignment", fake_align)
    monkeypatch.setattr(ac, "resolve_moved_camera",
                        lambda *a, **k: pytest.fail("must not auto-solve"))
    monkeypatch.setattr(app_mod, "CALIB_AUTOFIX_ENABLED", True)
    called = []
    monkeypatch.setattr(app_mod.InferenceJob, "_attempt_autofix",
                        lambda self, view, err: called.append(err) or None)
    assert app_mod.InferenceJob._verify_calibration_alignment(job) is False
    assert called == [1.4]
    assert "auto_attempted" not in job.needs_calibration
    assert "recalibrate at" in job.error
    assert field_store.load_permatch_log("evt", "main") == {}


def test_job_adopted_but_recheck_fails_unadopts(app_mod, tmp_path, monkeypatch):
    job, v0, fake_align = _main_job(app_mod, tmp_path, drift_seq=(1.4, 0.9))
    monkeypatch.setattr(app_mod, "_main_tag_alignment", fake_align)
    n_resolve = []
    monkeypatch.setattr(ac, "resolve_moved_camera",
                        lambda *a, **k: n_resolve.append(1) or _ready_result())
    monkeypatch.setattr(app_mod.InferenceJob, "_attempt_autofix",
                        lambda *a, **k: pytest.fail("staged autofix must not run"))
    inits = []
    _patch_init(app_mod, monkeypatch, inits)
    assert app_mod.InferenceJob._verify_calibration_alignment(job) is False
    assert n_resolve == [1] and inits == [1]              # no second attempt
    assert job.status == "error" and job.needs_calibration["auto_attempted"] is True
    assert "still reads 0.90 m" in job.needs_calibration["auto_reason"]
    assert [v["version_id"] for v in field_store.load_versions("evt", "main")] == [v0]
    e = field_store.load_permatch_log("evt", "main")["evt_qm45"]
    assert e["status"] == "failed" and e["version_id"] is None and e["unadopted_version"]


def test_job_aligned_never_auto_solves(app_mod, tmp_path, monkeypatch):
    job, v0, fake_align = _main_job(app_mod, tmp_path, drift_seq=(0.1,))
    monkeypatch.setattr(app_mod, "_main_tag_alignment", fake_align)
    monkeypatch.setattr(ac, "resolve_moved_camera",
                        lambda *a, **k: pytest.fail("must not auto-solve"))
    assert app_mod.InferenceJob._verify_calibration_alignment(job) is True
    assert field_store.load_permatch_log("evt", "main") == {}


def test_knob_default_on():
    from web.job_config import JobConfig
    assert JobConfig().calib_auto_on_move is True
    assert JobConfig({"calib_auto_on_move": False}).non_defaults() == {
        "calib_auto_on_move": False}


# ------------------------------------------------------------ corner fallback

def _epoch(cal=BASE, vid="v0"):
    return {"calibration": cal, "version_id": vid, "start_match": None,
            "created_at": "2026-01-01T00:00:00", "source": None}


def _corner(tmp_path, det, cfg, **kw):
    return cpm.corner_step(_video(tmp_path), "evt_qm5", "corner_left", CROP, (), FM,
                           _epoch(), (0, 29), 1.0, cfg, detect_fn=det,
                           roi_detect_fn=lambda f: {}, log=lambda m: None, **kw)


def test_corner_fallback_full_solve_adopts_relocated_camera(tmp_path):
    cfg = cpm.CornerStepConfig(detect_motion=False, auto_full_solve=True, jev_verify=False)
    out = _corner(tmp_path, _det(RELOCATED), cfg)
    assert out.status == "adopted", out.reason
    assert out.metrics["lens_from"] is None and out.metrics["method"] == "full_solve"
    assert "fresh full solve" in out.reason
    c = np.asarray(out.calibration.camera_center)
    assert np.linalg.norm(c - np.asarray(RELOCATED.camera_center)) < 0.2


def test_corner_fallback_off_drops_relocated_camera(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "resolve_moved_camera",
                        lambda *a, **k: pytest.fail("fallback is off"))
    cfg = cpm.CornerStepConfig(detect_motion=False, jev_verify=False)
    out = _corner(tmp_path, _det(RELOCATED), cfg)
    assert out.status == "dropped" and "camera center moved" in out.reason


def test_corner_fallback_failure_still_drops(tmp_path, monkeypatch):
    seen = {}

    def fake_resolve(video, mk, cam, crop, fm, versions, cfg, **k):
        seen.update(versions=versions, gate=cfg.drift_max_m, prep=k.get("prep"))
        return ac.AutoCalibResult(ac.new_record("failed", "tags span 0.6 m"), None, None)
    monkeypatch.setattr(ac, "resolve_moved_camera", fake_resolve)
    cfg = cpm.CornerStepConfig(detect_motion=False, auto_full_solve=True, jev_verify=False)
    out = _corner(tmp_path, _det(RELOCATED), cfg)
    assert out.status == "dropped"
    assert "fresh full solve failed too: tags span 0.6 m" in out.reason
    assert seen["versions"] == [] and seen["gate"] == cfg.resolve_max_m
    assert seen["prep"] is not None                        # shared input stage


def test_corner_lens_resolves_share_one_prep(tmp_path, monkeypatch):
    preps = []

    def fake_run(video, mk, cam, crop, fm, versions, **k):
        preps.append(k.get("prep"))
        return ac.AutoCalibResult(ac.new_record("failed", "no"), None, None)
    monkeypatch.setattr(ac, "run_camera", fake_run)
    other = truth_pinhole_camera(K=np.array([[1500., 0, 960], [0, 1500., 540], [0, 0, 1]]))
    cpm.corner_step(_video(tmp_path), "evt_qm5", "corner_left", CROP, (), FM,
                    _epoch(other, "chain"), (0, 29), 1.0,
                    cpm.CornerStepConfig(detect_motion=False), detect_fn=_det(RELOCATED),
                    roi_detect_fn=lambda f: {}, alt_epochs=[_epoch(BASE, "base")],
                    log=lambda m: None)
    assert len(preps) == 2 and preps[0] is not None and preps[0] is preps[1]


def test_job_wires_knob_into_corner_step(app_mod, tmp_path, monkeypatch):
    from web.job_config import JobConfig
    for over, want in (({}, True), ({"calib_auto_on_move": False}, False)):
        job = app_mod.InferenceJob.__new__(app_mod.InferenceJob)
        job.config = JobConfig(over)
        job.event_key, job.match_key = "evt", "evt_qm5"
        job.processing_ranges = [(0, 29)]
        job.video_fps = 1.0
        job.video_path = str(_video(tmp_path))
        job.field_model = FM
        job.timings = {}
        v0 = field_store.save_version("evt", "corner_left", BASE)
        job._view_selection = {"corner_left": v0}
        job.views = [CameraView("main", (0, 0, 1920, 723), (), BASE),
                     CameraView("corner_left", CROP, (), BASE)]
        seen = {}

        def fake(video, mk, cam, crop, masks, fm, epoch, frange, fps, ccfg, **k):
            seen["auto"] = ccfg.auto_full_solve
            return cpm.CornerOutcome("kept", "ok", "v0")
        monkeypatch.setattr(cpm, "corner_step", fake)
        app_mod.InferenceJob._corner_per_match_step(job)
        assert seen["auto"] is want
