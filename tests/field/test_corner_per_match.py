"""Per-match corner calibration: the pure step (web.field.corner_per_match), the
job integration (InferenceJob._corner_per_match_step), rejected versions in the
store, and the per-match routes. Synthetic video + projected tags; no network."""
import importlib

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from web.field import auto_calibrate as ac
from web.field import corner_per_match as cpm
from web.field import store as field_store
from web.field.camera_view import CameraView
from web.field.field_model import FieldModel
from tests._field_synth import truth_pinhole_camera, project_tag_corners

FM = FieldModel.load("2026-rebuilt-welded")
CROP = (0, 0, 1920, 1080)
BASE = truth_pinhole_camera()                                   # the covering epoch
MOVED = truth_pinhole_camera(target=(10.0, 4.03, 0.0))          # re-aimed this match


def _video(tmp_path, n=30):
    p = tmp_path / "evt_qm5.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (64, 36))
    for _ in range(n):
        w.write(np.full((36, 64, 3), 10, np.uint8))
    w.release()
    return p


def _det(cam, k=None):
    corr = project_tag_corners(cam, FM)
    ids = sorted(corr)[:k] if k else sorted(corr)
    return lambda frame: {t: corr[t][0].copy() for t in ids}


def _epoch(cal=BASE, vid="v0"):
    return {"calibration": cal, "version_id": vid, "start_match": None,
            "created_at": "2026-01-01T00:00:00", "source": None}


def _step(tmp_path, det, **kw):
    return cpm.corner_step(_video(tmp_path), "evt_qm5", "corner_left", CROP, (), FM,
                           _epoch(), (0, 29), 1.0, kw.pop("cfg", None),
                           detect_fn=det, roi_detect_fn=lambda f: {}, log=lambda m: None, **kw)


# ----------------------------------------------------------------- pure step

def test_check_pass_keeps_epoch(tmp_path):
    out = _step(tmp_path, _det(BASE))
    assert out.status == "kept" and out.calibration is None
    assert out.epoch_checked == "v0" and out.metrics["check_best_m"] < 0.05
    assert out.metrics["check_min_tags"] == 3
    assert out.overlay is not None


def test_check_fail_resolve_pass_adopts(tmp_path):
    out = _step(tmp_path, _det(MOVED))
    assert out.status == "adopted", out.reason
    assert out.metrics["check_best_m"] > 0.25
    assert out.metrics["val_best_m"] < 0.15
    c = np.asarray(out.calibration.camera_center)
    assert np.linalg.norm(c - np.asarray(MOVED.camera_center)) < 0.1
    assert np.allclose(out.calibration.K, BASE.K, rtol=0.05)          # lens kept/near
    assert out.auto_tags and "re-solved" in out.reason


def test_resolve_fail_drops(tmp_path):
    out = _step(tmp_path, _det(MOVED, k=2))           # only 2 tags: check + solve fail
    assert out.status == "dropped" and out.calibration is None
    assert "did not validate" in out.reason or "error" in out.reason


def test_resolve_fail_without_drop_is_failed_kept(tmp_path):
    out = _step(tmp_path, _det(MOVED, k=2), drop_on_fail=False)
    assert out.status == "failed_kept" and "kept anyway" in out.reason


def test_rejected_match_never_resolves(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "run_camera", lambda *a, **k: pytest.fail("must not re-solve"))
    out = _step(tmp_path, _det(MOVED), rejected=True)
    assert out.status == "dropped" and "rejected by the operator" in out.reason
    # ... but a covering epoch that checks out is still kept
    assert _step(tmp_path, _det(BASE), rejected=True).status == "kept"


def test_check_uses_two_tag_frames_only_when_nothing_better(tmp_path):
    chk = cpm.check_epoch(BASE, _video(tmp_path), CROP, (0, 29), FM, _det(BASE, k=2),
                          cpm.CornerStepConfig())
    assert chk["min_tags_used"] == 2 and chk["best_m"] < 0.05


# --------------------------------------------------------------------- store

def test_load_versions_filters_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    a = field_store.save_version("e", "corner_left", BASE)
    b = field_store.save_version("e", "corner_left", MOVED, start_match="e_qm5")
    assert field_store.set_version_rejected("e", "corner_left", b, True)
    assert [v["version_id"] for v in field_store.load_versions("e", "corner_left")] == [a]
    allv = field_store.load_versions("e", "corner_left", include_rejected=True)
    assert {v["version_id"]: v["rejected"] for v in allv} == {a: False, b: True}
    field_store.set_version_rejected("e", "corner_left", b, False)
    assert len(field_store.load_versions("e", "corner_left")) == 2
    assert not field_store.set_version_rejected("e", "corner_left", "nope", True)


# ---------------------------------------------------------- job integration

@pytest.fixture
def app_mod(tmp_path, monkeypatch):
    mod = importlib.import_module("web.app")
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(mod, "_auto_calib_judge", lambda: None)
    return mod


def _job(app_mod, tmp_path, overrides=None):
    from web.job_config import JobConfig
    job = app_mod.InferenceJob.__new__(app_mod.InferenceJob)
    job.config = JobConfig(overrides or {})
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
    return job, v0


def _fake_outcome(status, cal=None):
    return lambda *a, **k: cpm.CornerOutcome(
        status, f"fake {status}", "v0", cal, {"check_best_m": 1.2},
        auto_tags={"6": [[0, 0]] * 4}, overlay=np.zeros((8, 8, 3), np.uint8), seconds=0.1)


def test_job_adopts_and_uses_version_immediately(app_mod, tmp_path, monkeypatch):
    job, v0 = _job(app_mod, tmp_path)
    monkeypatch.setattr(cpm, "corner_step", _fake_outcome("adopted", MOVED))
    out = app_mod.InferenceJob._corner_per_match_step(job)
    assert out["corner_left"]["status"] == "adopted"
    view = next(v for v in job.views if v.name == "corner_left")
    assert view.calibration is MOVED
    from web.field.match_order import select_epoch
    vs = field_store.load_versions("evt", "corner_left")
    new = select_epoch(vs, "evt_qm5")
    assert new["version_id"] != v0 and new["start_match"] == "evt_qm5"
    assert new["label"] == cpm.LABEL and new["source"]["method"] == "auto_per_match"
    assert new["source"]["crop"] == list(CROP) and new["source"]["match_key"] == "evt_qm5"
    assert select_epoch(vs, "evt_qm4")["version_id"] == v0          # earlier matches untouched
    log = field_store.load_permatch_log("evt", "corner_left")["evt_qm5"]
    assert log["status"] == "adopted" and log["version_id"] == new["version_id"]
    assert log["preview"] and field_store.permatch_preview_path("evt", "corner_left", "evt_qm5").is_file()
    assert "corner_left" in job._camera_centers_xy and "corner_calib" in job.timings
    assert app_mod.InferenceJob._replay_provenance(job)["corner_calibration"]["corner_left"]["status"] == "adopted"


def test_job_drops_failed_corner(app_mod, tmp_path, monkeypatch):
    job, _ = _job(app_mod, tmp_path)
    monkeypatch.setattr(cpm, "corner_step", _fake_outcome("dropped"))
    app_mod.InferenceJob._corner_per_match_step(job)
    assert [v.name for v in job.views] == ["main"]
    assert field_store.load_permatch_log("evt", "corner_left")["evt_qm5"]["status"] == "dropped"
    assert len(field_store.load_versions("evt", "corner_left")) == 1


def test_job_step_error_drops_corner(app_mod, tmp_path, monkeypatch):
    job, _ = _job(app_mod, tmp_path)

    def boom(*a, **k):
        raise RuntimeError("bad video")
    monkeypatch.setattr(cpm, "corner_step", boom)
    out = app_mod.InferenceJob._corner_per_match_step(job)
    assert out["corner_left"]["status"] == "dropped" and "bad video" in out["corner_left"]["reason"]
    assert [v.name for v in job.views] == ["main"]


def test_job_passes_rejected_and_keeps_operator_decision(app_mod, tmp_path, monkeypatch):
    job, _ = _job(app_mod, tmp_path)
    field_store.save_permatch_entry("evt", "corner_left", "evt_qm5",
                                    {"status": "rejected", "rejected_version": "vX",
                                     "prev_status": "adopted"})
    seen = {}

    def fake(*a, **k):
        seen.update(k)
        return _fake_outcome("dropped")()
    monkeypatch.setattr(cpm, "corner_step", fake)
    app_mod.InferenceJob._corner_per_match_step(job)
    assert seen["rejected"] is True
    e = field_store.load_permatch_log("evt", "corner_left")["evt_qm5"]
    assert e["status"] == "rejected" and e["job_outcome"] == "dropped"
    assert e["rejected_version"] == "vX" and e["prev_status"] == "adopted"


def test_knob_off_is_old_behaviour(app_mod, tmp_path, monkeypatch):
    job, _ = _job(app_mod, tmp_path, {"corner_per_match_calib": False})
    monkeypatch.setattr(cpm, "corner_step", lambda *a, **k: pytest.fail("must not run"))
    before = [(v.name, v.calibration) for v in job.views]
    assert app_mod.InferenceJob._corner_per_match_step(job) == {}
    assert [(v.name, v.calibration) for v in job.views] == before
    assert field_store.load_permatch_log("evt", "corner_left") == {}
    assert "corner_calib" not in job.timings
    assert "corner_calibration" not in app_mod.InferenceJob._replay_provenance(job)


def test_main_view_is_never_touched(app_mod, tmp_path, monkeypatch):
    job, _ = _job(app_mod, tmp_path)
    calls = []
    monkeypatch.setattr(cpm, "corner_step",
                        lambda video, mk, cam, *a, **k: calls.append(cam) or _fake_outcome("kept")())
    app_mod.InferenceJob._corner_per_match_step(job)
    assert calls == ["corner_left"]


def test_knob_defaults():
    from web.job_config import JobConfig
    c = JobConfig()
    assert (c.corner_per_match_calib, c.corner_drop_on_fail, c.corner_check_max_m,
            c.corner_resolve_max_m) == (True, True, 0.10, 0.15)
    assert JobConfig({"corner_per_match_calib": False}).non_defaults() == {
        "corner_per_match_calib": False}


# -------------------------------------------------------------------- routes

def test_per_match_routes_reject_and_restore(app_mod, tmp_path):
    c = TestClient(app_mod.app)
    vid = field_store.save_version("evt", "corner_left", MOVED, start_match="evt_qm5")
    field_store.save_permatch_entry("evt", "corner_left", "evt_qm5",
                                    {"status": "adopted", "version_id": vid,
                                     "preview": field_store.save_permatch_preview(
                                         "evt", "corner_left", "evt_qm5", b"\xff\xd8jpg")})
    j = c.get("/api/calibration/per-match", params={"event": "evt"}).json()
    assert j["cameras"]["corner_left"]["evt_qm5"]["status"] == "adopted"
    r = c.get("/api/calibration/per-match/preview",
              params={"event": "evt", "camera": "corner_left", "match": "evt_qm5"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert c.get("/api/calibration/per-match/preview", params={
        "event": "evt", "camera": "corner_left", "match": "evt_qm9"}).status_code == 404

    r = c.post("/api/calibration/per-match/reject",
               json={"event": "evt", "camera": "corner_left", "match": "evt_qm5"})
    assert r.status_code == 200 and r.json()["entry"]["status"] == "rejected"
    assert field_store.load_versions("evt", "corner_left") == []          # ignored now
    r = c.post("/api/calibration/per-match/restore",
               json={"event": "evt", "camera": "corner_left", "match": "evt_qm5"})
    assert r.json()["entry"]["status"] == "adopted"
    assert [v["version_id"] for v in field_store.load_versions("evt", "corner_left")] == [vid]
    assert c.post("/api/calibration/per-match/reject", json={
        "event": "evt", "camera": "corner_left", "match": "evt_qm1"}).status_code == 404


def test_resolve_falls_back_to_base_lens(tmp_path, monkeypatch):
    calls = []

    def fake_run(video, mk, cam, crop, fm, versions, **k):
        calls.append(versions[0]["version_id"])
        ok = versions[0]["version_id"] == "base"
        rec = ac.new_record("ready" if ok else "failed", "r", metrics={"val_gate_best_m": 0.02})
        return ac.AutoCalibResult(rec, MOVED if ok else None, None)
    monkeypatch.setattr(ac, "run_camera", fake_run)
    other_lens = truth_pinhole_camera(K=np.array([[1500., 0, 960], [0, 1500., 540], [0, 0, 1]]))
    out = cpm.corner_step(_video(tmp_path), "evt_qm5", "corner_left", CROP, (), FM,
                          _epoch(other_lens, "chain"), (0, 29), 1.0, detect_fn=_det(MOVED),
                          roi_detect_fn=lambda f: {}, alt_epochs=[_epoch(BASE, "base")],
                          log=lambda m: None)
    assert calls == ["chain", "base"]
    assert out.status == "adopted" and out.metrics["lens_from"] == "base"
    assert "lens of base" in out.reason
    # an alternative with the SAME lens is not tried twice
    calls.clear()
    out = cpm.corner_step(_video(tmp_path), "evt_qm5", "corner_left", CROP, (), FM,
                          _epoch(BASE, "chain"), (0, 29), 1.0, detect_fn=_det(MOVED),
                          roi_detect_fn=lambda f: {}, alt_epochs=[_epoch(BASE, "base2")],
                          log=lambda m: None)
    assert calls == ["chain"] and out.status == "dropped"


def test_implausible_epoch_is_never_kept(tmp_path, monkeypatch):
    import cv2 as _cv2
    flipped = truth_pinhole_camera()
    R, _ = _cv2.Rodrigues(flipped.rvec)
    Rz = np.diag([-1.0, -1.0, 1.0])                 # roll the camera 180 deg
    flipped.rvec = _cv2.Rodrigues(Rz @ R)[0]
    flipped.tvec = Rz @ flipped.tvec
    monkeypatch.setattr(ac, "run_camera", lambda *a, **k: ac.AutoCalibResult(
        ac.new_record("failed", "no"), None, None))
    out = cpm.corner_step(_video(tmp_path), "evt_qm5", "corner_left", CROP, (), FM,
                          _epoch(flipped), (0, 29), 1.0, detect_fn=_det(flipped),
                          roi_detect_fn=lambda f: {}, log=lambda m: None)
    assert out.status == "dropped" and "implausible" in out.reason


# ------------------------------------------------------- same-view fallback
# 2025 corners see only 1-2 tags: the tag check is inconclusive every match. When
# this match's view registers onto the epoch's calibration frame, keep the epoch.

from tests.field.test_view_registration import _crop, _scene   # noqa: E402

SCENE = _scene()
REF_IMG = _crop(SCENE)                       # the epoch's calibration view
SAME_IMG = _crop(SCENE, dx=2)                # this match, unchanged camera
MOVED_IMG = _crop(SCENE, dx=30, dy=6)        # this match, camera re-aimed


def _ref_fn(img=REF_IMG, calls=None):
    def fn():
        if calls is not None:
            calls.append(1)
        return {"image": img, "match_key": "evt_qm1", "timestamp": 125.8,
                "reason": None if img is not None else "source video not local"}
    return fn


def _sv_step(tmp_path, monkeypatch, match_img, *, best_m=None, cfg=None, ref_fn="default",
             resolve_ok=False):
    """corner_step with the tag check stubbed to ``best_m`` and this match's median
    stubbed to ``match_img``; returns (outcome, run_camera call count)."""
    monkeypatch.setattr(cpm, "check_epoch", lambda *a, **k: {
        "best_m": best_m, "n_tags": 0 if best_m is None else 3,
        "min_tags_used": None if best_m is None else 3, "frames": 6,
        "frames_used": 0 if best_m is None else 6})
    monkeypatch.setattr(ac, "median_frame", lambda *a, **k: match_img.copy())
    runs = []

    def fake_run(*a, **k):
        runs.append(1)
        if resolve_ok:
            return ac.AutoCalibResult(ac.new_record("ready", "ok", metrics={}), MOVED, None)
        return ac.AutoCalibResult(ac.new_record(
            "failed", "only 1 known tags found (need >= 3)"), None, None)
    monkeypatch.setattr(ac, "run_camera", fake_run)
    cfg = cfg or cpm.CornerStepConfig(detect_motion=False)
    out = cpm.corner_step(_video(tmp_path), "evt_qm5", "corner_left", CROP, (), FM,
                          _epoch(), (0, 29), 1.0, cfg, detect_fn=lambda f: {},
                          roi_detect_fn=lambda f: {}, log=lambda m: None,
                          ref_view_fn=_ref_fn() if ref_fn == "default" else ref_fn)
    return out, len(runs)


def test_same_view_keeps_epoch_without_resolve(tmp_path, monkeypatch):
    out, runs = _sv_step(tmp_path, monkeypatch, SAME_IMG)
    assert out.status == "kept", out.reason
    assert runs == 0 and out.calibration is None and out.epoch_checked == "v0"
    assert "too few tags" in out.reason and "evt_qm1 @ 125.8 s" in out.reason
    m = out.metrics
    assert 1.0 < m["same_view_px"] < 3.0 and m["same_view_inliers"] >= 60
    assert m["same_view_ref"] == "evt_qm1"
    assert out.overlay is not None


def test_moved_view_falls_through_and_drops(tmp_path, monkeypatch):
    out, runs = _sv_step(tmp_path, monkeypatch, MOVED_IMG)
    assert out.status == "dropped" and runs == 1
    assert "did not validate" in out.reason
    assert "calibration frame" in out.reason            # the same-view result is explained
    assert out.metrics["same_view_ref"] == "evt_qm1"
    assert out.metrics.get("same_view_px") is None or out.metrics["same_view_px"] > 8.0


def test_moved_view_can_still_adopt_a_resolve(tmp_path, monkeypatch):
    out, runs = _sv_step(tmp_path, monkeypatch, MOVED_IMG, resolve_ok=True)
    assert out.status == "adopted" and runs == 1


def test_tags_measured_off_never_consult_the_view(tmp_path, monkeypatch):
    calls = []
    out, runs = _sv_step(tmp_path, monkeypatch, SAME_IMG, best_m=0.9,
                         ref_fn=_ref_fn(calls=calls))
    assert calls == [] and runs == 1 and out.status == "dropped"
    assert "same_view_px" not in out.metrics


def test_fallback_disabled_is_old_behaviour(tmp_path, monkeypatch):
    calls = []
    out, runs = _sv_step(tmp_path, monkeypatch, SAME_IMG, ref_fn=_ref_fn(calls=calls),
                         cfg=cpm.CornerStepConfig(detect_motion=False,
                                                  same_view_fallback=False))
    assert calls == [] and runs == 1 and out.status == "dropped"
    assert "same_view_ref" not in out.metrics


def test_no_reference_is_old_behaviour(tmp_path, monkeypatch):
    out, runs = _sv_step(tmp_path, monkeypatch, SAME_IMG, ref_fn=_ref_fn(img=None))
    assert out.status == "dropped" and runs == 1
    assert "source video not local" in out.metrics["same_view_reason"]
    out, runs = _sv_step(tmp_path, monkeypatch, SAME_IMG, ref_fn=None)
    assert out.status == "dropped" and runs == 1


def test_rejected_match_still_kept_by_same_view(tmp_path, monkeypatch):
    monkeypatch.setattr(cpm, "check_epoch", lambda *a, **k: {
        "best_m": None, "n_tags": 0, "min_tags_used": None, "frames": 6, "frames_used": 0})
    monkeypatch.setattr(ac, "median_frame", lambda *a, **k: SAME_IMG.copy())
    monkeypatch.setattr(ac, "run_camera", lambda *a, **k: pytest.fail("must not re-solve"))
    out = cpm.corner_step(_video(tmp_path), "evt_qm5", "corner_left", CROP, (), FM,
                          _epoch(), (0, 29), 1.0, cpm.CornerStepConfig(detect_motion=False),
                          detect_fn=lambda f: {}, roi_detect_fn=lambda f: {},
                          log=lambda m: None, rejected=True, ref_view_fn=_ref_fn())
    assert out.status == "kept"


# ------------------------------------------------------- reference view builder

def _src_epoch(**src):
    e = _epoch()
    e["source"] = {"match_key": "evt_qm1", "timestamp": 125.8, "crop": list(CROP), **src}
    return e


def test_reference_view_builds_median_and_caches(tmp_path, monkeypatch):
    vid = tmp_path / "evt_qm1.mp4"
    vid.write_bytes(b"x")
    seen = {}

    def fake_median(video, crop, lo, hi, n=31, **k):
        seen.update(video=video, crop=tuple(crop), lo=lo, hi=hi, n=n)
        return REF_IMG.copy()
    monkeypatch.setattr(ac, "median_frame", fake_median)
    saved = []
    r = cpm.reference_view(_src_epoch(), CROP, video_for=lambda mk: vid,
                           load_cached=lambda: None, save_cached=saved.append)
    assert r["image"] is not None and r["match_key"] == "evt_qm1" and r["timestamp"] == 125.8
    assert seen["hi"] == 125.8 and seen["lo"] < 125.8 and seen["lo"] >= 125.8 - 30
    assert saved and saved[0].startswith(b"\xff\xd8")       # cached as JPEG
    # cache hit: no video needed
    monkeypatch.setattr(ac, "median_frame", lambda *a, **k: pytest.fail("cache must win"))
    r2 = cpm.reference_view(_src_epoch(), CROP, video_for=lambda mk: None,
                            load_cached=lambda: saved[0], save_cached=None)
    assert r2["image"].shape == REF_IMG.shape and r2["reason"] is None


@pytest.mark.parametrize("epoch,why", [
    (_epoch(), "no calibration source"),
    (_src_epoch(timestamp=None), "no calibration source"),
    (_src_epoch(crop=[0, 733, 965, 347]), "crop"),
    (_src_epoch(), "not local"),
])
def test_reference_view_unavailable(epoch, why, monkeypatch):
    monkeypatch.setattr(ac, "median_frame", lambda *a, **k: pytest.fail("must not read"))
    r = cpm.reference_view(epoch, CROP, video_for=lambda mk: None,
                           load_cached=lambda: None, save_cached=None)
    assert r["image"] is None and why in r["reason"]


def test_refview_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    assert field_store.load_refview("e", "corner_left", "v1") is None
    name = field_store.save_refview("e", "corner_left", "v1", b"\xff\xd8jpg")
    assert (tmp_path / name).is_file() and name.endswith(".jpg") and "v1" in name
    assert field_store.refview_path("e", "corner_left", "v1") == tmp_path / name
    assert field_store.load_refview("e", "corner_left", "v1") == b"\xff\xd8jpg"
    assert field_store.load_refview("e", "corner_right", "v1") is None


# ------------------------------------------------------- job wiring

def test_same_view_knob_defaults():
    from web.job_config import JobConfig
    c = JobConfig()
    assert (c.corner_same_view_fallback, c.corner_same_view_max_px) == (True, 8.0)
    with pytest.raises(Exception):
        JobConfig({"corner_same_view_max_px": 0.1})


def test_job_passes_same_view_knobs_and_ref_fn(app_mod, tmp_path, monkeypatch):
    job, v0 = _job(app_mod, tmp_path, {"corner_same_view_fallback": False,
                                       "corner_same_view_max_px": 5.0})
    seen = {}

    def fake(video, mk, cam, crop, masks, fm, epoch, frange, fps, cfg, **k):
        seen["cfg"], seen["ref"] = cfg, k.get("ref_view_fn")
        return _fake_outcome("kept")()
    monkeypatch.setattr(cpm, "corner_step", fake)
    out = app_mod.InferenceJob._corner_per_match_step(job)
    assert seen["cfg"].same_view_fallback is False and seen["cfg"].same_view_max_px == 5.0
    assert callable(seen["ref"])
    assert out["corner_left"]["status"] == "kept" and out["corner_left"]["version_id"] == "v0"
    # the job's ref_view_fn caches through the store
    monkeypatch.setattr(ac, "local_video_for", lambda mk: None)
    r = seen["ref"]()
    assert r["image"] is None                         # _job's v0 has no source
    vid = field_store.save_version("evt", "corner_left", BASE, source={
        "match_key": "evt_qm5", "timestamp": 20.0, "crop": list(CROP)})
    field_store.save_refview("evt", "corner_left", vid, ac.encode_jpeg(REF_IMG, 95))
    job2, _ = _job(app_mod, tmp_path)
    job2._view_selection = {"corner_left": vid}
    monkeypatch.setattr(ac, "median_frame", lambda *a, **k: pytest.fail("cache must win"))
    app_mod.InferenceJob._corner_per_match_step(job2)
    r = seen["ref"]()
    assert r["image"] is not None and r["match_key"] == "evt_qm5"
