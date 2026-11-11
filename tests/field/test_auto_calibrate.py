"""Auto-calibration on event load: the pure per-camera pipeline (web.field.auto_calibrate)
and the auto-record store helpers.

Synthetic footage: a tiny black mp4 supplies frame timing, and an injected detect_fn
returns tag corners projected from a known camera — so the real aggregation, solve,
validation, pose re-solve and overlay code all run without real video, a GPU or the
network.
"""
import cv2
import numpy as np
import pytest

from web.field import auto_calibrate as ac
from web.field import store as field_store
from web.field.calibration import CameraCalibration
from web.field.field_model import FieldModel
from tests._field_synth import truth_pinhole_camera, project_tag_corners

FM = FieldModel.load("2026-rebuilt-welded")
CROP = (0, 0, 1920, 1080)          # truth_pinhole_camera image size


def _video(tmp_path, n=30, fps=1.0, name="clip.mp4"):
    p = tmp_path / name
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 36))
    for i in range(n):
        w.write(np.full((36, 64, 3), 10, np.uint8))
    w.release()
    return p


def _detector(cam, only=None):
    corr = project_tag_corners(cam, FM)
    dets = {tid: c2d for tid, (c2d, _) in corr.items() if only is None or tid in only}
    return lambda frame: {k: v.copy() for k, v in dets.items()}


def _version(cal, vid="v1", start_match=None):
    return {"calibration": cal, "start_match": start_match, "version_id": vid,
            "created_at": "2026-11-01T00:00:00", "label": None, "source": None}


# ----------------------------------------------------------------- full solve

def test_full_solve_ready_with_metrics_and_overlay(tmp_path):
    # The shared fixture lens (k1 -0.25, k2 -0.08 at f 1200) folds back inside its own
    # image (r_d peaks at 0.67, edge tags project to 0.72) — physically impossible,
    # and the full solve now refuses non-monotonic lenses. Use a realistic lens here.
    cam = truth_pinhole_camera(D=np.array([-0.12, 0.02, 0.0, 0.0, 0.0]))
    video = _video(tmp_path)
    res = ac.run_camera(video, "evt_qm1", "main", CROP, FM, versions=[],
                        detect_fn=_detector(cam))
    rec = res.record
    assert rec["status"] == "ready" and rec["method"] == "full_solve"
    assert res.calibration is not None and res.calibration.mode == "pinhole"
    assert rec["calibration"]["mode"] == "pinhole"
    m = rec["metrics"]
    used = set(project_tag_corners(cam, FM))
    assert m["n_tags"] == len(used) and m["rms_px"] < 2.0
    assert m["val_best_m"] < 0.05 and m["val_worst_m"] < 0.05 and m["val_frames"] > 0
    assert m["center_shift_m"] is None and m["rotation_delta_deg"] is None
    assert not set(m["unused_visible_tags"]) & used
    assert set(rec["auto_tags"]) == {str(t) for t in used}
    assert all(len(c) == 4 for c in rec["auto_tags"].values())
    assert set(rec["per_tag"]) == {str(t) for t in used}
    assert rec["match_key"] == "evt_qm1" and rec["timestamp"] >= 18.0
    assert rec["epoch"] is None and rec["resolved_version"] is None
    assert res.overlay is not None and res.overlay.shape[:2] == (36, 64)
    assert ac.encode_jpeg(res.overlay)[:2] == b"\xff\xd8"


def test_full_solve_too_few_tags_fails(tmp_path):
    cam = truth_pinhole_camera()
    two = sorted(project_tag_corners(cam, FM))[:2]
    res = ac.run_camera(_video(tmp_path), "evt_qm1", "main", CROP, FM, versions=[],
                        detect_fn=_detector(cam, only=two))
    assert res.record["status"] == "failed"
    assert "only 2 known tags" in res.record["reason"]
    assert res.calibration is None and res.record["calibration"] is None
    assert res.overlay is not None          # detections still drawn for the operator


def test_unreadable_video_raises(tmp_path):
    bad = tmp_path / "empty.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ac.AutoCalibError):
        ac.run_camera(bad, "evt_qm1", "main", CROP, FM, versions=[],
                      detect_fn=lambda f: {})


# --------------------------------------------------------------- verify branch

def test_verify_branch_verified_when_epoch_still_aligns(tmp_path):
    cam = truth_pinhole_camera()
    res = ac.run_camera(_video(tmp_path), "evt_qm4", "main", CROP, FM,
                        versions=[_version(cam, "v1")], detect_fn=_detector(cam))
    rec = res.record
    assert rec["status"] == "verified" and rec["method"] == "verify"
    assert rec["epoch"] == "v1"
    assert rec["calibration"] is None and res.calibration is None
    assert rec["metrics"]["val_gate_best_m"] < 0.05


def test_verify_picks_epoch_covering_reference_match(tmp_path):
    cam = truth_pinhole_camera()
    stale = truth_pinhole_camera(target=(4.0, 4.03, 0.0))
    versions = [_version(stale, "old", None), _version(cam, "new", "evt_qm3")]
    res = ac.run_camera(_video(tmp_path), "evt_qm5", "main", CROP, FM,
                        versions=versions, detect_fn=_detector(cam))
    assert res.record["status"] == "verified" and res.record["epoch"] == "new"


def test_verify_drifted_then_pose_resolve_ready(tmp_path):
    base = truth_pinhole_camera()                         # saved epoch
    moved = truth_pinhole_camera(target=(10.0, 4.03, 0.0))  # camera panned since
    res = ac.run_camera(_video(tmp_path), "evt_qm9", "main", CROP, FM,
                        versions=[_version(base, "v1")], detect_fn=_detector(moved))
    rec = res.record
    assert rec["status"] == "ready" and rec["method"] == "pose_resolve"
    assert rec["epoch"] == "v1"
    assert res.calibration is not None and rec["calibration"] is not None
    assert np.allclose(res.calibration.K, base.K)          # lens kept
    m = rec["metrics"]
    assert m["center_shift_m"] < 0.1 and m["rotation_delta_deg"] > 1.0
    assert m["val_gate_best_m"] < 0.05
    assert "misaligned" in rec["reason"]


def test_verify_drifted_homography_epoch_fails(tmp_path):
    hom = CameraCalibration(mode="homography", H=np.eye(3), image_size=(1920, 1080),
                            field_season="2026-rebuilt-welded")
    res = ac.run_camera(_video(tmp_path), "evt_qm2", "main", CROP, FM,
                        versions=[_version(hom, "h1")],
                        detect_fn=_detector(truth_pinhole_camera()))
    assert res.record["status"] == "failed"
    assert res.record["method"] == "pose_resolve"
    assert "homography" in res.record["reason"]


def test_verify_drifted_pose_resolve_fails_on_camera_relocation(tmp_path):
    base = truth_pinhole_camera()
    relocated = truth_pinhole_camera(eye=(4.0, -3.0, 6.0))   # moved 4 m sideways
    res = ac.run_camera(_video(tmp_path), "evt_qm9", "main", CROP, FM,
                        versions=[_version(base, "v1")], detect_fn=_detector(relocated))
    assert res.record["status"] == "failed"
    assert "camera center moved" in res.record["reason"]


# ------------------------------------------------------------------- geometry

def test_visible_model_tags_cover_used_tags_and_drop_back_facing():
    cam = truth_pinhole_camera()
    vis = set(ac.visible_model_tags(cam, FM, cam.image_size))
    in_frame = set(project_tag_corners(cam, FM))
    facing = {t for t in in_frame if ac._faces_camera(cam, FM, t)}
    assert facing and facing <= vis
    assert all(ac._faces_camera(cam, FM, t) for t in vis)


# --------------------------------------------------------- reference footage

def test_pick_reference_prefers_earliest_local_quals_first(tmp_path):
    (tmp_path / "evt_qm7.mp4").write_bytes(b"x")
    (tmp_path / "evt_sf1m1.mp4").write_bytes(b"x")
    (tmp_path / "evt_qm12.mp4").write_bytes(b"x")       # not in the TBA list
    matches = [{"key": "evt_qm1", "video_key": "a"}, {"key": "evt_qm7"},
               {"key": "evt_sf1m1"}]
    mk, path, dl = ac.pick_reference_match("evt", matches, dirs=[tmp_path])
    assert mk == "evt_qm7" and path == tmp_path / "evt_qm7.mp4" and dl is None


def test_pick_reference_local_file_without_match_list(tmp_path):
    (tmp_path / "evt_qm3.mp4").write_bytes(b"x")
    mk, path, _ = ac.pick_reference_match("evt", [], dirs=[tmp_path])
    assert mk == "evt_qm3"


def test_pick_reference_download_candidate_is_earliest_qual_with_video(tmp_path):
    matches = [{"key": "evt_f1m1", "video_key": "f"}, {"key": "evt_qm2", "video_key": "b"},
               {"key": "evt_qm1", "video_key": None}, {"key": "evt_qm5", "video_key": "e"}]
    mk, path, dl = ac.pick_reference_match("evt", matches, dirs=[tmp_path])
    assert mk == "evt_qm2" and path is None and dl["video_key"] == "b"


def test_pick_reference_none(tmp_path):
    assert ac.pick_reference_match("evt", [{"key": "evt_f1m1", "video_key": "f"}],
                                   dirs=[tmp_path]) == (None, None, None)


def test_local_video_for_rejects_unsafe_keys(tmp_path):
    (tmp_path / "evt_qm1.mp4").write_bytes(b"x")
    assert ac.local_video_for("evt_qm1", [tmp_path]) == tmp_path / "evt_qm1.mp4"
    for bad in ("../evt_qm1", "evt_qm1/../../x", "", None, "evt qm1"):
        assert ac.local_video_for(bad, [tmp_path]) is None


def test_resolve_reference_video_download_failure_carries_reason(tmp_path):
    def boom(video_key, out):
        raise ac.AutoCalibError(f"download of {video_key} failed: HTTP Error 403: Forbidden")
    with pytest.raises(ac.AutoCalibError) as ei:
        ac.resolve_reference_video("evt", [{"key": "evt_qm1", "video_key": "vk1"}],
                                   dirs=[tmp_path], download_dir=tmp_path, downloader=boom)
    assert "403" in str(ei.value)


def test_resolve_reference_video_downloads_to_match_key(tmp_path):
    calls = []

    def fake(video_key, out):
        calls.append((video_key, out))
        w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 1, (64, 36))
        w.write(np.zeros((36, 64, 3), np.uint8)); w.release()
        return out
    mk, path = ac.resolve_reference_video(
        "evt", [{"key": "evt_qm1", "video_key": "vk1"}], dirs=[tmp_path / "none"],
        download_dir=tmp_path, downloader=fake)
    assert mk == "evt_qm1" and path == tmp_path / "evt_qm1.mp4"
    assert calls == [("vk1", tmp_path / "evt_qm1.mp4")]


def test_resolve_reference_video_nothing_available(tmp_path):
    with pytest.raises(ac.AutoCalibError, match="no local footage"):
        ac.resolve_reference_video("evt", [], dirs=[tmp_path], downloader=None)


# ---------------------------------------------------------------------- store

def test_auto_record_store_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    assert field_store.load_auto_record("evt", "main") is None
    assert field_store.update_auto_record("evt", "main", status="failed") is None
    rec = ac.new_record("ready", "solved", match_key="evt_qm1", method="full_solve")
    field_store.save_auto_record("evt", "main", rec)
    assert (tmp_path / "calibration_auto_evt_main.json").is_file()
    assert field_store.load_auto_record("evt", "main") == rec
    up = field_store.update_auto_record("evt", "main", status="confirmed",
                                        resolved_version="v9")
    assert up["status"] == "confirmed" and up["resolved_version"] == "v9"
    assert field_store.load_auto_record("evt", "main")["match_key"] == "evt_qm1"
    name = field_store.save_auto_preview("evt", "main", b"\xff\xd8jpeg")
    assert name == "calibration_auto_evt_main.jpg"
    assert field_store.auto_preview_path("evt", "main").read_bytes() == b"\xff\xd8jpeg"
    # auto records never leak into calibration versions
    assert field_store.load_versions("evt", "main") == []


def test_new_record_has_spec_shape():
    rec = ac.new_record("running")
    for k in ("status", "method", "reason", "match_key", "timestamp", "calibration",
              "epoch", "metrics", "per_tag", "auto_tags", "preview", "created_at",
              "resolved_at", "resolved_version"):
        assert k in rec


# ------------------------------------------------------- phase 2: median frame

def test_median_frame_removes_transient_occluder(tmp_path):
    p = tmp_path / "occ.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 1, (64, 48))
    for i in range(40):
        f = np.full((48, 64, 3), 60, np.uint8)
        if i % 4 == 0:
            f[5:25, 5:25] = 250
        w.write(f)
    w.release()
    med = ac.median_frame(p, (0, 0, 64, 48), 0.0, 1e9, 31)
    assert med.shape == (48, 64, 3) and int(med[15, 15].mean()) < 100
    assert ac.median_frame(tmp_path / "nope.mp4", (0, 0, 64, 48), 0, 10) is None


def test_median_only_tag_joins_aggregate_and_record_carries_frame(tmp_path, monkeypatch):
    cam = truth_pinhole_camera()
    corr = project_tag_corners(cam, FM)
    ids = sorted(corr)
    extra_tag = ids[-1]
    marker = np.full((36, 64, 3), 77, np.uint8)
    monkeypatch.setattr(ac, "median_frame", lambda *a, **k: marker.copy())

    def detect(frame):
        out = {t: corr[t][0].copy() for t in ids[:-1]}
        if frame[0, 0, 0] == 77:                 # only the median shows the last tag
            out[extra_tag] = corr[extra_tag][0].copy()
        return out
    cfg = ac.AutoCalibConfig(locate_undecoded=False)
    res = ac.run_camera(_video(tmp_path), "evt_qm1", "main", CROP, FM, versions=[],
                        cfg=cfg, detect_fn=detect)
    rec = res.record
    assert str(extra_tag) in rec["auto_tags"]
    assert rec["per_tag"][str(extra_tag)]["source"] == "median"
    assert rec["metrics"]["n_tags"] == len(ids)
    assert res.median is not None and res.overlay.shape == marker.shape


# ------------------------------------------------------ phase 2: tag locator

from web.field import tag_locator as tl


def _render_tags(cam, ids, size, bg=40):
    """Synthetic frame: each tag = white border (1.4x) + black square + grey bits."""
    w, h = size
    img = np.full((h, w, 3), bg, np.uint8)
    for t in ids:
        pc = np.array([cam.field_to_image(tuple(p)) for p in FM.tag_corners_3d(t)])
        c = pc.mean(axis=0)
        cv2.fillConvexPoly(img, np.round((pc - c) * 1.4 + c).astype(np.int32), (235,) * 3)
        cv2.fillConvexPoly(img, np.round(pc).astype(np.int32), (25,) * 3)
        cv2.fillConvexPoly(img, np.round((pc - c) * 0.75 + c).astype(np.int32), (110,) * 3)
    return cv2.GaussianBlur(img, (0, 0), 0.8)


MONO_D = np.array([-0.3, 0.08, 0.0, 0.0, 0.0])     # radial lens, monotonic over the image


def _locator_scene():
    cam = truth_pinhole_camera(D=MONO_D)
    size = cam.image_size
    vis = [t for t in ac.visible_model_tags(cam, FM, size)
           if tl.predict_tag(cam, FM, t, size) is not None]
    img = _render_tags(cam, vis, size)
    centre = np.array(size) / 2.0
    by_dist = sorted(vis, key=lambda t: np.linalg.norm(
        np.array(cam.field_to_image(tuple(FM.tag_translation(t)))) - centre))
    corr = project_tag_corners(cam, FM)
    decoded = {t: corr[t][0] for t in [t for t in by_dist if t in corr][:5]}
    return cam, size, vis, img, decoded


def _truth_err(cal, cam, ids, size):
    out = []
    for t in ids:
        a = np.array(cal.field_to_image(tuple(FM.tag_translation(t))))
        b = np.array(cam.field_to_image(tuple(FM.tag_translation(t))))
        out.append(float(np.linalg.norm(a - b)))
    return max(out) if out else 0.0


def test_tag_pairs_from_field_model():
    pairs = tl.tag_pairs(FM, 0.6)
    for a, b in ((13, 14), (15, 16), (31, 32), (3, 4), (19, 20), (5, 8), (18, 27)):
        assert pairs.get(a) == b and pairs.get(b) == a
    assert 17 not in pairs


def test_distortion_monotonic_detects_fold_back():
    good = truth_pinhole_camera(D=MONO_D)
    assert tl.distortion_monotonic(good, good.image_size)
    bad = truth_pinhole_camera(D=np.array([-0.9, 0.0, 0.0, 0.0, 0.0]))
    assert not tl.distortion_monotonic(bad, bad.image_size)
    # checked only out to given points, a central region is still fine
    assert tl.distortion_monotonic(bad, bad.image_size, np.array([[960., 540.], [1000., 560.]]))


def test_pose_model_needs_lens_prior():
    corr = {f"a{t}": (c2, c3) for t, (c2, c3) in project_tag_corners(truth_pinhole_camera(), FM).items()}
    with pytest.raises(Exception):
        tl.solve_model(corr, (1920, 1080), "s", "pose")
    cal = tl.solve_model(corr, (1920, 1080), "s", "pose", lens=truth_pinhole_camera())
    assert cal.rms_error < 0.5


def test_locator_grows_solve_with_judge(tmp_path):
    cam, size, vis, img, decoded = _locator_scene()
    calls = []

    def judge(jpg):
        calls.append(len(jpg))
        return {"probs": {"single_apriltag": 0.85, "pair_of_apriltags": 0.85,
                          "not_an_apriltag": 0.05, "unclear": 0.1}}
    res = tl.locate_undecoded(img, FM, decoded, size, "2026-rebuilt-welded",
                              tl.LocatorConfig(), detect_fn=None, judge=judge, log=lambda m: None)
    located = {d["id"]: d for d in res.located}
    assert calls and len(located) >= 4
    assert set(located) <= set(vis) and not set(located) & set(decoded)
    for t, d in located.items():
        true = np.array(cam.field_to_image(tuple(FM.tag_translation(t))))
        assert np.hypot(d["u"] - true[0], d["v"] - true[1]) < 3.0
        assert d["method"] == "quad_jev" and d["jev_p"] in (0.85, 1.0)   # single: p(s)+p(pair)
        assert d["residual_px"] < 4.0
    # the grown solve predicts every visible tag within a few px
    assert _truth_err(res.calibration, cam, vis, size) < 5.0
    assert all(k[0] in "am" for k in res.corr)


def test_locator_without_judge_skips_template_candidates():
    cam, size, vis, img, decoded = _locator_scene()
    res = tl.locate_undecoded(img, FM, decoded, size, "2026-rebuilt-welded",
                              tl.LocatorConfig(jev_verify=True), detect_fn=None, judge=None,
                              log=lambda m: None)
    assert res.located == [] and any("Jev unavailable" in n for n in res.notes)


def test_locator_judge_rejection_blocks_candidates():
    cam, size, vis, img, decoded = _locator_scene()
    res = tl.locate_undecoded(
        img, FM, decoded, size, "2026-rebuilt-welded", tl.LocatorConfig(), detect_fn=None,
        judge=lambda jpg: {"probs": {"single_apriltag": 0.1, "pair_of_apriltags": 0.1,
                                     "not_an_apriltag": 0.7, "unclear": 0.1}},
        log=lambda m: None)
    assert res.located == []
    assert res.candidates and all("rejected" in c["verdict"] or "no candidate" in c["verdict"]
                                  for c in res.candidates)


def test_locator_judge_error_is_not_acceptance():
    cam, size, vis, img, decoded = _locator_scene()
    res = tl.locate_undecoded(img, FM, decoded, size, "2026-rebuilt-welded",
                              tl.LocatorConfig(), detect_fn=None,
                              judge=lambda jpg: {"error": "jev: 401"}, log=lambda m: None)
    assert res.located == []


def test_run_camera_records_located_tags(tmp_path, monkeypatch):
    cam, size, vis, img, decoded = _locator_scene()
    monkeypatch.setattr(ac, "median_frame", lambda *a, **k: img.copy())
    judge = lambda jpg: {"probs": {"single_apriltag": 0.9, "pair_of_apriltags": 0.9,
                                   "not_an_apriltag": 0.0, "unclear": 0.1}}
    res = ac.run_camera(_video(tmp_path), "evt_qm1", "main", CROP, FM, versions=[],
                        detect_fn=lambda f: {t: c.copy() for t, c in decoded.items()},
                        judge=judge, roi_detect_fn=lambda f: {})
    rec = res.record
    lt = rec["metrics"]["located_tags"]
    assert lt and all(set(d) == {"id", "method", "jev_p", "residual_px"} for d in lt)
    ids = {d["id"] for d in lt}
    assert {m["id"] for m in rec["manual_tags"]} == ids
    assert not ids & set(rec["metrics"]["unused_visible_tags"])
    assert "located" in rec["reason"] and rec["status"] == "ready"
    assert res.overlay.shape[:2] == (size[1], size[0])


def test_judge_p_single_counts_pair_but_pair_does_not_count_single():
    probs = {"single_apriltag": 0.25, "pair_of_apriltags": 0.5, "not_an_apriltag": 0.2,
             "unclear": 0.05}
    assert tl.judge_p(probs, "single_apriltag") == pytest.approx(0.75)
    assert tl.judge_p(probs, "pair_of_apriltags") == pytest.approx(0.5)
    assert tl.judge_p({"single_apriltag": 0.9, "pair_of_apriltags": 0.9}, "single_apriltag") == 1.0


def test_full_solve_from_one_hub_corner_is_never_ready(tmp_path):
    """2026casnd corner_left: 4 tags on two faces of ONE hub corner (~0.9 m apart)
    fit a free lens perfectly but extrapolate wildly. Too little field spread to
    determine a lens -> failed with a 'span' reason (candidate kept for inspection)."""
    cam = truth_pinhole_camera(D=np.array([-0.12, 0.02, 0.0, 0.0, 0.0]))
    hub = {5, 8, 9, 10}
    assert hub <= set(project_tag_corners(cam, FM))
    assert ac.tag_spread_m(FM, hub) < 1.0
    res = ac.run_camera(_video(tmp_path), "evt_qm1", "corner_right", CROP, FM, versions=[],
                        detect_fn=_detector(cam, only=hub))
    rec = res.record
    assert rec["status"] == "failed" and rec["method"] == "full_solve"
    assert "only span" in rec["reason"] and "far tags" in rec["reason"]
    assert res.overlay is not None


def test_tag_spread_m():
    assert ac.tag_spread_m(FM, []) == 0.0
    assert ac.tag_spread_m(FM, [5]) == 0.0
    assert ac.tag_spread_m(FM, [5, 8, 9, 10]) < 1.0
    assert ac.tag_spread_m(FM, [5, 25]) > 5.0
