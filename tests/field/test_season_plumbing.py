"""An event's season reaches calibration, the tag map, the clock search and GT
scoring (web.app + web.field.gt_scoring). No network, no real footage."""
import asyncio
import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests._field_synth import project_tag_corners, truth_pinhole_camera
from web.field import auto_calibrate as ac
from web.field import gt_scoring
from web.field import store as field_store
from web.field.field_model import FieldModel


@pytest.fixture
def app_mod(tmp_path, monkeypatch):
    mod = importlib.import_module("web.app")
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(mod, "_auto_calib_tasks", {})
    monkeypatch.setattr(mod, "_auto_calib_judge", lambda: None)
    # never probe footage for camera moves in these tests
    monkeypatch.setattr(mod, "_auto_calib_pick_steady",
                        lambda event, camera, mk, video, *a, **k: (mk, video, [], None))
    return mod


def _client(mod):
    return TestClient(mod.app)


def _fake_result():
    return ac.AutoCalibResult(record=ac.new_record("ready", "solved", method="full_solve"),
                              calibration=None, overlay=None)


# ------------------------------------------------------------- auto-calibration

def test_auto_calibration_solves_a_2025_event_against_the_2025_layout(app_mod, monkeypatch,
                                                                      tmp_path):
    monkeypatch.setattr(ac, "resolve_reference_video",
                        lambda *a, **k: ("2025hop_qm1", tmp_path / "2025hop_qm1.mp4"))
    monkeypatch.setattr(ac, "ensure_event_layout", lambda *a, **k: None)
    seen = {}

    def fake_run(video, match_key, camera, crop, fm, versions, cfg=None, **kw):
        seen[camera] = (fm.season, cfg.field_season, len(fm.tags))
        return _fake_result()
    monkeypatch.setattr(ac, "run_camera", fake_run)
    asyncio.run(app_mod._run_auto_calibrate("2025hop", [], ["main"]))
    assert seen == {"main": ("2025-reefscape-welded", "2025-reefscape-welded", 22)}
    assert field_store.load_auto_record("2025hop", "main")["status"] == "ready"


def test_auto_calibration_keeps_2026_for_2026_and_keyless_events(app_mod, monkeypatch,
                                                                  tmp_path):
    monkeypatch.setattr(ac, "resolve_reference_video",
                        lambda *a, **k: ("evt_qm1", tmp_path / "evt_qm1.mp4"))
    monkeypatch.setattr(ac, "ensure_event_layout", lambda *a, **k: None)
    seen = []
    monkeypatch.setattr(ac, "run_camera", lambda video, mk, cam, crop, fm, versions,
                        cfg=None, **kw: seen.append((fm.season, cfg.field_season))
                        or _fake_result())
    for ev in ("2026cur", "evt"):
        asyncio.run(app_mod._run_auto_calibrate(ev, [], ["main"]))
    assert seen == [("2026-rebuilt-welded", "2026-rebuilt-welded")] * 2


def test_auto_calibration_honours_an_andymark_override(app_mod, monkeypatch, tmp_path):
    from web.field import season as sn
    sn.save_override("2025hop", variant="andymark")
    monkeypatch.setattr(ac, "resolve_reference_video",
                        lambda *a, **k: ("2025hop_qm1", tmp_path / "2025hop_qm1.mp4"))
    monkeypatch.setattr(ac, "ensure_event_layout", lambda *a, **k: None)
    seen = []
    monkeypatch.setattr(ac, "run_camera", lambda video, mk, cam, crop, fm, versions,
                        cfg=None, **kw: seen.append(cfg.field_season) or _fake_result())
    asyncio.run(app_mod._run_auto_calibrate("2025hop", [], ["main"]))
    assert seen == ["2025-reefscape-andymark"]


@pytest.mark.parametrize("event", ["2019txho", "2031abc"])
def test_auto_calibration_without_a_tag_layout_fails_clearly(app_mod, monkeypatch, event):
    monkeypatch.setattr(ac, "resolve_reference_video",
                        lambda *a, **k: pytest.fail("must not look for footage"))
    monkeypatch.setattr(ac, "run_camera", lambda *a, **k: pytest.fail("must not solve"))
    asyncio.run(app_mod._run_auto_calibrate(event, [], ["main", "corner_left"]))
    year = event[:4]
    for cam in ("main", "corner_left"):
        rec = field_store.load_auto_record(event, cam)
        assert rec["status"] == "failed"
        assert rec["reason"] == f"no AprilTag layout for {year} — calibrate manually"


# --------------------------------------------------------------- manual solves

def _cam25():
    # undistorted: the shared fixture's strong k1 folds the 2025 edge tags
    cam = truth_pinhole_camera(eye=(8.77, -3.0, 6.0), target=(8.77, 4.03, 0.0),
                               D=np.zeros(5))
    cam.field_season = "2025-reefscape-welded"
    return cam


def test_solve_tags_uses_the_event_season(app_mod):
    fm = FieldModel.load("2025-reefscape-welded")
    cam = _cam25()
    from web.field.field_model import _quat_to_matrix
    c = cam.camera_center
    # only tags that face the camera (a real detector cannot decode the back faces)
    corr = {t: v for t, v in project_tag_corners(cam, fm).items()
            if (_quat_to_matrix(fm.tag_quaternion(t)) @ [1.0, 0.0, 0.0])
            @ (c - fm.tag_translation(t)) > 0}
    assert len(corr) >= 6
    body = {"event": "2025hop", "camera": "main", "image_size": list(cam.image_size),
            "auto_tags": {str(t): corr[t][0].tolist() for t in corr}, "manual_tags": []}
    r = _client(app_mod).post("/api/calibration/solve-tags", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["rms_error"] < 1.0
    cal = field_store.load_versions("2025hop", "main")[0]["calibration"]
    assert cal.field_season == "2025-reefscape-welded"


def test_solve_tags_refuses_a_field_without_tags(app_mod):
    r = _client(app_mod).post("/api/calibration/solve-tags", json={
        "event": "2019txho", "camera": "main", "image_size": [1920, 1080],
        "auto_tags": {}, "manual_tags": []})
    assert r.status_code == 400
    assert "no AprilTag layout for 2019" in r.json()["error"]


def test_masks_default_calibration_carries_the_event_season(app_mod):
    r = _client(app_mod).post("/api/calibration/masks",
                              json={"event": "2025hop", "masks": [[1, 2, 3, 4]]})
    assert r.status_code == 200
    cal = field_store.load_versions("2025hop", "main")[0]["calibration"]
    assert cal.field_season == "2025-reefscape-welded"


# --------------------------------------------------------------------- tag map

def test_tag_map_follows_the_event(app_mod):
    c = _client(app_mod)
    j = c.get("/api/calibration/tag-map", params={"event": "2025hop"}).json()
    assert j["season"] == "2025-reefscape-welded" and len(j["tags"]) == 22
    assert j["field"]["length"] == pytest.approx(17.548)
    assert ["17", "18", "19", "20", "21", "22"] in j["marker_groups"]
    d = c.get("/api/calibration/tag-map").json()
    assert d["season"] == "2026-rebuilt-welded" and len(d["tags"]) == 32
    assert d["season_info"]["source"] == "default"
    s = c.get("/api/calibration/tag-map", params={"season": "2024-crescendo"}).json()
    assert len(s["tags"]) == 16 and s["marker_groups"] == []
    r = c.get("/api/calibration/tag-map", params={"event": "2019txho"})
    assert r.status_code == 400 and "2019" in r.json()["error"]


def test_tag_map_overlay_legend_follows_the_season(app_mod):
    """The calibrate page's overlay legend lists the colours hud_draw draws on the
    event's field: 2025 GLB groups, 2026 hand-coded structures."""
    from web import hud_draw
    c = _client(app_mod)
    j = c.get("/api/calibration/tag-map", params={"event": "2025hop"}).json()
    leg = {d["label"]: d["color"] for d in j["wireframe_legend"]}
    for g in ("reef", "barge", "coral_station", "processor", "perimeter", "tape"):
        b, gg, r = hud_draw.WIREFRAME_GROUP_COLORS[g]
        assert leg[g.replace("_", " ")] == f"#{r:02x}{gg:02x}{b:02x}"
    assert "hubs" not in leg and "carpet grid" in leg
    d = c.get("/api/calibration/tag-map").json()
    assert [x["label"] for x in d["wireframe_legend"]] == [
        "carpet grid", "hubs", "bumps", "towers", "trenches"]


def test_field_season_override_endpoints(app_mod):
    c = _client(app_mod)
    j = c.get("/api/event/field-season", params={"event": "2025hop"}).json()
    assert j["season"]["variant"] == "welded" and j["override"] is None
    j = c.post("/api/event/field-season",
               json={"event": "2025hop", "variant": "andymark"}).json()
    assert j["season"]["tag_layout"] == "2025-reefscape-andymark"
    assert j["override"]["variant"] == "andymark"
    assert c.get("/api/calibration/tag-map",
                 params={"event": "2025hop"}).json()["season"] == "2025-reefscape-andymark"
    assert c.post("/api/event/field-season",
                  json={"event": "2025hop", "year": 2019}).status_code == 400
    j = c.post("/api/event/field-season", json={"event": "2025hop", "clear": True}).json()
    assert j["override"] is None and j["season"]["tag_layout"] == "2025-reefscape-welded"


def test_event_current_reports_field_season(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "current_event", {
        "event_key": "2025hop", "event_name": "Hopper", "matches": [], "teams": []})
    j = _client(app_mod).get("/api/event/current").json()
    assert j["field_season"]["year"] == 2025
    assert j["field_season"]["tag_layout"] == "2025-reefscape-welded"


# ------------------------------------------------------------------ match clock

def _countdown(start_frame, start_seconds, fps=10, first=None):
    first = start_seconds if first is None else first
    return [(start_frame + (start_seconds - v) * fps, v) for v in range(first, -1, -1)]


def test_clock_search_uses_the_2025_15_135_countdowns(app_mod):
    fps = 10
    # 2025: auto 0:15, teleop 2:15 whose first 8 s are unreadable (starts at 127)
    samples = _countdown(100, 15, fps) + [(300, None)] + _countdown(400, 135, fps, first=127)
    ranges, periods, fell_back, reason = app_mod.compute_active_windows(
        samples, fps, 2200, auto_s=15, teleop_s=135)
    assert not fell_back and reason is None
    assert [p["label"] for p in periods] == ["auto", "teleop"]
    assert periods[0]["start_frame"] == 100 and periods[0]["start_seconds"] == 15
    assert periods[1]["start_seconds"] == 127
    # the 2026 targets (20 / 140) miss that teleop
    _, _, fell_back26, reason26 = app_mod.compute_active_windows(samples, fps, 2200)
    assert fell_back26 and reason26 == "teleop countdown was not found"


def test_clock_search_2025_auto_seen_late(app_mod):
    # auto clock first legible at 0:11: a 15 s auto still counts (>= 9 s run),
    # the 2026 20 s search wants >= 12 s of countdown
    fps = 10
    samples = _countdown(100, 15, fps, first=11) + _countdown(400, 135, fps)
    _, periods, fell_back, _ = app_mod.compute_active_windows(
        samples, fps, 2200, auto_s=15, teleop_s=135)
    assert not fell_back and periods[0]["label"] == "auto"
    assert periods[0]["start_frame"] == 140 and periods[0]["start_seconds"] == 11
    assert periods[1]["label"] == "teleop" and periods[1]["start_frame"] == 400
    # the 2026 search skips the short auto (it then mistakes the end of teleop for
    # auto and never finds a teleop)
    _, _, fell_back26, _ = app_mod.compute_active_windows(samples, fps, 2200)
    assert fell_back26


# -------------------------------------------------------------------- GT scoring

def test_gt_auto_window_and_field_length_follow_the_match_season():
    assert gt_scoring.auto_seconds_for("2025hop_qm1") == 15.0
    assert gt_scoring.auto_seconds_for("2026cur_qm8") == 20.0
    assert gt_scoring.auto_seconds_for(None) == gt_scoring.AUTO_SECONDS == 20.0
    assert gt_scoring.field_length_for("2025hop_qm1") == pytest.approx(17.548)
    assert gt_scoring.field_length_for("2026cur_qm8") == pytest.approx(16.541)
    assert gt_scoring.field_length_for("2019txho_qm1") == pytest.approx(16.541)
