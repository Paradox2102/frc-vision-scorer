"""Draw list (HUD post-render spec sections 3-4): what InferenceJob._live_hud
records per frame, the metadata-only recorder that stores it, and hud_static."""
import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

import web.app as app_mod
from web import hud_draw
from web.replay_recorder import ReplayRecorder
from tests.test_hud_golden import DEFAULT_FLAGS, golden, scene


@pytest.fixture(autouse=True)
def _default_flags(monkeypatch):
    for k, v in DEFAULT_FLAGS.items():
        monkeypatch.setattr(app_mod, k, v)


def _line(job=None, kw=None):
    if job is None:
        job, kw = scene()
    line = {}
    img = job._live_hud(**kw, draw_line=line)
    return img, line


def test_recording_does_not_change_the_pixels():
    img, _ = _line()
    assert np.array_equal(img, golden())


def test_line_holds_what_was_drawn():
    job, kw = scene()
    _, line = _line(job, kw)
    assert line["frame_num"] == kw["frame_num"]
    assert line["main_hash"] == hud_draw.ahash(kw["overhead_frame"])
    main = {e["slot"]: e for e in line["main"]}
    assert set(main) == {"B1", "R2", "B3"}
    assert main["B1"]["box"] == [100.0, 80.0, 160.0, 150.0] and main["B1"]["score"] == 2
    assert main["B1"]["slot_id"] == 0 and main["B1"]["alliance"] == "blue"
    assert main["B1"]["ghost"] is False and len(main["B1"]["mask_poly"]) >= 3
    assert main["R2"]["mask_poly"] is None and main["R2"]["score"] == 1
    assert main["B3"]["ghost"] is True and main["B3"]["box"] == [250.0, 60.0, 300.0, 110.0]
    assert line["hubs"] == [{"alliance": "blue", "x": 200.0, "y": 120.0},
                            {"alliance": "red", "x": 440.0, "y": 60.0}]
    assert line["hub_boxes"] == [[182.0, 102.0, 218.0, 138.0, 0.9]]
    assert "ball_boxes" not in line
    assert line["corners"]["corner_left"] == [
        {"slot": "B1", "alliance": "blue", "box": [40.0, 30.0, 100.0, 90.0], "score": 2,
         "live_text": "3128 S2"}]
    assert line["corners"]["corner_right"] == [
        {"slot": None, "alliance": "red", "box": [200.0, 40.0, 260.0, 100.0], "score": None,
         "live_text": "red bumper no #"}]
    radar = {e["slot"]: e for e in line["radar"]}
    assert set(radar) == {"B1", "R2", "B3", "R1"}
    assert radar["B1"]["trail"] == [[2.0, 2.0], [2.5, 2.0], [3.0, 2.0]]
    assert radar["R1"]["corner_only"] is True and radar["B3"]["state"] == "coasting"
    json.dumps(line)                                    # JSON-ready


def test_hubs_are_recorded_per_frame():
    job, kw = scene()
    _, first = _line(job, kw)
    job.hubs[0].x, job.hubs[0].y = 210.0, 118.0          # the EMA moved it
    _, second = _line(job, kw)
    assert first["hubs"][0] == {"alliance": "blue", "x": 200.0, "y": 120.0}
    assert second["hubs"][0] == {"alliance": "blue", "x": 210.0, "y": 118.0}


def test_ball_boxes_only_with_the_flag(monkeypatch):
    monkeypatch.setattr(app_mod, "HUD_DRAW_BALL_BOXES", True)
    _, line = _line()
    assert line["ball_boxes"] == [[300.0, 150.0, 310.0, 160.0, 0.5]]


def test_metadata_only_recorder_spawns_no_encoder(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("ffmpeg spawned"))
    rec = ReplayRecorder(record_video=False)
    frame = np.full((241, 321, 3), 40, np.uint8)
    rec.open_lazy(frame, fps=8.0, out_dir=tmp_path)
    assert rec.is_open
    for i in range(3):
        rec.add_frame(frame, {"frame_num": 10 + i, "scores": {}, "robots": []},
                      draw={"frame_num": 10 + i, "main": []})
    art = rec.finalize(status="complete", summary={"match_label": "x"})
    assert not (tmp_path / "hud.mp4").exists() and (tmp_path / "thumb.jpg").exists()
    assert art["codec"] is None and (art["width"], art["height"]) == (320, 240)
    assert art["video_path"] == str(tmp_path / "hud.mp4") and art["frame_count"] == 3
    with gzip.open(tmp_path / "draw.jsonl.gz", "rt") as fh:
        lines = [json.loads(l) for l in fh]
    assert [(l["i"], l["frame_num"]) for l in lines] == [(0, 10), (1, 11), (2, 12)]


def test_metadata_only_meta_json_equals_the_live_encoded_one(tmp_path):
    out = {}
    for name, video in (("live", True), ("meta", False)):
        rec = ReplayRecorder(record_video=video)
        d = tmp_path / name
        frame = np.full((120, 160, 3), 40, np.uint8)
        rec.open_lazy(frame, fps=8.0, out_dir=d)
        for i in range(4):
            rec.add_frame(frame, {"frame_num": 10 + i, "scores": {"blue": i}, "robots": []},
                          draw={"main": []} if not video else None)
        rec.add_event({"frame_num": 12, "alliance": "blue"})
        rec.finalize(status="complete", summary={"match_label": "x"})
        out[name] = json.loads((d / "meta.json").read_text())
    assert out["live"] == out["meta"]
    assert not (tmp_path / "live" / "draw.jsonl.gz").exists()


def test_new_replay_recorder_follows_the_knob():
    job = app_mod.InferenceJob.__new__(app_mod.InferenceJob)
    job.config = SimpleNamespace(hud_post_render=False)
    rec = job._new_replay_recorder()
    assert rec.record_video is True and job._replay_draw_list is False
    job.config = SimpleNamespace(hud_post_render=True)
    rec = job._new_replay_recorder()
    assert rec.record_video is False and job._replay_draw_list is True


def test_hud_static_is_json_ready():
    job, _kw = scene()
    job.overhead_height = 240
    job.config = SimpleNamespace(field_wireframe_overlay=True)   # no calib -> off
    job._radar_reference_paths = [{"alliance": "blue", "team_number": "3128",
                                   "path_m": [[2.0, 2.0], [3.0, 2.5]]}]
    job.event_key, job.camera, job._calib_version_id = "2026cur", "main", "v1"
    st = json.loads(json.dumps(job._hud_static()))
    assert st["flags"] == hud_draw.flags_to_dict(hud_draw.HudFlags())
    assert st["ocr_regions"]["blue_score"] == [20, 10, 60, 20]
    assert st["field_wireframe_overlay"] is False and st["overhead_height"] == 240
    assert [v["name"] for v in st["views"]] == ["corner_left", "corner_right"]
    assert st["views"][1] == {"name": "corner_right", "crop": [320, 232, 320, 128],
                              "masks": [[250, 0, 70, 20]]}
    assert st["panel_max_width"] == 720 and st["reference_paths"][0]["team_number"] == "3128"
    assert (st["event_key"], st["camera"], st["calibration_version"]) == ("2026cur", "main", "v1")
