import json
import shutil
import subprocess
import numpy as np
import cv2
import pytest

from web.replay_recorder import ReplayRecorder


def _frame(w=320, h=240, val=50):
    return np.full((h, w, 3), val, np.uint8)


def test_records_playable_video_and_meta(tmp_path):
    rec = ReplayRecorder()
    rec.open_lazy(_frame(), fps=8.0, out_dir=tmp_path)
    for i in range(10):
        rec.add_frame(_frame(val=10 + i), {"scores": {"blue": i, "red": 0}, "robots": []})
    artifacts = rec.finalize(status="complete", summary={"match_label": "Test"})

    cap = cv2.VideoCapture(str(tmp_path / "hud.mp4"))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    cap.release()

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["summary"]["frame_count"] == 10
    assert meta["summary"]["fps"] == 8.0
    assert len(meta["frames"]) == 10
    assert meta["frames"][0]["i"] == 0 and meta["frames"][0]["t"] == 0.0
    assert meta["frames"][8]["t"] == pytest.approx(1.0)  # 8 / 8fps

    assert (tmp_path / "thumb.jpg").exists()
    assert artifacts["frame_count"] == 10
    assert artifacts["duration_sec"] == pytest.approx(10 / 8.0)
    assert artifacts["codec"] in ("h264", "avc1", "mp4v")


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                    reason="ffmpeg/ffprobe not available")
def test_output_is_browser_playable_h264(tmp_path):
    """With ffmpeg present, the mp4 must be H.264 so HTML5 <video> can decode it
    (regression: OpenCV's mp4v fallback produced un-playable MPEG-4 Part 2)."""
    rec = ReplayRecorder()
    rec.open_lazy(_frame(w=321, h=241), fps=8.0, out_dir=tmp_path)  # odd dims -> must be locked even
    for i in range(8):
        rec.add_frame(_frame(w=321, h=241, val=20 + i), {"scores": {"blue": 0, "red": 0}, "robots": []})
    artifacts = rec.finalize(status="complete", summary={})

    assert artifacts["codec"] == "h264"
    assert artifacts["width"] % 2 == 0 and artifacts["height"] % 2 == 0  # even for yuv420p
    codec_name = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0",
         str(tmp_path / "hud.mp4")],
        capture_output=True, text=True).stdout.strip()
    assert codec_name == "h264"


def test_add_frame_before_open_raises(tmp_path):
    rec = ReplayRecorder()
    with pytest.raises(RuntimeError):
        rec.add_frame(_frame(), {})


def test_event_times_use_recorded_index_not_video_timestamp(tmp_path):
    rec = ReplayRecorder()
    rec.open_lazy(_frame(), fps=8.0, out_dir=tmp_path)
    # Source video is 60fps, frame_skip 6 -> recorded frame i corresponds to frame_num=180+6*i
    for i in range(20):
        rec.add_frame(_frame(), {"frame_num": 180 + 6 * i, "scores": {"blue": 0, "red": 0}, "robots": []})
    # An event at source frame_num 240 -> recorded index (240-180)/6 = 10 -> t = 10/8 = 1.25s
    rec.add_event({"frame_num": 240, "alliance": "blue", "slot": "B2", "robot_label": "B2 9785", "confidence": "high"})
    rec.finalize(status="complete", summary={})

    meta = json.loads((tmp_path / "meta.json").read_text())
    ev = meta["events"][0]
    assert ev["t"] == pytest.approx(1.25)  # recorded-index based, NOT 240/60=4.0s


def test_partial_finalize_produces_valid_artifacts(tmp_path):
    rec = ReplayRecorder()
    rec.open_lazy(_frame(), fps=8.0, out_dir=tmp_path)
    rec.add_frame(_frame(), {"frame_num": 180, "scores": {"blue": 0, "red": 0}, "robots": []})
    artifacts = rec.finalize(status="partial", summary={})
    assert artifacts["status"] == "partial"
    cap = cv2.VideoCapture(str(tmp_path / "hud.mp4"))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 1
    cap.release()


def test_summary_preserves_timer_periods_and_autonomous_paths(tmp_path):
    rec = ReplayRecorder()
    rec.open_lazy(_frame(), fps=8.0, out_dir=tmp_path)
    rec.add_frame(_frame(), {"frame_num": 100, "scores": {}, "robots": []})
    summary = {
        "periods": [{"label": "auto", "start_frame": 100, "end_frame": 300}],
        "processing_ranges": [(95, 330)],
        "timer_gate_fallback": False,
        "autonomous_paths": {
            "B1": [
                {
                    "frame_num": 100,
                    "timestamp": 5.0,
                    "field_pos": [1.25, 2.5],
                }
            ]
        },
    }
    rec.finalize(status="complete", summary=summary)

    saved = json.loads((tmp_path / "meta.json").read_text())["summary"]
    assert saved["periods"][0]["label"] == "auto"
    assert saved["processing_ranges"] == [[95, 330]]
    assert saved["autonomous_paths"]["B1"][0]["field_pos"] == [1.25, 2.5]
