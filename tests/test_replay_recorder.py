import json
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
    assert artifacts["codec"] in ("avc1", "mp4v")


def test_add_frame_before_open_raises(tmp_path):
    rec = ReplayRecorder()
    with pytest.raises(RuntimeError):
        rec.add_frame(_frame(), {})
