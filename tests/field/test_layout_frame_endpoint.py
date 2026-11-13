"""GET /api/calibration/layout/frame: full-res composite frame for drawing the layout."""
import importlib

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from web.field import auto_calibrate as ac
from web.field import composite_layout as cl
from web.field import store as field_store

HIHO = {"frame_size": [1920, 1080],
        "cameras": {"main": {"crop": [0, 0, 1920, 460], "masks": []}},
        "confidence": {"main": 0.8}, "diag": {}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path / "up")
    (tmp_path / "up").mkdir()
    mod = importlib.import_module("web.app")
    monkeypatch.setattr(mod, "_auto_calib_tasks", {})
    return TestClient(mod.app)


def _write_video(path, n=60, fps=10, size=(320, 180)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), (i * 4) % 256, np.uint8))
    w.release()


def test_frame_from_footage_clamps_and_reports_duration(client, monkeypatch, tmp_path):
    foot = tmp_path / "foot"
    foot.mkdir()
    _write_video(foot / "evtF_qm1.mp4")
    monkeypatch.setattr(ac, "FOOTAGE_DIRS", (foot,))
    r = client.get("/api/calibration/layout/frame", params={"event": "evtF", "t": 2})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert abs(float(r.headers["X-Video-Duration"]) - 6.0) < 0.2
    assert abs(float(r.headers["X-Video-Time"]) - 2.0) < 0.01
    r = client.get("/api/calibration/layout/frame", params={"event": "evtF", "t": 999})
    assert r.status_code == 200 and float(r.headers["X-Video-Time"]) < 6.0


def test_record_video_outside_footage_dirs_is_ignored(client, monkeypatch, tmp_path):
    foot = tmp_path / "foot"
    foot.mkdir()
    outside = tmp_path / "secret.mp4"
    _write_video(outside)
    monkeypatch.setattr(ac, "FOOTAGE_DIRS", (foot,))
    monkeypatch.setattr(ac, "DOWNLOAD_DIR", foot)
    cl.save_event_layout("evtO", dict(HIHO, diag={"video": str(outside)}))
    r = client.get("/api/calibration/layout/frame", params={"event": "evtO"})
    assert r.status_code == 404


def test_falls_back_to_saved_median_then_404(client, monkeypatch, tmp_path):
    foot = tmp_path / "empty"
    foot.mkdir()
    monkeypatch.setattr(ac, "FOOTAGE_DIRS", (foot,))
    monkeypatch.setattr(ac, "DOWNLOAD_DIR", foot)
    assert client.get("/api/calibration/layout/frame", params={"event": "evtM"}).status_code == 404
    ac.ensure_event_layout("evtM", "v.mp4", log=lambda m: None,
                           detector=lambda v: (HIHO, np.full((1080, 1920, 3), 90, np.uint8)))
    r = client.get("/api/calibration/layout/frame", params={"event": "evtM"})
    assert r.status_code == 200 and r.headers["X-Frame-Source"] == "saved"
    assert client.get("/api/calibration/layout/frame", params={"event": "a/b"}).status_code == 400
