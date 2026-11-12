"""Mid-match camera-move detection (web.field.camera_motion) and what it changes:
the per-match corner step drops a corner that moved during its match, and event-level
auto-calibration moves a camera to a later, steady match. No video decoding of real
footage, no network."""
import numpy as np
import pytest

from web.field import auto_calibrate as ac
from web.field import camera_motion as cm
from web.field import corner_per_match as cpm
from web.field.field_model import FieldModel
from tests.field.test_corner_per_match import BASE, _det, _epoch, _video

FM = FieldModel.load("2026-rebuilt-welded")
RNG = np.random.default_rng(7)


def _track(t, base, noise=0.3, step_at=None, step=(0.0, 0.0)):
    rows = []
    for ti in t:
        p = np.asarray(base, float) + RNG.normal(0, noise, 2)
        if step_at is not None and ti > step_at:
            p = p + step
        rows.append((ti, p[0], p[1]))
    return rows


T = np.arange(18, 180, 3.0)


def test_steady_camera_is_not_a_move():
    tracks = {25: _track(T, (300, 120)), 26: _track(T, (340, 121)), 27: _track(T, (420, 118))}
    assert not cm.analyze_tracks(tracks)["moved"]


def test_occlusion_jitter_is_not_a_move():
    # partial occlusion: a few px of scatter on one tag, others steady
    tracks = {25: _track(T, (300, 120), noise=2.5), 26: _track(T, (340, 121)),
              27: _track(T, (420, 118))}
    assert not cm.analyze_tracks(tracks)["moved"]


def test_small_nudge_below_threshold_is_not_a_move():
    tracks = {8: _track(T, (100, 90), step_at=125, step=(2.0, 0.5)),
              9: _track(T, (140, 92), step_at=125, step=(2.0, 0.5))}
    assert not cm.analyze_tracks(tracks)["moved"]


def test_common_step_is_a_move_at_the_right_time():
    tracks = {k: _track(T, b, step_at=90, step=(14.0, -9.0))
              for k, b in {25: (300, 120), 26: (340, 121), 27: (420, 118)}.items()}
    r = cm.analyze_tracks(tracks)
    assert r["moved"] and abs(r["t_s"] - 90) <= 3 and r["shift_px"] > 14
    assert sorted(r["stepping"]) == [25, 26, 27]


def test_one_tag_alone_must_step_further():
    small = {25: _track(T, (300, 120), step_at=90, step=(10.0, 0.0))}
    big = {25: _track(T, (300, 120), step_at=90, step=(16.0, 0.0))}
    assert not cm.analyze_tracks(small)["moved"]
    assert cm.analyze_tracks(big)["moved"]


def test_tags_that_disagree_are_not_a_move():
    # one tag "jumps" (a misdetection pattern), two two-sided tags hold still
    tracks = {25: _track(T, (300, 120), step_at=90, step=(20.0, 0.0)),
              26: _track(T, (340, 121)), 27: _track(T, (420, 118))}
    assert not cm.analyze_tracks(tracks)["moved"]


def test_corner_step_drops_a_corner_that_moved(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "detect_camera_move", lambda *a, **k: {
        "moved": True, "t_s": 121.0, "shift_px": 18.0, "stepping": [9, 10],
        "two_sided": [9, 10], "per_tag": {}, "samples": 50})
    monkeypatch.setattr(cpm, "check_epoch", lambda *a, **k: pytest.fail("must not check"))
    monkeypatch.setattr(ac, "run_camera", lambda *a, **k: pytest.fail("must not re-solve"))
    out = cpm.corner_step(_video(tmp_path), "evt_qm5", "corner_right", (0, 0, 64, 36), (),
                          FM, _epoch(), (0, 29), 1.0, detect_fn=_det(BASE),
                          roi_detect_fn=lambda *a, **k: {})
    assert out.status == "dropped" and out.calibration is None
    assert out.metrics["moved_mid_match"]["t_s"] == 121.0
    assert "moved mid-match" in out.reason


def test_corner_step_ignores_motion_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "detect_camera_move", lambda *a, **k: pytest.fail("disabled"))
    out = cpm.corner_step(_video(tmp_path), "evt_qm5", "corner_left", (0, 0, 64, 36), (),
                          FM, _epoch(), (0, 29), 1.0,
                          cpm.CornerStepConfig(detect_motion=False), detect_fn=_det(BASE),
                          roi_detect_fn=lambda *a, **k: {})
    assert out.status == "kept"


def test_later_reference_videos_local_first_then_download(tmp_path):
    for k in ("ev_qm1", "ev_qm3"):
        (tmp_path / f"{k}.mp4").write_bytes(b"x")
    matches = [{"key": f"ev_qm{i}", "video_key": f"vk{i}"} for i in range(1, 6)]
    got = []

    def dl(vk, out):
        got.append(vk)
        return out

    out = list(ac.later_reference_videos("ev", matches, "ev_qm1", dirs=(tmp_path,),
                                         download_dir=tmp_path, downloader=dl, limit=3))
    assert [k for k, _ in out] == ["ev_qm3", "ev_qm2", "ev_qm4"]   # local first
    assert got == ["vk2", "vk4"]                                    # downloads only as reached


def test_event_level_moves_to_a_later_steady_match(monkeypatch):
    import web.app as app
    calls = []

    def fake_move(path, crop, rng, detect):
        calls.append(str(path))
        moved = str(path).endswith("qm1.mp4")
        return {"moved": moved, "t_s": 60.0, "shift_px": 20.0, "stepping": [1, 2],
                "two_sided": [1, 2], "per_tag": {}, "samples": 40}

    monkeypatch.setattr(cm, "detect_camera_move", fake_move)
    monkeypatch.setattr(ac, "later_reference_videos",
                        lambda *a, **k: iter([("ev_qm2", "/x/ev_qm2.mp4")]))
    monkeypatch.setattr(ac, "make_detect_fn", lambda: None)

    class CL:
        crop = (0, 0, 10, 10)

    mk, path, notes, endgame = app._auto_calib_pick_steady(
        "ev", "corner_left", "ev_qm1", "/x/ev_qm1.mp4", CL(), ac.AutoCalibConfig())
    assert (mk, path) == ("ev_qm2", "/x/ev_qm2.mp4")
    assert endgame is None
    assert notes == [{"match": "ev_qm1", "t_s": 60.0, "shift_px": 20.0}]
    assert calls == ["/x/ev_qm1.mp4", "/x/ev_qm2.mp4"]
