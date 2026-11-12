"""Seasons whose broadcast re-tiles the camera row at the endgame (2025) calibrate on
the pre-endgame framing: the aggregation / motion / validation window ends before
the endgame starts. The cutoff comes from the match clock, else a detected camera
move, else a conservative estimate. 2026 is untouched. No real footage."""
import pytest

from web.field import auto_calibrate as ac
from web.field.season import season_for_year

S25 = season_for_year(2025)
S26 = season_for_year(2026)
FPS = 30.0


def _clock(auto_start_t=9.0, pause_s=3.0, step_s=0.5, end_t=190.0, glitch=()):
    """Synthetic (frame, value) clock samples: pre-match None, auto 15 -> 0,
    pause, teleop 135 -> 0. ``glitch``: {t: value} misreads."""
    teleop_start = auto_start_t + S25.auto_s + pause_s
    out, t = [], 0.0
    glitch = dict(glitch)
    while t < end_t:
        if t < auto_start_t:
            v = None
        elif t < auto_start_t + S25.auto_s:
            v = int(S25.auto_s - (t - auto_start_t))
        elif t < teleop_start:
            v = 0
        elif t < teleop_start + S25.teleop_s:
            v = int(S25.teleop_s - (t - teleop_start))
        else:
            v = 0
        v = glitch.get(round(t, 1), v)
        out.append((int(round(t * FPS)), v))
        t += step_s
    return out, teleop_start + S25.teleop_s - S25.endgame_s


def _periods(samples, season=S25):
    import web.app as app
    _, periods, _, _ = app.compute_active_windows(samples, FPS, int(200 * FPS),
                                                  auto_s=season.auto_s,
                                                  teleop_s=season.teleop_s)
    return periods


def test_season_flag():
    assert S25.endgame_layout_switch is True
    assert S26.endgame_layout_switch is False
    assert season_for_year(2024).endgame_layout_switch is False
    assert season_for_year(2019).endgame_layout_switch is False
    assert S25.public()["endgame_layout_switch"] is True


def test_endgame_start_from_the_teleop_clock():
    samples, truth = _clock()
    got = ac.endgame_start_from_clock(samples, FPS, _periods(samples), S25)
    assert got["method"] == "match_clock"
    assert got["endgame_t_s"] == pytest.approx(truth, abs=1.0)


def test_endgame_start_ignores_isolated_misreads():
    samples, truth = _clock(glitch={100.0: 3, 120.0: 99, 135.5: 7})
    got = ac.endgame_start_from_clock(samples, FPS, _periods(samples), S25)
    assert got["endgame_t_s"] == pytest.approx(truth, abs=1.0)


def test_endgame_start_from_auto_only_is_a_lower_bound():
    samples, truth = _clock()
    auto_only = [p for p in _periods(samples) if p["label"] == "auto"]
    got = ac.endgame_start_from_clock(samples, FPS, auto_only, S25)
    assert got["method"] == "match_clock_auto"
    assert truth - 5.0 <= got["endgame_t_s"] <= truth + 0.5


def test_endgame_start_none_without_a_clock():
    assert ac.endgame_start_from_clock([], FPS, [], S25) is None
    assert ac.endgame_start_from_clock([(0, None)] * 50, FPS, [], S25) is None


def test_cutoff_prefers_the_clock():
    c = ac.endgame_cutoff(S25, (18.0, 1e9), clock={"endgame_t_s": 141.0,
                                                    "method": "match_clock"},
                          move_t=137.0)
    assert c["method"] == "match_clock"
    assert c["cutoff_s"] == pytest.approx(141.0 - ac.ENDGAME_CLOCK_MARGIN_S)
    assert c["endgame_t_s"] == pytest.approx(141.0)


def test_cutoff_falls_back_to_a_detected_move_then_the_estimate():
    est = S25.auto_s + S25.teleop_s - S25.endgame_s - ac.ENDGAME_CLOCK_MARGIN_S
    c = ac.endgame_cutoff(S25, (18.0, 1e9), clock=None, move_t=100.0)
    assert c["method"] == "camera_move"
    assert c["cutoff_s"] == pytest.approx(100.0 - ac.ENDGAME_MOVE_MARGIN_S)
    # a move reported AFTER the conservative estimate (sparse tags put the split
    # late: 2025hop qm2 corner_right said 161 s, the switch was at 145.5 s) never
    # pushes the cutoff past the estimate
    c = ac.endgame_cutoff(S25, (18.0, 1e9), clock=None, move_t=161.0)
    assert c["method"] == "estimate" and c["cutoff_s"] == pytest.approx(est)
    c = ac.endgame_cutoff(S25, (18.0, 1e9))
    assert c["method"] == "estimate" and c["cutoff_s"] == pytest.approx(est)


def test_cutoff_ignores_a_clock_that_leaves_no_window():
    c = ac.endgame_cutoff(S25, (18.0, 1e9), clock={"endgame_t_s": 30.0,
                                                    "method": "match_clock"})
    assert c["method"] == "estimate"


def test_cutoff_never_extends_a_shorter_window():
    c = ac.endgame_cutoff(S25, (18.0, 90.0), clock={"endgame_t_s": 141.0,
                                                     "method": "match_clock"})
    assert c["cutoff_s"] == pytest.approx(90.0)


# ----------------------------------------------------------- app: pick a steady match

class _CL:
    crop = (0, 0, 10, 10)


def _app(monkeypatch, moves, clock=None):
    """web.app with detect_camera_move answering from ``moves`` ({path: t or None}
    for a move at t on any range that contains it) and a fake clock."""
    import web.app as app
    from web.field import camera_motion as cm
    calls = []

    def fake_move(path, crop, rng, detect):
        calls.append((str(path), tuple(rng)))
        t = moves.get(str(path))
        moved = t is not None and rng[0] < t < (rng[1] if rng[1] is not None else 1e9)
        return {"moved": moved, "t_s": t if moved else None, "shift_px": 150.0 if moved else 0,
                "stepping": [1, 2] if moved else [], "two_sided": [1, 2], "per_tag": {},
                "samples": 40}

    monkeypatch.setattr(cm, "detect_camera_move", fake_move)
    monkeypatch.setattr(ac, "later_reference_videos",
                        lambda *a, **k: iter([("ev_qm2", "/x/ev_qm2.mp4")]))
    monkeypatch.setattr(ac, "make_detect_fn", lambda: None)
    clock_calls = []

    def fake_clock(path, season):
        clock_calls.append(str(path))
        return clock(path) if callable(clock) else clock
    monkeypatch.setattr(app, "_auto_calib_endgame_clock", fake_clock)
    return app, calls, clock_calls


def test_2025_calibrates_on_the_teleop_framing_of_the_same_match(monkeypatch):
    app, calls, clock_calls = _app(
        monkeypatch, {"/x/ev_qm1.mp4": 142.0},
        clock={"endgame_t_s": 140.8, "method": "match_clock"})
    mk, path, notes, eg = app._auto_calib_pick_steady(
        "2025ev", "corner_left", "ev_qm1", "/x/ev_qm1.mp4", _CL(), ac.AutoCalibConfig(),
        season=S25)
    assert (mk, path, notes) == ("ev_qm1", "/x/ev_qm1.mp4", [])
    assert eg["method"] == "match_clock" and eg["match"] == "ev_qm1"
    assert eg["cutoff_s"] == pytest.approx(140.8 - ac.ENDGAME_CLOCK_MARGIN_S)
    assert calls == [("/x/ev_qm1.mp4", (18.0, eg["cutoff_s"]))]   # clamped motion check
    assert clock_calls == ["/x/ev_qm1.mp4"]


def test_2025_without_a_clock_uses_the_move_and_rechecks_the_clamped_window(monkeypatch):
    app, calls, _ = _app(monkeypatch, {"/x/ev_qm1.mp4": 110.0}, clock=None)
    mk, path, notes, eg = app._auto_calib_pick_steady(
        "2025ev", "corner_left", "ev_qm1", "/x/ev_qm1.mp4", _CL(), ac.AutoCalibConfig(),
        season=S25)
    assert mk == "ev_qm1" and notes == []
    assert eg["method"] == "camera_move"
    assert eg["cutoff_s"] == pytest.approx(110.0 - ac.ENDGAME_MOVE_MARGIN_S)
    assert calls == [("/x/ev_qm1.mp4", (18.0, 1e9)), ("/x/ev_qm1.mp4", (18.0, eg["cutoff_s"]))]


def test_2025_a_real_move_inside_teleop_still_rejects_the_match(monkeypatch):
    app, calls, _ = _app(monkeypatch, {"/x/ev_qm1.mp4": 60.0},
                         clock={"endgame_t_s": 140.8, "method": "match_clock"})
    mk, path, notes, eg = app._auto_calib_pick_steady(
        "2025ev", "corner_left", "ev_qm1", "/x/ev_qm1.mp4", _CL(), ac.AutoCalibConfig(),
        season=S25)
    assert mk == "ev_qm2"
    assert notes == [{"match": "ev_qm1", "t_s": 60.0, "shift_px": 150.0}]
    assert eg["match"] == "ev_qm2"


def test_2026_is_unchanged_no_clock_no_clamp_moves_reject(monkeypatch):
    app, calls, clock_calls = _app(monkeypatch, {"/x/ev_qm1.mp4": 142.0},
                                   clock={"endgame_t_s": 140.8, "method": "match_clock"})
    mk, path, notes, eg = app._auto_calib_pick_steady(
        "2026ev", "corner_left", "ev_qm1", "/x/ev_qm1.mp4", _CL(), ac.AutoCalibConfig(),
        season=S26)
    assert (mk, path) == ("ev_qm2", "/x/ev_qm2.mp4")
    assert notes == [{"match": "ev_qm1", "t_s": 142.0, "shift_px": 150.0}]
    assert eg is None and clock_calls == []
    assert calls == [("/x/ev_qm1.mp4", (18.0, 1e9)), ("/x/ev_qm2.mp4", (18.0, 1e9))]


def test_auto_calibrate_camera_clamps_the_window_and_records_the_cutoff(monkeypatch, tmp_path):
    import web.app as app
    from web.field import store as field_store
    from web.field.field_model import FieldModel
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(app, "_auto_calib_judge", lambda: None)
    eg = {"cutoff_s": 137.8, "method": "match_clock", "endgame_t_s": 140.8,
          "match": "2025ev_qm1"}
    seen = {}
    monkeypatch.setattr(app, "_auto_calib_pick_steady",
                        lambda event, camera, mk, video, cl, cfg, matches=(), season=None:
                        (seen.setdefault("season", season) and mk, video, [], eg))

    def fake_run(video, mk, cam, crop, fm, versions, cfg=None, **kw):
        seen["window"] = cfg.window
        return ac.AutoCalibResult(record=ac.new_record("ready", "solved", method="full_solve"),
                                  calibration=None, overlay=None)
    monkeypatch.setattr(ac, "run_camera", fake_run)

    class CL:
        crop = (0, 733, 965, 347)
        masks = []
    rec = app._auto_calibrate_camera("2025ev", "corner_left", "2025ev_qm1", "/x/v.mp4", CL(),
                                     FieldModel.load("2025-reefscape-welded"))
    assert seen["season"].year == 2025
    assert seen["window"] == (18.0, 137.8)
    assert rec["endgame_cutoff"] == eg
    assert rec["status"] == "ready"
    assert "endgame" in rec["reason"] and "137.8" in rec["reason"]
