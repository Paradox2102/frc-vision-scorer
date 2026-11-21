"""Per-match video timeline for the scouting export (web/scouting/timeline.py)."""
from __future__ import annotations

import pytest

from web.scouting import timeline
from web.scouting.export import event_export
from web.scouting.model import MatchData

FPS = 60.0
# Clock-scan periods as the job writes them (2026cur qm8 shape).
PERIODS = [
    {"label": "auto", "start_frame": 420, "end_frame": 1650, "start_seconds": 20,
     "end_seconds": 0, "clock_first_frame": 210, "true_start_frame": 373,
     "start_method": "indicator"},
    {"label": "teleop", "start_frame": 1890, "end_frame": 10350, "start_seconds": 140,
     "end_seconds": 0, "clock_first_frame": 1620},
]


def _scan(auto_zero_s: float, tele_zero_s: float, rate: float = 1.0) -> list:
    """Synthetic clock OCR every 0.5 s: ceil(remaining) of a display countdown
    whose zero is at the given video seconds; ``rate`` = display seconds per
    video second (<1 = clock runs slow against the video)."""
    out = []
    for f in range(0, 12000, 30):
        t = f / FPS
        val = None
        for zero, length in ((auto_zero_s, 20), (tele_zero_s, 140)):
            rem = (zero - t) * rate
            if -0.6 < rem <= length + 1.5:
                val = max(0, min(length, -int(-rem // 1))) if rem > 0 else 0
                break
        out.append((f, val))
    return out


@pytest.fixture
def fake_video(monkeypatch):
    def install(samples, fps=FPS):
        monkeypatch.setattr(timeline, "video_info", lambda p: {
            "fps": fps, "frame_count": 12870, "duration_s": 214.5, "height": 1080})
        monkeypatch.setattr(timeline, "_clock_scan_samples", lambda p, f, h: samples)
    return install


def test_marks_and_phases_from_clock_ticks(fake_video):
    fake_video(_scan(26.75, 172.25))
    tl = timeline.build_timeline(PERIODS, 2026, "/v.mp4", "Pialfz0joK8")
    m = tl["marks"]
    assert m["match_start"]["t"] == pytest.approx(373 / FPS, abs=1e-3)
    assert m["match_start"]["source"] == "indicator"
    lag = timeline.DISPLAY_LAG_S
    assert m["auto_end"]["t"] == pytest.approx(26.75 - lag, abs=0.26)
    assert m["teleop_start"]["t"] == pytest.approx(172.25 - 140 - lag, abs=0.26)
    assert m["endgame_start"]["t"] == pytest.approx(172.25 - 30 - lag, abs=0.26)
    assert m["match_end"]["t"] == pytest.approx(172.25 - lag, abs=0.26)
    assert m["match_end"]["source"].startswith("clock_ticks(")
    for mk in m.values():
        assert mk["frame"] == round(mk["t"] * FPS)
    names = [p["phase"] for p in tl["phases"]]
    assert names[:3] == ["auto", "between", "teleop"]
    assert {"transition", "shift1", "shift2", "shift3", "shift4", "endgame"} <= set(names)
    ph = {p["phase"]: p for p in tl["phases"]}
    assert ph["between"]["start_s"] == m["auto_end"]["t"]
    assert ph["between"]["end_s"] == m["teleop_start"]["t"]
    assert ph["endgame"]["start_s"] == m["endgame_start"]["t"]
    assert ph["endgame"]["within"] == "teleop" and "within" not in ph["auto"]
    assert tl["video"]["youtube_url"] == "https://www.youtube.com/watch?v=Pialfz0joK8&t=6s"


def test_slow_broadcast_clock_is_fitted_locally(fake_video):
    # Display runs 1% slow: teleop spans ~141.4 s of video. Each boundary must
    # land where the display shows it, not at start + 140.
    rate = 0.99
    tele_zero = 172.0
    fake_video(_scan(26.75, tele_zero, rate=rate))
    tl = timeline.build_timeline(PERIODS, 2026, "/v.mp4")
    m = tl["marks"]
    lag = timeline.DISPLAY_LAG_S
    assert m["endgame_start"]["t"] == pytest.approx(tele_zero - 30 / rate - lag, abs=0.3)
    assert m["teleop_start"]["t"] == pytest.approx(tele_zero - 140 / rate - lag, abs=0.3)
    teleop = next(p for p in tl["phases"] if p["phase"] == "teleop")
    assert teleop["duration_s"] == pytest.approx(140 / rate, abs=0.5)


def test_fallback_without_clock_scan_uses_periods(fake_video):
    fake_video(None)
    tl = timeline.build_timeline(PERIODS, 2026, "/v.mp4")
    m = tl["marks"]
    assert m["match_start"]["source"] == "indicator"
    assert m["auto_end"]["t"] == pytest.approx(373 / FPS + 20, abs=1e-3)
    t0 = 1890 / FPS + timeline.TELEOP_START_FROM_LAST_START_S
    assert m["teleop_start"]["t"] == pytest.approx(t0, abs=1e-3)
    assert m["teleop_start"]["source"] == "clock_lead"
    assert m["endgame_start"]["t"] == pytest.approx(t0 + 110, abs=1e-3)
    assert m["match_end"]["t"] == pytest.approx(t0 + 140, abs=1e-3)


def test_no_video_estimates_fps_and_no_clock_returns_none():
    tl = timeline.build_timeline(PERIODS, 2026, None)
    assert tl["video"]["fps_source"] == "clock_estimate" and tl["video"]["youtube_url"] is None
    assert timeline.build_timeline([], 2026, None) is None


def test_auto_only_replay_has_no_teleop_marks(fake_video):
    fake_video(None)
    tl = timeline.build_timeline(PERIODS[:1], 2026, "/v.mp4")
    assert set(tl["marks"]) == {"match_start", "auto_end"}
    assert [p["phase"] for p in tl["phases"]] == ["auto"]


def test_2025_has_endgame_only_and_rule_lengths(fake_video):
    fake_video(None)
    periods = [dict(PERIODS[0], start_seconds=15, true_start_frame=None, start_method=None),
               dict(PERIODS[1], start_seconds=135)]
    tl = timeline.build_timeline(periods, 2025, "/v.mp4")
    ph = {p["phase"]: p for p in tl["phases"]}
    assert "shift1" not in ph and ph["endgame"]["duration_s"] == pytest.approx(20.0)
    assert ph["auto"]["duration_s"] == pytest.approx(15.0)


def test_event_export_orders_matches_and_attaches_timelines():
    def md(key, level, num, set_no=1):
        return MatchData(event_key="2026tst", match_key=key, comp_level=level,
                         match_number=num, set_number=set_no, year=2026, replay_id=key,
                         replay_created_at=None, legacy_labels=False,
                         alliances={"red": [1, 2, 3], "blue": [4, None, 6]},
                         hub_auto={}, frame_period_s=0.2)
    ms = [md("2026tst_f1m1", "f", 1), md("2026tst_qm10", "qm", 10), md("2026tst_qm2", "qm", 2)]
    out = event_export("2026tst", 2026, ms, [], [], {"2026tst_qm2": {"marks": {}}})
    assert out["schema"] == "frc-vision-scorer.scouting-export" and out["schema_version"] == 2
    assert [m["match_key"] for m in out["matches"]] == ["2026tst_qm2", "2026tst_qm10",
                                                         "2026tst_f1m1"]
    assert out["matches"][0]["timeline"] == {"marks": {}}
    assert out["matches"][1]["timeline"] is None
    assert out["matches"][0]["alliances"]["blue"] == [4, None, 6]
