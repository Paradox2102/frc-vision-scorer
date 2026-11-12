"""Core of the scouting-metrics framework: clock, geometry, 2026 shifts, loader, export."""
import json
import sqlite3

import pytest

from web.scouting import geometry
from web.scouting.clock import MatchClock
from web.scouting.export import JsonFileSink, RoboReconSink, envelope, filter_team
from web.scouting.games import rebuilt2026 as g
from web.scouting.loader import build_match, parse_match_key, pick_replay
from web.scouting.registry import Metric

PERIODS = [
    {"label": "auto", "start_frame": 600, "end_frame": 1800, "start_seconds": 20, "end_seconds": 0},
    {"label": "teleop", "start_frame": 2400, "end_frame": 10800, "start_seconds": 140, "end_seconds": 0},
]


# ------------------------------------------------------------------ clock
def test_clock_maps_frames_to_game_time():
    c = MatchClock(PERIODS, lead_s=0.0)
    assert c.frames_per_s() == pytest.approx(60.0)
    assert c.at(600) == ("auto", 0.0, 20.0)
    period, t, rem = c.at(1200)
    assert (period, t, rem) == ("auto", pytest.approx(10.0), pytest.approx(10.0))
    period, t, rem = c.at(2400 + 60 * 25)            # 25 s into teleop
    assert period == "teleop" and t == pytest.approx(45.0) and rem == pytest.approx(115.0)
    assert c.at(2000) is None                          # auto->teleop pause
    assert c.at(100) is None and c.at(20000) is None


def test_clock_lead_moves_period_start_earlier():
    c = MatchClock(PERIODS, lead_s=0.75)
    assert MatchClock(PERIODS).lead_s == pytest.approx(1.0)       # default
    period, t, rem = c.at(600 - 45)                    # 0.75 s before the scanned start
    assert (period, t, rem) == ("auto", pytest.approx(0.0), pytest.approx(20.0))
    assert c.at(600 - 60) is None                      # 1.0 s before: pre-match
    period, t, _ = c.at(600)
    assert t == pytest.approx(0.75)
    assert c.at(1800) is None                          # auto ended 0.75 s early
    assert c.at(1800 - 45)[0] == "auto"


def test_clock_uses_true_start_frame_when_present():
    periods = [dict(PERIODS[0], true_start_frame=600 - 72), PERIODS[1]]   # 1.2 s early
    c = MatchClock(periods)
    assert c.at(600 - 72)[:2] == ("auto", pytest.approx(0.0))
    assert c.at(600 - 78) is None
    period, t, _ = c.at(2400)                          # teleop keeps the default lead
    assert period == "teleop" and t == pytest.approx(20.0 + 1.0)


def test_clock_without_periods_is_not_ok():
    assert not MatchClock([]).ok
    assert not MatchClock([{"label": "auto", "start_frame": 5, "end_frame": 5,
                            "start_seconds": 20, "end_seconds": 0}]).ok


# ------------------------------------------------------------------ geometry
def test_symmetry_is_measured_from_tag_layouts():
    assert geometry.field_symmetry(2026) == geometry.ROTATIONAL
    assert geometry.field_symmetry(2025) == geometry.ROTATIONAL
    assert geometry.field_symmetry(2024) == geometry.MIRROR


def test_alliance_perspective_rotates_red_2026():
    L, W = geometry.field_dims(2026)
    assert geometry.to_alliance(2.0, 1.0, "blue", 2026) == (2.0, 1.0)
    ax, ay = geometry.to_alliance(L - 2.0, W - 1.0, "red", 2026)
    assert ax == pytest.approx(2.0) and ay == pytest.approx(1.0)
    ax, ay = geometry.to_alliance(L - 2.0, 1.0, "red", 2024)       # mirrored season
    assert ax == pytest.approx(2.0) and ay == pytest.approx(1.0)


def test_field_image_2026_exists():
    fi = geometry.field_image(2026)
    assert fi and fi["path"].exists() and fi["field_top_left_px"]


# ------------------------------------------------------------------ 2026 shifts
@pytest.mark.parametrize("remaining,shift", [
    (140, "transition"), (130.5, "transition"), (130, "shift1"), (105.01, "shift1"),
    (105, "shift2"), (80, "shift3"), (55, "shift4"), (30.5, "shift4"), (30, "endgame"), (0, "endgame")])
def test_teleop_shift_boundaries(remaining, shift):
    assert g.teleop_shift(remaining) == shift


def test_auto_winner_inactive_in_shifts_1_and_3():
    hub = {"red": 90, "blue": 55}                      # red won auto
    assert g.bucket("teleop", 120, "red", hub) == g.INACTIVE   # shift 1
    assert g.bucket("teleop", 120, "blue", hub) == g.ACTIVE
    assert g.bucket("teleop", 90, "red", hub) == g.ACTIVE      # shift 2
    assert g.bucket("teleop", 70, "red", hub) == g.INACTIVE    # shift 3
    assert g.bucket("teleop", 40, "blue", hub) == g.INACTIVE   # shift 4
    assert g.bucket("teleop", 135, "red", hub) == g.TRANSITION
    assert g.bucket("teleop", 10, "blue", hub) == g.ENDGAME
    assert g.bucket("auto", 10, "blue", hub) == g.AUTO


def test_tied_or_missing_auto_is_unknown_never_guessed():
    assert g.bucket("teleop", 120, "red", {"red": 60, "blue": 60}) == g.UNKNOWN
    assert g.bucket("teleop", 120, "red", {"red": None, "blue": 60}) == g.UNKNOWN
    assert g.bucket("teleop", 10, "red", {"red": 60, "blue": 60}) == g.ENDGAME


# ------------------------------------------------------------------ loader
def test_parse_match_key():
    assert parse_match_key("2026cur_qm8") == ("qm", 1, 8)
    assert parse_match_key("2026cur_sf3m1") == ("sf", 3, 1)
    assert parse_match_key("2026cur_f1m2") == ("f", 1, 2)


def _robot(slot, alliance, team, x, y, confirmed=False):
    return {"slot": slot, "alliance": alliance, "team_number": team,
            "confirmed": confirmed, "field_pos": [x, y]}


def _meta(frames, autos=None):
    return {"summary": {"periods": PERIODS, "window_labels": {"windows": 1},
                        "autonomous_paths": autos or {}}, "frames": frames}


LINEUP = {"comp_level": "qm", "match_number": 8, "set_number": 1,
          "alliances": {"red": [6497, 2724, 2637], "blue": [3128, 6773, 9066]},
          "hub_auto": {"red": 62, "blue": 82}}


def test_build_match_attributes_samples_and_stations():
    L, W = geometry.field_dims(2026)
    frames = []
    for i, fn in enumerate(range(600, 1800, 12)):       # auto, 0.2 s per frame
        frames.append({"frame_num": fn, "robots": [
            _robot("B1", "blue", "3128", 1.0 + i * 0.01, 2.0),
            _robot("R1", "red", "6497", L - 1.0, W - 2.0),
            _robot("R2", "red", None, 12.0, 4.0),          # unlabelled: tracked, not attributed
            _robot("R3", "red", "1234", 12.0, 5.0),        # not in lineup: dropped
        ]})
    # the pipeline's own auto path starts at the scanned start_frame; earlier
    # (lead) frames of the same slot must be prepended by the loader
    autos = {"B1": [{"frame_num": f["frame_num"], "field_pos": f["robots"][0]["field_pos"]}
                    for f in frames if f["frame_num"] >= 660]}
    md, why = build_match("2026cur", "2026cur_qm8", {"id": "r1", "created_at": "x"},
                          _meta(frames, autos), False, LINEUP, 2026)
    assert why is None
    assert md.frame_period_s == pytest.approx(0.2)
    assert set(md.teams) == {3128, 6497}
    blue, red = md.teams[3128], md.teams[6497]
    assert (blue.alliance, blue.station) == ("blue", 1)
    assert (red.alliance, red.station) == ("red", 1)
    s = red.samples[0]
    assert s.ax == pytest.approx(1.0) and s.ay == pytest.approx(2.0)   # rotated into own frame
    assert blue.auto_source == "autonomous_paths" and blue.auto_slot == "B1"
    clock = MatchClock(PERIODS)
    in_auto = [f["frame_num"] for f in frames if (clock.at(f["frame_num"]) or ("",))[0] == "auto"]
    assert blue.auto_label_share == 1.0
    assert [s.frame_num for s in blue.auto_path] == in_auto       # prepended + pipeline path
    assert red.auto_source == "frames"                                  # no summary path for R1
    assert md.off_roster_samples == len(in_auto)
    assert md.tracked_s["auto"] == pytest.approx(4 * len(in_auto) * 0.2)
    assert md.labelled_s["auto"] == pytest.approx(2 * len(in_auto) * 0.2)


def test_build_match_duplicate_team_keeps_confirmed_slot():
    frames = [{"frame_num": 2400 + 12 * i, "robots": [
        _robot("B1", "blue", "3128", 3.0, 3.0),
        _robot("B2", "blue", "3128", 5.0, 5.0, confirmed=True)]} for i in range(5)]
    md, _ = build_match("2026cur", "2026cur_qm8", {"id": "r"}, _meta(frames), False, LINEUP, 2026)
    xs = {s.x for s in md.teams[3128].samples}
    assert xs == {5.0} and md.duplicate_label_samples == 5


def test_build_match_without_clock_is_skipped():
    md, why = build_match("2026cur", "k", {"id": "r"}, {"summary": {}, "frames": []},
                          False, LINEUP, 2026)
    assert md is None and "clock" in why


def test_pick_replay_prefers_window_label_runs(tmp_path):
    def write(name, wl, full=True):
        p = tmp_path / f"{name}.json"
        summary = {"periods": PERIODS if full else PERIODS[:1]}
        if wl:
            summary["window_labels"] = {"w": 1}
        p.write_text(json.dumps({"summary": summary}))
        return {"id": name, "meta_path": str(p)}
    rows = [write("newest_legacy", False), write("older_wl", True)]
    row, _, legacy = pick_replay(rows)
    assert row["id"] == "older_wl" and legacy is False
    row, _, legacy = pick_replay(rows[:1])
    assert row["id"] == "newest_legacy" and legacy is True


def test_pick_replay_skips_newer_auto_only_runs(tmp_path):
    def write(name, summary):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({"summary": summary}))
        return {"id": name, "meta_path": str(p)}
    wl = {"window_labels": {"w": 1}}
    rows = [write("auto_only_flag", {**wl, "periods": PERIODS, "config": {"auto_period_only": True}}),
            write("auto_only_periods", {**wl, "periods": PERIODS[:1]}),
            write("older_full", {**wl, "periods": PERIODS})]
    assert pick_replay(rows)[0]["id"] == "older_full"
    assert pick_replay(rows[:2])[0]["id"] == "auto_only_flag"     # nothing better: newest


# ------------------------------------------------------------------ export
class _M(Metric):
    key, title, version = "m", "M", 3


def test_envelope_shape_and_team_filter(tmp_path):
    env = envelope(_M(), "2026cur", 2026, [], [{"match_key": "x", "reason": "y"}],
                   {"teams": {"3128": {"v": 1}, "6773": {"v": 2}}, "legend": {"a": 1}}, {"p": 1})
    assert env["schema"] == "frc-vision-scorer.scouting" and env["schema_version"] == 1
    assert env["metric_version"] == 3 and env["frame"]["symmetry"] == "rotational"
    assert env["teams"]["3128"]["team_number"] == 3128 and env["legend"] == {"a": 1}
    one = filter_team(env, 6773)
    assert list(one["teams"]) == ["6773"] and len(env["teams"]) == 2
    out = JsonFileSink(tmp_path).push([env])
    assert json.loads(open(out["written"][0]).read())["metric"] == "m"
    with pytest.raises(NotImplementedError):
        RoboReconSink().push([env])
