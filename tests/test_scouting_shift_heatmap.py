"""Shift heatmap metric: bucket split, station split, summaries, coverage, clamping."""
import sqlite3

import pytest

from web.scouting.geometry import field_dims
from web.scouting.metrics.shift_heatmap import ShiftHeatmap
from web.scouting.model import MatchData, Sample, TeamMatch

L, W = field_dims(2026)
DT = 0.2


def _teleop(ax, ay, n=700):
    """Robot parked at (ax, ay) for the whole teleop: remaining 140 -> 0.2."""
    return [Sample(frame_num=i, t_match=20 + i * DT, period="teleop",
                   remaining=round(140.0 - i * DT, 6), x=ax, y=ay, ax=ax, ay=ay, dt=DT)
            for i in range(n)]


def _auto(ax, ay, n=100):
    return [Sample(frame_num=i, t_match=i * DT, period="auto",
                   remaining=round(20.0 - i * DT, 6), x=ax, y=ay, ax=ax, ay=ay, dt=DT)
            for i in range(n)]


def _match(num, team=3128, alliance="blue", station=1, samples=(), hub_auto=None):
    lineup = {"red": [1, 2, 3], "blue": [4, 5, 6]}
    lineup[alliance][station - 1] = team
    md = MatchData(event_key="2026tst", match_key=f"2026tst_qm{num}", comp_level="qm",
                   match_number=num, set_number=1, year=2026, replay_id=f"r{num}",
                   replay_created_at=None, legacy_labels=False, alliances=lineup,
                   hub_auto=hub_auto if hub_auto is not None else {"red": 3, "blue": 9},
                   frame_period_s=DT)
    md.teams[team] = TeamMatch(team=team, alliance=alliance, station=station,
                               samples=list(samples))
    return md


def _run(matches, **params):
    m = ShiftHeatmap()
    return m.compute(matches, {**m.default_params, **params})


def test_parked_robot_splits_teleop_into_shift_buckets_in_one_cell():
    # Blue won auto fuel, robot is blue: inactive in shifts 1 and 3.
    out = _run([_match(1, samples=_teleop(3.2, 4.1))])
    t = out["teams"]["3128"]
    allb = t["by_station"]["all"]
    secs = {b: allb[b]["seconds"] for b in out["buckets"]}
    assert secs == {"auto": 0.0, "transition": 10.0, "active": 50.0, "inactive": 50.0,
                    "endgame": 30.0, "unknown": 0.0}
    for b in ("transition", "active", "inactive", "endgame"):
        assert allb[b]["cells"] == [[6, 8, secs[b]]]
    # which shifts are inactive: shift 1 (130..105) is inactive for the winner
    md = _match(2, samples=[s for s in _teleop(1.0, 1.0) if 105 < s.remaining <= 130])
    assert _run([md])["teams"]["3128"]["by_station"]["all"]["inactive"]["seconds"] == 25.0
    md = _match(3, samples=[s for s in _teleop(1.0, 1.0) if 80 < s.remaining <= 105])
    assert _run([md])["teams"]["3128"]["by_station"]["all"]["active"]["seconds"] == 25.0
    # the loser (red) sees the mirror: shift 1 active
    md = _match(4, alliance="red", samples=[s for s in _teleop(1.0, 1.0)
                                            if 105 < s.remaining <= 130])
    assert _run([md])["teams"]["3128"]["by_station"]["all"]["active"]["seconds"] == 25.0
    assert out["auto_winner_by_match"] == {"2026tst_qm1": "blue"}
    assert t["matches"][0]["seconds"]["inactive"] == 50.0


def test_output_shape_and_every_bucket_present():
    out = _run([_match(1, samples=_teleop(3.0, 3.0, n=10))])
    assert out["buckets"] == ["auto", "transition", "active", "inactive", "endgame", "unknown"]
    assert out["default_buckets"] == ["active", "inactive", "endgame"]
    assert out["grid"] == {"cell_m": 0.5, "nx": 34, "ny": 17, "field_m": [L, W]}
    assert set(out["station_layout"]) == {"1", "2", "3"}
    # lineup teams with no samples still appear, every bucket keyed
    assert set(out["teams"]) == {"1", "2", "3", "5", "6", "3128"}
    empty = out["teams"]["5"]["by_station"]["all"]["active"]
    assert empty == {"seconds": 0.0, "cells": [],
                     "summary": {"own_zone_frac": None, "neutral_zone_frac": None, "opp_zone_frac": None,
                                 "mean_ax": None, "mean_ay": None, "n_samples": 0}}
    for per in out["teams"]["3128"]["by_station"].values():
        assert set(per) == set(out["buckets"])


def test_tie_sends_shift_seconds_to_unknown():
    out = _run([_match(1, samples=_teleop(2.0, 2.0), hub_auto={"red": 5, "blue": 5})])
    allb = out["teams"]["3128"]["by_station"]["all"]
    assert allb["unknown"]["seconds"] == 100.0
    assert allb["active"]["seconds"] == allb["inactive"]["seconds"] == 0.0
    assert allb["transition"]["seconds"] == 10.0 and allb["endgame"]["seconds"] == 30.0
    assert out["auto_winner_by_match"]["2026tst_qm1"] is None
    cov = out["teams"]["3128"]["coverage"]
    assert cov["active"] is None and cov["inactive"] is None and cov["unknown"] == 1.0


def test_station_split_adds_up():
    m1 = _match(1, station=1, samples=_teleop(2.0, 2.0))
    m2 = _match(2, station=3, samples=_teleop(10.0, 6.0))
    out = _run([m2, m1])
    t = out["teams"]["3128"]
    assert set(t["by_station"]) == {"all", "1", "3"}
    assert t["matches_by_station"] == {"1": 1, "2": 0, "3": 1}
    assert [m["match_key"] for m in t["matches"]] == ["2026tst_qm1", "2026tst_qm2"]
    for b in out["buckets"]:
        s = t["by_station"]
        assert s["all"][b]["seconds"] == pytest.approx(s["1"][b]["seconds"] + s["3"][b]["seconds"])
    assert t["by_station"]["1"]["active"]["cells"] == [[4, 4, 50.0]]
    assert t["by_station"]["3"]["active"]["cells"] == [[20, 12, 50.0]]
    assert t["n_matches"] == 2


def test_zone_fractions_and_means():
    near = _teleop(2.0, 1.0)[:250]      # remaining 140..90.2: transition + shift1 + part shift2
    far = [Sample(**{**s.__dict__, "ax": L - 2.0, "ay": 7.0, "x": L - 2.0, "y": 7.0})
           for s in _teleop(0, 0)[250:]]
    out = _run([_match(1, samples=near + far)])
    allb = out["teams"]["3128"]["by_station"]["all"]
    assert allb["transition"]["summary"]["own_zone_frac"] == 1.0
    assert allb["inactive"]["summary"]["own_zone_frac"] == 0.5    # shift1 near, shift3 far
    assert allb["inactive"]["summary"]["opp_zone_frac"] == 0.5
    assert allb["inactive"]["summary"]["neutral_zone_frac"] == 0.0
    assert allb["endgame"]["summary"]["opp_zone_frac"] == 1.0
    assert allb["endgame"]["summary"]["mean_ax"] == round(L - 2.0, 3)
    assert allb["endgame"]["summary"]["mean_ay"] == 7.0
    assert allb["endgame"]["summary"]["n_samples"] == 150


def test_coverage_math():
    samples = _auto(1.0, 1.0, n=50) + _teleop(1.0, 1.0)[:50] + _teleop(1.0, 1.0)[550:]
    m1 = _match(1, samples=samples)
    m2 = _match(2, samples=[])                   # in lineup, never seen
    cov = _run([m1, m2])["teams"]["3128"]["coverage"]
    assert cov["auto"] == round(10.0 / 40.0, 3)
    assert cov["transition"] == 0.5
    assert cov["endgame"] == 0.5
    assert cov["active"] == 0.0 and cov["inactive"] == 0.0
    assert cov["unknown"] is None


def test_cells_clamp_at_field_edge():
    pts = [Sample(frame_num=0, t_match=30, period="teleop", remaining=120.0, x=0, y=0,
                  ax=-0.3, ay=-0.1, dt=DT),
           Sample(frame_num=1, t_match=30.2, period="teleop", remaining=119.8, x=0, y=0,
                  ax=L + 0.4, ay=W + 2.0, dt=DT)]
    out = _run([_match(1, samples=pts)])
    cells = out["teams"]["3128"]["by_station"]["all"]["inactive"]["cells"]
    assert cells == [[0, 0, 0.2], [33, 16, 0.2]]


def test_real_event_smoke():
    from web.scouting import registry
    try:
        env = registry.run_metric("2026cur", "shift_heatmap")
    except sqlite3.OperationalError as e:
        pytest.skip(f"no DB: {e}")
    if not env.get("teams"):
        pytest.skip("no 2026cur replays")
    assert len(env["teams"]) >= 10
    assert any(t["by_station"]["all"]["active"]["seconds"] > 0
               and t["by_station"]["all"]["inactive"]["seconds"] > 0
               for t in env["teams"].values())



def test_neutral_zone_between_hub_faces():
    # alliance zone = own wall .. hub near face (4.029); neutral = up to L - 4.029
    mid = [Sample(**{**s.__dict__, "ax": 8.0, "ay": 4.0, "x": 8.0, "y": 4.0})
           for s in _teleop(0, 0)]
    bump = [Sample(**{**s.__dict__, "ax": 4.6, "ay": 2.5, "x": 4.6, "y": 2.5})
            for s in _teleop(0, 0)]
    for samples in (mid, bump):
        out = _run([_match(1, samples=samples)])
        sm = out["teams"]["3128"]["by_station"]["all"]["endgame"]["summary"]
        assert (sm["own_zone_frac"], sm["neutral_zone_frac"], sm["opp_zone_frac"]) == (0.0, 1.0, 0.0)
    assert out["zones"]["own_max_x"] == pytest.approx(4.029, abs=0.01)
    assert out["zones"]["opp_min_x"] == pytest.approx(L - 4.029, abs=0.01)
