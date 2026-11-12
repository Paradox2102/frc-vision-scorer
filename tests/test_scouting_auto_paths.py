"""Tests for the auto_paths scouting metric (synthetic MatchData, no DB)."""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from web.scouting.metrics.auto_paths import AutoPaths, path_distance, resample
from web.scouting.model import MatchData, Sample, TeamMatch

P = dict(AutoPaths.default_params)
AUTO_S = 20.0


def mk_samples(fn, t0=0.0, t1=15.0, hz=5.0, delay=0.0, dx=0.0, dy=0.0, noise=0.0, seed=0,
               red=False):
    """Samples of position fn(t) at `hz`, observed with a start `delay`."""
    rng = np.random.default_rng(seed)
    out = []
    n = int(round((t1 - t0) * hz)) + 1
    for i in range(n):
        t = t0 + i / hz
        if t + delay > AUTO_S:
            break
        x, y = fn(t)
        x += dx + (rng.normal(0, noise) if noise else 0.0)
        y += dy + (rng.normal(0, noise) if noise else 0.0)
        ax, ay = x, y
        if red:   # absolute coords elsewhere; ax/ay already alliance-perspective
            x, y = 16.541 - ax, 8.069 - ay
        out.append(Sample(frame_num=i, t_match=round(t + delay, 4), period="auto",
                          remaining=AUTO_S - t - delay, x=x, y=y, ax=ax, ay=ay, dt=0.2))
    return out


def route_a(t):
    return 1.0 + 0.35 * min(t, 14.0), 1.0 + 0.1 * math.sin(t / 2)


def route_b(t):
    return 1.0 + 0.25 * min(t, 14.0), 6.5 - 0.2 * min(t, 14.0)


def mk_match(num, teams_paths: dict, red=(), blue=(), level="qm"):
    """teams_paths: {team: list[Sample]}; red/blue: lineup in station order."""
    alliances = {"red": list(red) + [None] * (3 - len(red)),
                 "blue": list(blue) + [None] * (3 - len(blue))}
    md = MatchData(event_key="2026test", match_key=f"2026test_{level}{num}", comp_level=level,
                   match_number=num, set_number=1, year=2026, replay_id=f"r{num}",
                   replay_created_at=None, legacy_labels=False, alliances=alliances,
                   hub_auto={"red": None, "blue": None}, frame_period_s=0.2)
    for team, path in teams_paths.items():
        alliance, station = md.station_of(team)
        md.teams[team] = TeamMatch(team=team, alliance=alliance or "blue", station=station,
                                   auto_path=path, auto_slot="B1", auto_label_share=0.9,
                                   auto_runner_up_share=0.1, auto_source="autonomous_paths")
    return md


def grid_of(path):
    return resample(path, AUTO_S, P["grid_s"], P["max_gap_s"])


def dist(a, b):
    return path_distance(grid_of(a), grid_of(b), P["grid_s"], P["max_shift_s"],
                         P["min_overlap_frac"], P["min_overlap_s"])


def test_resample_gap_larger_than_max_gap_is_nan():
    path = mk_samples(route_a, 0, 4) + mk_samples(route_a, 7, 12)   # 3 s hole
    g = grid_of(path)
    t = np.arange(len(g)) * P["grid_s"]
    valid = ~np.isnan(g[:, 0])
    assert valid[(t >= 0) & (t <= 4)].all()
    assert not valid[(t > 4) & (t < 7)].any()
    assert valid[(t >= 7) & (t <= 12)].all()
    assert not valid[t > 12].any()
    # interpolation inside a small gap (0.2 s spacing) lands on the line
    i = int(round(2.25 / P["grid_s"]))
    assert g[i, 0] == pytest.approx(route_a(2.25)[0], abs=0.01)


def test_identical_paths_distance_zero():
    a = mk_samples(route_a)
    assert dist(a, a) == pytest.approx(0.0, abs=1e-9)


def test_delayed_path_matches_with_shift():
    a = mk_samples(route_a)
    b = mk_samples(route_a, delay=0.5)
    assert dist(a, b) < 0.05
    no_shift = path_distance(grid_of(a), grid_of(b), P["grid_s"], 0.0,
                             P["min_overlap_frac"], P["min_overlap_s"])
    assert no_shift > 0.1


def test_truncated_path_still_compares():
    full = mk_samples(route_a, 0, 20)
    short = mk_samples(route_a, 0, 8)
    d = dist(full, short)
    assert math.isfinite(d) and d < 0.05


def test_short_overlap_is_incomparable():
    a = mk_samples(route_a, 0, 2)
    b = mk_samples(route_a, 0, 2)
    assert dist(a, b) == math.inf     # 2 s < min_overlap_s


def test_two_routines_cluster_with_representatives():
    # routine A on matches 1,3,5 ; routine B on 2,4,6. In each routine the
    # middle repeat is un-offset, so it is the medoid.
    offs = {1: 0.25, 3: 0.0, 5: -0.25, 2: 0.25, 4: 0.0, 6: -0.25}
    matches = []
    for num in range(1, 7):
        fn = route_a if num % 2 else route_b
        path = mk_samples(fn, dy=offs[num], noise=0.03, seed=num, delay=0.25 * (num % 3))
        matches.append(mk_match(num, {3128: path}, blue=[3128, 254, 1678]))
    out = AutoPaths().compute(matches, P)
    team = out["teams"]["3128"]
    assert team["n_matches"] == 6 and team["n_with_path"] == 6
    cl = team["clusters"]
    assert [c["count"] for c in cl] == [3, 3]
    assert cl[0]["members"] == ["2026test_qm1", "2026test_qm3", "2026test_qm5"]   # tie -> earliest
    assert cl[1]["members"] == ["2026test_qm2", "2026test_qm4", "2026test_qm6"]
    assert cl[0]["representative"] == "2026test_qm3"
    assert cl[1]["representative"] == "2026test_qm4"
    assert 0.2 < cl[0]["mean_spread_m"] < 0.5
    assert cl[0]["start"][0] == pytest.approx(1.0, abs=0.15)
    assert cl[1]["start"][1] == pytest.approx(6.5, abs=0.3)
    assert [m["cluster"] for m in team["matches"]] == [0, 1, 0, 1, 0, 1]
    # lineup teams without paths are listed
    assert out["teams"]["254"]["n_with_path"] == 0
    assert out["legend"] == {"perspective": "own alliance wall at x=0"}


def test_low_coverage_path_gets_no_cluster():
    full = mk_samples(route_a)
    tiny = mk_samples(route_a, 0, 3)        # 13 of 81 grid points < 0.25
    matches = [mk_match(1, {3128: full}, blue=[3128]), mk_match(2, {3128: full}, blue=[3128]),
               mk_match(3, {3128: tiny}, blue=[3128])]
    team = AutoPaths().compute(matches, P)["teams"]["3128"]
    m3 = team["matches"][2]
    assert m3["coverage"] < 0.25 and m3["cluster"] is None and m3["n_samples"] == len(tiny)
    assert team["clusters"][0]["count"] == 2


def test_lineup_team_without_path_listed():
    md = mk_match(1, {3128: mk_samples(route_a)}, red=[6773, 3128], blue=[254])
    out = AutoPaths().compute([md], P)
    t = out["teams"]["6773"]
    assert t["n_matches"] == 1 and t["n_with_path"] == 0 and t["clusters"] == []
    m = t["matches"][0]
    assert m["source"] is None and m["coverage"] == 0 and m["points"] == []
    assert m["cluster"] is None and m["alliance"] == "red" and m["station"] == 1
    assert set(out["teams"]) == {"3128", "6773", "254"}


def test_red_team_uses_alliance_perspective():
    path = mk_samples(route_a, red=True)
    md = mk_match(1, {3128: path}, red=[3128])
    m = AutoPaths().compute([md], P)["teams"]["3128"]["matches"][0]
    assert m["alliance"] == "red"
    t0, x0, y0 = m["points"][0]
    assert (x0, y0) == (round(route_a(0)[0], 2), round(route_a(0)[1], 2))
    assert m["points_norm"][0][0] == pytest.approx(route_a(0)[0] / 16.541, abs=1e-3)
    assert m["identity_conflict"] is False and m["label_share"] == 0.9


def test_matches_sorted_by_level():
    a = mk_samples(route_a)
    ms = [mk_match(1, {3128: a}, blue=[3128], level="sf"),
          mk_match(9, {3128: a}, blue=[3128]), mk_match(2, {3128: a}, blue=[3128])]
    keys = [m["match_key"] for m in AutoPaths().compute(ms, P)["teams"]["3128"]["matches"]]
    assert keys == ["2026test_qm2", "2026test_qm9", "2026test_sf1"]


def test_real_event_smoke():
    replays = Path(__file__).resolve().parents[1] / "web" / "replays"
    if not replays.is_dir() or not any(replays.iterdir()):
        pytest.skip("no replays on disk")
    from web.scouting import registry
    try:
        env = registry.run_metric("2026cur", "auto_paths")
    except sqlite3.OperationalError as e:
        pytest.skip(f"DB not usable: {e}")
    if not env["sources"]:
        pytest.skip("no 2026cur replays")
    assert env["schema_version"] == 1
    assert len(env["teams"]) >= 10


def route_a_mirrored(t):
    x, y = route_a(t)
    return x, 8.069 - y


def test_mirrored_routine_groups_with_its_reflection():
    mds = [mk_match(1, {100: mk_samples(route_a, noise=0.05, seed=1)}, blue=[100]),
           mk_match(2, {100: mk_samples(route_a_mirrored, noise=0.05, seed=2)}, blue=[100]),
           mk_match(3, {100: mk_samples(route_a, noise=0.05, seed=3)}, blue=[100])]
    team = AutoPaths().compute(mds, dict(P))["teams"]["100"]
    assert len(team["clusters"]) == 1
    c = team["clusters"][0]
    assert c["count"] == 3 and c["n_mirrored"] == 1
    flags = {m["match_key"]: m["mirrored"] for m in team["matches"]}
    assert flags == {"2026test_qm1": False, "2026test_qm2": True, "2026test_qm3": False}
    assert c["start"][1] < 2.0           # start in the representative's (unmirrored) side


def test_mirror_aware_off_keeps_reflections_apart():
    mds = [mk_match(1, {100: mk_samples(route_a)}, blue=[100]),
           mk_match(2, {100: mk_samples(route_a_mirrored)}, blue=[100])]
    team = AutoPaths().compute(mds, {**P, "mirror_aware": False})["teams"]["100"]
    assert len(team["clusters"]) == 2


def test_bool_query_params_parse():
    from web.scouting.registry import merged_params
    assert merged_params(AutoPaths(), {"mirror_aware": "false"})["mirror_aware"] is False
    assert merged_params(AutoPaths(), {"mirror_aware": "1"})["mirror_aware"] is True
    assert merged_params(AutoPaths(), {"threshold_m": "1.2"})["threshold_m"] == 1.2
