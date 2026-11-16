"""2026 ranking metrics (movement, cycles, defense, bump_stuck) + the shared motion cleaning."""
import math

import numpy as np
import pytest

from web.scouting import field_rules, motion, registry
from web.scouting.geometry import field_dims
from web.scouting.metrics.bump_stuck import BumpStuck
from web.scouting.metrics.cycles import Cycles
from web.scouting.metrics.defense import Defense
from web.scouting.metrics.movement import Movement
from web.scouting.model import MatchData, Sample, TeamMatch

L, W = field_dims(2026)
DT = 0.2
RULES = field_rules.auto_rules(2026)


def _samples(xy, t0=20.0, slot="B1"):
    """Samples every DT from t0 at the given (x, y) points (blue: ax = x)."""
    out = []
    for i, (x, y) in enumerate(xy):
        t = t0 + i * DT
        period = "auto" if t < 20.0 else "teleop"
        rem = (20.0 - t) if period == "auto" else (160.0 - t)
        out.append(Sample(frame_num=i, t_match=round(t, 6), period=period, remaining=round(rem, 6),
                          x=float(x), y=float(y), ax=float(x), ay=float(y), dt=DT, slot=slot))
    return out


def _jitter(n, x, y, sigma=0.2, seed=0):
    """A parked robot whose position jitters ~0.3-0.5 m frame to frame."""
    rng = np.random.default_rng(seed)
    return list(zip(x + rng.normal(0, sigma, n), y + rng.normal(0, sigma, n)))


def _circle(n, cx=8.0, cy=4.0, rad=2.0, v=2.0):
    w = v / rad
    return [(cx + rad * math.cos(w * i * DT), cy + rad * math.sin(w * i * DT)) for i in range(n)]


def _match(num, samples_by_team, alliance="blue"):
    lineup = {"red": [1, 2, 3], "blue": [4, 5, 6]}
    teams = list(samples_by_team)
    for i, t in enumerate(teams[:3]):
        lineup[alliance][i] = t
    md = MatchData(event_key="2026tst", match_key=f"2026tst_qm{num}", comp_level="qm",
                   match_number=num, set_number=1, year=2026, replay_id=f"r{num}",
                   replay_created_at=None, legacy_labels=False, alliances=lineup,
                   hub_auto={"red": 3, "blue": 9}, frame_period_s=DT)
    for i, (team, ss) in enumerate(samples_by_team.items()):
        md.teams[team] = TeamMatch(team=team, alliance=alliance, station=i + 1, samples=list(ss))
    return md


def _run(metric, matches, **params):
    return metric.compute(matches, {**metric.default_params, **params})


# ----------------------------------------------------------------- motion cleaning
def test_held_positions_are_excluded_but_short_repeats_kept():
    pts = _circle(100) + [(9.0, 4.0)] * 16 + _circle(100)      # 3.2 s frozen
    pts[20:22] = [(5.0, 5.0)] * 2                                 # 0.4 s repeat: kept
    tr = motion.clean_track(_samples(pts), "blue", 2026)
    assert tr.n_held_runs == 1
    assert tr.held_s == pytest.approx(15 * DT)                    # first sample stays valid
    assert tr.valid[100] and not tr.valid[101:116].any()
    assert tr.valid[20] and tr.valid[21]
    assert np.isnan(tr.speed[101:116]).all() and np.isnan(tr.ax[101:116]).all()
    # the held seconds count as neither moving nor stationary
    out = _run(Movement(), [_match(1, {3128: _samples(pts)})], min_tracked_s=10)
    row = out["teams"]["3128"]["matches"][0]
    assert row["held_s"] == pytest.approx(3.0)
    assert row["moving_s"] + row["stationary_s"] <= row["tracked_s"] + 1e-6
    assert row["tracked_s"] == pytest.approx((216 - 15) * DT, abs=0.05)


def test_jittery_parked_robot_is_not_moving_and_driving_robot_is():
    parked = _jitter(600, 2.0, 2.0)
    steps = np.hypot(*np.diff(np.array(parked), axis=0).T)
    assert np.median(steps) > 0.25          # raw frame-to-frame speed would read > 1 m/s
    out = _run(Movement(), [_match(1, {3128: _samples(parked), 254: _samples(_circle(600))})])
    p, d = out["teams"]["3128"], out["teams"]["254"]
    assert p["moving_frac"] < 0.05
    assert d["moving_frac"] > 0.97
    assert d["median_moving_mps"] == pytest.approx(2.0, abs=0.1)
    assert d["max_sustained_mps"] == pytest.approx(2.0, abs=0.15)
    assert d["avg_speed_mps"] > 1.8 and p["avg_speed_mps"] < 0.05
    assert out["ranking"][:2] == ["254", "3128"]


def test_speed_is_not_computed_across_a_gap_or_slot_change():
    a = _samples([(2.0, 2.0)] * 10, t0=20.0)
    b = _samples([(10.0, 2.0) if i % 2 else (10.01, 2.0) for i in range(10)], t0=22.0 + 0.6)
    tr = motion.clean_track(a + b, "blue", 2026)
    assert np.nanmax(tr.speed) < 0.1                     # the 8 m jump spans a gap
    c = _samples([(2.0 + 0.001 * i, 2.0) for i in range(10)], slot="B1")
    d = _samples([(6.0 + 0.001 * i, 2.0) for i in range(10)], t0=22.0, slot="B2")
    tr = motion.clean_track(c + d, "blue", 2026)
    assert len(set(tr.seg.tolist())) == 2 and np.nanmax(tr.speed) < 0.1


# ----------------------------------------------------------------- cycles
def _trip_path(n_trips, own_x=2.0, neu_x=8.0, dwell_n=25, transit_n=6, y=0.6):
    pts = []
    for _ in range(n_trips):
        pts += [(own_x + 0.01 * (i % 2), y) for i in range(dwell_n)]
        pts += [(own_x + (neu_x - own_x) * k / transit_n, y) for k in range(1, transit_n)]
        pts += [(neu_x + 0.01 * (i % 2), y) for i in range(dwell_n)]
        pts += [(neu_x - (neu_x - own_x) * k / transit_n, y) for k in range(1, transit_n)]
    pts += [(own_x + 0.01 * (i % 2), y) for i in range(dwell_n)]
    return pts


def test_three_real_trips_are_three_cycles():
    out = _run(Cycles(), [_match(1, {3128: _samples(_trip_path(3))})], min_tracked_s=10)
    t = out["teams"]["3128"]
    assert t["cycles_teleop_total"] == 3 and t["cycles_auto_total"] == 0
    row = t["matches"][0]
    assert row["cycles_teleop"] == 3 and row["resets"] == 0
    assert t["cycles_per_min"] == pytest.approx(60 * 3 / row["teleop_tracked_s"], abs=0.01)
    # neutral entry -> next neutral entry = one full loop: 2 dwells + 2 transits
    assert t["mean_cycle_s"] == pytest.approx((2 * 25 + 2 * 5) * DT, abs=0.25)
    # in-zone seconds of a visit: the 25-sample dwell + 2 transit samples each way
    assert t["mean_neutral_visit_s"] == pytest.approx((25 + 4) * DT, abs=0.25)


def test_zone_boundary_jitter_is_not_a_cycle():
    az, bf = RULES.alliance_zone_x, RULES.barrier_far_x
    # parked on the own-zone line, then on the neutral-zone edge, jittering across both
    pts = _jitter(150, az, 0.6, sigma=0.3, seed=1) + _jitter(150, bf + 0.2, 0.6, sigma=0.3, seed=2) \
        + _jitter(150, az, 0.6, sigma=0.3, seed=3)
    out = _run(Cycles(), [_match(1, {3128: _samples(pts)})], min_tracked_s=10)
    assert out["teams"]["3128"]["cycles_teleop_total"] == 0
    # a 0.6 s dip into the neutral zone (< min_dwell) between own-zone dwells
    dip = [(2.0, 0.6)] * 30 + [(8.0, 0.6)] * 3 + [(2.0, 0.6)] * 30
    dip = [(x + 0.01 * (i % 2), y) for i, (x, y) in enumerate(dip)]
    assert _run(Cycles(), [_match(1, {3128: _samples(dip)})], min_tracked_s=5)[
        "teams"]["3128"]["cycles_teleop_total"] == 0


def test_long_gap_resets_the_state_machine():
    neutral = [(8.0 + 0.01 * (i % 2), 0.6) for i in range(30)]
    home = [(2.0 + 0.01 * (i % 2), 0.6) for i in range(30)]
    start = [(2.0 + 0.01 * (i % 2), 0.6) for i in range(30)]
    ss = _samples(start + neutral) + _samples(home, t0=20.0 + 60 * DT + 4.0)   # 4 s gap
    out = _run(Cycles(), [_match(1, {3128: ss})], min_tracked_s=5)
    row = out["teams"]["3128"]["matches"][0]
    assert row["cycles_teleop"] == 0 and row["resets"] == 1
    # a 2 s gap (< max_gap_s) is bridged
    ss = _samples(start + neutral) + _samples(home, t0=20.0 + 60 * DT + 2.0)
    assert _run(Cycles(), [_match(1, {3128: ss})], min_tracked_s=5)[
        "teams"]["3128"]["matches"][0]["cycles_teleop"] == 1


def test_auto_cycles_are_counted_separately():
    ss = _samples(_trip_path(1, dwell_n=10, transit_n=4), t0=0.0)
    row = _run(Cycles(), [_match(1, {3128: ss})], min_tracked_s=0)["teams"]["3128"]["matches"][0]
    assert row["cycles_auto"] == 1 and row["cycles_teleop"] == 0


# ----------------------------------------------------------------- defense
def test_defense_threshold_and_fraction():
    edge = L - RULES.alliance_zone_x - 1.0
    inside = _jitter(300, edge + 0.3, 4.0, sigma=0.05, seed=4)
    outside = _jitter(300, edge - 0.3, 4.0, sigma=0.05, seed=5)
    out = _run(Defense(), [_match(1, {3128: _samples(inside + outside), 254: _samples(outside)})])
    t = out["teams"]["3128"]
    assert t["opp_side_s_per_match"] == pytest.approx(60.0, abs=0.5)
    assert t["opp_side_frac"] == pytest.approx(0.5, abs=0.01)
    assert t["matches_long"] == 1 and t["by_station"]["blue1"]["n_counted"] == 1
    assert out["teams"]["254"]["opp_side_frac"] == 0.0
    assert out["ranking"][:2] == ["3128", "254"]
    # near_m = 0: only the zone itself counts
    assert _run(Defense(), [_match(1, {3128: _samples(inside)})], near_m=0.0)[
        "teams"]["3128"]["opp_side_frac"] == 0.0


# ----------------------------------------------------------------- bump stuck
def _bump_centre():
    x0, y0, x1, y1 = RULES.bumps[0]
    return (x0 + x1) / 2, (y0 + y1) / 2


def test_bump_episode_over_five_seconds_counts_four_does_not():
    cx, cy = _bump_centre()
    away = [(1.5 + 0.01 * (i % 2), 0.6) for i in range(100)]
    six = _jitter(30, cx, cy, sigma=0.08, seed=6)                 # 6 s
    four = _jitter(20, cx, cy, sigma=0.08, seed=7)                # 4 s
    out = _run(BumpStuck(), [_match(1, {3128: _samples(away + six + away),
                                        254: _samples(away + four + away)})], min_tracked_s=10)
    t = out["teams"]["3128"]
    assert t["episodes"] == 1 and t["matches_with_episode"] == 1
    assert t["stuck_s_total"] == pytest.approx(6.0, abs=0.5)
    ep = t["episode_list"][0]
    assert ep["match_key"] == "2026tst_qm1" and ep["period"] == "teleop"
    assert ep["t_start"] == pytest.approx(20.0 + 100 * DT, abs=0.5)
    assert ep["bump"] == "own right" and ep["bump_field_side"] == "blue"
    assert out["teams"]["254"]["episodes"] == 0
    assert out["ranking"][0] == "3128"


def test_driving_over_a_bump_and_shooting_at_the_barrier_do_not_count():
    cx, cy = _bump_centre()
    x0, _, x1, _ = RULES.bumps[0]
    over = [(2.0 + 1.5 * i * DT, cy) for i in range(30)]         # 1.5 m/s across it
    face = _jitter(100, x0 + 0.05, cy, sigma=0.1, seed=8)         # 20 s on the barrier face
    out = _run(BumpStuck(), [_match(1, {3128: _samples(over + face)})], min_tracked_s=5)
    assert out["teams"]["3128"]["episodes"] == 0


def test_held_positions_on_a_bump_do_not_count():
    cx, cy = _bump_centre()
    frozen = [(cx, cy)] * 40                                      # 8 s identical = stalled
    out = _run(BumpStuck(), [_match(1, {3128: _samples([(1.5, 0.6 + 0.01 * (i % 2))
                                                        for i in range(200)] + frozen)})],
               min_tracked_s=10)
    assert out["teams"]["3128"]["episodes"] == 0


# ----------------------------------------------------------------- ranking / registry
def test_ranking_order_low_data_and_unseen_teams():
    m1 = _match(1, {11: _samples(_circle(600, v=2.5)), 22: _samples(_circle(600, v=1.0)),
                    33: _samples(_jitter(600, 2.0, 2.0))})
    m2 = _match(2, {11: _samples(_circle(600, v=2.5)), 22: _samples(_circle(600, v=1.0))})
    out = _run(Movement(), [m1, m2])
    assert out["ranking"][:3] == ["11", "22", "33"]
    t = out["teams"]
    assert [t[k]["rank"] for k in ("11", "22", "33")] == [1, 2, 3]
    assert not t["11"]["low_data"] and t["33"]["low_data"]       # 1 counted match < 2
    assert t["11"]["n_counted"] == 2
    # lineup-only team (never labelled): listed, unranked, last
    assert t["1"]["rank"] is None and t["1"]["n_counted"] == 0
    assert out["ranking"].index("1") > out["ranking"].index("33")
    # sparse match below min_tracked_s does not count
    sparse = _match(3, {11: _samples(_circle(100, v=2.5))})
    out = _run(Movement(), [m1, m2, sparse])
    assert out["teams"]["11"]["n_counted"] == 2 and out["teams"]["11"]["n_matches"] == 3


def test_registered_for_2026_only_with_bounds():
    keys = {m.key for m in registry.available_metrics(2026)}
    assert {"movement", "cycles", "defense", "bump_stuck"} <= keys
    assert not {"movement", "cycles", "defense", "bump_stuck"} & {
        m.key for m in registry.available_metrics(2025)}
    with pytest.raises(ValueError):
        registry.merged_params(registry.get_metric("movement"), {"move_speed_mps": 50})
    for key in ("movement", "cycles", "defense", "bump_stuck"):
        m = registry.get_metric(key)
        assert set(m.param_bounds) <= set(m.default_params)


def test_empty_event():
    for metric in (Movement(), Cycles(), Defense(), BumpStuck()):
        out = _run(metric, [])
        assert out["teams"] == {} and out["ranking"] == [] and out["legend"]


def test_field_rules_bumps_and_barrier():
    assert len(RULES.bumps) == 4
    assert RULES.barrier_far_x == pytest.approx(5.222, abs=0.01)
    xs = sorted({round(b[0], 3) for b in RULES.bumps})
    assert xs == [pytest.approx(4.062, abs=0.01), pytest.approx(11.352, abs=0.01)]


def test_settled_held_run_counts_as_stationary():
    # tracking froze while the robot sat still: the next real sample is where
    # it was held -> the frozen seconds are stationary observations, not unknown
    pts = _circle(100) + [(9.0, 4.0)] * 16 + [(9.05, 4.02)] + _jitter(50, 9.0, 4.0, sigma=0.05, seed=3)
    tr = motion.clean_track(_samples(pts), "blue", 2026)
    assert tr.n_held_runs == 0 and tr.held_s == 0.0
    assert tr.held_kept_s == pytest.approx(15 * DT)
    assert tr.valid[101:116].all()
    assert (tr.speed[103:114] < 0.5).all()


def test_settled_held_run_on_a_bump_counts():
    cx, cy = _bump_centre()
    approach = [(1.5, 0.6 + 0.01 * (i % 2)) for i in range(200)]
    beached = [(cx, cy)] * 40 + _jitter(10, cx, cy, sigma=0.03, seed=5)   # froze, then still there
    out = _run(BumpStuck(), [_match(1, {3128: _samples(approach + beached)})], min_tracked_s=10)
    assert out["teams"]["3128"]["episodes"] == 1


def test_track_cache_is_bounded():
    from web.scouting import ranking
    md = _match(1, {3128: _samples(_circle(100))})
    tm = md.teams[3128]
    for k in range(10):
        ranking.cached_track(tm, "blue", 2026, {"smooth_s": 0.5 + 0.1 * k})
    assert len(tm.__dict__["_motion_cache"]) == ranking._TRACK_CACHE_MAX
