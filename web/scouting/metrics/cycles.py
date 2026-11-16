"""Neutral <-> alliance-zone cycles (2026): collect in the NEUTRAL ZONE, come home to shoot.

A zone state machine runs over each team's valid samples in match order, on
the smoothed alliance-perspective x (own wall at x = 0):

* "own"     when ax <= alliance_zone_x - zone_margin_m   (inside own ALLIANCE ZONE)
* "neutral" when barrier_far_x + m <= ax <= L - barrier_far_x - m (between the barriers)
* anything else (barrier: hub / bumps / trench, or the opponent's zone) keeps
  the current state.

A new state is only entered once the robot has dwelt in it for ``min_dwell_s``
of valid samples (dwell resets if it goes back to the current state's zone
first), so boundary jitter never flips it. One CYCLE = a counted neutral visit
followed by a counted own-zone visit; it belongs to the period in which the
own visit starts. A tracking gap > ``max_gap_s`` resets the machine (no state)
so a trip is never invented across a long gap.
"""
from __future__ import annotations

import numpy as np

from web.scouting import field_rules, ranking
from web.scouting.geometry import field_dims
from web.scouting.ranking import r
from web.scouting.registry import Metric, register

OWN, NEUTRAL = "own", "neutral"


def run_cycles(tr, own_max: float, neu_lo: float, neu_hi: float, min_dwell: float,
               max_gap: float) -> dict:
    """The zone state machine over one cleaned track. Returns counts + visit lists."""
    out = {"cycles": {"auto": 0, "teleop": 0}, "gaps": 0, "cycle_times": [],
           "visits": {OWN: [], NEUTRAL: []}, "cycle_return_times": []}
    if tr is None:
        return out
    vi = np.flatnonzero(tr.valid)
    state = None
    cand, cand_t0, cand_dwell = None, None, 0.0
    visit_dwell = 0.0            # in-zone seconds of the current counted visit
    last_neutral_t = None        # entry time of the counted neutral visit before now
    prev_t = None
    for i in vi:
        t, ax, dt = float(tr.t[i]), float(tr.ax[i]), float(tr.dt[i])
        if prev_t is not None and t - prev_t > max_gap:
            if state is not None:
                out["visits"][state].append(visit_dwell)
            out["gaps"] += 1
            state, cand, cand_dwell, visit_dwell, last_neutral_t = None, None, 0.0, 0.0, None
        prev_t = t
        if ax <= own_max:
            z = OWN
        elif neu_lo <= ax <= neu_hi:
            z = NEUTRAL
        else:
            z = None
        if z is not None and z == state:
            visit_dwell += dt
            cand, cand_dwell = None, 0.0          # back home: pending switch cancelled
            continue
        if z is None:
            continue                               # barrier / opponent zone: no change
        if cand != z:
            cand, cand_t0, cand_dwell = z, t, 0.0
        cand_dwell += dt
        if cand_dwell + 1e-9 >= min_dwell:
            if state is not None:
                out["visits"][state].append(visit_dwell)
            if state == NEUTRAL and z == OWN:
                period = "teleop" if _period_at(tr, cand_t0) else "auto"
                out["cycles"][period] += 1
                out["cycle_return_times"].append(cand_t0)
            if z == NEUTRAL:
                if last_neutral_t is not None and state == OWN:
                    out["cycle_times"].append(cand_t0 - last_neutral_t)
                last_neutral_t = cand_t0
            state, visit_dwell = z, cand_dwell
            cand, cand_dwell = None, 0.0
    if state is not None:
        out["visits"][state].append(visit_dwell)
    return out


def _period_at(tr, t0: float) -> bool:
    """True when match time t0 is in teleop (first sample at/after t0)."""
    j = int(np.searchsorted(tr.t, t0 - 1e-9))
    return bool(tr.teleop[min(j, len(tr.t) - 1)])


@register
class Cycles(Metric):
    key = "cycles"
    title = "Neutral ↔ alliance-zone cycles"
    version = 1
    years = (2026,)
    description = ("Completed trips neutral zone -> own alliance zone (collect, come back to "
                   "shoot) per match and per tracked teleop minute, from a hysteresis zone "
                   "state machine on smoothed positions.")
    default_params = {**ranking.COMMON_DEFAULTS, "zone_margin_m": 0.3, "min_dwell_s": 1.0,
                      "max_gap_s": 3.0}
    param_bounds = {**ranking.COMMON_BOUNDS, "zone_margin_m": (0.0, 2.0),
                    "min_dwell_s": (0.0, 10.0), "max_gap_s": (0.6, 30.0)}

    def compute(self, matches: list, params: dict) -> dict:
        p = {**self.default_params, **params}
        year = matches[0].year if matches else 2026
        L, _ = field_dims(year)
        rules = field_rules.auto_rules(year)
        az = rules.alliance_zone_x if rules and rules.alliance_zone_x else L / 3
        bf = rules.barrier_far_x if rules and rules.barrier_far_x else az
        m = float(p["zone_margin_m"])
        own_max, neu_lo, neu_hi = az - m, bf + m, L - bf - m
        min_tr = float(p["min_tracked_s"])
        exp = ranking.expected_seconds(year)
        rows: dict = {}
        acc: dict = {}
        for md, team, alliance, station, tr in ranking.team_tracks(matches, p):
            row = ranking.match_head(md, alliance, station)
            tele_s = ranking.valid_seconds(tr, tr.teleop) if tr is not None else 0.0
            counted = tr is not None and tele_s >= min_tr
            res = run_cycles(tr, own_max, neu_lo, neu_hi, float(p["min_dwell_s"]),
                             float(p["max_gap_s"]))
            row.update({"teleop_tracked_s": r(tele_s, 1),
                        "coverage": r(tele_s / exp["teleop"], 3), "counted": counted,
                        "cycles_teleop": res["cycles"]["teleop"],
                        "cycles_auto": res["cycles"]["auto"], "resets": res["gaps"],
                        "cycles_per_min": r(60 * res["cycles"]["teleop"] / tele_s, 2)
                        if tele_s > 0 else None,
                        "cycle_return_times": [r(x, 1) for x in res["cycle_return_times"]]})
            rows.setdefault(team, []).append(row)
            a = acc.setdefault(team, {"tele_s": 0.0, "cyc_t": 0, "cyc_a": 0, "gaps": 0,
                                      "ctimes": [], "own": [], "neutral": []})
            if counted:
                a["tele_s"] += tele_s
                a["cyc_t"] += res["cycles"]["teleop"]
                a["cyc_a"] += res["cycles"]["auto"]
                a["gaps"] += res["gaps"]
                a["ctimes"] += res["cycle_times"]
                a["own"] += res["visits"][OWN]
                a["neutral"] += res["visits"][NEUTRAL]

        teams = {}
        for team, rs in rows.items():
            a = acc[team]
            n = sum(1 for x in rs if x["counted"])
            teams[str(team)] = {
                "team_number": team, "n_matches": len(rs), "n_counted": n,
                "low_data": n < int(p["min_matches"]),
                "cycles_per_min": r(60 * a["cyc_t"] / a["tele_s"], 2) if a["tele_s"] > 0 else None,
                "cycles_teleop_per_match": r(a["cyc_t"] / n, 2) if n else None,
                "cycles_auto_per_match": r(a["cyc_a"] / n, 2) if n else None,
                "cycles_teleop_total": a["cyc_t"], "cycles_auto_total": a["cyc_a"],
                "teleop_tracked_s_per_match": r(a["tele_s"] / n, 1) if n else None,
                "coverage": r(sum(x["teleop_tracked_s"] or 0 for x in rs)
                              / (exp["teleop"] * len(rs)), 3) if rs else None,
                "mean_cycle_s": r(np.mean(a["ctimes"]), 1) if a["ctimes"] else None,
                "median_cycle_s": r(np.median(a["ctimes"]), 1) if a["ctimes"] else None,
                "mean_neutral_visit_s": r(np.mean(a["neutral"]), 1) if a["neutral"] else None,
                "mean_own_visit_s": r(np.mean(a["own"]), 1) if a["own"] else None,
                "resets": a["gaps"],
                "matches": sorted(rs, key=ranking.match_sort_key)}
        order = ranking.assign_ranks(
            teams, lambda t: (t["cycles_per_min"], t["cycles_teleop_per_match"] or 0)
            if t["cycles_per_min"] is not None else None)
        return {
            "teams": teams, "ranking": order, "score_key": "cycles_per_min",
            "zones": {"own_max_x": r(own_max, 3), "neutral_min_x": r(neu_lo, 3),
                      "neutral_max_x": r(neu_hi, 3)},
            "legend": (
                f"Zone state machine on smoothed alliance-perspective x: own = x <= "
                f"{own_max:.2f} m (alliance zone minus {m:g} m), neutral = {neu_lo:.2f}–"
                f"{neu_hi:.2f} m (between the two barriers, {m:g} m in); the barrier "
                f"(hub, bumps, trench) and the opponent's zone change nothing. A zone counts "
                f"after {p['min_dwell_s']:g} s of dwell. One cycle = a counted neutral visit "
                f"then a counted own-zone visit; ranked on teleop cycles per tracked teleop "
                f"minute (pooled over counted matches). A tracking gap > {p['max_gap_s']:g} s "
                f"resets the machine (counted as a reset), so trips across gaps are never "
                f"invented — low-coverage matches undercount. Cycle time = neutral entry to "
                f"the next neutral entry. " + ranking.motion_legend(p))}
