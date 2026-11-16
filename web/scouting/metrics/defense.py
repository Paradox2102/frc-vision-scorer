"""Time on the opponent's side (2026): a defense proxy.

Teleop seconds with the smoothed alliance-perspective x >= (L - alliance_zone_x)
- ``near_m``: inside the opponent's ALLIANCE ZONE or within ``near_m`` of its
edge (the far face of their hub). Held / unknown samples never count. Ranked
on the fraction of tracked teleop time spent there (pooled over counted
matches), with the raw seconds per match alongside.
"""
from __future__ import annotations

from web.scouting import field_rules, ranking
from web.scouting.geometry import field_dims
from web.scouting.ranking import r
from web.scouting.registry import Metric, register


@register
class Defense(Metric):
    key = "defense"
    title = "Time on the opponent's side"
    version = 1
    years = (2026,)
    description = ("Teleop seconds in (or within near_m of) the opponent's alliance zone, "
                   "per match and as a fraction of tracked teleop time.")
    default_params = {**ranking.COMMON_DEFAULTS, "near_m": 1.0, "long_visit_s": 15.0}
    param_bounds = {**ranking.COMMON_BOUNDS, "near_m": (-2.0, 6.0),
                    "long_visit_s": (1.0, 140.0)}

    def compute(self, matches: list, params: dict) -> dict:
        p = {**self.default_params, **params}
        year = matches[0].year if matches else 2026
        L, _ = field_dims(year)
        rules = field_rules.auto_rules(year)
        az = rules.alliance_zone_x if rules and rules.alliance_zone_x else L / 3
        x_min = L - az - float(p["near_m"])
        min_tr, long_s = float(p["min_tracked_s"]), float(p["long_visit_s"])
        exp = ranking.expected_seconds(year)
        rows: dict = {}
        acc: dict = {}
        for md, team, alliance, station, tr in ranking.team_tracks(matches, p):
            row = ranking.match_head(md, alliance, station)
            if tr is not None:
                tele = tr.valid & tr.teleop
                tele_s = float(tr.dt[tele].sum())
                opp = tele & (tr.ax >= x_min)
                opp_s = float(tr.dt[opp].sum())
            else:
                tele_s = opp_s = 0.0
            counted = tr is not None and tele_s >= min_tr
            row.update({"teleop_tracked_s": r(tele_s, 1), "coverage": r(tele_s / exp["teleop"], 3),
                        "counted": counted, "opp_side_s": r(opp_s, 1),
                        "opp_side_frac": r(opp_s / tele_s, 3) if tele_s > 0 else None})
            rows.setdefault(team, []).append(row)
            a = acc.setdefault(team, {"tele": 0.0, "opp": 0.0, "long": 0, "split": {}})
            if counted:
                a["tele"] += tele_s
                a["opp"] += opp_s
                a["long"] += opp_s >= long_s
                k = f"{alliance or '?'}{station or ''}"
                sp = a["split"].setdefault(k, {"n": 0, "tele": 0.0, "opp": 0.0})
                sp["n"] += 1
                sp["tele"] += tele_s
                sp["opp"] += opp_s

        teams = {}
        for team, rs in rows.items():
            a = acc[team]
            n = sum(1 for x in rs if x["counted"])
            frac = a["opp"] / a["tele"] if a["tele"] > 0 else None
            teams[str(team)] = {
                "team_number": team, "n_matches": len(rs), "n_counted": n,
                "low_data": n < int(p["min_matches"]),
                "opp_side_frac": r(frac, 3),
                "opp_side_s_per_match": r(a["opp"] / n, 1) if n else None,
                "opp_side_s_norm": r(frac * exp["teleop"], 1) if frac is not None else None,
                "matches_long": a["long"],
                "teleop_tracked_s_per_match": r(a["tele"] / n, 1) if n else None,
                "coverage": r(sum(x["teleop_tracked_s"] or 0 for x in rs)
                              / (exp["teleop"] * len(rs)), 3) if rs else None,
                "by_station": {k: {"n_counted": v["n"], "opp_side_s_per_match": r(v["opp"] / v["n"], 1),
                                   "opp_side_frac": r(v["opp"] / v["tele"], 3) if v["tele"] > 0 else None}
                               for k, v in sorted(a["split"].items())},
                "matches": sorted(rs, key=ranking.match_sort_key)}
        order = ranking.assign_ranks(
            teams, lambda t: (t["opp_side_frac"], t["opp_side_s_per_match"] or 0)
            if t["opp_side_frac"] is not None else None)
        return {
            "teams": teams, "ranking": order, "score_key": "opp_side_frac",
            "zones": {"opp_min_x": r(x_min, 3)},
            "legend": (
                f"Teleop time with smoothed alliance-perspective x >= {x_min:.2f} m: inside the "
                f"opponent's alliance zone or within {p['near_m']:g} m of it. Ranked on the "
                f"share of tracked teleop time (pooled over counted matches); s/match is raw, "
                f"'per 140 s' = share × 140 s (what a fully tracked match would show). "
                f"Long = matches with >= {long_s:g} s there. Being there is not proof of "
                f"defense (a robot can also feed or get pushed). " + ranking.motion_legend(p))}
