"""Beached on a bump (2026): time a robot spends STOPPED on a BUMP.

Score = seconds per counted match with the robot's smoothed centre on any of
the four bumps AND its smoothed speed below ``stop_speed_mps`` (driving over
a bump at speed does not count), summed over the whole match — not one
contiguous stretch. A beached robot's position wobbles in and out of the bump
outline (it is raised and tilted), so a "> 5 s unbroken, edges trimmed" rule
missed most beachings: validated against Manta (3128) scout notes on 2026cur
(702 robot-matches, 38 noted "beached"), per-team Spearman with the beached
note rate was +0.26 for that rule vs +0.42 for this one (edge trim 0.1 m:
+0.36, 0.2 m: +0.34 — so ``bump_inset_m`` defaults to 0). Held positions that
ended elsewhere are unknown and never count; settled holds (the robot was
still there when tracking resumed) do.

Episodes are for checking the video only (they do not drive the ranking):
runs of stopped-on-one-bump samples no more than ``max_dropout_s`` apart with
at least ``min_episode_s`` stopped seconds.

Bumps are tested in absolute field coordinates (all four count) and labelled
from the team's side: own / opponent half, and left / right as seen from its
own driver station.
"""
from __future__ import annotations

import numpy as np

from web.scouting import field_rules, ranking
from web.scouting.geometry import field_dims, to_alliance
from web.scouting.ranking import r
from web.scouting.registry import Metric, register


def bump_label(rect, alliance: str, year: int) -> tuple:
    """(label, field_side) of an absolute bump rect for a team on ``alliance``."""
    L, W = field_dims(year)
    cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    ax, ay = to_alliance(cx, cy, alliance or "blue", year)
    side = "own" if ax < L / 2 else "opponent"
    lr = "left" if ay > W / 2 else "right"          # driver faces +x: +y is on the left
    return f"{side} {lr}", ("blue" if cx < L / 2 else "red")


def stopped_on_bump(tr, rects: list, stop_speed: float) -> np.ndarray:
    """Per sample: index of the bump the robot is stopped on, -1 otherwise."""
    which = np.full(len(tr), -1, dtype=int)
    v = tr.valid & ~np.isnan(tr.x) & ~np.isnan(tr.speed) & (tr.speed < stop_speed)
    for k, (x0, y0, x1, y1) in enumerate(rects):
        inside = v & (tr.x > x0) & (tr.x < x1) & (tr.y > y0) & (tr.y < y1)
        which[inside] = k
    return which


def find_episodes(tr, which: np.ndarray, min_seen: float, max_dropout: float) -> list:
    """[(bump index, i_first, i_last, stopped_s)] runs of stopped-on-one-bump
    samples no more than max_dropout apart with >= min_seen stopped seconds."""
    if tr is None:
        return []
    out = []
    cur = None                                     # [bump, first, last, seen]
    for i in np.flatnonzero(which >= 0):
        k = int(which[i])
        if cur is not None and cur[0] == k and tr.t[i] - tr.t[cur[2]] < max_dropout:
            cur[2] = i
            cur[3] += float(tr.dt[i])
            continue
        if cur is not None:
            out.append(tuple(cur))
        cur = [k, i, i, float(tr.dt[i])]
    if cur is not None:
        out.append(tuple(cur))
    return [e for e in out if e[3] >= min_seen - 1e-9]


def _remaining(md, team, t_match: float):
    """On-screen countdown at match time t (for finding the moment in the video)."""
    tm = md.teams.get(team)
    if tm is None or not tm.samples:
        return None
    best = min(tm.samples, key=lambda s: abs(s.t_match - t_match))
    return best.remaining


@register
class BumpStuck(Metric):
    key = "bump_stuck"
    title = "Stuck on a bump"
    version = 2
    years = (2026,)
    description = ("Seconds per match a robot spends stopped on a bump (beached), plus "
                   "episodes with match / time / bump for video checks.")
    default_params = {**ranking.COMMON_DEFAULTS, "bump_inset_m": 0.0, "stop_speed_mps": 0.5,
                      "min_episode_s": 2.0, "max_dropout_s": 2.0}
    param_bounds = {**ranking.COMMON_BOUNDS, "bump_inset_m": (0.0, 0.5),
                    "stop_speed_mps": (0.1, 2.0), "min_episode_s": (0.5, 60.0),
                    "max_dropout_s": (0.2, 5.0)}

    def compute(self, matches: list, params: dict) -> dict:
        p = {**self.default_params, **params}
        year = matches[0].year if matches else 2026
        rules = field_rules.auto_rules(year)
        inset = float(p["bump_inset_m"])
        rects = [(x0 + inset, y0 + inset, x1 - inset, y1 - inset)
                 for x0, y0, x1, y1 in (rules.bumps if rules else ())]
        rects = [q for q in rects if q[2] > q[0] and q[3] > q[1]]
        min_tr = float(p["min_tracked_s"])
        exp = ranking.expected_seconds(year)
        rows: dict = {}
        eps: dict = {}
        for md, team, alliance, station, tr in ranking.team_tracks(matches, p):
            row = ranking.match_head(md, alliance, station)
            tracked = ranking.valid_seconds(tr)
            counted = tr is not None and tracked >= min_tr
            which = (stopped_on_bump(tr, rects, float(p["stop_speed_mps"]))
                     if tr is not None else None)
            stuck = float(tr.dt[which >= 0].sum()) if tr is not None else 0.0
            found = (find_episodes(tr, which, float(p["min_episode_s"]), float(p["max_dropout_s"]))
                     if tr is not None else [])
            m_eps = []
            for k, i0, i1, seen in found:
                label, field_side = bump_label(rules.bumps[k], alliance, year)
                t0 = float(tr.t[i0])
                m_eps.append({**ranking.match_head(md, alliance, station),
                              "period": "teleop" if tr.teleop[i0] else "auto",
                              "t_start": r(t0, 1), "remaining_at_start": r(_remaining(md, team, t0), 1),
                              "duration_s": r(float(tr.t[i1] - tr.t[i0] + tr.dt[i1]), 1),
                              "stopped_s": r(seen, 1), "bump": label, "bump_field_side": field_side,
                              "counted": counted})
            row.update({"tracked_s": r(tracked, 1), "coverage": r(tracked / exp["all"], 3),
                        "counted": counted, "n_episodes": len(m_eps),
                        "stuck_s": r(stuck, 1)})
            rows.setdefault(team, []).append(row)
            eps.setdefault(team, []).extend(m_eps)

        teams = {}
        for team, rs in rows.items():
            n = sum(1 for x in rs if x["counted"])
            te = sorted(eps[team], key=lambda e: (ranking.match_sort_key(e), e["t_start"]))
            ce = [e for e in te if e["counted"]]
            total = sum(x["stuck_s"] or 0.0 for x in rs if x["counted"])
            teams[str(team)] = {
                "team_number": team, "n_matches": len(rs), "n_counted": n,
                "low_data": n < int(p["min_matches"]),
                "stuck_s_total": r(total, 1), "episodes": len(ce),
                "matches_with_episode": len({e["match_key"] for e in ce}),
                "longest_s": r(max((e["stopped_s"] for e in ce), default=0.0), 1),
                "matches_stuck": sum(1 for x in rs if x["counted"] and (x["stuck_s"] or 0) >= 1.0),
                "stuck_s_per_match": r(total / n, 1) if n else None,
                "tracked_s_per_match": r(sum(x["tracked_s"] or 0 for x in rs if x["counted"]) / n, 1)
                if n else None,
                "coverage": r(sum(x["tracked_s"] or 0 for x in rs) / (exp["all"] * len(rs)), 3)
                if rs else None,
                "episode_list": te,
                "matches": sorted(rs, key=ranking.match_sort_key)}
        order = ranking.assign_ranks(
            teams, lambda t: (t["stuck_s_per_match"], t["stuck_s_total"])
            if t["n_counted"] else None)
        return {
            "teams": teams, "ranking": order, "score_key": "stuck_s_per_match",
            "bumps": [list(b) for b in (rules.bumps if rules else ())],
            "legend": (
                f"Rank score = seconds per counted match the robot is STOPPED (smoothed speed "
                f"< {p['stop_speed_mps']:g} m/s) with its centre on any bump, added up over the "
                f"whole match (a beached robot's position wobbles on and off the bump, so it "
                f"need not be one unbroken stretch); driving over at speed does not count. "
                f"Checked against Manta scout notes on 2026cur: per-team rank correlation "
                f"+0.42 with how often scouts wrote 'beached'. Episodes (for finding it in the "
                f"video) = stretches with >= {p['min_episode_s']:g} s stopped on one bump, gaps "
                f"< {p['max_dropout_s']:g} s allowed. A robot can also park on a bump on "
                f"purpose — check the video. " + ranking.motion_legend(p))}
