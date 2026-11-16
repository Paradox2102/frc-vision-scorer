"""Beached on a bump (2026): robots that sit on a BUMP for a long time.

An episode = a run of valid samples whose smoothed centre lies inside one bump
rect shrunk by ``bump_inset_m`` on every side (so a robot shooting against the
barrier face with ~0.3 m position noise, or brushing the hub / trench-block
edge, is not on it), with in-bump samples never more than ``max_dropout_s``
apart, lasting longer than ``min_stuck_s``. Held (stalled-tracking) positions
are not observations and never count. Driving over a bump takes ~1 s.

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


def find_episodes(tr, rects: list, min_stuck: float, max_dropout: float) -> list:
    """[(bump index, i_first, i_last, seen_s)] for one cleaned track."""
    if tr is None:
        return []
    which = np.full(len(tr), -1, dtype=int)
    v = tr.valid & ~np.isnan(tr.x)
    for k, (x0, y0, x1, y1) in enumerate(rects):
        inside = v & (tr.x > x0) & (tr.x < x1) & (tr.y > y0) & (tr.y < y1)
        which[inside] = k
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
    return [e for e in out
            if tr.t[e[2]] - tr.t[e[1]] + tr.dt[e[2]] > min_stuck + 1e-9]


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
    version = 1
    years = (2026,)
    description = ("Episodes of a robot's centre staying on a bump (inset from its edges) "
                   "for more than min_stuck_s, with match / time / bump for video checks.")
    default_params = {**ranking.COMMON_DEFAULTS, "bump_inset_m": 0.2, "min_stuck_s": 5.0,
                      "max_dropout_s": 1.0}
    param_bounds = {**ranking.COMMON_BOUNDS, "bump_inset_m": (0.0, 0.5),
                    "min_stuck_s": (0.5, 60.0), "max_dropout_s": (0.2, 5.0)}

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
            found = find_episodes(tr, rects, float(p["min_stuck_s"]), float(p["max_dropout_s"]))
            m_eps = []
            for k, i0, i1, seen in found:
                label, field_side = bump_label(rules.bumps[k], alliance, year)
                t0 = float(tr.t[i0])
                m_eps.append({**ranking.match_head(md, alliance, station),
                              "period": "teleop" if tr.teleop[i0] else "auto",
                              "t_start": r(t0, 1), "remaining_at_start": r(_remaining(md, team, t0), 1),
                              "duration_s": r(float(tr.t[i1] - tr.t[i0] + tr.dt[i1]), 1),
                              "seen_s": r(seen, 1), "bump": label, "bump_field_side": field_side,
                              "counted": counted})
            row.update({"tracked_s": r(tracked, 1), "coverage": r(tracked / exp["all"], 3),
                        "counted": counted, "n_episodes": len(m_eps),
                        "stuck_s": r(sum(e["duration_s"] for e in m_eps), 1)})
            rows.setdefault(team, []).append(row)
            eps.setdefault(team, []).extend(m_eps)

        teams = {}
        for team, rs in rows.items():
            n = sum(1 for x in rs if x["counted"])
            te = sorted(eps[team], key=lambda e: (ranking.match_sort_key(e), e["t_start"]))
            ce = [e for e in te if e["counted"]]
            total = sum(e["duration_s"] for e in ce)
            teams[str(team)] = {
                "team_number": team, "n_matches": len(rs), "n_counted": n,
                "low_data": n < int(p["min_matches"]),
                "stuck_s_total": r(total, 1), "episodes": len(ce),
                "matches_with_episode": len({e["match_key"] for e in ce}),
                "longest_s": r(max((e["duration_s"] for e in ce), default=0.0), 1),
                "stuck_s_per_match": r(total / n, 1) if n else None,
                "tracked_s_per_match": r(sum(x["tracked_s"] or 0 for x in rs if x["counted"]) / n, 1)
                if n else None,
                "coverage": r(sum(x["tracked_s"] or 0 for x in rs) / (exp["all"] * len(rs)), 3)
                if rs else None,
                "episode_list": te,
                "matches": sorted(rs, key=ranking.match_sort_key)}
        order = ranking.assign_ranks(
            teams, lambda t: (t["stuck_s_total"], t["episodes"])
            if t["n_counted"] else None)
        return {
            "teams": teams, "ranking": order, "score_key": "stuck_s_total",
            "bumps": [list(b) for b in (rules.bumps if rules else ())],
            "legend": (
                f"Episode = smoothed robot centre inside a bump shrunk by {inset:g} m on every "
                f"side (so shooting against the barrier face or brushing the hub / trench "
                f"block does not count) for > {p['min_stuck_s']:g} s, gaps < "
                f"{p['max_dropout_s']:g} s allowed inside. Driving over takes ~1 s. Totals use "
                f"counted matches only; episodes from uncounted (sparse) matches are still "
                f"listed. A robot can also park on a bump on purpose — check the video. "
                + ranking.motion_legend(p))}
