"""Time moving (2026): how much of the match each robot spends driving, and how fast.

Per sample (``motion.clean_track``): smoothed speed >= ``move_speed_mps`` =
moving, below = stationary; samples with no defined speed (held positions,
segment edges, gaps) are neither. Per team, over counted matches:

* moving / stationary seconds per match (raw) and ``moving_frac`` = moving ÷
  (moving + stationary), pooled;
* median / p90 speed while moving (pooled samples), typical top speed =
  median over matches of the highest speed held for ``sustain_s``;
* RANK SCORE ``avg_speed_mps`` = average speed over all speed-known time =
  moving_frac × mean speed while moving (stationary time counts as 0). Faster
  when moving and less time stopped both raise it.

Whole match (auto + teleop) by default, with the teleop-only numbers alongside.
"""
from __future__ import annotations

import numpy as np

from web.scouting import motion, ranking
from web.scouting.ranking import r
from web.scouting.registry import Metric, register


class _Pool:
    """Pooled moving / stationary seconds and moving speeds of one team + scope."""
    __slots__ = ("moving_s", "stationary_s", "speed_dt", "speeds")

    def __init__(self):
        self.moving_s = self.stationary_s = self.speed_dt = 0.0
        self.speeds = []

    def add(self, tr, mask, thr):
        known = mask & ~np.isnan(tr.speed)
        mv = known & (tr.speed >= thr)
        st = known & (tr.speed < thr)
        m_s, s_s = float(tr.dt[mv].sum()), float(tr.dt[st].sum())
        self.moving_s += m_s
        self.stationary_s += s_s
        self.speed_dt += float((tr.speed[mv] * tr.dt[mv]).sum())
        self.speeds.append(tr.speed[mv])
        return m_s, s_s, float((tr.speed[mv] * tr.dt[mv]).sum())

    def summary(self, n_counted: int) -> dict:
        known = self.moving_s + self.stationary_s
        sp = np.concatenate(self.speeds) if self.speeds else np.zeros(0)
        frac = self.moving_s / known if known > 0 else None
        mean_mv = self.speed_dt / self.moving_s if self.moving_s > 0 else None
        return {"moving_s_per_match": r(self.moving_s / n_counted) if n_counted else None,
                "stationary_s_per_match": r(self.stationary_s / n_counted) if n_counted else None,
                "moving_frac": r(frac, 3),
                "mean_moving_mps": r(mean_mv),
                "median_moving_mps": r(np.median(sp)) if len(sp) else None,
                "p90_moving_mps": r(np.percentile(sp, 90)) if len(sp) else None,
                "avg_speed_mps": r(self.speed_dt / known, 3) if known > 0 else None,
                "speed_known_s": r(known, 1)}


@register
class Movement(Metric):
    key = "movement"
    title = "Time moving"
    version = 1
    years = (2026,)
    description = ("Seconds moving vs stationary per match, speed while moving and the "
                   "average speed over all tracked time (rank score), from smoothed positions.")
    default_params = {**ranking.COMMON_DEFAULTS, "move_speed_mps": 0.5, "sustain_s": 1.5}
    param_bounds = {**ranking.COMMON_BOUNDS, "move_speed_mps": (0.05, 3.0),
                    "sustain_s": (0.2, 10.0)}

    def compute(self, matches: list, params: dict) -> dict:
        p = {**self.default_params, **params}
        thr, min_tr = float(p["move_speed_mps"]), float(p["min_tracked_s"])
        year = matches[0].year if matches else 2026
        exp = ranking.expected_seconds(year)
        rows: dict = {}
        pools: dict = {}
        tops: dict = {}
        stats = {"held_s": 0.0, "held_kept_s": 0.0, "labelled_s": 0.0, "implausible_samples": 0}
        for md, team, alliance, station, tr in ranking.team_tracks(matches, p):
            row = ranking.match_head(md, alliance, station)
            pool = pools.setdefault(team, {"all": _Pool(), "teleop": _Pool()})
            tracked = ranking.valid_seconds(tr)
            row.update({"tracked_s": r(tracked, 1), "coverage": r(tracked / exp["all"], 3),
                        "held_s": r(tr.held_s if tr else 0.0, 1)})
            counted = tr is not None and tracked >= min_tr
            row["counted"] = counted
            if tr is not None:
                stats["held_s"] += tr.held_s
                stats["held_kept_s"] += tr.held_kept_s
                stats["labelled_s"] += float(tr.dt.sum())
                stats["implausible_samples"] += tr.n_implausible
            if counted:
                for scope, mask in (("all", tr.valid), ("teleop", tr.valid & tr.teleop)):
                    m_s, s_s, sdt = pool[scope].add(tr, mask, thr)
                    known = m_s + s_s
                    blk = {"moving_s": r(m_s, 1), "stationary_s": r(s_s, 1),
                           "moving_frac": r(m_s / known, 3) if known > 0 else None,
                           "avg_speed_mps": r(sdt / known, 3) if known > 0 else None}
                    if scope == "all":
                        row.update(blk)
                    else:
                        row["teleop"] = {**blk, "tracked_s": r(ranking.valid_seconds(tr, tr.teleop), 1)}
                top = motion.sustained_max(tr, float(p["sustain_s"]))
                row["max_sustained_mps"] = r(top)
                if top is not None:
                    tops.setdefault(team, []).append(top)
            rows.setdefault(team, []).append(row)

        teams = {}
        for team, rs in rows.items():
            counted = [x for x in rs if x["counted"]]
            n = len(counted)
            tracked = sum(x["tracked_s"] or 0 for x in counted)
            all_tracked = sum(x["tracked_s"] or 0 for x in rs)
            t_tops = tops.get(team, [])
            teams[str(team)] = {
                "team_number": team, "n_matches": len(rs), "n_counted": n,
                "low_data": n < int(p["min_matches"]),
                "tracked_s_per_match": r(tracked / n, 1) if n else None,
                "coverage": r(all_tracked / (exp["all"] * len(rs)), 3) if rs else None,
                **pools[team]["all"].summary(n),
                "max_sustained_mps": r(float(np.median(t_tops))) if t_tops else None,
                "max_sustained_best_mps": r(max(t_tops)) if t_tops else None,
                "teleop": pools[team]["teleop"].summary(n),
                "matches": sorted(rs, key=ranking.match_sort_key)}
        ranking_order = ranking.assign_ranks(
            teams, lambda t: (t["avg_speed_mps"],) if t["avg_speed_mps"] is not None else None)
        held = stats["held_s"]
        lab = stats["labelled_s"]
        return {
            "teams": teams, "ranking": ranking_order,
            "score_key": "avg_speed_mps",
            "event_stats": {"labelled_s": r(lab, 1), "held_s": r(held, 1),
                            "held_frac": r(held / lab, 3) if lab else None,
                            "held_kept_s": r(stats["held_kept_s"], 1),
                            "implausible_speed_samples": stats["implausible_samples"]},
            "legend": (
                f"Moving = smoothed speed >= {thr:g} m/s (a parked robot's smoothed speed "
                f"reads below this ~98% of the time). Moving fraction = moving ÷ (moving + "
                f"stationary) seconds, pooled over counted matches; seconds per match are "
                f"raw (they shrink with low coverage, the fraction does not). Rank score = "
                f"average speed over all tracked time = moving fraction × mean speed while "
                f"moving: faster when moving and less time stopped both raise it. Top speed "
                f"= median over matches of the highest speed held for {p['sustain_s']:g} s. "
                + ranking.motion_legend(p))}
