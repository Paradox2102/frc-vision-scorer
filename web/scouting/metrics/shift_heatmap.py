"""Shift heatmaps (2026 REBUILT): where each robot spends time per hub shift.

Every labelled sample adds its ``dt`` seconds to a grid cell (alliance
perspective, own wall at x=0) of its shift bucket — auto, transition, active,
inactive, endgame or unknown (``games/rebuilt2026.bucket`` decides; a hub-auto
tie or missing TBA data leaves the four shifts ``unknown``). Each team gets the
grid for all its matches ("all") and per TBA driver station ("1".."3", only
stations it played), plus a per-bucket summary (time split over the three
zones — own ALLIANCE ZONE, NEUTRAL ZONE, opponent's ALLIANCE ZONE — and mean
position) and coverage = labelled seconds / expected seconds, so a sparse map
reads as sparse rather than as "the robot sat still".
"""
from __future__ import annotations

import math
from collections import defaultdict

from web.field.season import season_for_year
from web.scouting import field_rules
from web.scouting.games import rebuilt2026
from web.scouting.games.rebuilt2026 import BUCKETS, auto_winner, bucket
from web.scouting.geometry import field_dims
from web.scouting.registry import Metric, register

DEFAULT_BUCKETS = ["active", "inactive", "endgame"]
_LEVEL_ORDER = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}
TRANSITION_S, ENDGAME_S, SHIFT_PAIR_S = 10.0, 30.0, 50.0   # 2 shifts x 25 s each


def _expected_seconds(winner, auto_s: float) -> dict:
    """Seconds a robot could be seen in each bucket in one match."""
    known = winner is not None
    return {"auto": auto_s, "transition": TRANSITION_S, "endgame": ENDGAME_S,
            "active": SHIFT_PAIR_S if known else 0.0,
            "inactive": SHIFT_PAIR_S if known else 0.0,
            "unknown": 2 * SHIFT_PAIR_S if not known else 0.0}


class _Acc:
    """Seconds per cell plus the running sums for one (station, bucket)."""
    __slots__ = ("cells", "seconds", "own_s", "opp_s", "sum_ax", "sum_ay", "n")

    def __init__(self):
        self.cells = defaultdict(float)
        self.seconds = self.own_s = self.opp_s = self.sum_ax = self.sum_ay = 0.0
        self.n = 0

    def add(self, ix, iy, ax, ay, dt, zones):
        self.cells[(ix, iy)] += dt
        self.seconds += dt
        self.sum_ax += ax * dt
        self.sum_ay += ay * dt
        own_max, opp_min = zones
        self.own_s += dt if ax <= own_max else 0.0
        self.opp_s += dt if ax >= opp_min else 0.0
        self.n += 1

    def to_json(self) -> dict:
        cells = sorted([ix, iy, round(s, 2)] for (ix, iy), s in self.cells.items()
                       if round(s, 2) > 0)
        if self.n and self.seconds > 0:
            own, opp = self.own_s / self.seconds, self.opp_s / self.seconds
            summary = {"own_zone_frac": round(own, 3),
                       "neutral_zone_frac": round(max(0.0, 1 - own - opp), 3),
                       "opp_zone_frac": round(opp, 3),
                       "mean_ax": round(self.sum_ax / self.seconds, 3),
                       "mean_ay": round(self.sum_ay / self.seconds, 3),
                       "n_samples": self.n}
        else:
            summary = {"own_zone_frac": None, "neutral_zone_frac": None, "opp_zone_frac": None,
                       "mean_ax": None, "mean_ay": None, "n_samples": 0}
        return {"seconds": round(self.seconds, 2), "cells": cells, "summary": summary}


def _match_sort_key(m: dict) -> tuple:
    return (_LEVEL_ORDER.get(m["comp_level"], 9), m["set_number"], m["match_number"],
            m["match_key"])


def _lineup_teams(md) -> set:
    out = {int(t) for a in ("red", "blue") for t in (md.alliances.get(a) or [])
           if t is not None}
    return out | {int(t) for t in md.teams if t is not None}


@register
class ShiftHeatmap(Metric):
    key = "shift_heatmap"
    title = "Shift heatmaps"
    version = 2
    years = (2026,)
    description = ("Where each robot spends time per 2026 hub shift (active / inactive / "
                   "endgame, plus auto, transition, unknown), split by driver station.")
    default_params = {"cell_m": 0.5}
    param_bounds = {"cell_m": (0.1, 5.0)}

    def compute(self, matches: list, params: dict) -> dict:
        cell = float(params.get("cell_m") or self.default_params["cell_m"])
        year = matches[0].year if matches else 2026
        L, W = field_dims(year)
        nx, ny = max(1, math.ceil(L / cell)), max(1, math.ceil(W / cell))
        auto_s = float(season_for_year(year).auto_s)
        rules = field_rules.auto_rules(year)
        own_max = rules.alliance_zone_x if rules and rules.alliance_zone_x else L / 3
        zones = (own_max, L - own_max)      # alliance perspective: own wall at x=0

        accs: dict = defaultdict(lambda: defaultdict(lambda: {b: _Acc() for b in BUCKETS}))
        expected: dict = defaultdict(lambda: dict.fromkeys(BUCKETS, 0.0))
        rows: dict = defaultdict(list)
        winners = {}

        for md in matches:
            winner = auto_winner(md.hub_auto)
            winners[md.match_key] = winner
            exp = _expected_seconds(winner, auto_s)
            for team in sorted(_lineup_teams(md)):
                tm = md.teams.get(team)
                lineup_alliance, lineup_station = md.station_of(team)
                alliance = (tm.alliance if tm else None) or lineup_alliance
                station = (tm.station if tm else None) or lineup_station
                if lineup_alliance is not None:
                    for b in BUCKETS:
                        expected[team][b] += exp[b]
                keys = ["all"] + ([str(station)] if station else [])
                for k in keys:          # a station the team played shows up even if unseen
                    accs[team][k]
                secs = dict.fromkeys(BUCKETS, 0.0)
                for s in (tm.samples if tm else []):
                    b = bucket(s.period, s.remaining, alliance, md.hub_auto)
                    ix = min(max(int(math.floor(s.ax / cell)), 0), nx - 1)
                    iy = min(max(int(math.floor(s.ay / cell)), 0), ny - 1)
                    for k in keys:
                        accs[team][k][b].add(ix, iy, s.ax, s.ay, s.dt, zones)
                    secs[b] += s.dt
                rows[team].append({
                    "match_key": md.match_key, "comp_level": md.comp_level,
                    "match_number": md.match_number, "set_number": md.set_number,
                    "alliance": alliance, "station": station, "auto_winner": winner,
                    "seconds": {b: round(v, 2) for b, v in secs.items()}})

        teams = {}
        for team in sorted(rows):
            by_station = {k: {b: acc.to_json() for b, acc in per.items()}
                          for k, per in sorted(accs[team].items())}
            by_station.setdefault("all", {b: _Acc().to_json() for b in BUCKETS})
            coverage = {}
            for b in BUCKETS:
                exp_s = expected[team][b]
                coverage[b] = (round(by_station["all"][b]["seconds"] / exp_s, 3)
                               if exp_s > 0 else None)
            matches_by_station = {str(n): sum(1 for r in rows[team] if r["station"] == n)
                                  for n in (1, 2, 3)}
            teams[str(team)] = {
                "team_number": team, "n_matches": len(rows[team]),
                "matches": sorted(rows[team], key=_match_sort_key),
                "matches_by_station": matches_by_station,
                "by_station": by_station, "coverage": coverage}

        return {"teams": teams,
                "grid": {"cell_m": cell, "nx": nx, "ny": ny, "field_m": [L, W]},
                "zones": {"own_max_x": round(zones[0], 3), "opp_min_x": round(zones[1], 3)},
                "buckets": list(BUCKETS),
                "default_buckets": list(DEFAULT_BUCKETS),
                "station_layout": {str(k): v for k, v in rebuilt2026.STATION_Y_ORDER.items()},
                "auto_winner_by_match": winners}
