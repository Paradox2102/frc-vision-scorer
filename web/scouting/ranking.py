"""Shared plumbing for the 2026 team-ranking metrics (movement, cycles,
defense, bump_stuck): lineup iteration with cleaned tracks, the per-match row
header, the shared params and the rank assembly.

Coverage rules every ranking follows: a match COUNTS toward a team's averages
only when it has at least ``min_tracked_s`` of valid (non-held) tracking in the
metric's scope; teams with fewer than ``min_matches`` counted matches are
flagged ``low_data`` (still ranked, the page greys them out). Rates and
fractions are pooled over counted matches (sum of numerators / sum of
denominators), so a sparsely tracked team is compared on what was seen.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from web.field.season import season_for_year
from web.scouting import motion

LEVEL_ORDER = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}

COMMON_DEFAULTS = {**motion.DEFAULTS, "min_tracked_s": 30.0, "min_matches": 2}
COMMON_BOUNDS = {**motion.BOUNDS, "min_tracked_s": (0.0, 160.0), "min_matches": (1, 20)}


def match_sort_key(m: dict) -> tuple:
    return (LEVEL_ORDER.get(m["comp_level"], 9), m["set_number"], m["match_number"],
            m["match_key"])


def lineup_teams(md) -> list:
    out = {int(t) for a in ("red", "blue") for t in (md.alliances.get(a) or [])
           if t is not None}
    return sorted(out | {int(t) for t in md.teams if t is not None})


def expected_seconds(year: int) -> dict:
    s = season_for_year(year)
    return {"auto": float(s.auto_s), "teleop": float(s.teleop_s),
            "all": float(s.auto_s) + float(s.teleop_s)}


def team_tracks(matches: list, params: dict):
    """Yield (md, team, alliance, station, Track | None) for every lineup team of
    every match (None = in the lineup but never labelled)."""
    for md in matches:
        for team in lineup_teams(md):
            tm = md.teams.get(team)
            la, ls = md.station_of(team)
            alliance = (tm.alliance if tm else None) or la
            station = (tm.station if tm else None) or ls
            tr = None
            if tm is not None and tm.samples:
                tr = cached_track(tm, alliance or "blue", md.year, params)
            yield md, team, alliance, station, tr


def cached_track(tm, alliance: str, year: int, params: dict):
    """``motion.clean_track`` memoised on the TeamMatch (which lives in the
    registry's event load cache), so the four ranking metrics clean each
    track once per event instead of once each. Tracks are read-only."""
    mp = motion.motion_params(params)
    key = (alliance, int(year), len(tm.samples), tuple(sorted(mp.items())))
    cache = tm.__dict__.setdefault("_motion_cache", {})
    tr = cache.get(key)
    if tr is None:
        tr = cache[key] = motion.clean_track(tm.samples, alliance, year, mp)
    return tr


def match_head(md, alliance, station) -> dict:
    return {"match_key": md.match_key, "comp_level": md.comp_level,
            "set_number": md.set_number, "match_number": md.match_number,
            "alliance": alliance, "station": station}


def valid_seconds(tr, mask: Optional[np.ndarray] = None) -> float:
    if tr is None:
        return 0.0
    m = tr.valid if mask is None else (tr.valid & mask)
    return float(tr.dt[m].sum())


def r(v, nd: int = 2):
    """Round a float for JSON (None / nan -> None)."""
    if v is None:
        return None
    v = float(v)
    return None if not np.isfinite(v) else round(v, nd)


def assign_ranks(teams: dict, score: Callable[[dict], Optional[tuple]]) -> list:
    """Sort team keys by ``score`` (a tuple, larger first; None = unrankable,
    listed last), write ``rank`` (1-based, None when unrankable) and return
    the ordered team keys."""
    def key(k):
        s = score(teams[k])
        return (0, tuple(-x for x in s), int(k)) if s is not None else (1, (), int(k))
    order = sorted(teams, key=key)
    n = 0
    for k in order:
        if score(teams[k]) is None:
            teams[k]["rank"] = None
        else:
            n += 1
            teams[k]["rank"] = n
    return order


MOTION_LEGEND = (
    "Positions are median-smoothed over {smooth_s:g} s; speed = smoothed displacement "
    "across a centred {speed_window_s:g} s window ÷ its duration, never across a "
    "tracking gap > {gap_s:g} s or a tracker-slot change. Runs of bit-identical "
    "positions longer than {held_s:g} s are stalled tracking: if the robot is "
    "still within {held_settle_m:g} m of that spot when tracking resumes it sat "
    "still (counted as stationary), otherwise those seconds are unknown and "
    "excluded. A match counts only with >= {min_tracked_s:g} s of valid "
    "tracking; fewer than {min_matches} counted matches = low data.")


def motion_legend(params: dict) -> str:
    p = {**COMMON_DEFAULTS, **params}
    return MOTION_LEGEND.format(**{k: p[k] for k in ("smooth_s", "speed_window_s", "gap_s",
                                                    "held_s", "held_settle_m", "min_tracked_s",
                                                    "min_matches")})
