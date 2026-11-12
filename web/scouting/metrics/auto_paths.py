"""Auto paths: each team's autonomous path per match, grouped into routines.

Every path is in ALLIANCE PERSPECTIVE (``Sample.ax/ay``: own wall at x=0), so a
team's red and blue matches overlay.

Similarity measure
------------------
1. Each path is resampled onto a fixed time grid (``grid_s``, default 0.25 s)
   over the auto window ``0..auto_s``. Between two consecutive samples the
   position is linearly interpolated only when they are ≤ ``max_gap_s`` apart;
   every other grid point (long gaps, before the first / after the last
   sample) is NaN = "not seen".
2. The distance between paths a and b is the mean Euclidean distance between
   a(t) and b(t + s) over the grid points where both are valid, minimised over
   time shifts s in ``[-max_shift_s, +max_shift_s]`` (a late start still
   matches). A shift only counts when the overlap is at least
   ``max(min_overlap_s, min_overlap_frac × the shorter path's valid time)``;
   with no qualifying shift the pair is incomparable (distance = inf). A
   truncated path (tracking lost mid-auto) therefore still compares to a full
   one over the part both saw.
3. Paths with coverage ≥ ``min_coverage`` are grouped per team with
   average-linkage agglomerative clustering cut at ``threshold_m``.
   Incomparable pairs never merge (inf is stood in by a huge finite value, so
   any cluster containing such a pair averages far above the threshold).
4. MIRROR-AWARE (``mirror_aware``, default on): teams commonly run one routine
   from either side of the field (left / right trench). Each pair is also
   compared with one path reflected across the field's long centre line
   (ay -> W - ay) and the smaller distance counts. Each clustered match gets
   ``mirrored`` = whether it is the reflection of its cluster's representative;
   cluster start/end are computed in the representative's orientation.
"""
from __future__ import annotations

import math
from statistics import median
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from web.field.season import season_for_year
from web.scouting import geometry
from web.scouting.model import MatchData, Sample
from web.scouting.registry import Metric, register

_LEVEL_ORDER = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}
_BIG = 1e6                 # finite stand-in for an incomparable (inf) distance
_CONFLICT_SHARE = 0.25     # runner-up label share that flags an identity conflict


def match_sort_key(md: MatchData) -> tuple:
    return (_LEVEL_ORDER.get(md.comp_level, 9), md.set_number, md.match_number)


def resample(path: list[Sample], auto_s: float, grid_s: float, max_gap_s: float) -> np.ndarray:
    """(n_grid, 2) array of (ax, ay) on t = 0, grid_s, .. auto_s; NaN where unseen."""
    n = int(round(auto_s / grid_s)) + 1
    out = np.full((n, 2), np.nan)
    if not path:
        return out
    pts = sorted(path, key=lambda s: s.t_match)
    ts, xs, ys = [], [], []
    for s in pts:                        # drop duplicate timestamps (keep the first)
        if ts and s.t_match <= ts[-1]:
            continue
        ts.append(float(s.t_match)); xs.append(float(s.ax)); ys.append(float(s.ay))
    t = np.asarray(ts)
    grid = np.arange(n) * grid_s
    eps = 1e-9
    for i, g in enumerate(grid):
        if g < t[0] - eps or g > t[-1] + eps:
            continue
        j = int(np.searchsorted(t, g + eps, side="right")) - 1   # t[j] <= g
        if abs(t[j] - g) <= eps:
            out[i] = (xs[j], ys[j])
            continue
        if j + 1 >= len(t):
            continue
        gap = t[j + 1] - t[j]
        if gap > max_gap_s + eps:
            continue
        w = (g - t[j]) / gap
        out[i] = (xs[j] + w * (xs[j + 1] - xs[j]), ys[j] + w * (ys[j + 1] - ys[j]))
    return out


def _valid(grid: np.ndarray) -> np.ndarray:
    return ~np.isnan(grid[:, 0])


def path_distance(a: np.ndarray, b: np.ndarray, grid_s: float, max_shift_s: float,
                  min_overlap_frac: float, min_overlap_s: float) -> float:
    """Mean point distance of a(t) vs b(t+s), minimised over shifts s; inf if
    no shift has enough overlap."""
    va, vb = _valid(a), _valid(b)
    shorter_s = min(int(va.sum()), int(vb.sum())) * grid_s
    need = max(min_overlap_s, min_overlap_frac * shorter_s)
    n = len(a)
    k_max = int(round(max_shift_s / grid_s))
    best = math.inf
    for k in range(-k_max, k_max + 1):
        # pair a[i] with b[i + k]
        lo, hi = max(0, -k), min(n, n - k)
        if hi <= lo:
            continue
        both = va[lo:hi] & vb[lo + k:hi + k]
        cnt = int(both.sum())
        if cnt == 0 or cnt * grid_s < need - 1e-9:
            continue
        d = np.linalg.norm(a[lo:hi][both] - b[lo + k:hi + k][both], axis=1).mean()
        best = min(best, float(d))
    return best


def _reflect(grid: np.ndarray, width: float) -> np.ndarray:
    """The path reflected across the field's long centre line (ay -> W - ay)."""
    out = grid.copy()
    out[:, 1] = width - out[:, 1]
    return out


def cluster_labels(dist: np.ndarray, threshold_m: float) -> list:
    """Average-linkage cluster ids (arbitrary 1-based labels) for a square
    distance matrix that may contain inf."""
    n = len(dist)
    if n == 0:
        return []
    if n == 1:
        return [1]
    d = np.where(np.isfinite(dist), dist, _BIG)
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d, checks=False), method="average")
    return [int(c) for c in fcluster(z, t=threshold_m, criterion="distance")]


def _r(v: Optional[float], nd: int) -> Optional[float]:
    return None if v is None else round(float(v), nd)


def _median_xy(points: list) -> list:
    return [round(float(median(float(p[0]) for p in points)), 2),
            round(float(median(float(p[1]) for p in points)), 2)]


def _span(grid: np.ndarray) -> tuple:
    """(first valid idx, last valid idx) or (None, None)."""
    idx = np.flatnonzero(_valid(grid))
    if len(idx) == 0:
        return None, None
    return int(idx[0]), int(idx[-1])


@register
class AutoPaths(Metric):
    key = "auto_paths"
    title = "Auto paths"
    version = 1
    years = None
    description = ("Each team's autonomous path per match (alliance perspective), "
                   "with repeated routines grouped into clusters.")
    default_params = {"threshold_m": 0.9, "grid_s": 0.25, "max_gap_s": 1.5,
                      "max_shift_s": 1.0, "min_overlap_frac": 0.4,
                      "min_overlap_s": 3.0, "min_coverage": 0.25, "mirror_aware": True}

    def compute(self, matches: list, params: dict) -> dict:
        p = {**self.default_params, **(params or {})}
        grid_s = float(p["grid_s"])
        ordered = sorted(matches, key=match_sort_key)

        teams: dict = {}
        for md in ordered:
            lineup = {int(t) for row in (md.alliances or {}).values() for t in (row or [])
                      if t is not None}
            for team in sorted(lineup | {int(t) for t in md.teams}):
                teams.setdefault(team, []).append(md)

        out = {str(team): self._team(team, mds, p, grid_s) for team, mds in sorted(teams.items())}
        return {"teams": out, "legend": {"perspective": "own alliance wall at x=0"}}

    def _team(self, team: int, mds: list, p: dict, grid_s: float) -> dict:
        entries: list = []
        grids: list = []            # (entry index, grid) for clusterable paths
        for md in mds:
            tm = md.teams.get(team)
            path: list = list(tm.auto_path) if tm and tm.auto_path else []
            if tm is not None:
                alliance, station = tm.alliance, tm.station
            else:
                alliance, station = md.station_of(team)
            auto_s = float(season_for_year(md.year).auto_s)
            grid = resample(path, auto_s, grid_s, float(p["max_gap_s"]))
            coverage = float(_valid(grid).mean()) if path else 0.0
            path.sort(key=lambda s: s.t_match)
            runner = tm.auto_runner_up_share if tm and path else None
            entries.append({
                "match_key": md.match_key, "comp_level": md.comp_level,
                "match_number": md.match_number, "set_number": md.set_number,
                "alliance": alliance, "station": station,
                "source": tm.auto_source if tm and path else None,
                "coverage": round(coverage, 3),
                "label_share": _r(tm.auto_label_share, 3) if tm and path else None,
                "runner_up_share": _r(runner, 3),
                "identity_conflict": runner is not None and runner >= _CONFLICT_SHARE,
                "legacy_labels": md.legacy_labels,
                "n_samples": len(path),
                "cluster": None,
                "mirrored": None,
                "points": [[round(s.t_match, 2), round(s.ax, 2), round(s.ay, 2)] for s in path],
                "points_norm": [list(geometry.normalise(s.ax, s.ay, md.year)) for s in path],
            })
            if path and coverage >= float(p["min_coverage"]):
                grids.append((len(entries) - 1, grid))

        width = geometry.field_dims(mds[0].year)[1] if mds else 0.0
        clusters = self._clusters(entries, grids, p, grid_s, width)
        return {"team_number": team, "n_matches": len(entries),
                "n_with_path": sum(1 for e in entries if e["n_samples"]),
                "matches": entries, "clusters": clusters}

    def _clusters(self, entries: list, grids: list, p: dict, grid_s: float,
                  width: float) -> list:
        n = len(grids)
        if n == 0:
            return []
        args = (grid_s, float(p["max_shift_s"]), float(p["min_overlap_frac"]),
                float(p["min_overlap_s"]))
        dist = np.zeros((n, n))
        flip = np.zeros((n, n), dtype=bool)      # pair matched best with one reflected
        for i in range(n):
            for j in range(i + 1, n):
                d = path_distance(grids[i][1], grids[j][1], *args)
                if p.get("mirror_aware"):
                    dm = path_distance(grids[i][1], _reflect(grids[j][1], width), *args)
                    if dm < d:
                        d = dm
                        flip[i, j] = flip[j, i] = True
                dist[i, j] = dist[j, i] = d
        labels = cluster_labels(dist, float(p["threshold_m"]))
        groups: dict = {}
        for i, lab in enumerate(labels):          # i follows match order
            groups.setdefault(lab, []).append(i)
        ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))

        out = []
        for cid, members in enumerate(ordered):
            sub = dist[np.ix_(members, members)]
            finite_sub = np.where(np.isfinite(sub), sub, _BIG)
            medoid = members[int(np.argmin(finite_sub.sum(axis=1)))]   # ties -> earliest
            pairs = [sub[a, b] for a in range(len(members)) for b in range(a + 1, len(members))
                     if math.isfinite(sub[a, b])]
            starts, ends, spans = [], [], []
            for m in members:
                mirrored = bool(flip[medoid, m])
                g = _reflect(grids[m][1], width) if mirrored else grids[m][1]
                first, last = _span(g)
                starts.append(g[first]); ends.append(g[last])
                spans.append((last - first) * grid_s)
                entries[grids[m][0]]["cluster"] = cid
                entries[grids[m][0]]["mirrored"] = mirrored
            out.append({
                "id": cid, "count": len(members),
                "members": [entries[grids[m][0]]["match_key"] for m in members],
                "representative": entries[grids[medoid][0]]["match_key"],
                "mean_spread_m": round(float(np.mean(pairs)), 2) if pairs else 0.0,
                "n_mirrored": int(sum(bool(flip[medoid, m]) for m in members)),
                "start": _median_xy(starts),
                "end": _median_xy(ends),
                "duration_s": round(float(median(spans)), 2),
            })
        return out
