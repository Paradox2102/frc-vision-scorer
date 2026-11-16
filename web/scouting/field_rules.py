"""Physical field rules for auto paths (2026 REBUILT): start line + obstacles.

Vision auto paths break two rules no real robot can:

1. START LINE. Every robot starts with part of itself touching its ROBOT
   STARTING LINE: in alliance perspective the full-width tape at x ≈ 4.00 m
   (tape centre, 1" on the own-wall side of the hub cabinet's near face).
2. OBSTACLES. Robots cannot drive through the hub cabinet base or the trench
   support blocks between each bump's outer end and the trench opening (the
   trench lane itself is drivable: robots pass UNDER the arm; the bumps are
   drivable too). The field is rotationally symmetric, so the same six
   absolute rectangles apply in alliance perspective for both alliances.

Measured on 2026hop (39 paths): 24 started > 0.8 m from the line, 8 had points
inside an obstacle, 15 had a straight segment through one. Causes: straight
lines drawn across coverage gaps (qm84 1287 "cuts" through the hub after going
over the bump unseen), 0.3–0.5 m position noise inside the thin trench blocks,
first samples at t ≈ 0.5 s after the robot already moved, paths that are
another robot entirely (qm115 4143 starts at the red wall) and paths first
seen late (qm45 401 at 13 s — that first point is not a start).

``constrain_auto_path`` therefore (1) clamps points out of the obstacles
inflated by half a robot width, (2) checks the start against the line and,
when the first point is reachable from the line in the time elapsed, prepends
an inferred start on it, and (3) routes every segment that crosses an inflated
obstacle around it (shortest path over a visibility graph of the inflated
corners). Every point carries a kind: seen / clamped / detour / start.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

TAPE_OFFSET_M = 0.0254         # start-line tape centre: 1" own-wall side of the hub face
START_V_MAX = 4.5              # m/s, fastest plausible auto robot
START_A_MAX = 10.0             # m/s^2
_CORNER_EPS = 1e-3             # detour corners pushed this far off the inflated rect
_SHRINK_EPS = 1e-6             # rect shrink for the segment-entry test

DEFAULTS = {"obstacle_margin_m": 0.25, "start_tol_m": 0.8, "start_window_s": 1.5,
            "start_noise_m": 0.5, "start_v_max": START_V_MAX, "start_a_max": START_A_MAX}

SEEN, CLAMPED, DETOUR, START = "seen", "clamped", "detour", "start"
KIND_CODE = {SEEN: "s", CLAMPED: "c", DETOUR: "d", START: "i"}


@dataclass(frozen=True)
class AutoRules:
    start_line_x: float               # alliance-perspective x of the starting line
    obstacles: tuple                  # ((x0, y0, x1, y1), ...) absolute field metres
    field_m: tuple                    # (L, W)
    # ALLIANCE ZONE depth: own wall to the hub cabinet's near face (the
    # starting line's far edge); the opponent zone mirrors it, NEUTRAL ZONE
    # = everything between (bumps / trenches / hubs included)
    alliance_zone_x: float = 0.0


@lru_cache(maxsize=None)
def auto_rules(year: int) -> Optional[AutoRules]:
    """The season's auto rules, or None (no rules modelled: callers skip all
    constraint logic). Derived from the field geometry, not hard-coded."""
    from web.field.field_model import FieldModel
    from web.field.occlusion import hub_prisms
    from web.field.overlay_geometry import TRENCH_SPAN_M, structure_footprints
    from web.field.season import season_for_year

    if int(year) != 2026:
        return None
    season = season_for_year(int(year))
    if not season.tag_layout:
        return None
    fm = FieldModel.load(season.tag_layout)
    fp = structure_footprints(fm, hub_prisms(fm))
    hubs, bumps = fp.get("hubs") or [], fp.get("bumps") or []
    if not hubs:
        return None
    L, W = float(fm.length_m), float(fm.width_m)

    def bbox(poly):
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        return min(xs), min(ys), max(xs), max(ys)

    obstacles = []
    for hub in sorted((bbox(h) for h in hubs), key=lambda r: r[0]):
        hx0, hy0, hx1, hy1 = hub
        obstacles.append(hub)
        mine = [bbox(b) for b in bumps if hx0 <= (bbox(b)[0] + bbox(b)[2]) / 2 <= hx1]
        low = [b for b in mine if b[3] <= hy0 + 1e-6]
        high = [b for b in mine if b[1] >= hy1 - 1e-6]
        if low:   # block between the low bump's outer edge and the trench opening
            obstacles.append((hx0, TRENCH_SPAN_M, hx1, min(b[1] for b in low)))
        if high:
            obstacles.append((hx0, max(b[3] for b in high), hx1, W - TRENCH_SPAN_M))
    near_face = min(bbox(h)[0] for h in hubs)        # own-side hub's near face
    return AutoRules(start_line_x=round(near_face - TAPE_OFFSET_M, 4),
                     obstacles=tuple(tuple(round(v, 4) for v in r) for r in obstacles),
                     field_m=(L, W), alliance_zone_x=round(near_face, 4))


# ----------------------------------------------------------------- geometry
def _inflate(rect, m):
    x0, y0, x1, y1 = rect
    return (x0 - m, y0 - m, x1 + m, y1 + m)


def _strictly_inside(x, y, r) -> bool:
    return r[0] < x < r[2] and r[1] < y < r[3]


def _push_out(x, y, r):
    """Nearest point on rect r's boundary (shortest axis move)."""
    moves = ((x - r[0], (r[0], y)), (r[2] - x, (r[2], y)),
             (y - r[1], (x, r[1])), (r[3] - y, (x, r[3])))
    return min(moves, key=lambda m: m[0])[1]


def segment_enters(ax, ay, bx, by, r, shrink: float = _SHRINK_EPS) -> bool:
    """True when segment a-b passes through the interior of rect r (Liang–Barsky
    against r shrunk by ``shrink``; a positive-length overlap is required, so a
    segment that only grazes the boundary does not count)."""
    x0, y0, x1, y1 = r[0] + shrink, r[1] + shrink, r[2] - shrink, r[3] - shrink
    dx, dy = bx - ax, by - ay
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
        if p == 0.0:
            if q < 0.0:
                return False
            continue
        t = q / p
        if p < 0.0:
            if t > t1:
                return False
            t0 = max(t0, t)
        else:
            if t < t0:
                return False
            t1 = min(t1, t)
    return t1 - t0 > 1e-12


def _blocked(a, b, rects) -> bool:
    return any(segment_enters(a[0], a[1], b[0], b[1], r) for r in rects)


@lru_cache(maxsize=32)
def _corner_graph(rules: AutoRules, margin: float):
    """Inflated rects, their (eps-pushed, in-field) corners and the corner-corner
    visibility edges."""
    L, W = rules.field_m
    rects = tuple(_inflate(r, margin) for r in rules.obstacles)
    e = _CORNER_EPS
    corners = []
    for x0, y0, x1, y1 in rects:
        for cx, cy in ((x0 - e, y0 - e), (x1 + e, y0 - e), (x1 + e, y1 + e), (x0 - e, y1 + e)):
            if margin - 1e-9 <= cx <= L - margin + 1e-9 and margin - 1e-9 <= cy <= W - margin + 1e-9:
                if not any(_strictly_inside(cx, cy, r) for r in rects):
                    corners.append((cx, cy))
    edges = {i: [] for i in range(len(corners))}
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            if not _blocked(corners[i], corners[j], rects):
                d = math.dist(corners[i], corners[j])
                edges[i].append((j, d))
                edges[j].append((i, d))
    return rects, tuple(corners), edges


def _shortest_detour(a, b, rules: AutoRules, margin: float) -> Optional[list]:
    """Corner waypoints of the shortest obstacle-free route a -> b (excluding
    a and b), or None if none exists."""
    rects, corners, cedges = _corner_graph(rules, margin)
    n = len(corners)
    A, B = n, n + 1                   # node ids for the endpoints
    pts = list(corners) + [a, b]
    adj = {k: list(v) for k, v in cedges.items()}
    adj[A], adj[B] = [], []
    for i, c in enumerate(corners):
        for end in (A, B):
            if not _blocked(pts[end], c, rects):
                d = math.dist(pts[end], c)
                adj[end].append((i, d))
                adj[i].append((end, d))
    dist = {A: 0.0}
    prev: dict = {}
    heap = [(0.0, A)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == B:
            break
        if d > dist.get(u, math.inf):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, math.inf) - 1e-12:
                dist[v], prev[v] = nd, u
                heapq.heappush(heap, (nd, v))
    if B not in dist:
        return None
    route, u = [], prev[B]
    while u != A:
        route.append(pts[u])
        u = prev[u]
    return route[::-1]


def _clamp_point(x, y, rects, L, W, m):
    """(x, y) moved into the shrunk field and out of every inflated rect."""
    for _ in range(4):
        nx = min(max(x, m), L - m)
        ny = min(max(y, m), W - m)
        for r in rects:
            if _strictly_inside(nx, ny, r):
                nx, ny = _push_out(nx, ny, r)
        if (nx, ny) == (x, y):
            break
        x, y = nx, ny
    return x, y


def _nearest_start(x, y, bx0, bx1, rects, W, m):
    """Nearest legal start point to (x, y): inside the start band
    bx0 <= x <= bx1, inside the field, outside every inflated rect.

    A robot may start UNDER the trench (trench lane, back bumper on the line)
    or beside the hub, so the nearest legal point can be sideways out of a
    trench block into the lane, not only back toward the own wall. The
    nearest point of (band minus open rects) is the plain band projection or
    lies on some rect edge, so those are the candidates."""
    def clamp(v, lo, hi):
        return min(max(v, lo), hi)
    ylo, yhi = m, W - m
    cands = [(clamp(x, bx0, bx1), clamp(y, ylo, yhi))]
    for r in rects:
        ex0, ex1 = max(r[0], bx0), min(r[2], bx1)
        if ex0 <= ex1:                                  # bottom / top edges in the band
            for ey in (r[1], r[3]):
                cands.append((clamp(x, ex0, ex1), clamp(ey, ylo, yhi)))
        for ex in (r[0], r[2]):                         # left / right edges in the band
            if bx0 <= ex <= bx1:
                cands.append((ex, clamp(clamp(y, r[1], r[3]), ylo, yhi)))
    legal = [c for c in cands if not any(_strictly_inside(c[0], c[1], r) for r in rects)]
    if not legal:                                       # cannot happen on the 2026 field
        return cands[0]
    return min(legal, key=lambda c: math.hypot(c[0] - x, c[1] - y))


def _reach(t: float, v_max: float, a_max: float) -> float:
    return min(v_max * t, 0.5 * a_max * t * t)


# ----------------------------------------------------------------- main entry
def constrain_auto_path_indexed(points: list, rules: AutoRules, params: Optional[dict] = None):
    """``constrain_auto_path`` plus ``src``: per output point the index of the
    input point it came from (None for inserted detour / prepended start)."""
    p = {**DEFAULTS, **{k: v for k, v in (params or {}).items() if k in DEFAULTS}}
    m = float(p["obstacle_margin_m"])
    L, W = rules.field_m
    rects = tuple(_inflate(r, m) for r in rules.obstacles)
    report = {"start_check": None, "start_offset_m": None, "n_clamped": 0, "n_detour": 0}
    if not points:
        return [], [], report

    # 1. clamp
    pts, src = [], []
    for i, (t, x, y) in enumerate(points):
        cx, cy = _clamp_point(float(x), float(y), rects, L, W, m)
        moved = abs(cx - x) > 1e-9 or abs(cy - y) > 1e-9
        report["n_clamped"] += int(moved)
        pts.append((float(t), cx, cy, CLAMPED if moved else SEEN))
        src.append(i)

    # 2. start check (on the clamped path)
    t0, fx, fy, _ = pts[0]
    line, tol = float(rules.start_line_x), float(p["start_tol_m"])
    if t0 > float(p["start_window_s"]):
        report["start_check"] = "unseen"
    else:
        sx, sy = _nearest_start(fx, fy, line - tol, line + tol, rects, W, m)
        d = math.hypot(sx - fx, sy - fy)
        report["start_offset_m"] = round(d, 2)
        if d < 1e-9:
            report["start_check"] = "ok"
        elif d <= _reach(t0, float(p["start_v_max"]), float(p["start_a_max"])) + float(p["start_noise_m"]):
            report["start_check"] = "inferred"
            if t0 > 0:
                pts.insert(0, (0.0, sx, sy, START))
                src.insert(0, None)
            else:
                pts[0] = (t0, sx, sy, START)
        else:
            report["start_check"] = "violation"

    # 3. detour around obstacles
    out, out_src = [pts[0]], [src[0]]
    for k in range(1, len(pts)):
        a, b = pts[k - 1], pts[k]
        if _blocked(a[1:3], b[1:3], rects):
            route = _shortest_detour(a[1:3], b[1:3], rules, m)
            if route:
                chain = [a[1:3]] + route + [b[1:3]]
                seg = [math.dist(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
                total = sum(seg) or 1.0
                acc = 0.0
                for i, (wx, wy) in enumerate(route):
                    acc += seg[i]
                    out.append((a[0] + (b[0] - a[0]) * acc / total, wx, wy, DETOUR))
                    out_src.append(None)
                report["n_detour"] += len(route)
        out.append(b)
        out_src.append(src[k])
    return out, out_src, report


def constrain_auto_path(points: list, rules: AutoRules, params: Optional[dict] = None):
    """Apply the field rules to one auto path.

    ``points``: time-sorted ``(t, x, y)`` in alliance perspective. Returns
    ``(points_out, report)``: ``points_out`` = ``(t, x, y, kind)`` with kind
    seen / clamped / detour / start, ``report`` = {start_check: ok | inferred |
    unseen | violation | None (empty path), start_offset_m, n_clamped, n_detour}.
    """
    out, _, report = constrain_auto_path_indexed(points, rules, params)
    return out, report
