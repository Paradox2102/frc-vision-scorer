#!/usr/bin/env python3
"""Extract a 3D field wireframe from an AdvantageScope Field3d GLB.

Turns AdvantageScope's field model (github.com/Mechanical-Advantage/
AdvantageScopeAssets, release archive-v1, e.g. Field3d_2025FRCFieldWeldedV3.zip)
into line segments in the WPILib field frame (meters: origin at the blue-alliance
corner, +x toward red, +y across, +z up), for the HUD wireframe overlay and
calibration checks of seasons that have no hand-coded geometry in
web/field/overlay_geometry.py.

Pipeline:
  1. Load every scene node, apply its scene-graph transform, then the config's
     `rotations` (in order, each left-multiplied), then translate by
     (L/2, W/2, 0) with L/W from the WPILib tag layout's `field`.
  2. Verify every AprilTag of the WPILib layout lies on the mesh (<= 5 mm);
     abort otherwise (wrong transform / wrong layout variant).
  3. Drop game pieces, carpet, loose hardware and anything wholly outside the
     field rectangle grown by 0.3 m (stands, driver-station tables).
  4. Per node: weld vertices, keep feature edges (dihedral > --angle deg (60)),
     boundary and non-manifold edges; chain them into polylines,
     Douglas-Peucker simplify (--dp 2 cm), drop polylines shorter than
     --min-len (0.2 m) or with a bbox diagonal under --min-extent (0.3 m).
  5. De-duplicate near-identical segments on a --dedupe grid (4 cm; collapses
     both faces of plates/tape and the edges of thin tubes), group segments (reef / barge / hub / tower / tape / perimeter ...)
     by node name + location relative to the season's tag anchors.

Output JSON:
  {source, season, field:{length,width}, transform, max_tag_residual_mm,
   params, counts, groups:{name: [[x0,y0,z0,x1,y1,z1], ...]}}   (mm-rounded)

Run (the repo .venv has no trimesh; use a throwaway env):
  uv run --with trimesh --with rtree --with scipy --with matplotlib \\
    python -I scripts/extract_field_wireframe.py \\
      --glb  <dir>/Field3d_2025FRCFieldWeldedV3/model.glb \\
      --config <dir>/Field3d_2025FRCFieldWeldedV3/config.json \\
      --tags data/field/2025-reefscape-welded.json \\
      --out  data/field/wireframes/2025-reefscape-welded.json \\
      --preview data/field/wireframes/2025-reefscape-welded.png

Downloaded GLBs are data only; run with `python -I`.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

EXCLUDE_NAME = re.compile(
    r"coral|algae|ge-2550|carpet|washer|eyebolt|eyenut|carabiner|^hex\b|^hex[_\d.-]*$"
    r"|game ?piece|fuel",
    re.IGNORECASE,
)
TAPE_NAME = re.compile(r"tape", re.IGNORECASE)
FIELD_MARGIN_M = 0.3
TAG_TOL_M = 0.005


# ---------------------------------------------------------------- transform
def config_rotation(cfg: dict) -> np.ndarray:
    R = np.eye(4)
    for r in cfg.get("rotations", []):
        axis = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[r["axis"]]
        R = trimesh.transformations.rotation_matrix(math.radians(r["degrees"]), axis) @ R
    return R


def load_nodes(glb: Path, M: np.ndarray):
    """Yield (node_name, vertices(N,3) in field frame, faces(F,3))."""
    scene = trimesh.load(str(glb), force="scene")
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph.get(node)
        geom = scene.geometry[gname]
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        A = M @ T
        v = np.asarray(geom.vertices, float) @ A[:3, :3].T + A[:3, 3]
        yield f"{node}|{gname}", v, np.asarray(geom.faces, np.int64)


def tag_points(tags: dict) -> dict[int, np.ndarray]:
    return {int(t["ID"]): np.array([t["pose"]["translation"][k] for k in "xyz"], float)
            for t in tags["tags"]}


# ---------------------------------------------------------------- edges
def feature_polylines(v: np.ndarray, f: np.ndarray, angle_deg: float,
                      dp_tol: float, min_len: float, min_extent: float = 0.0
                      ) -> list[np.ndarray]:
    # weld at 0.1 mm (GLB exports split vertices per normal / uv)
    q = np.round(v / 1e-4).astype(np.int64)
    uq, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    vw = uq.astype(float) * 1e-4
    fw = inv[f]
    ok = (fw[:, 0] != fw[:, 1]) & (fw[:, 1] != fw[:, 2]) & (fw[:, 0] != fw[:, 2])
    fw = fw[ok]
    if len(fw) == 0:
        return []
    n = np.cross(vw[fw[:, 1]] - vw[fw[:, 0]], vw[fw[:, 2]] - vw[fw[:, 0]])
    nn = np.linalg.norm(n, axis=1)
    good = nn > 1e-12
    fw, n = fw[good], n[good] / nn[good, None]
    if len(fw) == 0:
        return []
    e = np.concatenate([fw[:, [0, 1]], fw[:, [1, 2]], fw[:, [2, 0]]])
    e.sort(axis=1)
    face_of = np.tile(np.arange(len(fw)), 3)
    nv = len(vw)
    key = e[:, 0] * nv + e[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    _, start, cnt = np.unique(ks, return_index=True, return_counts=True)
    sel_edges = []
    # boundary + non-manifold
    odd = cnt != 2
    sel_edges.append(e[order[start[odd]]])
    two = cnt == 2
    fa = face_of[order[start[two]]]
    fb = face_of[order[start[two] + 1]]
    cosang = np.einsum("ij,ij->i", n[fa], n[fb])
    sharp = cosang < math.cos(math.radians(angle_deg))
    sel_edges.append(e[order[start[two]]][sharp])
    E = np.concatenate(sel_edges)
    if len(E) == 0:
        return []
    return [p for p in (dp_simplify(vw[c], dp_tol) for c in chain_edges(E))
            if polyline_len(p) >= min_len
            and float(np.linalg.norm(np.ptp(p, axis=0))) >= min_extent]


def chain_edges(E: np.ndarray) -> list[list[int]]:
    """Chain an undirected edge list into vertex-index polylines (cycles closed)."""
    adj: dict[int, list[int]] = {}
    for a, b in E.tolist():
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    used: set[tuple[int, int]] = set()

    def walk(a: int, b: int) -> list[int]:
        path = [a, b]
        used.add((min(a, b), max(a, b)))
        prev, cur = a, b
        while len(adj[cur]) == 2:
            nxt = adj[cur][0] if adj[cur][1] == prev else adj[cur][1]
            k = (min(cur, nxt), max(cur, nxt))
            if k in used:
                break
            used.add(k)
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    chains = []
    for s, nbrs in adj.items():          # open chains from junctions / ends
        if len(nbrs) != 2:
            for t in nbrs:
                if (min(s, t), max(s, t)) not in used:
                    chains.append(walk(s, t))
    for s, nbrs in adj.items():          # remaining pure cycles
        for t in nbrs:
            if (min(s, t), max(s, t)) not in used:
                chains.append(walk(s, t))
    return chains


def dp_simplify(p: np.ndarray, tol: float) -> np.ndarray:
    if len(p) < 3:
        return p
    keep = np.zeros(len(p), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(p) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = p[i], p[j]
        ab = b - a
        L2 = float(ab @ ab)
        seg = p[i + 1:j] - a
        if L2 < 1e-18:          # closed loop: distance to the start point
            d = np.linalg.norm(seg, axis=1)
        else:
            t = np.clip(seg @ ab / L2, 0, 1)
            d = np.linalg.norm(seg - t[:, None] * ab, axis=1)
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return p[keep]


def polyline_len(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


# ---------------------------------------------------------------- grouping
def season_rules(season: str, tp: dict[int, np.ndarray], L: float, W: float):
    """List of (group, predicate(centroid xyz, bbox_min, bbox_max)) tried in order."""
    def near(ids, r):
        pts = np.array([tp[i][:2] for i in ids if i in tp])
        return lambda c, lo, hi: bool(len(pts)) and float(
            np.min(np.linalg.norm(pts - c[:2], axis=1))) < r

    def around_centre(ids, r):
        ctr = np.mean([tp[i][:2] for i in ids if i in tp], axis=0)
        return lambda c, lo, hi: float(np.linalg.norm(c[:2] - ctr)) < r

    if season.startswith("2025"):
        reef_b, reef_r = around_centre(range(17, 23), 1.35), around_centre(range(6, 12), 1.35)
        return [
            ("reef", lambda c, lo, hi: reef_b(c, lo, hi) or reef_r(c, lo, hi)),
            ("coral_station", near((1, 2, 12, 13), 1.2)),
            ("processor", near((3, 16), 0.9)),
            ("barge", lambda c, lo, hi: abs(c[0] - L / 2) < 1.3
             and -0.1 < c[1] < W + 0.1 and hi[2] > 0.25),
        ]
    if season.startswith("2026"):
        hub_b = around_centre((18, 19, 20, 21, 24, 25, 26, 27), 1.1)
        hub_r = around_centre((2, 3, 4, 5, 8, 9, 10, 11), 1.1)
        trench_x = [tp[i][0] for i in (1, 6, 7, 12, 17, 22, 23, 28) if i in tp]

        def trench(c, lo, hi):
            return (min(abs(c[0] - x) for x in trench_x) < 0.5
                    and (c[1] < 2.0 or c[1] > W - 2.0) and hi[2] > 0.3)

        def bump(c, lo, hi):
            return (min(abs(c[0] - x) for x in trench_x) < 1.0 and hi[2] < 0.3
                    and 1.0 < c[1] < W - 1.0)
        return [
            ("hub", lambda c, lo, hi: hub_b(c, lo, hi) or hub_r(c, lo, hi)),
            ("tower", near((15, 16, 31, 32), 1.3)),
            ("outpost", near((13, 14, 29, 30), 0.9)),
            ("trench", trench),
            ("bump", bump),
        ]
    return []


def classify(name: str, poly: np.ndarray, rules, L: float, W: float) -> str:
    if TAPE_NAME.search(name):
        return "tape"
    c = poly.mean(axis=0)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    for g, pred in rules:
        if pred(c, lo, hi):
            return g
    edge = min(c[0], L - c[0], c[1], W - c[1])
    if edge < 0.35:
        return "perimeter"
    if hi[2] < 0.03:
        return "floor"
    return "other"


# ---------------------------------------------------------------- main
def extract(glb, config, tags_path, angle, dp, min_len, min_extent=0.0, dedupe=0.005,
            season=None):
    t0 = time.time()
    cfg = json.loads(Path(config).read_text())
    tags = json.loads(Path(tags_path).read_text())
    L, W = float(tags["field"]["length"]), float(tags["field"]["width"])
    M = config_rotation(cfg)
    M[:3, 3] += [L / 2, W / 2, 0.0]
    season = season or Path(tags_path).stem
    tp = tag_points(tags)

    nodes = list(load_nodes(Path(glb), M))
    t_load = time.time() - t0

    # --- tag residual check against all non-game-piece geometry
    vs, fs, off = [], [], 0
    for name, v, f in nodes:
        if re.search(r"coral|algae|ge-2550|fuel", name, re.I):
            continue
        vs.append(v); fs.append(f + off); off += len(v)
    full = trimesh.Trimesh(np.concatenate(vs), np.concatenate(fs), process=False)
    ids = sorted(tp)
    _, dist, _ = trimesh.proximity.closest_point(full, np.array([tp[i] for i in ids]))
    resid = {i: float(d) for i, d in zip(ids, dist)}
    worst = max(resid.values())
    if worst > TAG_TOL_M:
        bad = {i: round(d * 1000, 1) for i, d in resid.items() if d > TAG_TOL_M}
        sys.exit(f"tag residual check FAILED (mm): {bad} -- transform or layout variant wrong")
    del full, vs, fs

    rules = season_rules(season, tp, L, W)
    lo_lim = np.array([-FIELD_MARGIN_M, -FIELD_MARGIN_M])
    hi_lim = np.array([L + FIELD_MARGIN_M, W + FIELD_MARGIN_M])
    groups: dict[str, list] = {}
    n_kept = n_excl_name = n_excl_out = 0
    for name, v, f in nodes:
        short = name.split("|")[0]
        if EXCLUDE_NAME.search(short) or EXCLUDE_NAME.search(name.split("|")[1]):
            n_excl_name += 1
            continue
        bmin, bmax = v[:, :2].min(0), v[:, :2].max(0)
        if np.any(bmax < lo_lim) or np.any(bmin > hi_lim):
            n_excl_out += 1
            continue
        n_kept += 1
        for p in feature_polylines(v, f, angle, dp, min_len, min_extent):
            # clip-free: keep polylines whose centroid is near the field
            c = p.mean(0)
            if not (lo_lim[0] <= c[0] <= hi_lim[0] and lo_lim[1] <= c[1] <= hi_lim[1]):
                continue
            g = classify(name, p, rules, L, W)
            groups.setdefault(g, []).append(np.hstack([p[:-1], p[1:]]))

    # --- dedupe segments on a `dedupe` grid, direction-agnostic (collapses the two
    #     faces of plates / tape and adjacent edges of thin tubes)
    out = {}
    for g, segs in groups.items():
        S = np.concatenate(segs)
        if g in ("tape", "floor"):
            S[:, 2] = S[:, 5] = 0.0
        a = np.round(S[:, :3] / dedupe).astype(np.int64)
        b = np.round(S[:, 3:] / dedupe).astype(np.int64)
        swap = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & ((a[:, 1] > b[:, 1]) |
                ((a[:, 1] == b[:, 1]) & (a[:, 2] > b[:, 2]))))
        k = np.where(swap[:, None], np.hstack([b, a]), np.hstack([a, b]))
        _, first = np.unique(k, axis=0, return_index=True)
        S = S[np.sort(first)]
        S = S[np.linalg.norm(S[:, 3:] - S[:, :3], axis=1) > 1e-3]
        out[g] = np.round(S, 3).tolist()

    meta = {
        "source": f"AdvantageScope {Path(glb).parent.name} (github.com/Mechanical-Advantage/"
                  "AdvantageScopeAssets, release archive-v1)",
        "season": season,
        "field": {"length": L, "width": W},
        "transform": {"advantagescope_rotations": cfg.get("rotations", []),
                      "translation": [L / 2, W / 2, 0.0],
                      "matrix": np.round(M, 9).tolist()},
        "max_tag_residual_mm": round(worst * 1000, 2),
        "tag_residual_mm": {str(i): round(d * 1000, 2) for i, d in resid.items()},
        "params": {"dihedral_deg": angle, "dp_tol_m": dp, "min_polyline_m": min_len,
                   "min_extent_m": min_extent,
                   "field_margin_m": FIELD_MARGIN_M, "dedupe_grid_m": dedupe},
        "nodes": {"total": len(nodes), "kept": n_kept, "excluded_by_name": n_excl_name,
                  "excluded_outside": n_excl_out},
        "counts": {g: len(s) for g, s in sorted(out.items())},
        "total_segments": sum(len(s) for s in out.values()),
        "runtime_s": {"load": round(t_load, 1), "total": round(time.time() - t0, 1)},
    }
    meta["groups"] = dict(sorted(out.items()))
    return meta, tp


# ---------------------------------------------------------------- preview
COLOURS = {"reef": "#c040ff", "barge": "#ff8c00", "coral_station": "#00b0b0",
           "processor": "#2a9d2a", "hub": "#e0b000", "tower": "#ff00ff",
           "outpost": "#00b0b0", "trench": "#8050ff", "bump": "#ff6020",
           "tape": "#3060ff", "floor": "#7090c0", "perimeter": "#30a030",
           "other": "#909090"}


def render_preview(meta: dict, tp: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    L, W = meta["field"]["length"], meta["field"]["width"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 15),
                                   gridspec_kw={"height_ratios": [1, 1.1]})
    # top-down
    for g, segs in meta["groups"].items():
        S = np.asarray(segs)
        if len(S):
            ax1.add_collection(LineCollection(S[:, [0, 1, 3, 4]].reshape(-1, 2, 2),
                                              colors=COLOURS.get(g, "k"), lw=0.6, label=g))
    ax1.plot([0, L, L, 0, 0], [0, 0, W, W, 0], "k--", lw=0.8)
    T = np.array([tp[i] for i in sorted(tp)])
    ax1.scatter(T[:, 0], T[:, 1], c="red", s=14, zorder=5)
    for i in sorted(tp):
        ax1.annotate(str(i), tp[i][:2], fontsize=7, color="red")
    ax1.set_aspect("equal"); ax1.set_xlim(-0.6, L + 0.6); ax1.set_ylim(-0.6, W + 0.6)
    ax1.set_title(f"{meta['season']} top-down (WPILib frame, m) — "
                  f"{meta['total_segments']} segs, tag resid ≤ {meta['max_tag_residual_mm']} mm")
    ax1.legend(loc="upper right", fontsize=7, ncol=2)

    # oblique: broadcast-ish camera behind/above the scoring-table side (y<0)
    eye = np.array([L / 2, -5.0, 5.0]); tgt = np.array([L / 2, W / 2, 0.0])
    fwd = tgt - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    Rcam = np.stack([right, -up, fwd])
    fpx = 1.0

    def proj(P):
        c = (P - eye) @ Rcam.T
        return np.stack([fpx * c[:, 0] / c[:, 2], -fpx * c[:, 1] / c[:, 2]], 1)

    for g, segs in meta["groups"].items():
        S = np.asarray(segs)
        if len(S):
            a, b = proj(S[:, :3]), proj(S[:, 3:])
            ax2.add_collection(LineCollection(np.stack([a, b], 1),
                                              colors=COLOURS.get(g, "k"), lw=0.5))
    pt = proj(T)
    ax2.scatter(pt[:, 0], pt[:, 1], c="red", s=10, zorder=5)
    corners = proj(np.array([[0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0], [0, 0, 0]]))
    ax2.plot(corners[:, 0], corners[:, 1], "k--", lw=0.8)
    ax2.set_aspect("equal"); ax2.autoscale_view()
    ax2.set_title("oblique from scoring-table side (eye x=L/2, y=-5, z=5)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--glb", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview")
    ap.add_argument("--season", help="defaults to the tag layout file stem")
    ap.add_argument("--angle", type=float, default=60.0, help="dihedral threshold, deg")
    ap.add_argument("--dp", type=float, default=0.02, help="Douglas-Peucker tolerance, m")
    ap.add_argument("--min-len", type=float, default=0.20, help="min polyline length, m")
    ap.add_argument("--min-extent", type=float, default=0.30,
                    help="min polyline bounding-box diagonal, m (drops bolt rings, small parts)")
    ap.add_argument("--dedupe", type=float, default=0.04,
                    help="endpoint grid for segment de-duplication, m")
    a = ap.parse_args()
    meta, tp = extract(a.glb, a.config, a.tags, a.angle, a.dp, a.min_len, a.min_extent,
                       a.dedupe, a.season)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(meta, separators=(",", ":")))
    if a.preview:
        render_preview(meta, tp, Path(a.preview))
    summary = {k: meta[k] for k in ("season", "max_tag_residual_mm", "nodes", "counts",
                                    "total_segments", "runtime_s")}
    print(json.dumps(summary, indent=1))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
