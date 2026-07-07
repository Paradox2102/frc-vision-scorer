"""Offline camera-coverage / blind-spot map for a calibrated event.

Rasterizes the field into a grid, scores each cell's effective camera
coverage (sum of per-camera weights — see web/field/coverage.py for the exact
per-camera formula), and renders a heatmap: red = uncovered, yellow = partial
(< 1 effective camera), green = >= 1 effective camera. Also emits the raw
grid + blind-cell list as JSON beside the PNG, and prints a text summary
(total blind area, largest blind region centroid) — this is what explains
coverage failures like the corner close-range dead zone or an unseeded robot,
and answers "where should the next camera point".

Usage
-----
    .venv/bin/python scripts/coverage_map.py --event 2026cur \
        [--cell 0.25] [--out coverage_2026cur.png]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")
from web.field.camera_view import load_views
from web.field.coverage import coverage_grid
from web.field.field_model import FieldModel
from web.field.occlusion import hub_prisms
from web.field.overlay_geometry import structure_footprints

PX_PER_M = 80.0


def _to_px(x: float, y: float) -> tuple[int, int]:
    return (int(round(x * PX_PER_M)), int(round(y * PX_PER_M)))


def _draw_polygon(img, poly, color, thickness=2):
    pts = np.array([_to_px(x, y) for (x, y) in poly], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def _largest_blind_region(xs, ys, grid, cell_m: float, blind_thresh: float = 0.5):
    """Connected-component (4-neighbor) centroid of the largest blind region,
    plus total blind area in m^2. Returns (total_area_m2, centroid_xy_or_None)."""
    blind = grid < blind_thresh
    if not blind.any():
        return 0.0, None
    visited = np.zeros_like(blind, dtype=bool)
    best_size, best_cells = 0, []
    ny, nx = blind.shape
    for j in range(ny):
        for i in range(nx):
            if not blind[j, i] or visited[j, i]:
                continue
            stack = [(j, i)]
            visited[j, i] = True
            cells = []
            while stack:
                cj, ci = stack.pop()
                cells.append((cj, ci))
                for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nj, ni = cj + dj, ci + di
                    if 0 <= nj < ny and 0 <= ni < nx and blind[nj, ni] and not visited[nj, ni]:
                        visited[nj, ni] = True
                        stack.append((nj, ni))
            if len(cells) > best_size:
                best_size, best_cells = len(cells), cells
    total_area = float(blind.sum()) * cell_m * cell_m
    if not best_cells:
        return total_area, None
    cy = float(np.mean([ys[j] for (j, _i) in best_cells]))
    cx = float(np.mean([xs[i] for (_j, i) in best_cells]))
    return total_area, (cx, cy)


def render(field_model, views, xs, ys, grid, prisms, out_path: Path, cell_m: float):
    w_px = int(round(field_model.length_m * PX_PER_M)) + 1
    h_px = int(round(field_model.width_m * PX_PER_M)) + 1
    img = np.zeros((h_px, w_px, 3), dtype=np.uint8)

    half = cell_m / 2.0
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            cov = grid[j, i]
            if cov < 0.5:
                color = (0, 0, 200)      # red (BGR): uncovered
            elif cov < 1.0:
                color = (0, 200, 200)    # yellow: partial
            else:
                color = (0, 160, 0)      # green: >= 1 effective camera
            p0 = _to_px(x - half, y - half)
            p1 = _to_px(x + half, y + half)
            cv2.rectangle(img, p0, p1, color, thickness=-1)

    footprints = structure_footprints(field_model, prisms)
    for key, polys in footprints.items():
        for poly in polys:
            _draw_polygon(img, poly, (255, 255, 255), thickness=2)

    for view in views:
        cam_center = getattr(view.calibration, "camera_center", None)
        if cam_center is None:
            continue
        cx, cy = float(cam_center[0]), float(cam_center[1])
        if not (0.0 <= cx <= field_model.length_m and 0.0 <= cy <= field_model.width_m):
            continue
        px = _to_px(cx, cy)
        cv2.circle(img, px, 8, (255, 0, 255), thickness=-1)
        cv2.putText(img, view.name, (px[0] + 10, px[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", required=True)
    ap.add_argument("--cell", type=float, default=0.25, help="grid cell size in meters")
    ap.add_argument("--out", default=None, help="output PNG path (default coverage_<event>.png)")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else Path(f"coverage_{args.event}.png")

    views = load_views(args.event)
    if not views:
        raise SystemExit(f"no calibrated views found for event '{args.event}'")

    field_season = next((v.calibration.field_season for v in views
                         if getattr(v.calibration, "field_season", None)), None)
    if not field_season:
        raise SystemExit("no view carries a field_season; cannot load FieldModel")
    field_model = FieldModel.load(field_season)

    prisms = hub_prisms(field_model)
    xs, ys, grid = coverage_grid(views, field_model, cell_m=args.cell, prisms=prisms)

    render(field_model, views, xs, ys, grid, prisms, out_path, args.cell)

    blind_cells = [[float(xs[i]), float(ys[j])]
                   for j in range(grid.shape[0]) for i in range(grid.shape[1])
                   if grid[j, i] < 0.5]
    out_json = out_path.with_suffix(".json")
    out_json.write_text(json.dumps({
        "cell_m": args.cell,
        "coverage": grid.tolist(),
        "blind_cells": blind_cells,
    }))

    total_blind_area, centroid = _largest_blind_region(xs, ys, grid, args.cell)
    print(f"event: {args.event}  cameras: {[v.name for v in views]}")
    print(f"grid: {grid.shape[1]}x{grid.shape[0]} cells @ {args.cell} m")
    print(f"total blind area: {total_blind_area:.2f} m^2  ({len(blind_cells)} cells)")
    if centroid is not None:
        print(f"largest blind region centroid: ({centroid[0]:.2f}, {centroid[1]:.2f})")
    else:
        print("largest blind region centroid: none (fully covered)")
    print(f"wrote {out_path}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
