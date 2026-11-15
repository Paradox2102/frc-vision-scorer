"""Camera blind zones (job knob ``blind_zone_coast``).

Some events' main views cut off the near field corners, and the 2026 corner
cameras sit ~0.6-0.9 m outside those corners, 0.6-0.7 m up, looking along the
field: the carpet within ~2.5 m of them is below their frame, and a robot that
is in frame closer than ~3 m spans > 50% of the short corner tile's height
(det_frac_backstop "always" drops it). A robot that drives into such a gap is
lost by every camera; it is still there.

This module turns the calibrations into a per-job field grid:

* ``view_useful_grid`` — per view, the cells where a robot standing there is
  USEFULLY visible: its ground point (and, with ``top_probe_m`` > 0, a point
  that far up — a robot cut off by the frame edge) projects inside the view's
  frame inset by ``margin_px`` and passes the depth / lens-fold checks
  (CameraCalibration.project_checked); the robot body
  (``ROBOT_MID_PROBE_M`` up) is not under one of the view's overlay masks; the
  ground point is below the view's SAM top crop; and, for a non-main view,
  the cell is at least ``min_range_m`` (ground distance) from the camera.
* ``BlindZone`` — the cells no view usefully sees (the AND over views of "not
  useful"), with a distance field for re-acquisition gating, plus each view's
  "out of view" set dilated by ``margin_m`` for per-camera occlusion
  (web.field.occlusion.resolve_occlusion ``out_of_view``): a slot out of view in
  EVERY camera is occluded by the existing ALL-cameras merge.

Everything is computed once per active view set (the 2025 corner views go
inactive at the endgame cutoff, so a job may have two zones)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

BLIND_ZONE_CELL_M = 0.25
# Optional second probe height (top_probe_m): require a point this far up
# (bumper top / lower frame) in frame too. Default OFF in the job knob: hand GT
# (2026cur_qm8) has annotators marking robots "visible" with their ground point
# 1-30 px inside main's left/right edges, where the 0.4 m point leans out of
# frame — the probe blanked 779 GT-visible keyframes (margin 10 px) vs 61 for
# ground point only at margin 0.
ROBOT_TOP_PROBE_M = 0.4
# The robot body point tested against overlay masks.
ROBOT_MID_PROBE_M = 0.2


def field_grid(field_dims, cell_m: float = BLIND_ZONE_CELL_M):
    """Cell-centre coordinates (xs, ys) covering the field rectangle."""
    L, W = float(field_dims[0]), float(field_dims[1])
    nx = max(1, int(np.ceil(L / cell_m)))
    ny = max(1, int(np.ceil(W / cell_m)))
    xs = (np.arange(nx) + 0.5) * cell_m
    ys = (np.arange(ny) + 0.5) * cell_m
    return xs, ys


def view_useful_grid(cal, xs, ys, *, margin_px: float = 0.0,
                     min_range_m: float = 0.0, masks: Sequence = (),
                     top_crop_frac: float = 0.0, image_size=None,
                     top_probe_m: float = 0.0) -> np.ndarray:
    """(len(ys), len(xs)) bool: cells where a robot is usefully visible in the
    view with calibration ``cal`` (see module docstring). ``image_size`` (w, h)
    defaults to cal.image_size; ``masks`` are (x, y, w, h) in that frame;
    ``min_range_m`` > 0 also blanks cells within that ground distance of the
    camera (pose calibrations only)."""
    X, Y = np.meshgrid(np.asarray(xs, float), np.asarray(ys, float))
    n = X.size
    if n == 0:
        return np.zeros(X.shape, bool)
    w, h = image_size if image_size is not None else cal.image_size
    w, h = float(w), float(h)
    g = np.stack([X.ravel(), Y.ravel(), np.zeros(n)], axis=1)
    mid = g.copy(); mid[:, 2] = ROBOT_MID_PROBE_M
    m = -float(margin_px)   # project_checked grows the frame by margin_px
    uv_g, ok = cal.project_checked(g, margin_px=m, image_size=(w, h))
    if top_probe_m and top_probe_m > 0:
        top = g.copy(); top[:, 2] = float(top_probe_m)
        _uv_t, ok_t = cal.project_checked(top, margin_px=m, image_size=(w, h))
        ok = ok & ok_t
    if top_crop_frac and top_crop_frac > 0:
        with np.errstate(invalid="ignore"):
            ok &= uv_g[:, 1] >= float(top_crop_frac) * h
    if masks:
        uv_m, _ = cal.project_checked(mid, margin_px=None, image_size=(w, h))
        for (mx, my, mw, mh) in masks:
            with np.errstate(invalid="ignore"):
                inside = ((uv_m[:, 0] >= mx) & (uv_m[:, 0] < mx + mw)
                          & (uv_m[:, 1] >= my) & (uv_m[:, 1] < my + mh))
            ok &= ~inside
    if min_range_m and min_range_m > 0:
        cc = getattr(cal, "camera_center", None)
        if cc is not None:
            ok &= np.hypot(g[:, 0] - float(cc[0]), g[:, 1] - float(cc[1])) >= float(min_range_m)
    return ok.reshape(X.shape)


def _dilate(mask: np.ndarray, radius_m: float, cell_m: float) -> np.ndarray:
    r = int(np.ceil(max(0.0, float(radius_m)) / cell_m))
    if r <= 0 or not mask.any():
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


@dataclass
class BlindZone:
    """Blind cells of one active view set. ``useful[name]`` is each view's
    usefully-visible grid; ``blind`` = no view usefully sees the cell."""
    xs: np.ndarray
    ys: np.ndarray
    cell_m: float
    useful: dict                       # view name -> (ny, nx) bool
    margin_m: float = 0.3
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        shape = (len(self.ys), len(self.xs))
        if self.useful:
            any_useful = np.zeros(shape, bool)
            for g in self.useful.values():
                any_useful |= g
        else:
            any_useful = np.zeros(shape, bool)
        self.blind = ~any_useful
        # Per view: out of useful view, dilated by margin_m (the per-camera
        # occlusion test; their AND is ~ the dilated blind zone).
        self._oov = {name: _dilate(~g, self.margin_m, self.cell_m)
                     for name, g in self.useful.items()}
        # Distance (m) from each cell centre to the nearest blind cell centre.
        if self.blind.any():
            src = (~self.blind).astype(np.uint8)
            self._dist = cv2.distanceTransform(src, cv2.DIST_L2, 5).astype(float) * self.cell_m
        else:
            self._dist = None

    @property
    def views(self) -> list:
        return sorted(self.useful)

    def _cell(self, pos) -> Optional[tuple]:
        """Grid (row, col) of ``pos``, clamped onto the field; None if non-finite."""
        try:
            x, y = float(pos[0]), float(pos[1])
        except (TypeError, ValueError, IndexError):
            return None
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        c = int(np.clip(np.floor(x / self.cell_m), 0, len(self.xs) - 1))
        r = int(np.clip(np.floor(y / self.cell_m), 0, len(self.ys) - 1))
        return r, c

    def out_of_view(self, view_name: str, pos) -> bool:
        """True when ``pos`` is (within margin_m of) a cell ``view_name`` cannot
        usefully see. A view this zone was not built with: False."""
        g = self._oov.get(view_name)
        rc = self._cell(pos)
        if g is None or rc is None:
            return False
        return bool(g[rc])

    def contains(self, pos, margin_m: float = 0.0) -> bool:
        """``pos`` lies in a blind cell, or within ``margin_m`` of one."""
        return self.distance_m(pos) <= float(margin_m)

    def distance_m(self, pos) -> float:
        """Approximate distance (m) from ``pos`` to the blind zone (0 inside;
        inf when the zone is empty). Cell-quantised (+-cell/2)."""
        if self._dist is None:
            return float("inf")
        rc = self._cell(pos)
        if rc is None:
            return float("inf")
        return float(self._dist[rc])

    @property
    def area_m2(self) -> float:
        return float(self.blind.sum()) * self.cell_m * self.cell_m

    def components(self) -> list[dict]:
        """Connected blind regions: [{area_m2, bbox [x0, y0, x1, y1], polygon
        [[x, y], ...]}] in field meters (bbox/polygon on cell edges), largest first."""
        if not self.blind.any():
            return []
        m = self.blind.astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = []
        cm = self.cell_m
        for k in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[k])
            comp = (labels == k).astype(np.uint8)
            # Pad so contours of edge components close on the field boundary.
            cnts, _ = cv2.findContours(np.pad(comp, 1), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            poly = []
            if cnts:
                c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
                # Contour runs through cell centres (pad offset 1): cell index
                # i -> centre (i + 0.5) * cm.
                poly = [[round((float(px) - 1 + 0.5) * cm, 3),
                         round((float(py) - 1 + 0.5) * cm, 3)] for px, py in c]
            out.append({
                "area_m2": round(area * cm * cm, 3),
                "bbox": [round(x * cm, 3), round(y * cm, 3),
                         round((x + w) * cm, 3), round((y + h) * cm, 3)],
                "polygon": poly,
            })
        out.sort(key=lambda d: -d["area_m2"])
        return out

    def summary(self) -> dict:
        """JSON-able record for logs / the replay summary."""
        return {
            "views": self.views,
            "cell_m": self.cell_m,
            "margin_m": self.margin_m,
            "area_m2": round(self.area_m2, 3),
            "useful_frac": {k: round(float(v.mean()), 4) for k, v in sorted(self.useful.items())},
            "components": self.components(),
            "params": dict(self.params),
            # Row-major blind cells (rows = y) for drawing: "#" blind, "." seen.
            "grid": {"nx": int(len(self.xs)), "ny": int(len(self.ys)),
                     "rows": ["".join("#" if b else "." for b in row) for row in self.blind]},
        }

    @classmethod
    def from_summary(cls, d: dict) -> "BlindZone":
        """Rebuild a zone from ``summary()`` (TRACKER_TRACE replay). Only the
        blind grid round-trips: ``distance_m`` / ``contains`` / ``area_m2`` are
        identical; ``out_of_view`` is not (per-view grids are not recorded)."""
        cm = float(d["cell_m"])
        g = d["grid"]
        blind = np.array([[ch == "#" for ch in row] for row in g["rows"]], dtype=bool)
        blind = blind.reshape(int(g["ny"]), int(g["nx"]))
        xs = (np.arange(int(g["nx"])) + 0.5) * cm
        ys = (np.arange(int(g["ny"])) + 0.5) * cm
        return cls(xs=xs, ys=ys, cell_m=cm, useful={"_seen": ~blind},
                   margin_m=float(d.get("margin_m", 0.3)),
                   params=dict(d.get("params") or {}))

    def describe(self) -> str:
        comps = self.components()
        parts = ", ".join(
            f"{c['area_m2']:.1f} m2 x[{c['bbox'][0]:.2f},{c['bbox'][2]:.2f}] "
            f"y[{c['bbox'][1]:.2f},{c['bbox'][3]:.2f}]" for c in comps) or "none"
        return (f"views={self.views} area={self.area_m2:.1f} m2 "
                f"({len(comps)} region(s)): {parts}")


def compute_blind_zone(views, field_dims, *, margin_px: float = 0.0,
                       corner_min_range_m: float = 3.0, margin_m: float = 0.3,
                       cell_m: float = BLIND_ZONE_CELL_M,
                       top_crop: bool = True, top_probe_m: float = 0.0) -> BlindZone:
    """BlindZone over ``views`` (CameraView-like: name, crop (x, y, w, h), masks,
    calibration). Views without a calibration are skipped (they never feed
    the occlusion merge either). ``corner_min_range_m`` applies to every view
    but "main". ``top_crop``: honour each calibration's robot_track_top_crop."""
    xs, ys = field_grid(field_dims, cell_m)
    useful = {}
    for v in views:
        cal = getattr(v, "calibration", None)
        if cal is None:
            continue
        iw, ih = (float(s) for s in cal.image_size)
        crop = getattr(v, "crop", None)
        if crop is not None and len(crop) == 4:
            # The detector sees the view crop; never test pixels it lacks.
            iw, ih = min(iw, float(crop[2])), min(ih, float(crop[3]))
        tc = float(getattr(cal, "robot_track_top_crop", 0.0) or 0.0) if top_crop else 0.0
        useful[v.name] = view_useful_grid(
            cal, xs, ys, margin_px=margin_px,
            min_range_m=(0.0 if v.name == "main" else corner_min_range_m),
            masks=tuple(getattr(v, "masks", ()) or ()), top_crop_frac=tc,
            image_size=(iw, ih), top_probe_m=top_probe_m)
    return BlindZone(xs=xs, ys=ys, cell_m=cell_m, useful=useful, margin_m=margin_m,
                     params={"margin_px": float(margin_px),
                             "corner_min_range_m": float(corner_min_range_m),
                             "top_probe_m": float(top_probe_m),
                             "top_crop": bool(top_crop)})
