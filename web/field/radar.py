"""Top-down field map: exact structure footprints + track-state rendering.

Structures come from overlay_geometry.structure_footprints — the same
CAD-derived placements the /calibrate overlay draws — so the map and the
camera overlays can never disagree about where the field furniture is.

Per-track rendering encodes tracking state for replay review:
  solid dot            visible, main-camera-backed position
  hollow / faded dot   coasting (occluded or missed)
  dashed outer ring    position currently sourced from corner cameras only
  fading polyline      recent filtered positions (~2 s trail)
  label                team number, slot id fallback
"""
from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np

from web.field.occlusion import hub_prisms
from web.field.overlay_geometry import structure_footprints


@dataclass
class RadarTrack:
    pos: tuple
    alliance: str | None = None
    team: str | None = None
    slot_id: int | None = None
    state: str = "visible"       # "visible" | "coasting"
    corner_only: bool = False
    trail: tuple = ()            # oldest-first recent positions

_BG = 235
_ALLIANCE = {"blue": (255, 120, 0), "red": (0, 0, 255), None: (150, 150, 150)}
# Scouted (Manta) ghost routes: a wide, translucent band the crisp live trail rides
# on top of. 4x the 2px live stroke, blended at half opacity.
_REF_WIDTH = 8
_REF_ALPHA = 0.5
# Match the /calibrate overlay legend: cyan hubs, orange bumps,
# magenta towers, pink trenches (BGR).
_STRUCTURE = {
    "hubs": (200, 200, 0),
    "bumps": (0, 140, 230),
    "towers": (200, 0, 200),
    "trenches": (180, 170, 230),
}


def _smooth_arc(points, step_m: float = 0.10, sigma_m: float = 0.40):
    """Low-pass a grid-quantized polyline into flowing arcs.

    Manta paths are drawn as axis-aligned grid steps (up/right/down/right ...), so
    corner-cutting alone leaves a wavy staircase. Grid quantization is high-frequency
    noise (~0.17 m wavelength) riding on the real multi-meter trajectory: resampling
    to uniform arc-length spacing and Gaussian-smoothing the coordinates (sigma_m
    between the two scales) cancels the axis-aligned oscillation and yields arcs,
    while genuine turns survive. The true endpoints are pinned so the start/end
    markers stay put.
    """
    p = np.asarray(points, float)
    if len(p) < 3:
        return [(float(x), float(y)) for x, y in p]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-6:
        return [(float(x), float(y)) for x, y in p]
    # Resample at uniform arc length so sigma means the same everywhere.
    m = max(int(total / step_m) + 1, 3)
    su = np.linspace(0.0, total, m)
    rx = np.interp(su, s, p[:, 0])
    ry = np.interp(su, s, p[:, 1])
    # Gaussian kernel, sigma in resampled-point units; edge-pad so ends hold.
    sigma = max(sigma_m / step_m, 0.5)
    rad = max(int(round(3 * sigma)), 1)
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2)
    k /= k.sum()
    sx = np.convolve(np.pad(rx, rad, mode="edge"), k, mode="valid")
    sy = np.convolve(np.pad(ry, rad, mode="edge"), k, mode="valid")
    sx[0], sy[0] = p[0]
    sx[-1], sy[-1] = p[-1]
    return list(zip(sx.tolist(), sy.tolist()))


class Radar:
    def __init__(self, field_length: float, field_width: float,
                 px_per_m: float = 56, field_model=None):
        self.L, self.W, self.s = field_length, field_width, px_per_m
        self.w = round(field_length * px_per_m)
        self.h = round(field_width * px_per_m)
        footprints = None
        if field_model is not None:
            try:
                footprints = structure_footprints(field_model, hub_prisms(field_model))
            except Exception:
                footprints = None
        self._base = self._render_base(footprints)
        # Optional scouted reference routes (e.g. Manta auto paths), precomputed
        # to pixels once — they are static for the whole match.
        self._reference: list[dict] = []

    def field_to_pixel(self, xy) -> tuple:
        # WPILib field coordinates use +Y across the field. For the observer HUD,
        # draw +Y upward on screen so the map matches the main broadcast view.
        return (xy[0] * self.s, (self.W - xy[1]) * self.s)

    def _poly_px(self, poly) -> np.ndarray:
        return np.array([self.field_to_pixel(p) for p in poly], np.int32)

    def _render_base(self, footprints) -> np.ndarray:
        img = np.full((self.h, self.w, 3), _BG, np.uint8)
        cv2.rectangle(img, (0, 0), (self.w - 1, self.h - 1), (80, 80, 80), 2)
        cv2.line(img, (self.w // 2, 0), (self.w // 2, self.h), (185, 185, 185), 2)

        for kind, polys in (footprints or {}).items():
            color = _STRUCTURE.get(kind, (120, 120, 120))
            for poly in polys:
                pts = self._poly_px(poly)
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)
                cv2.polylines(img, [pts], True, color, 2)

        cv2.putText(img, "BLUE", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, _ALLIANCE["blue"], 1, cv2.LINE_AA)
        cv2.putText(img, "RED", (self.w - 48, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, _ALLIANCE["red"], 1, cv2.LINE_AA)
        return img

    def set_reference_paths(self, paths) -> None:
        """Store scouted reference routes (``MantaAutoPath``-like: ``alliance``,
        ``team_number``, ``path_m`` in field meters) to draw as ghost routes."""
        self._reference = []
        for p in paths or []:
            # Smooth the grid-quantized human path in meter space before projecting.
            smoothed = _smooth_arc(list(getattr(p, "path_m", ())))
            pts = [self.field_to_pixel(xy) for xy in smoothed]
            if len(pts) < 1:
                continue
            self._reference.append({
                "alliance": getattr(p, "alliance", None),
                "team": str(p.team_number) if getattr(p, "team_number", None) else None,
                "pts_px": pts,
            })

    def render(self, tracks) -> np.ndarray:
        img = self._base.copy()
        if self._reference:
            # Ghost routes go on a separate layer blended at half opacity, so they
            # read as a translucent band the crisp live trail rides on top of.
            overlay = img.copy()
            for ref in self._reference:
                self._draw_reference_path(overlay, ref)
            cv2.addWeighted(overlay, _REF_ALPHA, img, 1 - _REF_ALPHA, 0, img)
            # Labels drawn after the blend, at full opacity, so they stay legible.
            for ref in self._reference:
                self._draw_reference_label(img, ref)
            self._draw_reference_legend(img)
        for t in tracks:
            self._draw_trail(img, t)
        for t in tracks:
            self._draw_track(img, t)
        return img

    @staticmethod
    def _dashed_line(img, a, b, color, thickness=2, on=9, off=5) -> None:
        a = np.asarray(a, float)
        d = np.asarray(b, float) - a
        length = float(np.hypot(d[0], d[1]))
        if length < 1e-6:
            return
        step = d / length
        pos = 0.0
        while pos < length:
            end = min(pos + on, length)
            p0 = a + step * pos
            p1 = a + step * end
            cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                     color, thickness, cv2.LINE_AA)
            pos = end + off

    def _draw_reference_path(self, img, ref: dict) -> None:
        pts = ref["pts_px"]
        color = _ALLIANCE.get(ref["alliance"], _ALLIANCE[None])
        for i in range(len(pts) - 1):
            self._dashed_line(img, pts[i], pts[i + 1], color,
                              thickness=_REF_WIDTH, on=16, off=10)
        start = (int(pts[0][0]), int(pts[0][1]))
        cv2.circle(img, start, 8, color, 2, cv2.LINE_AA)      # start: open ring
        end = (int(pts[-1][0]), int(pts[-1][1]))
        cv2.drawMarker(img, end, color, cv2.MARKER_TILTED_CROSS, 16, 3, cv2.LINE_AA)

    def _draw_reference_label(self, img, ref: dict) -> None:
        if not ref["team"]:
            return
        x, y = ref["pts_px"][0]
        org = (int(x) + 10, int(y) - 8)
        color = _ALLIANCE.get(ref["alliance"], _ALLIANCE[None])
        cv2.putText(img, ref["team"], org, cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (250, 250, 250), 3, cv2.LINE_AA)
        cv2.putText(img, ref["team"], org, cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    color, 1, cv2.LINE_AA)

    def _draw_reference_legend(self, img) -> None:
        cv2.putText(img, "- - scouted (Manta)", (8, self.h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)

    def _draw_trail(self, img, t: RadarTrack) -> None:
        trail = list(t.trail or ())
        if len(trail) < 2:
            return
        color = np.array(_ALLIANCE.get(t.alliance, _ALLIANCE[None]), float)
        bg = np.full(3, _BG, float)
        n = len(trail)
        for i in range(n - 1):
            a = self.field_to_pixel(trail[i])
            b = self.field_to_pixel(trail[i + 1])
            # Older segments fade toward the carpet.
            mix = 0.25 + 0.55 * (i + 1) / n
            seg = tuple(int(v) for v in (bg + (color - bg) * mix))
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     seg, 2, cv2.LINE_AA)

    def _draw_track(self, img, t: RadarTrack) -> None:
        u, v = self.field_to_pixel(t.pos)
        c = (int(u), int(v))
        color = _ALLIANCE.get(t.alliance, _ALLIANCE[None])
        r = 8

        if t.state == "coasting":
            # Hollow, washed-out dot: last known position, not a live fix.
            faded = tuple(int(_BG + (ch - _BG) * 0.55) for ch in color)
            cv2.circle(img, c, r, faded, 2, cv2.LINE_AA)
            cv2.circle(img, c, 2, faded, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, c, r + 1, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.circle(img, c, r, color, -1, cv2.LINE_AA)

        if t.corner_only:
            # Dashed outer ring: position is corner-camera-sourced (poor range).
            for a0 in range(0, 360, 90):
                cv2.ellipse(img, c, (r + 5, r + 5), 0, a0, a0 + 50,
                            (60, 60, 60), 1, cv2.LINE_AA)

        label = t.team or (f"S{t.slot_id}" if t.slot_id is not None else None)
        if label:
            org = (c[0] - 4 * len(label), c[1] - r - 6)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (250, 250, 250), 3, cv2.LINE_AA)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (20, 20, 20), 1, cv2.LINE_AA)
