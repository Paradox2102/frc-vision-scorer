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
# Match the /calibrate overlay legend: cyan hubs, orange bumps,
# magenta towers, pink trenches (BGR).
_STRUCTURE = {
    "hubs": (200, 200, 0),
    "bumps": (0, 140, 230),
    "towers": (200, 0, 200),
    "trenches": (180, 170, 230),
}


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

    def render(self, tracks) -> np.ndarray:
        img = self._base.copy()
        for t in tracks:
            self._draw_trail(img, t)
        for t in tracks:
            self._draw_track(img, t)
        return img

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
