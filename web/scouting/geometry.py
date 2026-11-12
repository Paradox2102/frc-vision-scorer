"""Field geometry for scouting: size, symmetry, alliance perspective.

ALLIANCE PERSPECTIVE puts the robot's own alliance wall at x=0 so a team's
red-alliance and blue-alliance matches overlay. Blue is already in that frame
(WPILib origin is the blue-right corner). Red is transformed with the field's
symmetry:
  rotational (2026, 2025):  (x, y) -> (L - x, W - y)
  mirror     (2024, 2023):  (x, y) -> (L - x, y)

The symmetry is MEASURED from the season's AprilTag layout (every tag must map
onto another tag), so a new season needs no code here. Seasons without a tag
layout fall back to rotational.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

from web.field.season import season_for_year

FIELD_DIR = Path(__file__).resolve().parents[2] / "data" / "field"
FIELD2D_DIR = FIELD_DIR / "advantagescope"

ROTATIONAL = "rotational"
MIRROR = "mirror"
_DEFAULT_FIELD = (16.541, 8.069)


@lru_cache(maxsize=None)
def field_dims(year: int) -> tuple:
    s = season_for_year(year)
    return tuple(s.field_m) if s.field_m else _DEFAULT_FIELD


@lru_cache(maxsize=None)
def field_symmetry(year: int) -> str:
    s = season_for_year(year)
    path = FIELD_DIR / f"{s.tag_layout}.json" if s.tag_layout else None
    if not path or not path.exists():
        return ROTATIONAL
    d = json.loads(path.read_text())
    L, W = d["field"]["length"], d["field"]["width"]
    pts = [(t["pose"]["translation"]["x"], t["pose"]["translation"]["y"]) for t in d["tags"]]

    def worst(f):
        return max(min(math.dist(f(p), q) for q in pts) for p in pts)

    rot = worst(lambda p: (L - p[0], W - p[1]))
    mir = worst(lambda p: (L - p[0], p[1]))
    return ROTATIONAL if rot <= mir else MIRROR


def to_alliance(x: float, y: float, alliance: str, year: int) -> tuple:
    """Absolute field metres -> alliance perspective (own wall at x=0)."""
    if alliance != "red":
        return float(x), float(y)
    L, W = field_dims(year)
    if field_symmetry(year) == MIRROR:
        return L - float(x), float(y)
    return L - float(x), W - float(y)


def normalise(x: float, y: float, year: int) -> tuple:
    """Metres -> 0..1 of the full field (RoboRecon PathValue convention)."""
    L, W = field_dims(year)
    return round(float(x) / L, 4), round(float(y) / W, 4)


def frame_info(year: int) -> dict:
    """The coordinate-frame block every export envelope carries."""
    return {"units": "m", "field_m": list(field_dims(year)),
            "perspective": "alliance", "symmetry": field_symmetry(year),
            "note": ("x=0 is the team's own alliance wall, +x toward the opponent "
                     "wall; y follows WPILib after the symmetry transform; "
                     "*_norm values are 0..1 of the full field")}


def field_image(year: int) -> Optional[dict]:
    """AdvantageScope field2d image (blue wall on the left, WPILib x right / y
    up) + the pixel corners of the field rectangle, or None."""
    meta = FIELD2D_DIR / f"{int(year)}_field2d.json"
    img = FIELD2D_DIR / f"{int(year)}_field2d.jpg"
    if not meta.exists() or not img.exists():
        return None
    d = json.loads(meta.read_text())
    return {"path": img, "image_size": d.get("image_size"),
            "field_top_left_px": d.get("field_top_left_px"),
            "field_bottom_right_px": d.get("field_bottom_right_px"),
            "field_m": [d.get("field_length_m"), d.get("field_width_m")]}
