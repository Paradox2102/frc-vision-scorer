"""AdvantageScope 2D field drawings (data/field/advantagescope/).

Each ``{year}[_andymark]_field2d.{jpg,json}`` pair is pre-rotated so the blue
wall is on the LEFT and WPILib +y is UP (image top = y=W), the same frame as
``Radar.field_to_pixel``. The JSON gives the pixel rect of the field carpet.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

FIELD2D_DIR = Path(__file__).resolve().parents[2] / "data" / "field" / "advantagescope"


def field2d_names() -> set[str]:
    """Stems (e.g. "2026_field2d") that have both a .json and a .jpg."""
    return {p.stem for p in FIELD2D_DIR.glob("*_field2d.json")
            if p.with_suffix(".jpg").is_file()}


def _stem_for_layout(layout_stem: str) -> str | None:
    year = str(layout_stem or "")[:4]
    if not year.isdigit():
        return None
    names = field2d_names()
    if "andymark" in str(layout_stem).lower() and f"{year}_andymark_field2d" in names:
        return f"{year}_andymark_field2d"
    return f"{year}_field2d" if f"{year}_field2d" in names else None


@lru_cache(maxsize=None)
def _load_meta(name: str) -> dict | None:
    try:
        cfg = json.loads((FIELD2D_DIR / f"{name}.json").read_text())
        return {
            "name": name,
            "image_path": str(FIELD2D_DIR / f"{name}.jpg"),
            "image_size": [int(v) for v in cfg["image_size"]],
            "field_top_left_px": [float(v) for v in cfg["field_top_left_px"]],
            "field_bottom_right_px": [float(v) for v in cfg["field_bottom_right_px"]],
            "field_m": [float(cfg["field_length_m"]), float(cfg["field_width_m"])],
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def field2d_for_layout(layout_stem: str) -> dict | None:
    """Drawing metadata for a tag-layout stem (``FieldModel.season``, e.g.
    "2026-rebuilt-welded", "2025-reefscape-andymark"), or None if none exists."""
    name = _stem_for_layout(layout_stem)
    meta = _load_meta(name) if name else None
    return dict(meta) if meta else None


@lru_cache(maxsize=8)
def field2d_image(image_path: str) -> np.ndarray | None:
    """BGR image (cached; treat as read-only) or None if unreadable."""
    import cv2
    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)
