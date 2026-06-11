"""Static-structure zones in field coordinates (meters).

A zone marks a known field element (e.g. a fuel/ball structure) that the robot
detector false-positives on. The rule the tracker enforces: a NEW slot is never
seeded from a detection inside a zone — a real robot is never *born* inside a
field element, it drives in (and is therefore already an active track). Zones
are field-space, so one definition applies across every camera via fusion.

Persisted per-event as JSON under web/uploads/, mirroring field/store.py.
"""
from __future__ import annotations

import json
from pathlib import Path

Zone = tuple  # (xmin, ymin, xmax, ymax) in field meters

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"


def point_in_zones(xy, zones) -> bool:
    """True if field point (x, y) falls inside any zone (corners any order)."""
    if not zones:
        return False
    x, y = float(xy[0]), float(xy[1])
    for x1, y1, x2, y2 in zones:
        if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
    return False


def prism_bounding_zones(prisms, margin=0.2) -> list:
    """Convert static prism footprints to rectangular no-robot zones."""
    zones = []
    for prism in prisms:
        xs = prism.footprint[:, 0]
        ys = prism.footprint[:, 1]
        zones.append((
            float(xs.min()) - margin,
            float(ys.min()) - margin,
            float(xs.max()) + margin,
            float(ys.max()) + margin,
        ))
    return zones


def _path(event_key: str) -> Path:
    safe = "".join(c for c in event_key if c.isalnum() or c in "_-")
    return UPLOAD_DIR / f"zones_{safe}.json"


def save_zones(event_key: str, zones) -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"zones": [[float(v) for v in rect] for rect in zones]}
    _path(event_key).write_text(json.dumps(payload, indent=2))


def load_zones(event_key: str) -> list:
    p = _path(event_key)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return [tuple(float(v) for v in rect) for rect in data.get("zones", [])]
