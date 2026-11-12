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


def start_zones(field_dims, depth_m) -> list:
    """Legal autonomous starting bands: robots start touching (or near) their
    alliance wall, so seed-time priors only apply within ``depth_m`` of each
    end wall, full field width. Season rules vary; depth is a knob, and these
    complement (never replace) the exclusion zones."""
    L, W = float(field_dims[0]), float(field_dims[1])
    d = float(depth_m)
    return [(0.0, 0.0, d, W), (L - d, 0.0, L, W)]


def prism_bounding_zones(prisms, margin=0.2) -> list:
    """Convert static prism footprints to rectangular no-robot zones."""
    zones = []
    for prism in prisms:
        xs = prism.footprint[:, 0]
        ys = prism.footprint[:, 1]
        # A reef is a hexagon: its bounding rect already overhangs the four angled
        # faces, where robots score with their centres ~0.4 m off the face, so a
        # margin would block seeding them (rect corners reach past the faces).
        m = 0.0 if getattr(prism, "kind", "hub") == "reef" else margin
        zones.append((
            float(xs.min()) - m,
            float(ys.min()) - m,
            float(xs.max()) + m,
            float(ys.max()) + m,
        ))
    return zones


def inflate_zones(zones, margin: float) -> list:
    """Grow each zone rect outward by ``margin`` meters, normalizing corner
    order. Used at seed-gating sites: corner cams project a structure's ground
    contact with enough range error (2026hop qm28: 17cm) to land just outside
    a tightly drawn rect."""
    if not zones:
        return []
    out = []
    for x1, y1, x2, y2 in zones:
        out.append((
            min(x1, x2) - margin,
            min(y1, y2) - margin,
            max(x1, x2) + margin,
            max(y1, y2) + margin,
        ))
    return out


def ray_enters_zones(camera_xy, contact_xy, zones,
                     max_beyond_m: float = 3.0, step_m: float = 0.25) -> bool:
    """True if the ground ray from ``camera_xy`` through ``contact_xy``,
    continued past the contact, enters any zone within ``max_beyond_m``.

    Low cameras see a structure's base occluded (balls, wall lip), so the
    detected bottom edge projects to a contact point SHORT of the structure —
    always between camera and structure, never beyond. A contact whose ray
    pierces a static zone just behind it is that occluded-base signature
    (2026hop qm28: blue outpost contact landed 1.9m short of its zone)."""
    if not zones:
        return False
    cx, cy = float(camera_xy[0]), float(camera_xy[1])
    px, py = float(contact_xy[0]), float(contact_xy[1])
    dx, dy = px - cx, py - cy
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-6:
        return False
    dx, dy = dx / norm, dy / norm
    t = step_m
    while t <= max_beyond_m:
        if point_in_zones((px + t * dx, py + t * dy), zones):
            return True
        t += step_m
    return False


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
