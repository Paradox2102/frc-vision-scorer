"""Manta scouting-data reconciliation + cache-only auto-path loader.

Manta (team 3128, https://manta-api.vercel.app) serves human-scouted match data.
Its richest signal is ``auto_field``: a hand-drawn robot trajectory for the
autonomous period, stored as ``"x,y"`` strings on a fixed integer grid.

This module is the single source of truth for turning Manta's grid coordinates
into our WPILib field-meter frame. The network client (``MantaClient``) lives in
``scripts/manta_client.py`` and re-imports these pure pieces; the web app imports
this module directly to overlay cached paths on the Field Map. The loader here is
deliberately **cache-only** — it never performs network I/O, so it is safe to call
during a job without touching Manta's daily-rate-limited API.

Coordinate reconciliation
-------------------------
Manta's grid spans X in [0,100] along the field *length* and Y in [0,52.5] across
the *width*, with RED at LOW X and BLUE at HIGH X. Our WPILib frame
(data/field/2026-rebuilt-welded.json) has the origin at the BLUE corner, +x toward
RED, so blue is at LOW x. Manta-X and our-x therefore run opposite -> flip X. The Y
orientation was validated against 2026cur_qm2 (flip_y=False).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# --- field geometry (kept in sync with data/field/2026-rebuilt-welded.json) ---
FIELD_LENGTH_M = 16.541
FIELD_WIDTH_M = 8.069

# data/manta_cache relative to the repo root (web/field/manta.py -> parents[2]).
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "manta_cache"


@dataclass(frozen=True)
class GridCalibration:
    """Maps Manta grid coords -> WPILib field meters.

    Defaults assume the grid's full declared extent maps onto the field. The X
    axis is flipped because Manta's red(low-X)/blue(high-X) is opposite to our
    blue(low-x)/red(high-x) convention. flip_y validated False on 2026cur_qm2.
    """

    grid_x_min: float = 0.0
    grid_x_max: float = 100.0
    grid_y_min: float = 0.0
    grid_y_max: float = 52.5
    flip_x: bool = True
    flip_y: bool = False

    def to_meters(self, gx: float, gy: float) -> tuple[float, float]:
        fx = (gx - self.grid_x_min) / (self.grid_x_max - self.grid_x_min)
        fy = (gy - self.grid_y_min) / (self.grid_y_max - self.grid_y_min)
        if self.flip_x:
            fx = 1.0 - fx
        if self.flip_y:
            fy = 1.0 - fy
        return fx * FIELD_LENGTH_M, fy * FIELD_WIDTH_M


def parse_auto_field(raw: Iterable[str], cal: GridCalibration) -> list[tuple[float, float]]:
    """Convert a Manta ``auto_field`` list of ``"x,y"`` strings to field meters."""
    pts: list[tuple[float, float]] = []
    for item in raw or []:
        try:
            gx, gy = item.split(",")
            pts.append(cal.to_meters(float(gx), float(gy)))
        except (ValueError, AttributeError):
            continue
    return pts


def match_number_from_key(match_key: str) -> Optional[tuple[str, int]]:
    """``2026cur_qm2`` -> ``('qual', 2)``. Returns None for unsupported levels."""
    if not match_key or "_" not in match_key:
        return None
    tail = match_key.split("_", 1)[1]
    m = re.match(r"^(qm|qf|sf|f)(\d+)", tail)
    if not m:
        return None
    level = {"qm": "qual", "qf": "qf", "sf": "sf", "f": "final"}[m.group(1)]
    return level, int(m.group(2))


@dataclass
class MantaAutoPath:
    """One robot's human-scouted autonomous path, in WPILib field meters."""

    alliance: str
    team_number: Optional[int]
    position: Optional[int]
    path_m: list[tuple[float, float]]


def _best_scout_path(robot: dict, cal: GridCalibration) -> list[tuple[float, float]]:
    """Longest usable (>=2 point) auto_field among a robot's scouts, in meters."""
    best: list[tuple[float, float]] = []
    for sc in robot.get("scouts", []):
        raw = sc.get("data", {}).get("auto_field")
        if not isinstance(raw, list) or len(raw) < 2:
            continue
        path = parse_auto_field(raw, cal)
        if len(path) >= 2 and len(path) > len(best):
            best = path
    return best


def load_match_auto_paths(
    match_key: Optional[str],
    cache_dir: Path = DEFAULT_CACHE,
    cal: GridCalibration = GridCalibration(),
) -> list[MantaAutoPath]:
    """Cache-only: return the scouted auto paths for ``match_key`` in field meters.

    Returns ``[]`` if the key is unparseable or no cache file exists. Never hits
    the network. Robots without a usable (>=2 point) auto_field are omitted.
    """
    parsed = match_number_from_key(match_key or "")
    if parsed is None:
        return []
    mtype, number = parsed
    event_key = match_key.split("_", 1)[0]
    cache_file = Path(cache_dir) / f"{event_key}_{mtype}_{number}.json"
    if not cache_file.exists():
        return []
    try:
        data = json.loads(cache_file.read_text())
    except (ValueError, OSError):
        return []

    out: list[MantaAutoPath] = []
    for robot in data.get("robots", []):
        path = _best_scout_path(robot, cal)
        if not path:
            continue
        out.append(MantaAutoPath(
            alliance=robot.get("alliance"),
            team_number=robot.get("team_number"),
            position=robot.get("position"),
            path_m=path,
        ))
    return out


def overlay_paths_for_job(
    auto_period_only: bool,
    match_key: Optional[str],
    cache_dir: Path = DEFAULT_CACHE,
) -> list[MantaAutoPath]:
    """The Field Map overlay activates only for auto-only runs with cached Manta
    data. Returns the paths to draw, or ``[]`` when the overlay is inactive."""
    if not auto_period_only or not match_key:
        return []
    return load_match_auto_paths(match_key, cache_dir=cache_dir)
