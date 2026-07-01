"""Manta scouting-API client + coordinate reconciliation.

Manta (https://manta-api.vercel.app) serves manual scouting data from team 3128.
We use it as ground truth to validate our vision pipeline. The richest signal is
``auto_field``: a human-drawn robot trajectory for the autonomous period, stored
as a list of ``"x,y"`` strings on a fixed integer grid.

This module does three things:
  1. Pulls + on-disk caches Manta data (auth via the MANTA_API key in .env).
  2. Converts Manta's grid coordinates into our WPILib field-meter frame.
  3. Exposes auto_field paths keyed by (event, match, alliance, position, team).

Coordinate reconciliation
-------------------------
Manta frontend constants (decoded from its SvelteKit bundle)::

    {image: "2026RebuiltField.png", aspect: [600,315], width: 100,
     xOffset: 1, redZoneMaxX: 29, blueZoneMinX: 71}

So Manta's grid spans X in [0,100] along the field *length* and Y in [0,52.5]
across the *width* (52.5 = 100 * 315/600). Crucially the RED zone sits at LOW X
(0-29) and BLUE at HIGH X (71-100).

Our field frame (WPILib, data/field/2026-rebuilt-welded.json): origin at the BLUE
alliance corner, +x toward RED (length 16.541 m), +y across width (8.069 m). So
blue is at LOW x. That means Manta-X and our-x run OPPOSITE directions -> we flip.

The grid->meter scale is mildly anisotropic (the field image has vertical margin),
so we keep per-axis extents configurable in GridCalibration and default to the
declared full-grid extents. Refine GRID_* once a real overlapping match exists.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# The pure Manta coordinate reconciliation (grid->meter, path parsing, field
# dimensions) lives in web/field/manta.py — the single source of truth shared with
# the web app's Field Map overlay. Ensure the repo root is importable when this
# script runs standalone (python scripts/manta_client.py ...), where sys.path[0]
# is scripts/, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from web.field.manta import (  # noqa: E402
    DEFAULT_CACHE,
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    GridCalibration,
    parse_auto_field,
)

BASE_URL = "https://manta-api.vercel.app/api/v1"


class RateLimitError(RuntimeError):
    """Raised on HTTP 429 — Manta's daily quota is exhausted (resets UTC midnight)."""


def _load_api_key() -> str:
    """Read MANTA_API from the repo .env without importing extra deps."""
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("MANTA_API="):
                return line.split("=", 1)[1].strip()
    key = os.environ.get("MANTA_API")
    if key:
        return key
    raise RuntimeError("MANTA_API not found in .env or environment")


# --------------------------------------------------------------------------- #
# API client  (GridCalibration + parse_auto_field re-exported from web.field.manta)
# --------------------------------------------------------------------------- #
@dataclass
class MantaClient:
    api_key: str = field(default_factory=_load_api_key)
    cache_dir: Path = DEFAULT_CACHE
    min_interval_s: float = 0.2  # be polite; respect X-RateLimit headers
    _last_call: float = 0.0

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, path: str, *, cache_key: Optional[str] = None, refresh: bool = False) -> dict:
        cache_file = self.cache_dir / f"{cache_key}.json" if cache_key else None
        if cache_file and cache_file.exists() and not refresh:
            return json.loads(cache_file.read_text())

        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(
            BASE_URL + path, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                self._last_call = time.monotonic()
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            if e.code == 429:
                # Daily quota exhausted — a hard stop, NOT "no more matches".
                raise RateLimitError(f"Manta rate limit hit on {path}: {body}") from e
            raise RuntimeError(f"Manta {path} -> {e.code}: {body}") from e

        if cache_file:
            cache_file.write_text(json.dumps(data))
        return data

    # -- endpoints --
    def events(self, refresh: bool = False) -> list[dict]:
        return self._get("/events", cache_key="events", refresh=refresh)["events"]

    def teams(self, event_key: str, refresh: bool = False) -> list[dict]:
        return self._get(
            f"/events/{event_key}/teams", cache_key=f"{event_key}_teams", refresh=refresh
        )["teams"]

    def match(self, event_key: str, number: int, mtype: str = "qual", refresh: bool = False) -> dict:
        return self._get(
            f"/events/{event_key}/matches/{mtype}/{number}",
            cache_key=f"{event_key}_{mtype}_{number}",
            refresh=refresh,
        )

    # -- high-level: trajectories for an event --
    def auto_paths(
        self,
        event_key: str,
        cal: GridCalibration = GridCalibration(),
        mtype: str = "qual",
        max_matches: int = 200,
        refresh: bool = False,
    ) -> list[dict]:
        """Return one record per (match, robot, scout) that has an auto_field path.

        Each record: event, match_type, match_number, alliance, position,
        team_number, scout_team, n_points, path_m (list of (x,y) meters).
        """
        out: list[dict] = []
        misses = 0
        for n in range(1, max_matches + 1):
            try:
                m = self.match(event_key, n, mtype, refresh=refresh)
            except RateLimitError:
                raise  # daily quota — let the caller see it, don't pretend we're done
            except RuntimeError:
                misses += 1
                if misses > 5:
                    break
                continue
            misses = 0
            for robot in m.get("robots", []):
                for sc in robot.get("scouts", []):
                    raw = sc.get("data", {}).get("auto_field")
                    if not isinstance(raw, list) or len(raw) < 2:
                        continue
                    path = parse_auto_field(raw, cal)
                    if len(path) < 2:
                        continue
                    out.append(
                        {
                            "event": event_key,
                            "match_type": mtype,
                            "match_number": n,
                            "alliance": robot.get("alliance"),
                            "position": robot.get("position"),
                            "team_number": robot.get("team_number"),
                            "scout_team": sc.get("scout_team"),
                            "n_points": len(path),
                            "path_m": path,
                        }
                    )
        return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Cache Manta auto_field trajectories for an event.")
    ap.add_argument("event_key", help="e.g. 2026cur")
    ap.add_argument("--type", default="qual")
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    client = MantaClient()
    try:
        paths = client.auto_paths(args.event_key, mtype=args.type, max_matches=args.max, refresh=args.refresh)
    except RateLimitError as e:
        raise SystemExit(f"Manta daily quota exhausted (resets ~UTC midnight): {e}")
    print(f"{args.event_key}: {len(paths)} robot auto-paths cached under {client.cache_dir}")

    # validation: red robots should land on high-x (red) half, blue on low-x half
    half = FIELD_LENGTH_M / 2
    side_ok = side_tot = 0
    for rec in paths:
        if rec["alliance"] not in ("red", "blue"):
            continue
        mean_x = sum(p[0] for p in rec["path_m"]) / len(rec["path_m"])
        on_own = (rec["alliance"] == "red" and mean_x > half) or (
            rec["alliance"] == "blue" and mean_x < half
        )
        side_ok += int(on_own)
        side_tot += 1
    if side_tot:
        print(f"alliance-side sanity: {side_ok}/{side_tot} robots' mean-x on own half "
              f"({100*side_ok/side_tot:.0f}%) — expect high if flip_x is correct")
