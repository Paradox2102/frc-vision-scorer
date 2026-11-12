"""Field season registry: which field an event is played on.

The pipeline used to hard-code the 2026 field ("2026-rebuilt-welded"). An event's
season now follows its year (the first 4 characters of the TBA event key), so a
2025 event calibrates against the 2025 AprilTag layout, draws the 2025 wireframe,
builds 2025 structures and searches the 2025 match clock.

  season_for_event("2025hop")   -> the 2025 REEFSCAPE season (welded layout)
  season_for_event(None / "")   -> the 2026 default (no event: today's behaviour)
  season_for_event("test")      -> the 2026 default (key carries no year)
  season_for_event("2019txho")  -> a season with NO tag layout: auto-calibration
                                   fails with "no AprilTag layout for 2019 ..."

An event can carry an override in web/uploads/field_season_<event>.json (same
upload dir as the per-event composite layouts, so tests redirect it the same way):
{"year": 2025, "variant": "andymark"} plays the event on another year's field
and/or the AndyMark tag-layout variant. Welded is the default variant.

``season_for_layout(stem)`` maps a tag-layout stem (CameraCalibration.field_season,
FieldModel.season) back to its season so drawing / structure code can dispatch.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_YEAR = 2026
DEFAULT_LAYOUT = "2026-rebuilt-welded"

WIREFRAME_DIR = Path(__file__).resolve().parents[2] / "data" / "field" / "wireframes"

# structure kinds: what fixed structures (no-robot zones, occluders, radar shapes)
# a season's field has. None -> no structures; the pipeline still runs.
STRUCTURE_2026_HUB = "2026_hub"
STRUCTURE_2025_REEF = "2025_reef"

VARIANTS = ("welded", "andymark")


class SeasonError(ValueError):
    """Bad override input (unknown year / variant)."""


@dataclass(frozen=True)
class Season:
    year: int
    name: str
    tag_layout: Optional[str]            # data/field/<stem>.json, None = no AprilTags
    tag_layout_andymark: Optional[str]   # AndyMark-field variant, if one exists
    wireframe: Optional[str]             # data/field/wireframes/<stem>.json, or None
    auto_s: float
    teleop_s: float
    endgame_s: float
    structures: Optional[str]            # STRUCTURE_* or None
    field_m: Optional[tuple] = None      # (length, width) of the welded field
    known: bool = True                   # False: year not in the registry
    variant: str = "welded"
    source: str = "year"                 # "year" | "default" | "override"

    @property
    def has_tags(self) -> bool:
        return bool(self.tag_layout)

    def public(self) -> dict:
        """JSON-able summary for API responses."""
        return {"year": self.year, "name": self.name, "tag_layout": self.tag_layout,
                "variant": self.variant, "has_tags": self.has_tags,
                "wireframe": self.wireframe, "structures": self.structures,
                "auto_s": self.auto_s, "teleop_s": self.teleop_s,
                "endgame_s": self.endgame_s, "known": self.known, "source": self.source,
                "has_andymark": bool(self.tag_layout_andymark)}


SEASONS: dict = {
    # 2026 endgame_s is not used by the pipeline yet — verify before relying on it.
    2026: Season(2026, "2026 REBUILT", "2026-rebuilt-welded", None,
                 "2026-rebuilt-welded", auto_s=20.0, teleop_s=140.0, endgame_s=30.0,
                 structures=STRUCTURE_2026_HUB, field_m=(16.541, 8.069)),
    2025: Season(2025, "2025 REEFSCAPE", "2025-reefscape-welded",
                 "2025-reefscape-andymark", "2025-reefscape-welded",
                 auto_s=15.0, teleop_s=135.0, endgame_s=20.0,
                 structures=STRUCTURE_2025_REEF, field_m=(17.548, 8.052)),
    2024: Season(2024, "2024 CRESCENDO", "2024-crescendo", None, None,
                 auto_s=15.0, teleop_s=135.0, endgame_s=20.0, structures=None,
                 field_m=(16.541, 8.211)),
    2023: Season(2023, "2023 CHARGED UP", "2023-chargedup", None, None,
                 auto_s=15.0, teleop_s=135.0, endgame_s=30.0, structures=None,
                 field_m=(16.54175, 8.0137)),
}


def _unknown_season(year: int) -> Season:
    """A year with no registry entry: no AprilTags (FRC fields carried none before
    2023), no structures, the 15 s / 135 s clock every pre-2026 game used."""
    why = "no AprilTags" if year <= 2022 else "unknown season"
    return Season(year, f"{year} ({why})", None, None, None, auto_s=15.0,
                  teleop_s=135.0, endgame_s=30.0, structures=None, field_m=None,
                  known=False)


def season_for_year(year: int, variant: str = "welded") -> Season:
    year = int(year)
    s = SEASONS.get(year) or _unknown_season(year)
    if variant == "andymark" and s.tag_layout_andymark:
        s = replace(s, tag_layout=s.tag_layout_andymark, variant="andymark")
    return s


def default_season() -> Season:
    return replace(SEASONS[DEFAULT_YEAR], source="default")


def event_year(event_key) -> Optional[int]:
    """The year from a TBA event key ("2025hop" -> 2025), or None when the key does
    not start with 4 digits."""
    k = str(event_key or "")
    if len(k) >= 4 and k[:4].isdigit():
        return int(k[:4])
    return None


# ----------------------------------------------------------------- per-event override

def _store_helpers():
    # composite_layout._upload_dir reads store.UPLOAD_DIR at call time so tests
    # that point it at a tmp dir also redirect overrides
    from web.field.composite_layout import _safe, _upload_dir
    return _safe, _upload_dir


def override_path(event_key: str) -> Path:
    _safe, _upload_dir = _store_helpers()
    return _upload_dir() / f"field_season_{_safe(event_key)}.json"


def load_override(event_key) -> Optional[dict]:
    _safe, _ = _store_helpers()
    if not event_key or not _safe(event_key):
        return None
    p = override_path(event_key)
    if not p.is_file():
        return None
    try:
        rec = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def save_override(event_key: str, year: Optional[int] = None,
                  variant: str = "welded") -> Path:
    """Persist {"year", "variant"} for ``event_key``. ``year`` None keeps the key's
    own year (a variant-only override)."""
    _safe, _ = _store_helpers()
    if not event_key or not _safe(event_key):
        raise SeasonError(f"bad event key {event_key!r}")
    if variant not in VARIANTS:
        raise SeasonError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if year is not None:
        year = int(year)
        if year not in SEASONS:
            raise SeasonError(f"no field data for {year}; known: {sorted(SEASONS)}")
        if variant == "andymark" and not SEASONS[year].tag_layout_andymark:
            raise SeasonError(f"{year} has no AndyMark layout")
    rec = {"event_key": event_key, "year": year, "variant": variant,
           "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    p = override_path(event_key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, p)
    return p


def clear_override(event_key: str) -> bool:
    _safe, _ = _store_helpers()
    if not event_key or not _safe(event_key):
        return False
    p = override_path(event_key)
    if p.is_file():
        p.unlink()
        return True
    return False


# ------------------------------------------------------------------------- resolve

def season_for_event(event_key) -> Season:
    """The season ``event_key`` is played on: its override, else its key's year,
    else (no event / no year in the key) the 2026 default."""
    rec = load_override(event_key) if event_key else None
    if rec:
        variant = rec.get("variant") if rec.get("variant") in VARIANTS else "welded"
        year = rec.get("year")
        if year is None:
            year = event_year(event_key)
        if year is not None:
            try:
                return replace(season_for_year(int(year), variant), source="override")
            except (TypeError, ValueError):
                pass
    year = event_year(event_key)
    if year is None:
        return default_season()
    return season_for_year(year)


def layout_for_event(event_key) -> Optional[str]:
    """The tag-layout stem for ``event_key`` (None: its season has no AprilTags)."""
    return season_for_event(event_key).tag_layout


def no_layout_reason(season: Season) -> str:
    return f"no AprilTag layout for {season.year} — calibrate manually"


def season_for_layout(stem) -> Optional[Season]:
    """The season a tag-layout stem belongs to (exact registry match first, then the
    stem's year prefix). ``""``/None -> None (a FieldModel built by hand)."""
    if not stem:
        return None
    for s in SEASONS.values():
        if stem == s.tag_layout:
            return s
        if stem == s.tag_layout_andymark:
            return replace(s, tag_layout=stem, variant="andymark")
    year = event_year(stem)
    return season_for_year(year) if year is not None else None


def structure_kind(field_model) -> Optional[str]:
    """Which fixed structures ``field_model``'s field has. A FieldModel without a
    season (constructed by hand, e.g. in tests) keeps the historical 2026 hub
    behaviour; a season we do not know has none."""
    stem = getattr(field_model, "season", "") or ""
    if not stem:
        return STRUCTURE_2026_HUB
    s = season_for_layout(stem)
    return s.structures if s is not None else None


def is_2026(field_model) -> bool:
    return structure_kind(field_model) == STRUCTURE_2026_HUB


# ------------------------------------------------------------------------ wireframe

_WIREFRAME_CACHE: dict = {}


def load_wireframe(stem) -> Optional[dict]:
    """{group: (N, 6) float array of [x0,y0,z0,x1,y1,z1]} from
    data/field/wireframes/<stem>.json (cached), or None when there is none."""
    if not stem:
        return None
    if stem in _WIREFRAME_CACHE:
        return _WIREFRAME_CACHE[stem]
    out = None
    p = WIREFRAME_DIR / f"{stem}.json"
    if p.is_file():
        try:
            import numpy as np
            data = json.loads(p.read_text())
            out = {}
            for name, segs in (data.get("groups") or {}).items():
                arr = np.asarray(segs, float).reshape(-1, 6)
                out[name] = arr
        except (OSError, ValueError):
            out = None
    _WIREFRAME_CACHE[stem] = out
    return out


def wireframe_for(field_model) -> Optional[dict]:
    """The GLB wireframe groups for ``field_model``'s season, or None."""
    s = season_for_layout(getattr(field_model, "season", "") or "")
    if s is None:
        return None
    # a variant layout (andymark) uses its own wireframe when one exists
    stem = getattr(field_model, "season", "")
    if (WIREFRAME_DIR / f"{stem}.json").is_file():
        return load_wireframe(stem)
    return load_wireframe(s.wireframe)
