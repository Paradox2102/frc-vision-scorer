"""Where each physical camera lives inside the composite broadcast frame, and
which graphics overlays cover field area. Pure data + a thin loader.

The global default (data/field/composite_layout.json) was measured on the 2026
Championship broadcast. Other events tile their cameras differently, so an
event can carry its own layout in web/uploads/composite_layout_<event>.json
(written by save_event_layout, e.g. from web.field.layout_detect). load_layout
(event_key=...) prefers that record when its status is "auto" or "confirmed"
and falls back to the global default otherwise.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import json
import os

_LAYOUT_PATH = Path(__file__).resolve().parents[2] / "data/field/composite_layout.json"

# Statuses an event layout record can carry. Only USABLE ones replace the
# global default; "rejected" keeps the record (and its diag) for review, and
# "kept_default" records a detection that was NOT adopted because the event's
# existing calibrations were solved in the default layout's crops (calibrations
# are crop-local pixels, so moving a crop origin would silently break them).
EVENT_LAYOUT_STATUSES = ("auto", "confirmed", "rejected", "kept_default")
USABLE_STATUSES = ("auto", "confirmed")


@dataclass(frozen=True)
class CameraLayout:
    name: str
    crop: tuple          # (x, y, w, h) in full-composite pixels
    masks: tuple         # tuple of (x, y, w, h) rects, relative to the crop origin


def layout_to_cameras(layout: dict) -> dict[str, CameraLayout]:
    """{"cameras": {name: {"crop": [...], "masks": [...]}}} (the shape of
    composite_layout.json, a detect_layout() result or an event record) ->
    {name: CameraLayout}."""
    out = {}
    for name, spec in (layout.get("cameras") or {}).items():
        crop = tuple(int(v) for v in spec["crop"])
        if len(crop) != 4 or crop[2] <= 0 or crop[3] <= 0:
            raise ValueError(f"bad crop for camera {name!r}: {spec['crop']!r}")
        out[name] = CameraLayout(
            name=name,
            crop=crop,
            masks=tuple(tuple(int(v) for v in m) for m in spec.get("masks", [])),
        )
    return out


def _safe(event_key: str) -> str:
    # same convention as web/field/store.py
    return "".join(c for c in str(event_key) if c.isalnum() or c in "_-")


def _upload_dir() -> Path:
    # read store.UPLOAD_DIR at call time so tests that point it at a tmp dir
    # also redirect event layouts
    from web.field import store
    return Path(store.UPLOAD_DIR)


def event_layout_path(event_key: str) -> Path:
    return _upload_dir() / f"composite_layout_{_safe(event_key)}.json"


def load_event_layout_record(event_key: str) -> Optional[dict]:
    """The full per-event record (cameras, status, source, confidence, diag,
    saved_at), or None when the event has none / it is unreadable."""
    if not event_key or not _safe(event_key):
        return None
    p = event_layout_path(event_key)
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def event_layout_preview_path(event_key: str) -> Path:
    """JPEG of the composite median frame with the layout's crops/masks drawn
    (rendered once at detection time, next to the record)."""
    return _upload_dir() / f"composite_layout_{_safe(event_key)}.jpg"


def save_event_layout(event_key: str, layout_dict: dict, status: str = "auto",
                      source: str = "detect", **extra) -> Path:
    """Persist an event's composite layout. layout_dict is a detect_layout()
    result or any {"frame_size", "cameras", ...} dict; confidence and diag are
    kept when present. ``extra`` keys (e.g. reason, detected) are stored as-is.
    Returns the written path."""
    if not event_key or not _safe(event_key):
        raise ValueError(f"bad event key {event_key!r}")
    if status not in EVENT_LAYOUT_STATUSES:
        raise ValueError(f"status must be one of {EVENT_LAYOUT_STATUSES}, got {status!r}")
    cams = layout_to_cameras(layout_dict)            # validates crops
    rec = {
        "event_key": event_key,
        "status": status,
        "source": source,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frame_size": list(layout_dict.get("frame_size") or [1920, 1080]),
        "cameras": {n: {"crop": list(c.crop), "masks": [list(m) for m in c.masks]}
                    for n, c in cams.items()},
        "confidence": dict(layout_dict.get("confidence") or {}),
        "diag": layout_dict.get("diag") or {},
    }
    rec.update(extra)
    p = event_layout_path(event_key)
    p.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:       # unique temp name per writer
            f.write(json.dumps(rec, indent=1, default=str))
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def load_layout(path: Path = _LAYOUT_PATH, *, event_key: Optional[str] = None) -> dict[str, CameraLayout]:
    """Camera layouts by name. With event_key, the event's own layout when it
    has a usable record (status auto/confirmed, with a main camera); otherwise
    the layout at path (the global default)."""
    if event_key:
        rec = load_event_layout_record(event_key)
        if rec and rec.get("status") in USABLE_STATUSES:
            try:
                cams = layout_to_cameras(rec)
            except (KeyError, TypeError, ValueError):
                cams = {}
            if "main" in cams:
                return cams
    raw = json.loads(Path(path).read_text())
    return layout_to_cameras(raw)
