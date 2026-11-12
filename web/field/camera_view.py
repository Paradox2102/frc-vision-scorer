"""A physical camera = a crop rect into the composite + overlay masks + its own
calibration. extract() turns a composite frame into this camera's working frame."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from web.field.composite_layout import load_layout
from web.field import store as field_store


@dataclass
class CameraView:
    name: str
    crop: tuple                  # (x, y, w, h) in full-composite pixels
    masks: tuple                 # tuple of (x, y, w, h), relative to crop origin
    calibration: Optional[object]  # web.field.calibration.CameraCalibration or None
    # First source frame this view must NOT feed tracking from (None: the whole
    # match). Set on corner views of a season whose broadcast re-tiles them at the
    # endgame (Season.endgame_layout_switch, 2025): after it their pixels no
    # longer match the crop/calibration. The HUD still shows the raw panel.
    active_until_frame: Optional[int] = None

    def active_at(self, frame_num: int) -> bool:
        return self.active_until_frame is None or frame_num < self.active_until_frame

    def extract(self, composite_frame: np.ndarray) -> np.ndarray:
        x, y, w, h = self.crop
        crop = composite_frame[y:y + h, x:x + w].copy()  # copy: never mutate source
        for (mx, my, mw, mh) in self.masks:
            crop[my:my + mh, mx:mx + mw] = 0
        return crop


def load_views(event_key: str, selection: dict | None = None) -> list["CameraView"]:
    """Build the list of calibrated CameraViews for an event. Cameras without a
    saved calibration are omitted. 'main' is always ordered first so it remains the
    reference view for fusion.

    selection: optional {camera_name: version_id} map. When provided, each named
    camera uses the specified calibration version. Cameras not in the map (and
    cameras when selection is None) default to the latest version by created_at
    (with version_id as a microsecond-resolution tie-breaker).

    Falls back to the legacy field_store.load() when no versioned records exist,
    for backward compatibility with the single-file calibration format.

    Crops come from the event's own composite layout when it has one
    (web.field.composite_layout.load_layout(event_key=...)), else the global one.
    """
    layout = load_layout(event_key=event_key)   # per-event layout, else global
    views = []
    for name, cl in layout.items():
        versions = field_store.load_versions(event_key, name)
        if versions:
            chosen = None
            if selection and name in selection:
                chosen = next((v for v in versions if v["version_id"] == selection[name]), None)
            if chosen is None:
                chosen = max(versions,
                             key=lambda v: (v.get("created_at") or "", v.get("version_id") or ""))
            cal = chosen["calibration"]
        else:
            # Legacy fallback: single-file calibration (supports test mocks of field_store.load)
            cal = field_store.load(event_key, name)
            if cal is None:
                continue
        masks = cal.masks if (hasattr(cal, "masks") and cal.masks) else cl.masks
        views.append(CameraView(name=name, crop=cl.crop, masks=tuple(masks), calibration=cal))
    views.sort(key=lambda v: (v.name != "main", v.name))  # main first, then stable
    return views
