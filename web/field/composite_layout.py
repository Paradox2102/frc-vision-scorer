"""Where each physical camera lives inside the composite broadcast frame, and
which graphics overlays cover field area. Pure data + a thin loader."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

_LAYOUT_PATH = Path(__file__).resolve().parents[2] / "data/field/composite_layout.json"


@dataclass(frozen=True)
class CameraLayout:
    name: str
    crop: tuple          # (x, y, w, h) in full-composite pixels
    masks: tuple         # tuple of (x, y, w, h) rects, relative to the crop origin


def load_layout(path: Path = _LAYOUT_PATH) -> dict[str, CameraLayout]:
    raw = json.loads(Path(path).read_text())
    out = {}
    for name, spec in raw["cameras"].items():
        out[name] = CameraLayout(
            name=name,
            crop=tuple(spec["crop"]),
            masks=tuple(tuple(m) for m in spec.get("masks", [])),
        )
    return out
