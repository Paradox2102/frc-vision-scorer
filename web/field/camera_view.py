"""A physical camera = a crop rect into the composite + overlay masks + its own
calibration. extract() turns a composite frame into this camera's working frame."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class CameraView:
    name: str
    crop: tuple                  # (x, y, w, h) in full-composite pixels
    masks: tuple                 # tuple of (x, y, w, h), relative to crop origin
    calibration: Optional[object]  # web.field.calibration.CameraCalibration or None

    def extract(self, composite_frame: np.ndarray) -> np.ndarray:
        x, y, w, h = self.crop
        crop = composite_frame[y:y + h, x:x + w].copy()  # copy: never mutate source
        for (mx, my, mw, mh) in self.masks:
            crop[my:my + mh, mx:mx + mw] = 0
        return crop
