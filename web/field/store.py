"""Persist camera calibrations as JSON under web/uploads/."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from web.field.calibration import CameraCalibration

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"


def _path(event_key: str, camera: str) -> Path:
    safe = "".join(c for c in f"{event_key}_{camera}" if c.isalnum() or c in "_-")
    return UPLOAD_DIR / f"calibration_{safe}.json"


def save(event_key: str, camera: str, cal: CameraCalibration) -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _path(event_key, camera).write_text(json.dumps(cal.to_json(), indent=2))


def load(event_key: str, camera: str) -> CameraCalibration | None:
    p = _path(event_key, camera)
    if not p.exists():
        return None
    return CameraCalibration.from_json(json.loads(p.read_text()))


def load_intrinsics(event_key: str, camera: str):
    cal = load(event_key, camera)
    return (None, None) if cal is None else (cal.K, np.asarray(cal.D).reshape(-1))
