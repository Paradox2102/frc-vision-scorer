"""Season field model: WPILib AprilTag layout + dimensions + alliance geometry."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# FRC 36h11 tag edge length (meters). TODO: confirm from 2026 game manual.
TAG_SIZE_M = 0.1651  # 6.5 in (provisional)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "field"


@dataclass
class FieldModel:
    tags: dict[int, dict]   # id -> {"t": (3,) array, "q": (w,x,y,z)}
    length_m: float
    width_m: float

    @classmethod
    def load(cls, season: str) -> "FieldModel":
        path = DATA_DIR / f"{season}.json"
        data = json.loads(path.read_text())
        tags = {}
        for tag in data["tags"]:
            tr = tag["pose"]["translation"]
            q = tag["pose"]["rotation"]["quaternion"]
            tags[int(tag["ID"])] = {
                "t": np.array([tr["x"], tr["y"], tr["z"]], dtype=float),
                "q": np.array([q["W"], q["X"], q["Y"], q["Z"]], dtype=float),
            }
        return cls(tags=tags, length_m=float(data["field"]["length"]),
                   width_m=float(data["field"]["width"]))

    def tag_translation(self, tag_id: int) -> np.ndarray:
        return self.tags[tag_id]["t"]

    def tag_quaternion(self, tag_id: int) -> np.ndarray:
        return self.tags[tag_id]["q"]

    def alliance_side(self, point_xy) -> str:
        return "blue" if point_xy[0] < self.length_m / 2 else "red"
