"""Season field model: WPILib AprilTag layout + dimensions + alliance geometry."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# FRC 36h11 tag edge length (meters). TODO: confirm from 2026 game manual.
TAG_SIZE_M = 0.1651  # 6.5 in (provisional)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "field"


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Hamilton quaternion [W, X, Y, Z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


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

    def tag_corners_3d(self, tag_id: int) -> np.ndarray:
        """4 corner positions (4,3) in field meters. Local corner order
        (provisional, reconciled with the detector in Task 3):
        [-y-z, +y-z, +y+z, -y+z] with normal along local +x."""
        s = TAG_SIZE_M / 2.0
        local = np.array([
            [0.0, -s, -s],
            [0.0,  s, -s],
            [0.0,  s,  s],
            [0.0, -s,  s],
        ])
        R = _quat_to_matrix(self.tag_quaternion(tag_id))
        return (R @ local.T).T + self.tag_translation(tag_id)
