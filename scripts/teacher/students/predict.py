"""Inference helper for the S2 legibility gate and S3 robot/not-robot students.

Input crops are BGR uint8 arrays cut like the harvester cuts them: the detector box
padded by 25% (bumper) or 10% (whole robot) of its size on each side, clamped to the
frame (``pad_crop``). Features: frozen DINOv3-S explicit CLS, letterbox 224 (features.py).

  from scripts.teacher.students.predict import Students
  st = Students()                       # loads data/teacher/models/s2_legibility.pt, s3_robot.pt
  out = st.predict(crops_bgr)           # {"s2": [N,3] (legible, illegible, not_bumper), "s3": [N]}
  keep = out["s2"][:, 0] >= st.s2_threshold      # S2 gate (keeps ~95% of legible on b1 CV)
  robot = out["s3"] >= st.s3_threshold           # S3 (>= 99.5% robot recall on b1 CV)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from scripts.teacher.students import features as F
from scripts.teacher.students.heads import MODELS, Head, size_feats_from_wh

PADS = {"bumper": 0.25, "robot": 0.10}


def pad_crop(frame: np.ndarray, box, detector: str) -> np.ndarray:
    """Cut ``box`` from ``frame`` with the harvester's padding for ``detector``."""
    from scripts.teacher.harvest import pad_box
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = pad_box(box, PADS[detector], w, h)
    return frame[y0:y1, x0:x1]


class Students:
    def __init__(self, models_dir: Path = MODELS, device: Optional[str] = None,
                 backbone: Optional[F.Backbone] = None, s2_ckpt: Optional[dict] = None,
                 s3_ckpt: Optional[dict] = None):
        import torch
        load = lambda n: torch.load(Path(models_dir) / n, map_location="cpu", weights_only=True)  # noqa: E731
        self.ck2 = s2_ckpt if s2_ckpt is not None else load("s2_legibility.pt")
        self.ck3 = s3_ckpt if s3_ckpt is not None else load("s3_robot.pt")
        for ck in (self.ck2, self.ck3):
            f = ck["feature"]
            if (f["mode"], f["size"]) != (F.DEFAULT_MODE, F.DEFAULT_SIZE):
                raise ValueError(f"checkpoint features {f} != features.py defaults")
        self.s2 = Head.from_ckpt(self.ck2["head"])
        self.s3 = Head.from_ckpt(self.ck3["head"])
        self.s2_threshold = float(self.ck2["threshold_keep95"])
        self.s3_threshold = float(self.ck3["threshold_recall995"])
        self.bb = backbone or F.Backbone(device=device)

    def features(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
        return self.bb.embed_crops(list(crops_bgr))

    def _x(self, X: np.ndarray, wh: np.ndarray, use_size: bool) -> np.ndarray:
        return np.concatenate([X, size_feats_from_wh(wh)], 1) if use_size else X

    def predict(self, crops_bgr: Sequence[np.ndarray]) -> dict:
        if not len(crops_bgr):
            return {"s2": np.zeros((0, 3), np.float32), "s3": np.zeros(0, np.float32)}
        X = self.features(crops_bgr)
        wh = np.array([(c.shape[1], c.shape[0]) for c in crops_bgr], np.float32)
        p2 = self.s2.predict_proba(self._x(X, wh, self.ck2["use_size"]))
        p3 = self.s3.predict_proba(self._x(X, wh, self.ck3["use_size"]))[:, 1]
        return {"s2": p2.astype(np.float32), "s3": p3.astype(np.float32)}


def predict(crops_bgr: Sequence[np.ndarray], _cache={}) -> dict:  # noqa: B006
    """Module-level convenience: lazily loads the default checkpoints once."""
    if "st" not in _cache:
        _cache["st"] = Students()
    return _cache["st"].predict(crops_bgr)
