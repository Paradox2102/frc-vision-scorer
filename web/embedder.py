"""DINOv2 appearance embeddings for robot re-identification.

FRC-agnostic: images in, L2-normalized vectors out. Swap the embedding backend
by changing DEFAULT_MODEL (e.g. to a SigLIP checkpoint) — no caller changes.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

DEFAULT_MODEL = "facebook/dinov2-small"  # 384-dim CLS embedding
MIN_CROP_PX = 10  # boxes smaller than this on either side yield no embedding
BOX_INSET = 0.08  # shrink box by this fraction per side to shed background


def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Cosine similarity of two vectors. Returns 0.0 if either is None.

    Assumes inputs are L2-normalized (embeddings are), but normalizes defensively.
    """
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _inset_box(box, w: int, h: int):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < MIN_CROP_PX or bh < MIN_CROP_PX:
        return None
    dx, dy = int(bw * BOX_INSET), int(bh * BOX_INSET)
    return x1 + dx, y1 + dy, x2 - dx, y2 - dy


class RobotEmbedder:
    """Lazy-loaded DINOv2 embedder. The model loads on first embed() call."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._device = None
        self._torch = None
        self._load_failed = False  # latch: don't retry a failed load every frame

    def _ensure_model(self) -> bool:
        """Load model/processor once. Returns False on failure (graceful degradation)."""
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            self._model.train(False)  # inference mode: disable dropout etc.
            self._torch = torch
            return True
        except Exception as e:  # pragma: no cover - exercised only on env failure
            print(f"  RobotEmbedder: failed to load {self.model_name} ({e}); "
                  f"embeddings disabled, falling back to position-only re-ID.")
            self._model = None
            self._load_failed = True
            return False

    def embed(self, frame: np.ndarray, box) -> Optional[np.ndarray]:
        """Return an L2-normalized embedding for the crop, or None if invalid/unavailable."""
        return self.embed_batch(frame, [box])[0]

    def embed_batch(self, frame: np.ndarray, boxes) -> list[Optional[np.ndarray]]:
        """Embed multiple crops from one frame in a single forward pass.

        Returns a list aligned with `boxes`; entries are None for invalid crops.
        """
        h, w = frame.shape[:2]
        insets = [_inset_box(b, w, h) for b in boxes]
        valid_idx = [i for i, b in enumerate(insets) if b is not None]
        out: list[Optional[np.ndarray]] = [None] * len(boxes)
        if not valid_idx:
            return out
        if not self._ensure_model():
            return out  # all None — caller falls back to position-only

        torch = self._torch  # cached during _ensure_model()
        crops = []
        for i in valid_idx:
            x1, y1, x2, y2 = insets[i]
            crop_bgr = frame[y1:y2, x1:x2]
            crops.append(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

        inputs = self._processor(images=crops, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        cls = outputs.last_hidden_state[:, 0]            # CLS token, [N, 384]
        cls = torch.nn.functional.normalize(cls, dim=1)  # L2-normalize
        vecs = cls.cpu().numpy().astype(np.float32)
        for slot, i in enumerate(valid_idx):
            out[i] = vecs[slot]
        return out
