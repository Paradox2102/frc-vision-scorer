"""Appearance embeddings (DINOv3 ViT-S by default) for robot re-identification.

FRC-agnostic: images in, L2-normalized vectors out. Swap the embedding backend
by setting ROBOT_EMBEDDER_MODEL (e.g. to a SigLIP checkpoint, a transformers
name like facebook/dinov2-small, or a "timm/..." name for the timm backend
below) — no caller changes.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

# DINOv3 ViT-S, 384-dim CLS via the timm backend (the facebook/ repos are HF-
# gated; the timm/ mirror is open). Beats dinov2-small on the production-crop
# re-ID eval — relink AUC 0.9525 vs 0.9420; at the 0.55 relink floor 82.8%/4.86%
# true/false pass vs 80.3%/5.92% — with the same similarity scale.
DEFAULT_MODEL = "timm/vit_small_patch16_dinov3.lvd1689m"
# Active model, resolved once at import time. Lets an embedder rollout/rollback
# be an environment flip (e.g. ROBOT_EMBEDDER_MODEL=facebook/dinov2-small)
# with no code change; RobotEmbedder() with no args picks this up.
ACTIVE_MODEL = os.environ.get("ROBOT_EMBEDDER_MODEL", DEFAULT_MODEL)
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


def _clamp_box(box, w: int, h: int):
    """Clamp a box to the frame. Returns None if it's below MIN_CROP_PX."""
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
        return None
    return x1, y1, x2, y2


def _inset_box(box, w: int, h: int):
    clamped = _clamp_box(box, w, h)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    bw, bh = x2 - x1, y2 - y1
    dx, dy = int(bw * BOX_INSET), int(bh * BOX_INSET)
    return x1 + dx, y1 + dy, x2 - dx, y2 - dy


def _apply_mask(crop_bgr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Black out pixels outside `mask` (the SAM2 robot segmentation).

    `mask` is box-local (covers the same region as the crop) but may be at a
    different resolution; it's nearest-resized to the crop. Returns `crop_bgr`
    unchanged (same object) when `mask` is None so the no-mask path is free.
    """
    if mask is None:
        return crop_bgr
    ch, cw = crop_bgr.shape[:2]
    m = mask
    if m.shape[:2] != (ch, cw):
        m = cv2.resize(m.astype(np.uint8), (cw, ch),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    out = crop_bgr.copy()
    out[~m] = 0
    return out


class RobotEmbedder:
    """Lazy-loaded appearance embedder. The model loads on first embed() call."""

    def __init__(self, model_name: str = ACTIVE_MODEL):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._transform = None  # timm path only; None for the transformers path
        self._image_cls = None  # timm path only: PIL.Image, loaded alongside timm
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
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.model_name.startswith("timm/"):
                # The official facebook/dinov3-* repos are gated on HF; the
                # timm/ mirrors are open and load through timm instead.
                import timm
                from PIL import Image
                self._image_cls = Image
                self._model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
                self._model.to(self._device)
                self._model.eval()  # torch.nn.Module.eval() (inference mode): disable dropout etc.
                data_config = timm.data.resolve_model_data_config(self._model)
                self._transform = timm.data.create_transform(**data_config, is_training=False)
            else:
                from transformers import AutoImageProcessor, AutoModel
                self._processor = AutoImageProcessor.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
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

    def embed(self, frame: np.ndarray, box, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Return an L2-normalized embedding for the crop, or None if invalid/unavailable.

        When `mask` (a box-local SAM2 segmentation) is given, background pixels
        are blacked out so the signature describes the robot, not the field.
        """
        return self.embed_batch(frame, [box], masks=[mask])[0]

    def embed_batch(self, frame: np.ndarray, boxes, masks=None) -> list[Optional[np.ndarray]]:
        """Embed multiple crops from one frame in a single forward pass.

        Returns a list aligned with `boxes`; entries are None for invalid crops.
        `masks`, when provided, is aligned with `boxes`; each entry is either a
        box-local boolean SAM2 mask or None. Masked crops skip the background
        inset (the mask already sheds background); unmasked crops keep the inset.
        """
        h, w = frame.shape[:2]
        if masks is None:
            masks = [None] * len(boxes)
        # Masked crops use the full clamped box (the mask removes background);
        # unmasked crops inset to shed background heuristically.
        crop_boxes = [
            _clamp_box(b, w, h) if m is not None else _inset_box(b, w, h)
            for b, m in zip(boxes, masks)
        ]
        valid_idx = [i for i, b in enumerate(crop_boxes) if b is not None]
        out: list[Optional[np.ndarray]] = [None] * len(boxes)
        if not valid_idx:
            return out
        if not self._ensure_model():
            return out  # all None — caller falls back to position-only

        torch = self._torch  # cached during _ensure_model()
        crops = []
        for i in valid_idx:
            x1, y1, x2, y2 = crop_boxes[i]
            crop_bgr = _apply_mask(frame[y1:y2, x1:x2], masks[i])
            crops.append(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

        if self.model_name.startswith("timm/"):
            batch = torch.stack(
                [self._transform(self._image_cls.fromarray(c)) for c in crops]).to(self._device)
            with torch.no_grad():
                # forward_features() returns the raw token sequence (no head
                # pooling applied). Do NOT call the model directly / use its
                # pooled forward() output: timm's default global_pool for
                # DINOv3 is 'avg' (mean over patch tokens), which regresses
                # badly on our re-ID eval (AUC 0.9345 vs 0.9525 for CLS; at
                # the 0.55 cosine floor it lets 76% of false pairs through
                # vs 4.86% for CLS). Index 0 is the CLS token.
                feats = self._model.forward_features(batch)
            cls = feats[:, 0]
        else:
            inputs = self._processor(images=crops, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            cls = outputs.last_hidden_state[:, 0]  # CLS token, [N, 384]

        cls = torch.nn.functional.normalize(cls, dim=1)  # L2-normalize
        vecs = cls.cpu().numpy().astype(np.float32)
        for slot, i in enumerate(valid_idx):
            out[i] = vecs[slot]
        return out
