"""Appearance embeddings (DINOv3 ViT-S by default) for robot re-identification.

FRC-agnostic: images in, L2-normalized vectors out. Swap the embedding backend
by setting ROBOT_EMBEDDER_MODEL (e.g. to a SigLIP checkpoint, a transformers
name like facebook/dinov2-small, or a "timm/..." name for the timm backend
below) — no caller changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
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

# Per-job embedder backend (JobConfig.robot_embedder). "dinov3" = RobotEmbedder
# (ACTIVE_MODEL, env-routable); "s4" = the teacher-distilled S4 same-robot
# embedder (web/students/s4_embedder.py, 128-d, own crop rule).
EMBEDDER_BACKENDS = ("dinov3", "s4")


@dataclass(frozen=True)
class EmbedThresholds:
    """Every cosine threshold / margin / weight that reads an appearance
    similarity, per embedding space. Cosines from different models live on
    different scales, so each backend carries its own table (tuned offline:
    scripts/teacher/students/s4_thresholds.py, design spec §16)."""
    app_floor: float           # legacy re-ID merge floor (app.APP_FLOOR, reid.py)
    relink_floor: float        # FusedTrackRegistry appearance relink floor
    identity_floor: float      # IdentityRegistry appearance veto / bonus floor
    update_margin: float       # IdentityRegistry EMA-update relink-margin gate
    confident_margin: float    # confidence rise / low-margin immediate read
    divergence_sim: float      # obs-vs-identity "appearance divergent" floor
    gallery_confirm_sim: float
    gallery_confirm_gap: float
    gallery_supplement_sim: float
    gallery_dedup_sim: float
    selective_app_scale: float  # multiplies JobConfig.selective_app_weight


# DINOv3 (and the dinov2 env route): today's hand-set constants, unchanged.
DINOV3_THRESHOLDS = EmbedThresholds(
    app_floor=0.55, relink_floor=0.55, identity_floor=0.55,
    update_margin=0.05, confident_margin=0.10, divergence_sim=0.70,
    gallery_confirm_sim=0.88, gallery_confirm_gap=0.06,
    gallery_supplement_sim=0.80, gallery_dedup_sim=0.92,
    selective_app_scale=1.0,
)
# S4: tuned on the S4 VAL events (2026bcvi, 2024mimil — whole events never
# trained on; the test events are not touched) by
# scripts/teacher/students/s4_thresholds.py (data/teacher/models/
# s4_thresholds.json). Rule per threshold: among S4 values whose false-pass
# rate <= DINOv3's at today's value AND true-pass >= DINOv3's, take the
# Youden-J (true - false) maximiser. Proxy operating points (DINOv3 -> S4):
#   relink 0.55 -> 0.59: true 80.2% -> 88.9%, false 40.2% -> 9.4%
#   divergence 0.70 -> 0.624: not-divergent true 34.9% -> 81.5%, false 3.6% -> 3.6%
#   confident margin 0.10 -> 0.081: correct 22.6% -> 80.0%, wrong 2.7% -> 2.7%
#   gallery supplement 0.80 -> 0.845: true 2.2% -> 18.2%, false 0.08% -> 0.08%
# dedup: same same-team positive quantile; selective_app_scale: equal mean
# (neg - pos) cost differential (S4 gap 0.366 vs DINOv3 0.061).
S4_THRESHOLDS = EmbedThresholds(
    app_floor=0.59, relink_floor=0.59, identity_floor=0.59,
    update_margin=0.023, confident_margin=0.081, divergence_sim=0.624,
    gallery_confirm_sim=0.943, gallery_confirm_gap=0.01,
    gallery_supplement_sim=0.845, gallery_dedup_sim=0.965,
    selective_app_scale=0.166,
)


def backend_of_model(model_name: Optional[str]) -> str:
    """Embedding-space family of an ``embed_model`` tag ("s4" | "dinov3")."""
    return "s4" if (model_name or "").startswith("s4:") else "dinov3"


def thresholds_for(backend_or_model: Optional[str]) -> EmbedThresholds:
    """Threshold table for a backend name ("dinov3"/"s4") or an embed_model tag."""
    key = backend_or_model or "dinov3"
    if key not in EMBEDDER_BACKENDS:
        key = backend_of_model(key)
    return S4_THRESHOLDS if key == "s4" else DINOV3_THRESHOLDS


def resolve_embedder_backend(requested: Optional[str]) -> tuple[str, str]:
    """(backend actually used, reason). "s4" needs its checkpoint and CUDA;
    otherwise it falls back to "dinov3" with a reason the job prints as a
    warning. Unknown names fall back too."""
    req = (requested or "dinov3").lower()
    if req not in EMBEDDER_BACKENDS:
        return "dinov3", f"unknown robot_embedder {requested!r}; using dinov3"
    if req == "dinov3":
        return "dinov3", ""
    try:
        from web.students import s4_embedder as s4
    except Exception as e:  # pragma: no cover - import failure only
        return "dinov3", f"S4 embedder unavailable ({e}); using dinov3"
    if not s4.DEFAULT_CKPT.exists():
        return "dinov3", f"S4 checkpoint missing ({s4.DEFAULT_CKPT}); using dinov3"
    try:
        import torch
        if not torch.cuda.is_available():
            return "dinov3", "S4 embedder needs CUDA (none available); using dinov3"
    except Exception as e:  # pragma: no cover
        return "dinov3", f"torch unavailable ({e}); using dinov3"
    return "s4", ""


def make_robot_embedder(backend: str = "dinov3"):
    """Embedder instance for a RESOLVED backend (see resolve_embedder_backend)."""
    if backend == "s4":
        from web.students import s4_embedder as s4
        return s4.S4RobotEmbedder(ckpt=s4.DEFAULT_CKPT)
    return RobotEmbedder()


def embedder_model_name(backend: str) -> str:
    """The embed_model tag a backend's vectors carry, without loading weights."""
    if backend == "s4":
        from web.students import s4_embedder as s4
        return s4.model_tag(s4.DEFAULT_CKPT)
    return ACTIVE_MODEL


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

    backend = "dinov3"

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

    def release(self) -> None:
        """Drop the model (the job's GPU-release path calls this)."""
        self._model = None
        self._processor = None
        self._transform = None

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

        # Pin the batch dtype to the model's own parameter dtype rather than
        # whatever torch's ambient default dtype happens to be: a concurrent
        # transformers model load (e.g. AutoModel.from_pretrained(...,
        # torch_dtype=torch.float16) in another thread, as the local bumper-
        # reader fallback does) temporarily flips torch's global default
        # dtype, and building/stacking the batch inside that window would
        # otherwise silently produce e.g. a float16 batch against float32
        # weights -> "Input type (c10::Half) and bias type (float) should be
        # the same" in the model's first conv/linear layer.
        model_dtype = next(self._model.parameters()).dtype
        if self.model_name.startswith("timm/"):
            batch = torch.stack(
                [self._transform(self._image_cls.fromarray(c)) for c in crops]
            ).to(self._device, model_dtype)
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
            inputs = self._processor(images=crops, return_tensors="pt").to(self._device, model_dtype)
            with torch.no_grad():
                outputs = self._model(**inputs)
            cls = outputs.last_hidden_state[:, 0]  # CLS token, [N, 384]

        cls = torch.nn.functional.normalize(cls, dim=1)  # L2-normalize
        vecs = cls.cpu().numpy().astype(np.float32)
        for slot, i in enumerate(valid_idx):
            out[i] = vecs[slot]
        return out
