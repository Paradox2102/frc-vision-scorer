"""S4 same-robot embedder: runtime inference (teacher-distillation design §13).

DINOv3 ViT-S/16 with its last 8 blocks fine-tuned (SupCon on (event, team) identities),
explicit CLS -> MLP -> 128-d L2-normalised vector. Training, evaluation and threshold tuning
live in ``scripts/teacher/students/embedder.py`` / ``s4_thresholds.py``, which re-export the
inference pieces from here (nothing under ``web/`` imports ``scripts``).

Crop rule (what the checkpoint was trained on, NOT ``web.embedder.BOX_INSET``): the robot box
padded by 10% per side (the harvester crop), letterboxed onto a mean-grey square and resized
to 224, ImageNet-normalised. SAM masks are ignored (training crops are unmasked). The
harvester deinterlaced combing crops of woven videos; ``deinterlace=True`` applies a cheap
equivalent (comb gate -> keep one field, stretch back to full height).

Checkpoints are ``torch.save`` dicts of plain tensors / config (``weights_only=True``), so
moving this code does not affect loading.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from web.students.paths import MODELS

BACKBONE = "timm/vit_small_patch16_dinov3.lvd1689m"
DEFAULT_CKPT = MODELS / "s4_embedder.pt"
EMB_DIM = 128
ROBOT_PAD = 0.10
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MIN_CROP_PX = 10          # same floor as web.embedder: smaller boxes yield no embedding
COMB_THR = 0.9            # web.bumper_register.deinterlace_crop's comb gate


# ----------------------------------------------------------------------------- crop + preprocess
def pad_box(box: Sequence[float], frac: float, w: int, h: int) -> Tuple[int, int, int, int]:
    """Box padded by ``frac`` of its size per side, clamped to the frame (harvester rule)."""
    bw, bh = float(box[2]) - float(box[0]), float(box[3]) - float(box[1])
    x0 = int(max(0, np.floor(box[0] - frac * bw)))
    y0 = int(max(0, np.floor(box[1] - frac * bh)))
    x1 = int(min(w, np.ceil(box[2] + frac * bw)))
    y1 = int(min(h, np.ceil(box[3] + frac * bh)))
    return x0, y0, x1, y1


def letterbox(crop_bgr: np.ndarray, size: int = SIZE) -> np.ndarray:
    """BGR uint8 crop -> normalised CHW float32 (mean-grey letterbox to ``size``).
    Identical to ``scripts.teacher.students.features.preprocess(.., "letterbox")``."""
    if crop_bgr is None or crop_bgr.size == 0:
        crop_bgr = np.zeros((size, size, 3), np.uint8)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = max(h, w)
    canvas = np.empty((s, s, 3), np.uint8)
    canvas[:] = (MEAN * 255).round().astype(np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = rgb
    interp = cv2.INTER_CUBIC if s < size else cv2.INTER_AREA
    img = cv2.resize(canvas, (size, size), interpolation=interp)
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def field_deinterlace(crop_bgr: np.ndarray, comb_thr: float = COMB_THR) -> np.ndarray:
    """Cheap deinterlace: when the crop combs, keep the even field stretched back to full
    height (the harvester's detection-side rule); otherwise return the crop unchanged."""
    if crop_bgr is None or crop_bgr.ndim < 2 or crop_bgr.shape[0] < 8 or crop_bgr.shape[1] < 4:
        return crop_bgr
    from web.bumper_register import comb_ratio
    if comb_ratio(crop_bgr) <= comb_thr:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    return cv2.resize(crop_bgr[::2], (w, h), interpolation=cv2.INTER_LINEAR)


def robot_crop(frame_bgr: np.ndarray, box, pad: float = ROBOT_PAD,
               min_px: int = MIN_CROP_PX) -> Optional[np.ndarray]:
    """The S4 input crop for ``box`` (harvester 10% pad), or None when the clamped box is
    smaller than ``min_px`` on either side."""
    h, w = frame_bgr.shape[:2]
    bx0, by0 = max(0, int(box[0])), max(0, int(box[1]))
    bx1, by1 = min(w, int(box[2])), min(h, int(box[3]))
    if bx1 - bx0 < min_px or by1 - by0 < min_px:
        return None
    x0, y0, x1, y1 = pad_box(box, pad, w, h)
    return frame_bgr[y0:y1, x0:x1]


# ----------------------------------------------------------------------------- model
def build_net(pretrained: bool = True, train_blocks: int = 4, dim: int = EMB_DIM, hidden: int = 512):
    import timm
    import torch.nn as nn
    import torch.nn.functional as tF

    class S4Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(BACKBONE, pretrained=pretrained, num_classes=0)
            for p in self.backbone.parameters():
                p.requires_grad = False
            n = len(self.backbone.blocks)
            for b in self.backbone.blocks[max(0, n - train_blocks):]:
                for p in b.parameters():
                    p.requires_grad = True
            if train_blocks > 0:
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True
            d = self.backbone.num_features
            self.proj = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, dim))

        def cls(self, x):
            return self.backbone.forward_features(x)[:, 0]  # explicit CLS

        def forward(self, x):
            return tF.normalize(self.proj(self.cls(x)).float(), dim=1)

    return S4Net()


def load_net(path: Path, device: str = "cuda"):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["cfg"]
    net = build_net(pretrained=False, train_blocks=cfg["train_blocks"], dim=cfg["dim"])
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state_dict"].items()})
    return net.to(device).eval(), ck


def model_tag(ckpt: Path = DEFAULT_CKPT) -> str:
    """Stable ``embed_model`` tag for vectors from ``ckpt``: ``s4:<file stem>:<sha256[:10]>``.
    The content hash makes a retrained checkpoint under the same file name a different vector
    space (gallery / crop rows never mix them)."""
    p = Path(ckpt)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"s4:{p.stem}:{h.hexdigest()[:10]}"


class S4Embedder:
    """Inference wrapper: BGR crops (or frame + robot boxes) -> [N, 128] L2-normalised."""

    def __init__(self, ckpt: Path = DEFAULT_CKPT, device: Optional[str] = None, half: bool = True):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net, self.ck = load_net(Path(ckpt), self.device)
        self.half = half and self.device.startswith("cuda")
        if self.half:
            self.net.half()

    def embed_crops(self, crops_bgr: Sequence[np.ndarray], batch: int = 64) -> np.ndarray:
        torch = self.torch
        out = []
        for i in range(0, len(crops_bgr), batch):
            x = np.stack([letterbox(c) for c in crops_bgr[i:i + batch]])
            x = torch.from_numpy(x).to(self.device, torch.float16 if self.half else torch.float32)
            with torch.inference_mode():
                out.append(self.net(x).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.ck["dim"]), np.float32)

    def embed_boxes(self, frame_bgr: np.ndarray, boxes) -> np.ndarray:
        """Cut robot boxes the way the checkpoint was trained (harvester 10% pad) and embed."""
        h, w = frame_bgr.shape[:2]
        crops = []
        for b in boxes:
            x0, y0, x1, y1 = pad_box(b, ROBOT_PAD, w, h)
            crops.append(frame_bgr[y0:y1, x0:x1])
        return self.embed_crops(crops)


class S4RobotEmbedder:
    """Drop-in for ``web.embedder.RobotEmbedder`` (``embed`` / ``embed_batch`` / ``model_name``)
    backed by the S4 checkpoint. Lazy: the net loads on the first call; a failed load latches
    and every later call returns all-None (callers already degrade to position-only)."""

    backend = "s4"

    def __init__(self, ckpt: Path = DEFAULT_CKPT, device: Optional[str] = None,
                 deinterlace: bool = True, model_name: Optional[str] = None):
        self.ckpt = Path(ckpt)
        self.device = device
        self.deinterlace = bool(deinterlace)
        self.model_name = model_name or model_tag(self.ckpt)
        self._impl: Optional[S4Embedder] = None
        self._load_failed = False

    def _ensure_model(self) -> bool:
        if self._impl is not None:
            return True
        if self._load_failed:
            return False
        try:
            self._impl = S4Embedder(self.ckpt, device=self.device)
            return True
        except Exception as e:  # pragma: no cover - env failure only
            print(f"  S4RobotEmbedder: failed to load {self.ckpt} ({e}); "
                  f"embeddings disabled, falling back to position-only re-ID.")
            self._impl = None
            self._load_failed = True
            return False

    def embed(self, frame: np.ndarray, box, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        return self.embed_batch(frame, [box], masks=[mask])[0]

    def embed_batch(self, frame: np.ndarray, boxes, masks=None) -> List[Optional[np.ndarray]]:
        """Embed every box of one frame in one forward pass; ``masks`` is accepted for
        interface parity and ignored (S4 was trained on unmasked padded crops)."""
        out: List[Optional[np.ndarray]] = [None] * len(boxes)
        crops, idx = [], []
        for i, b in enumerate(boxes):
            if b is None:
                continue
            c = robot_crop(frame, b)
            if c is None:
                continue
            crops.append(field_deinterlace(c) if self.deinterlace else c)
            idx.append(i)
        if not crops or not self._ensure_model():
            return out
        vecs = self._impl.embed_crops(crops)
        for slot, i in enumerate(idx):
            out[i] = vecs[slot].astype(np.float32)
        return out

    def release(self) -> None:
        """Drop the net (the job's GPU-release path calls this)."""
        self._impl = None
