"""S1 student digit reader for the window-label stage (window_labels_reader knob).

Teacher-distillation spec section 14 (docs/superpowers/specs/2026-11-10-teacher-distillation-design.md).

S1 (scripts/teacher/students/digits.py) was trained on the harvester's bumper
crops: the 2026cur_v3 bumper detector box (conf >= 0.40, imgsz 640 on the
whole frame, one field of woven video), padded 25% per side, deinterlaced.
The window stage's digit requests are wide tiles around a tracked slot, so
the reader cuts S1 inputs the harvester's way instead: the bumper detector
runs on each sample's main-view frame, the bumpers whose centre is near the
slot's projected target are kept, and each is padded with the harvester's
``pad_box``. One S1 call per sample; a call names a team only when S1 names
one and every nearby bumper S1 names agrees.

Everything above ``StudentReader`` is pure (no torch, no files)."""
from __future__ import annotations

import gc
import math
from typing import Optional, Sequence

import numpy as np

BUMPER_CONF = 0.40        # harvester crop floor (harvest.BUMPER_CONF)
DET_IMGSZ = 640           # harvester detector input size
MIN_SIDE_PX = 20          # harvester MIN_SIDE_PX: smaller boxes were never cropped
PAD_BUMPER = 0.25         # harvester PAD_BUMPER
NEAR_MAX = 0.45           # bumper centre within this x tile side of the target
NEAR_BAND = 0.15          # bumpers within nearest + this x side are read together
MAX_PER_SAMPLE = 3        # at most this many bumpers read per sample
OTHER = "other"
ABSTAIN = {OTHER: 1.0}    # a returned call that names nobody


def colour_of(cls_name: str) -> Optional[str]:
    c = str(cls_name).lower()
    return "red" if c.startswith("red") else "blue" if c.startswith("blue") else None


def pad_box(box: Sequence[float], frac: float, w: int, h: int) -> tuple:
    """harvest.pad_box: ``box`` grown by ``frac`` of its size per side, clamped."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    x0 = int(max(0, math.floor(box[0] - frac * bw)))
    y0 = int(max(0, math.floor(box[1] - frac * bh)))
    x1 = int(min(w, math.ceil(box[2] + frac * bw)))
    y1 = int(min(h, math.ceil(box[3] + frac * bh)))
    return x0, y0, x1, y1


def near_bumpers(dets: Sequence, u: float, v: float, side: float) -> list:
    """The detections to read for a sample whose target is (u, v) with tile
    ``side``: boxes at BUMPER_CONF or above, at least MIN_SIDE_PX on their
    short side, centre within NEAR_MAX * side of the target; of those the
    nearest and every one within NEAR_BAND * side of it (at most
    MAX_PER_SAMPLE), nearest first. ``dets``: [(box xyxy, conf, colour)]."""
    cands = []
    for box, conf, colour in dets:
        if conf < BUMPER_CONF or min(box[2] - box[0], box[3] - box[1]) < MIN_SIDE_PX:
            continue
        d = math.hypot((box[0] + box[2]) / 2 - u, (box[1] + box[3]) / 2 - v) / max(1.0, side)
        if d <= NEAR_MAX:
            cands.append((d, tuple(float(c) for c in box), float(conf), colour))
    cands.sort(key=lambda c: c[0])
    if not cands:
        return []
    lim = cands[0][0] + NEAR_BAND
    return [c for c in cands if c[0] <= lim][:MAX_PER_SAMPLE]


def sample_call(reads: Sequence[dict], roster: Sequence[str]) -> Optional[dict]:
    """One digit call (the format ``window_labels.digit_decision`` takes)
    from the S1 reads of a sample's nearby bumpers, nearest first. None: no
    bumper (no call). A call names a team only when S1 named one and no other
    nearby bumper was named as a different team; it then carries S1's
    probabilities over the roster (plus "other"). Otherwise ABSTAIN."""
    if not reads:
        return None
    named = [r for r in reads if r.get("team") is not None]
    if not named or len({r["team"] for r in named}) > 1:
        return dict(ABSTAIN)
    probs = named[0].get("probs") or {}
    out = {str(t): float(probs.get(str(t), 0.0)) for t in roster}
    out[OTHER] = float(probs.get(OTHER, 0.0))
    return out


class StudentReader:
    """Bumper detector + S1, loaded once per stage run (GPU, fp16).

    ``detect(images) -> [[(box, conf, colour)]]`` (one list per image, image
    pixels); ``read(crops, rosters, colours) -> [S1 result]``; ``close()``
    drops the models and returns the CUDA cache. Raises on construction when
    CUDA, the checkpoint or the detector weights are missing (the stage then
    falls back to Jev)."""

    def __init__(self, device: Optional[str] = None, require_cuda: bool = True):
        import torch
        if device is None:
            if not torch.cuda.is_available():
                if require_cuda:
                    raise RuntimeError("CUDA unavailable")
                device = "cpu"
            else:
                device = "cuda"
        from scripts.teacher.harvest import BUMPER_WEIGHTS
        from scripts.teacher.students.digits import DigitReader
        from scripts.teacher.students.heads import MODELS
        ckpt = MODELS / "s1_digits.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"S1 checkpoint missing: {ckpt}")
        if not BUMPER_WEIGHTS.exists():
            raise FileNotFoundError(f"bumper detector weights missing: {BUMPER_WEIGHTS}")
        from ultralytics import YOLO
        self.device = device
        self.detector = YOLO(str(BUMPER_WEIGHTS))
        self.names = self.detector.names
        self.s1 = DigitReader(ckpt, device=device)
        self.model_key = f"s1:{ckpt.name}@{int(ckpt.stat().st_mtime)}"

    def detect(self, images: Sequence[np.ndarray]) -> list:
        if not len(images):
            return []
        dev = 0 if str(self.device).startswith("cuda") else "cpu"
        res = self.detector(list(images), conf=BUMPER_CONF, imgsz=DET_IMGSZ, device=dev,
                            verbose=False)
        out = []
        for r in res:
            b = r.boxes
            out.append([(tuple(box), float(c), colour_of(self.names[int(k)]))
                        for box, c, k in zip(b.xyxy.cpu().numpy().tolist(),
                                             b.conf.cpu().numpy().tolist(),
                                             b.cls.cpu().numpy().astype(int).tolist())])
        return out

    def read(self, crops: Sequence[np.ndarray], rosters: Sequence, colours: Sequence) -> list:
        return self.s1.read(list(crops), list(rosters), list(colours))

    def close(self) -> None:
        self.detector = self.s1 = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def field_frame(img: np.ndarray) -> np.ndarray:
    """harvest.field_frame: one field (even rows) stretched back to full
    height, the image the harvester detected on for woven video."""
    import cv2
    h, w = img.shape[:2]
    return cv2.resize(img[::2], (w, h), interpolation=cv2.INTER_LINEAR)
