#!/usr/bin/env python3
"""Frozen DINOv3 ViT-S/16 CLS features for harvested crops (design §5, S2/S3 inputs).

Backbone = ``timm/vit_small_patch16_dinov3.lvd1689m`` loaded like web/embedder.py
(timm, num_classes=0, eval) with the CLS token taken EXPLICITLY:
``forward_features(x)[:, 0]`` (timm's default pooled output is an avg over patch
tokens, a known regression in this repo). Features are stored raw (not
L2-normalised) as float16.

Preprocessing (``--mode``):
  letterbox  pad the crop to a square with the ImageNet-mean colour, resize to
             ``--size`` (default 224). Keeps the whole crop; default.
  squash     resize straight to size x size (aspect distorted).
  center     production embedder transform: shorter side -> 256, centre crop 256
             (cuts the ends of wide bumper crops).
The mode was chosen by grouped CV on gold b1 (see heads.py ``modes``).

Cache layout (``--out``, default data/teacher/features/dinov3s_cls):
  <event>/<match>.npy          float16 [N, 384], rows aligned with ...
  <event>/<match>.paths.json   [crop path relative to harvest_v2, ...] (manifest order)
  index.json                   {model, mode, size, matches: {match_key: {event, n, manifest_rows}}}
A match is (re)computed when its manifest row count differs from the cache;
matches still being harvested (no summary.json) are skipped.

Usage:
  .venv/bin/python -m scripts.teacher.students.features --events 2025cur,2026cur
  .venv/bin/python -m scripts.teacher.students.features            # every event in harvest_v2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.teacher.harvest import MAIN  # noqa: E402

MODEL_NAME = "timm/vit_small_patch16_dinov3.lvd1689m"
FEAT_DIM = 384
DEFAULT_ROOT = MAIN / "data/teacher/harvest_v2"
DEFAULT_OUT = MAIN / "data/teacher/features/dinov3s_cls"
DEFAULT_MODE = "letterbox"
DEFAULT_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(crop_bgr: np.ndarray, mode: str = DEFAULT_MODE, size: int = DEFAULT_SIZE) -> np.ndarray:
    """BGR uint8 crop -> normalised CHW float32 array for the backbone."""
    if crop_bgr is None or crop_bgr.size == 0:
        crop_bgr = np.zeros((size, size, 3), np.uint8)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if mode == "letterbox":
        s = max(h, w)
        canvas = np.empty((s, s, 3), np.uint8)
        canvas[:] = (MEAN * 255).round().astype(np.uint8)
        y0, x0 = (s - h) // 2, (s - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = rgb
        interp = cv2.INTER_CUBIC if s < size else cv2.INTER_AREA
        img = cv2.resize(canvas, (size, size), interpolation=interp)
    elif mode == "squash":
        interp = cv2.INTER_CUBIC if min(h, w) < size else cv2.INTER_AREA
        img = cv2.resize(rgb, (size, size), interpolation=interp)
    elif mode == "center":
        size = 256
        sc = size / min(h, w)
        nw, nh = max(size, round(w * sc)), max(size, round(h * sc))
        img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
        y0, x0 = (nh - size) // 2, (nw - size) // 2
        img = img[y0:y0 + size, x0:x0 + size]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


class Backbone:
    """Frozen DINOv3-S; ``__call__(batch_chw) -> [N, 384]`` raw CLS (float32 numpy)."""

    def __init__(self, device: Optional[str] = None, half: bool = True):
        import timm
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = half and self.device.startswith("cuda")
        m = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        m.eval()
        self.model = m.to(self.device, torch.float16 if self.half else torch.float32)

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(batch).to(self.device, torch.float16 if self.half else torch.float32)
        with torch.inference_mode():
            cls = self.model.forward_features(x)[:, 0]  # explicit CLS, never the pooled output
        return cls.float().cpu().numpy()

    def embed_crops(self, crops_bgr: Sequence[np.ndarray], mode: str = DEFAULT_MODE,
                    size: int = DEFAULT_SIZE, batch: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(crops_bgr), batch):
            out.append(self(np.stack([preprocess(c, mode, size) for c in crops_bgr[i:i + batch]])))
        return np.concatenate(out) if out else np.zeros((0, FEAT_DIM), np.float32)


class _CropSet:
    def __init__(self, root: Path, paths: List[str], mode: str, size: int):
        self.root, self.paths, self.mode, self.size = root, paths, mode, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(str(self.root / self.paths[i]))
        return preprocess(img, self.mode, self.size)


def embed_paths(bb: Backbone, root: Path, paths: List[str], mode: str, size: int,
                batch: int = 128, workers: int = 4) -> np.ndarray:
    import torch
    if not paths:
        return np.zeros((0, FEAT_DIM), np.float16)
    dl = torch.utils.data.DataLoader(_CropSet(root, paths, mode, size), batch_size=batch,
                                     num_workers=workers, shuffle=False)
    out = [bb(b.numpy()) for b in dl]
    return np.concatenate(out).astype(np.float16)


def match_dirs(root: Path, events: Optional[Sequence[str]]):
    evs = events or sorted(p.name for p in root.iterdir() if p.is_dir())
    for ev in evs:
        d = root / ev
        if not d.is_dir():
            continue
        for m in sorted(d.iterdir()):
            if (m / "manifest.jsonl").exists() and (m / "summary.json").exists():
                yield ev, m


def manifest_paths(mdir: Path) -> List[str]:
    return [json.loads(l)["path"] for l in (mdir / "manifest.jsonl").open() if l.strip()]


def load_cache(out: Path = DEFAULT_OUT, events: Optional[Sequence[str]] = None):
    """{crop path: row index} + stacked float32 features for the cached matches of ``events``."""
    idx = json.loads((out / "index.json").read_text())
    feats, paths = [], []
    for mk, meta in sorted(idx["matches"].items()):
        if events and meta["event"] not in events:
            continue
        feats.append(np.load(out / meta["event"] / f"{mk}.npy"))
        paths.extend(json.loads((out / meta["event"] / f"{mk}.paths.json").read_text()))
    X = np.concatenate(feats).astype(np.float32) if feats else np.zeros((0, FEAT_DIM), np.float32)
    return {p: i for i, p in enumerate(paths)}, X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=None, help="comma list (default: all in --root)")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--mode", default=DEFAULT_MODE, choices=["letterbox", "squash", "center"])
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ipath = out / "index.json"
    idx = json.loads(ipath.read_text()) if ipath.exists() else {}
    if idx and (idx.get("mode"), idx.get("size")) != (a.mode, a.size):
        sys.exit(f"cache at {out} is mode={idx.get('mode')} size={idx.get('size')}; use another --out")
    idx.update({"model": MODEL_NAME, "cls": "forward_features[:,0]", "mode": a.mode, "size": a.size,
                "dtype": "float16"})
    idx.setdefault("matches", {})
    bb = Backbone()
    tot_n, tot_t = 0, 0.0
    for ev, mdir in match_dirs(root, a.events.split(",") if a.events else None):
        mk = mdir.name
        paths = manifest_paths(mdir)
        meta = idx["matches"].get(mk)
        if meta and meta["n"] == len(paths) and not a.force and (out / ev / f"{mk}.npy").exists():
            continue
        t0 = time.time()
        X = embed_paths(bb, root, paths, a.mode, a.size, a.batch, a.workers)
        dt = time.time() - t0
        (out / ev).mkdir(exist_ok=True)
        np.save(out / ev / f"{mk}.npy", X)
        (out / ev / f"{mk}.paths.json").write_text(json.dumps(paths))
        idx["matches"][mk] = {"event": ev, "n": len(paths)}
        ipath.write_text(json.dumps(idx, indent=1))
        tot_n += len(paths)
        tot_t += dt
        print(f"{mk}: {len(paths)} crops in {dt:.1f}s ({len(paths) / max(dt, 1e-6):.0f}/s)", flush=True)
    if tot_n:
        print(f"TOTAL {tot_n} crops in {tot_t:.0f}s = {tot_n / tot_t:.0f} crops/s")
    print(f"cache: {len(idx['matches'])} matches, {sum(m['n'] for m in idx['matches'].values())} crops -> {out}")


if __name__ == "__main__":
    main()
