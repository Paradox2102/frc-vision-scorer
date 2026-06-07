#!/usr/bin/env python3
"""Assemble a hard-negative-augmented bumper dataset:
  base positives (Benjamin)  +  subsampled robot-free static-structure crops as
  background images (empty label files, which YOLO treats as negatives).

Reads the negatives folder LIVE, so pruning bad crops there and re-running just works.

Usage:
  .venv/bin/python scripts/build_hardneg_dataset.py \\
    --base data/bumper_benjamin \\
    --negs data/hard_negatives/static_negs/2026hop_qm30 \\
    --out data/bumper_hardneg --per-cluster 5 --val-frac 0.15
"""
import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")
CL_RE = re.compile(r"_cl(\d+)_")


def _even_sample(items, k):
    """Pick up to k items evenly spaced across the (frame-sorted) list."""
    n = len(items)
    if n <= k:
        return list(items)
    step = n / k
    return [items[int(i * step)] for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/bumper_benjamin")
    ap.add_argument("--negs", required=True)
    ap.add_argument("--out", default="data/bumper_hardneg")
    ap.add_argument("--per-cluster", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    base = Path(args.base)
    negs = Path(args.negs)
    out = Path(args.out)
    if not (base / "train" / "images").exists():
        sys.exit(f"base dataset not found under {base}")
    if not negs.exists():
        sys.exit(f"negatives dir not found: {negs}")

    if out.exists():
        shutil.rmtree(out)
    for sp in SPLITS:
        (out / sp / "images").mkdir(parents=True, exist_ok=True)
        (out / sp / "labels").mkdir(parents=True, exist_ok=True)

    # 1) copy base positives verbatim
    pos_counts = {}
    for sp in SPLITS:
        imgs = sorted((base / sp / "images").glob("*"))
        for img in imgs:
            shutil.copy2(img, out / sp / "images" / img.name)
            lbl = base / sp / "labels" / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, out / sp / "labels" / lbl.name)
        pos_counts[sp] = len(imgs)

    # 2) gather + subsample negatives per cluster (exclude *_context.jpg)
    crops = [p for p in negs.glob("*.jpg") if not p.stem.endswith("_context")]
    by_cluster = defaultdict(list)
    for p in crops:
        m = CL_RE.search(p.name)
        by_cluster[int(m.group(1)) if m else -1].append(p)
    chosen = []
    for cl, items in sorted(by_cluster.items()):
        items.sort(key=lambda p: p.name)  # frame-ordered
        chosen.extend(_even_sample(items, args.per_cluster))

    # 3) split negatives train/val and write as background images (empty labels)
    neg_counts = {"train": 0, "valid": 0}
    val_every = max(2, round(1 / args.val_frac)) if args.val_frac > 0 else 0
    for i, crop in enumerate(chosen):
        sp = "valid" if (val_every and i % val_every == 0) else "train"
        shutil.copy2(crop, out / sp / "images" / crop.name)
        (out / sp / "labels" / (crop.stem + ".txt")).write_text("")  # empty == background
        neg_counts[sp] += 1

    # 4) yaml (same class scheme as base)
    (out / "bumper_dataset.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n\n"
        "names:\n  0: blue_bumper\n  1: red_bumper\n\nnc: 2\n"
    )

    train_total = pos_counts["train"] + neg_counts["train"]
    print(f"base positives: train={pos_counts['train']} valid={pos_counts['valid']} test={pos_counts['test']}")
    print(f"negatives added: {len(chosen)} (from {len(by_cluster)} clusters, "
          f"<= {args.per_cluster}/cluster) -> train={neg_counts['train']} valid={neg_counts['valid']}")
    print(f"train background ratio: {neg_counts['train']}/{train_total} = "
          f"{100*neg_counts['train']/max(1,train_total):.1f}%")
    print(f"dataset -> {out}/bumper_dataset.yaml")


if __name__ == "__main__":
    main()
