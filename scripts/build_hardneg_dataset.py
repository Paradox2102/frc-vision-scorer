#!/usr/bin/env python3
"""Assemble a hard-negative-augmented bumper dataset:
  base positives (Benjamin)  +  subsampled hub/outpost crops (from
  scripts/crop_negatives_from_boxes.py) + the user's seed crops, all added as
  background images (empty label files = negatives).

Subsamples ~per-event crops spread across that event's boxes for venue + structure
variety while keeping the background ratio near ~10%. Emits a single review montage
of exactly the negatives selected.

Usage:
  .venv/bin/python scripts/build_hardneg_dataset.py --per-event 4
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

SPLITS = ("train", "valid", "test")


def _even(items, k):
    n = len(items)
    if n <= k:
        return list(items)
    step = n / k
    return [items[int(i * step)] for i in range(k)]


def _sample_event(event_dir, per_event):
    """Pick ~per_event crops from an event, spread across its box subfolders."""
    boxes = sorted([d for d in event_dir.iterdir() if d.is_dir()])
    by_box = {b.name: sorted(b.glob("frame*.jpg")) for b in boxes}
    chosen, i = [], 0
    # round-robin across boxes, taking evenly-spaced frames, until per_event reached
    picks_per_box = max(1, per_event // max(1, len(by_box)))
    for name, crops in by_box.items():
        chosen.extend(_even(crops, picks_per_box))
    # if we still need more (few boxes), top up from the largest boxes
    while len(chosen) < per_event:
        added = False
        for name, crops in by_box.items():
            extra = [c for c in crops if c not in chosen]
            if extra:
                chosen.append(extra[len(extra) // 2]); added = True
            if len(chosen) >= per_event:
                break
        if not added:
            break
    return chosen[:per_event] if per_event else chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/bumper_benjamin")
    ap.add_argument("--negs-root", default="data/hard_negatives/boxes")
    ap.add_argument("--seeds", default="data/hard_negatives/user_negs")
    ap.add_argument("--out", default="data/bumper_hardneg")
    ap.add_argument("--per-event", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    base, out = Path(args.base), Path(args.out)
    if not (base / "train" / "images").exists():
        sys.exit(f"base dataset not found under {base}")
    if out.exists():
        shutil.rmtree(out)
    for sp in SPLITS:
        (out / sp / "images").mkdir(parents=True, exist_ok=True)
        (out / sp / "labels").mkdir(parents=True, exist_ok=True)

    pos = {}
    for sp in SPLITS:
        imgs = sorted((base / sp / "images").glob("*"))
        for img in imgs:
            shutil.copy2(img, out / sp / "images" / img.name)
            lbl = base / sp / "labels" / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, out / sp / "labels" / lbl.name)
        pos[sp] = len(imgs)

    # gather negatives: per-event subsample + all seed crops
    chosen = []
    negs_root = Path(args.negs_root)
    for ev in sorted([d for d in negs_root.iterdir() if d.is_dir()]) if negs_root.exists() else []:
        picks = _sample_event(ev, args.per_event)
        chosen += [(ev.name, p) for p in picks]
    seeds = Path(args.seeds)
    if seeds.exists():
        for s in sorted(seeds.glob("seed_*.png")):
            chosen.append(("seed", s))

    neg = {"train": 0, "valid": 0}
    val_every = max(2, round(1 / args.val_frac)) if args.val_frac > 0 else 0
    for i, (tag, crop) in enumerate(chosen):
        sp = "valid" if (val_every and i % val_every == 0) else "train"
        name = f"neg_{tag}_{crop.parent.name}_{crop.stem}{crop.suffix}"
        shutil.copy2(crop, out / sp / "images" / name)
        (out / sp / "labels" / (Path(name).stem + ".txt")).write_text("")
        neg[sp] += 1

    (out / "bumper_dataset.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n\n"
        "names:\n  0: blue_bumper\n  1: red_bumper\n\nnc: 2\n"
    )

    # review montage of selected negatives
    tiles = []
    for tag, crop in chosen:
        t = cv2.resize(cv2.imread(str(crop)), (140, 105))
        cv2.putText(t, tag.replace("2026", "").replace("_qm", "q")[:13], (2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
        tiles.append(t)
    cols = 10
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    montage = ROOT / "data" / "hard_negatives" / "selected_negatives.jpg"
    cv2.imwrite(str(montage), np.vstack(rows))

    train_total = pos["train"] + neg["train"]
    print(f"positives: train={pos['train']} valid={pos['valid']} test={pos['test']}")
    print(f"negatives: {len(chosen)} -> train={neg['train']} valid={neg['valid']} "
          f"(background {100*neg['train']/max(1,train_total):.1f}% of train)")
    print(f"dataset -> {out}/bumper_dataset.yaml")
    print(f"review montage -> {montage}")


if __name__ == "__main__":
    main()
