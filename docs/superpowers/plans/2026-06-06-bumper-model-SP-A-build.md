# SP-A: Build the Bumper Detection Model — Implementation Plan

## ✅ STATUS: COMPLETE (2026-06-06) — as-built differs from the original plan below

**What was actually built:** a **2-class** YOLOv8m **bumper** detector
(`{0: blue_bumper, 1: red_bumper}`) trained on the **Benjamin dataset only**.
Weights: `runs/train/frc_bumpers/weights/best.pt`. Full write-up:
`runs/train/frc_bumpers/EVAL.md`.

**Result — GO:** overall mAP50 **0.977** (blue 0.970 / red 0.984), **0 red↔blue swaps**
on 748 val instances. Inference ~6 ms/image on the 3090.

**Why it diverged from the original 3-class merge plan (below):**
- **Karthik (`frc-bumpers-red-blue`) was rejected as a bumper source** — on download it
  proved to be a *fuel* dataset (~18k fuel labels, only ~10 robot labels). It also had no
  published Roboflow version; we forked it to `2102frcrobots` and generated v1 to inspect.
- **Benjamin (`frc-bumpers-h0p1g` v4)** is the clean red/blue **bumper** source (545 imgs,
  1004 red + 1159 blue boxes) — but bumpers only, no fuel (REEFSCAPE footage).
- The two datasets are mirror images (each unlabeled for the other's class), so merging
  poisons both. Per user decision (robot tracking is the priority), we trained **Benjamin
  bumpers only** and **keep the existing model for fuel** (hub is static, not needed).

**Consequences for SP-B:** the detector is **2-class (no fuel/hub)**, so SP-B is now an
**"augment"** integration — robots + alliance color from this model; fuel from the
existing model; no hub. The SP-B plan has been rewritten accordingly.

**Tasks actually executed:** download script (`scripts/download_bumper_datasets.py`),
train (`scripts/train_bumper_model.py`), eval (`scripts/eval_bumper_model.py`).
**Dropped (not needed for single-source Benjamin):** `web/training/class_remap.py`, the
merge script, and the fuel-label inspection (no merge, no fuel training). `DetectionSchema`
is name-driven, so Benjamin's `blue=0/red=1` order needs no remap.

---

# (Original plan — superseded above; kept for provenance)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a YOLOv8m model that detects `red_bumper`, `blue_bumper`, and `fuel`, by merging two Roboflow datasets into one unified schema and training on the RTX 3090.

**Architecture:** Download two Roboflow datasets via the SDK → normalize each dataset's class names to a unified `{0: red_bumper, 1: blue_bumper, 2: fuel}` schema with a pure, unit-tested name→id mapper → merge images/labels into `data/bumper_merged/` (modeled on `scripts/merge_roboflow_dataset.py`) → train with ultralytics (modeled on `scripts/train_yolo_v5.py`) → evaluate with per-class mAP + a red↔blue confusion check.

**Tech Stack:** Python, `roboflow` SDK 1.2.16, `ultralytics` 8.4.21, YOLOv8m, CUDA (RTX 3090, 24 GB). Tests run with `.venv/bin/python -m pytest` (the base conda python lacks pytest — see project memory).

**Contract this produces (consumed by SP-B):** a `.pt` at `runs/train/frc_bumpers/weights/best.pt` whose `model.names == {0: "red_bumper", 1: "blue_bumper", 2: "fuel"}`.

**Spec:** `docs/superpowers/specs/2026-06-06-bumper-color-detection-model-design.md`

---

## File Structure

- Create: `scripts/download_bumper_datasets.py` — downloads both Roboflow datasets into `data/` using `ROBOFLOW_API` from `.env`.
- Create: `web/training/class_remap.py` — pure helpers: `normalize_class_name()`, `build_class_map()`. Importable + unit-testable (no I/O).
- Create: `tests/test_class_remap.py` — unit tests for the remap helpers.
- Create: `scripts/merge_bumper_datasets.py` — merges both downloaded datasets into `data/bumper_merged/`, using `web/training/class_remap.py`.
- Create: `scripts/train_bumper_model.py` — trains YOLOv8m, outputs to `runs/train/frc_bumpers/`.
- Create: `scripts/inspect_fuel_labels.py` — samples the Benjamin dataset for visible-but-unlabeled fuel (the data-quality risk gate).
- Create (downloaded, git-ignored): `data/bumper_benjamin/`, `data/bumper_karthik/`, `data/bumper_merged/`.

> Note: `web/training/` is a new package. Put helpers there (not in `scripts/`) so they import cleanly under `web.*` like the rest of the codebase and are unit-testable without running a script. Add an empty `web/training/__init__.py`.

---

## Task 1: Dataset download script

**Files:**
- Create: `scripts/download_bumper_datasets.py`

This task is I/O/exploratory (network download); verification is inspecting the result, not a unit test.

- [ ] **Step 1: Write the download script**

```python
#!/usr/bin/env python3
"""Download the two Roboflow bumper datasets used to train the FRC bumper model.

Reads ROBOFLOW_API from the project .env. Downloads in YOLOv8 format. Defaults to
each project's latest version. Override a version with --benjamin-version /
--karthik-version if the latest changes.

Usage:
    .venv/bin/python scripts/download_bumper_datasets.py
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# (workspace_slug, project_slug, local_dir_name) parsed from the Universe URLs.
BENJAMIN = ("benjamins-workspace-h5vao", "frc-bumpers-h0p1g", "bumper_benjamin")
KARTHIK = ("karthiks-workspace-gez2e", "frc-bumpers-red-blue", "bumper_karthik")


def download_one(rf, workspace, project_slug, dest_name, version_num):
    project = rf.workspace(workspace).project(project_slug)
    if version_num is None:
        versions = project.versions()
        if not versions:
            raise SystemExit(f"No versions found for {workspace}/{project_slug}")
        # versions() is newest-first; take the highest version id to be safe.
        version_num = max(int(v.version.split("/")[-1]) for v in versions)
    dest = PROJECT_ROOT / "data" / dest_name
    print(f"Downloading {workspace}/{project_slug} v{version_num} -> {dest}")
    project.version(version_num).download("yolov8", location=str(dest), overwrite=True)
    yaml = dest / "data.yaml"
    print(f"  done. data.yaml present: {yaml.exists()}")
    if yaml.exists():
        print("  --- data.yaml ---")
        print("  " + yaml.read_text().replace("\n", "\n  "))
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benjamin-version", type=int, default=None)
    ap.add_argument("--karthik-version", type=int, default=None)
    args = ap.parse_args()

    api_key = os.environ.get("ROBOFLOW_API")
    if not api_key:
        raise SystemExit("ROBOFLOW_API not set in .env")
    rf = Roboflow(api_key=api_key)

    download_one(rf, *BENJAMIN, args.benjamin_version)
    download_one(rf, *KARTHIK, args.karthik_version)
    print("\nBoth datasets downloaded. Inspect the printed data.yaml class names "
          "before running the merge.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add download dirs to .gitignore**

Append to `.gitignore` (datasets are large and reproducible from the script):

```
/data/bumper_benjamin/
/data/bumper_karthik/
/data/bumper_merged/
```

- [ ] **Step 3: Run the download**

Run: `.venv/bin/python scripts/download_bumper_datasets.py`
Expected: two datasets under `data/bumper_benjamin/` and `data/bumper_karthik/`, each with a `data.yaml`, and the printed `names:` lists for both.

> If the roboflow `.versions()` API shape differs in 1.2.16 and the script errors locating the version, fall back to passing explicit `--benjamin-version N --karthik-version N` (the version number is the trailing integer in the dataset URL on Roboflow Universe).

- [ ] **Step 4: Record the actual class names**

Read both `data.yaml` files and write the exact `names` lists into a comment block at the top of `scripts/merge_bumper_datasets.py` (created in Task 3). These drive the alias table — do not assume; use what was downloaded.

- [ ] **Step 5: Commit**

```bash
git add scripts/download_bumper_datasets.py .gitignore
git commit -m "feat(training): roboflow downloader for the two bumper datasets"
```

---

## Task 2: Class-name normalization + mapping (pure, TDD)

**Files:**
- Create: `web/training/__init__.py` (empty)
- Create: `web/training/class_remap.py`
- Test: `tests/test_class_remap.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_class_remap.py
from web.training.class_remap import normalize_class_name, build_class_map

UNIFIED = {"red_bumper": 0, "blue_bumper": 1, "fuel": 2}


def test_normalize_collapses_case_space_underscore():
    assert normalize_class_name("Red Bumper") == "red_bumper"
    assert normalize_class_name("blue-bumper") == "blue_bumper"
    assert normalize_class_name("  FUEL ") == "fuel"


def test_build_class_map_remaps_by_name_not_source_id():
    # Source lists fuel first, red/blue after — ids differ from unified.
    source_names = {0: "fuel", 1: "red bumper", 2: "blue bumper"}
    cmap, dropped = build_class_map(source_names, UNIFIED)
    assert cmap == {0: 2, 1: 0, 2: 1}
    assert dropped == []


def test_build_class_map_drops_unknown_classes():
    source_names = {0: "red_bumper", 1: "blue_bumper", 2: "referee"}
    cmap, dropped = build_class_map(source_names, UNIFIED)
    assert cmap == {0: 0, 1: 1}
    assert dropped == [(2, "referee")]


def test_build_class_map_applies_aliases():
    # "fuel_ball"/"ball"/"cargo" all map to fuel.
    source_names = {0: "fuel_ball"}
    cmap, _ = build_class_map(source_names, UNIFIED)
    assert cmap == {0: 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_class_remap.py -v`
Expected: FAIL (`ModuleNotFoundError: web.training.class_remap`).

- [ ] **Step 3: Write the implementation**

```python
# web/training/class_remap.py
"""Pure helpers to normalize Roboflow class names and remap source class ids
to a unified schema. No I/O — unit-testable."""
from __future__ import annotations
import re

# Aliases collapse to a canonical unified key. Extend if a source uses new names.
_ALIASES = {
    "red": "red_bumper", "redbumper": "red_bumper", "red_bumper": "red_bumper",
    "blue": "blue_bumper", "bluebumper": "blue_bumper", "blue_bumper": "blue_bumper",
    "fuel": "fuel", "fuel_ball": "fuel", "fuelball": "fuel",
    "ball": "fuel", "cargo": "fuel",
}


def normalize_class_name(name: str) -> str:
    """Lowercase, trim, and collapse spaces/hyphens to single underscores."""
    s = name.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def build_class_map(source_names: dict[int, str], unified: dict[str, int]):
    """Map a source model's {id: name} to {source_id: unified_id} by normalized
    name (via aliases). Returns (class_map, dropped) where dropped is a list of
    (source_id, original_name) for classes with no unified target.
    """
    class_map: dict[int, int] = {}
    dropped: list[tuple[int, str]] = []
    for src_id, raw in source_names.items():
        canon = _ALIASES.get(normalize_class_name(raw))
        if canon is not None and canon in unified:
            class_map[src_id] = unified[canon]
        else:
            dropped.append((src_id, raw))
    return class_map, dropped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_class_remap.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Reconcile aliases with the real data**

Compare the `_ALIASES` keys against the actual normalized names recorded in Task 1 Step 4. If a downloaded dataset uses a name not covered (e.g. `red_robot`), add it to `_ALIASES` and add a test case. Re-run the test.

- [ ] **Step 6: Commit**

```bash
git add web/training/__init__.py web/training/class_remap.py tests/test_class_remap.py
git commit -m "feat(training): pure class-name normalizer + remap for unified bumper schema"
```

---

## Task 3: Merge script

**Files:**
- Create: `scripts/merge_bumper_datasets.py`

Verification is the printed per-source/per-split/per-class counts plus a spot-check of remapped labels (not a unit test — the pure logic is already tested in Task 2).

- [ ] **Step 1: Write the merge script**

```python
#!/usr/bin/env python3
"""Merge the two Roboflow bumper datasets into data/bumper_merged/ under the
unified schema {0: red_bumper, 1: blue_bumper, 2: fuel}.

Models scripts/merge_roboflow_dataset.py but remaps BY NORMALIZED NAME (via
web/training/class_remap) so it is robust to the two sources ordering classes
differently. Source class names (recorded from data.yaml during download):

    Benjamin (data/bumper_benjamin/data.yaml): <FILL IN from Task 1 Step 4>
    Karthik  (data/bumper_karthik/data.yaml):  <FILL IN from Task 1 Step 4>
"""
import shutil
from collections import Counter
from pathlib import Path

import yaml  # PyYAML (ships with ultralytics)

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from web.training.class_remap import build_class_map  # noqa: E402

UNIFIED = {"red_bumper": 0, "blue_bumper": 1, "fuel": 2}
MERGED = PROJECT_ROOT / "data" / "bumper_merged"

# (local dir, prefix, {split_name_in_source: unified_split}) — Roboflow uses
# train/valid/test; we normalize "valid" -> "val".
SOURCES = [
    (PROJECT_ROOT / "data" / "bumper_benjamin", "benj"),
    (PROJECT_ROOT / "data" / "bumper_karthik", "kart"),
]
SPLIT_MAP = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def load_source_names(src_dir: Path) -> dict[int, str]:
    data = yaml.safe_load((src_dir / "data.yaml").read_text())
    names = data["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def remap_label_file(src_label: Path, dst_label: Path, class_map: dict[int, int]) -> Counter:
    counts: Counter = Counter()
    out = []
    for line in src_label.read_text().strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(parts[0])
        if cls in class_map:
            parts[0] = str(class_map[cls])
            out.append(" ".join(parts))
            counts[class_map[cls]] += 1
    # Write even if empty (a legitimately empty negative image), so the image is kept.
    dst_label.write_text(("\n".join(out) + "\n") if out else "")
    return counts


def copy_source(src_dir: Path, prefix: str) -> Counter:
    names = load_source_names(src_dir)
    class_map, dropped = build_class_map(names, UNIFIED)
    print(f"\n{src_dir.name}: names={names}")
    print(f"  class_map (src_id->unified)={class_map}")
    if dropped:
        print(f"  DROPPED classes (no unified target): {dropped}")
    total = Counter()
    for src_split, dst_split in SPLIT_MAP.items():
        split_dir = src_dir / src_split
        img_dir = split_dir / "images"
        if not img_dir.exists():
            continue
        out_img = MERGED / dst_split / "images"
        out_lbl = MERGED / dst_split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        n = 0
        split_counts = Counter()
        for img in img_dir.glob("*"):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl = split_dir / "labels" / f"{img.stem}.txt"
            shutil.copy2(img, out_img / f"{prefix}_{img.name}")
            if lbl.exists():
                split_counts += remap_label_file(lbl, out_lbl / f"{prefix}_{img.stem}.txt", class_map)
            else:
                (out_lbl / f"{prefix}_{img.stem}.txt").write_text("")
            n += 1
        total += split_counts
        print(f"  {src_split}->{dst_split}: {n} images, per-class boxes={dict(split_counts)}")
    return total


def write_yaml():
    (MERGED / "dataset.yaml").write_text(
        "# Merged FRC bumper dataset (Benjamin + Karthik), unified schema.\n"
        f"path: {MERGED}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "names:\n  0: red_bumper\n  1: blue_bumper\n  2: fuel\n\n"
        "nc: 3\n"
    )


def main():
    if MERGED.exists():
        print(f"Removing existing {MERGED}")
        shutil.rmtree(MERGED)
    grand = Counter()
    for src_dir, prefix in SOURCES:
        grand += copy_source(src_dir, prefix)
    write_yaml()
    print(f"\nMerged dataset.yaml written. Grand total per-class boxes: {dict(grand)}")
    for split in ["train", "val", "test"]:
        d = MERGED / split / "images"
        if d.exists():
            print(f"  {split}: {len(list(d.glob('*')))} images")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Fill in the recorded class names**

Replace the `<FILL IN ...>` comment placeholders with the actual `names` from Task 1 Step 4.

- [ ] **Step 3: Run the merge**

Run: `.venv/bin/python scripts/merge_bumper_datasets.py`
Expected: prints each source's `class_map`, any DROPPED classes (investigate if a bumper/fuel class was unexpectedly dropped — that means an alias is missing, go fix Task 2), per-split image counts, and a non-zero per-class box total for all three unified classes.

- [ ] **Step 4: Spot-check remapped labels**

Run: `head -3 data/bumper_merged/train/labels/benj_*.txt | head -20` and confirm class ids are only 0/1/2. Open one image+label pair mentally against the printed `class_map` to confirm red/blue weren't swapped.

- [ ] **Step 5: Commit**

```bash
git add scripts/merge_bumper_datasets.py
git commit -m "feat(training): merge both bumper datasets into unified red/blue/fuel schema"
```

---

## Task 4: Fuel-labeling-risk inspection (data-quality gate)

**Files:**
- Create: `scripts/inspect_fuel_labels.py`

The spec flags that the Benjamin set (bumpers-only) may contain unlabeled fuel, which would teach the model "fuel = background." This task surfaces the risk so a decision is recorded before training.

- [ ] **Step 1: Write the inspection script**

```python
#!/usr/bin/env python3
"""Sample Benjamin-source images in the merged dataset and report how many have
ANY fuel (class 2) label. The Benjamin set has no fuel class, so its images will
show 0 fuel labels by construction — the point is to eyeball the saved montage and
judge whether fuel balls are VISIBLE but unlabeled in those images.

Writes up to N sample images (with any existing boxes drawn) to /tmp/benj_fuel_check/.
"""
import random
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MERGED = PROJECT_ROOT / "data" / "bumper_merged"
OUT = Path("/tmp/benj_fuel_check")
N = 30


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = sorted((MERGED / "train" / "images").glob("benj_*"))
    random.seed(42)
    sample = random.sample(imgs, min(N, len(imgs)))
    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl = MERGED / "train" / "labels" / f"{img_path.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().strip().splitlines():
                if not line:
                    continue
                c, cx, cy, bw, bh = (float(x) for x in line.split())
                x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
                color = [(0, 0, 255), (255, 0, 0), (0, 255, 0)][int(c)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.imwrite(str(OUT / img_path.name), img)
    print(f"Wrote {len(sample)} annotated samples to {OUT}")
    print("Open them and check: are there fuel balls in these images that have NO "
          "green box? If many, apply the spec's mitigation (pseudo-label or accept).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and review**

Run: `.venv/bin/python scripts/inspect_fuel_labels.py`
Then open `/tmp/benj_fuel_check/` images. Decide:
- **No/rare visible fuel in Benjamin images** → no action; note this in the commit message.
- **Frequent unlabeled fuel** → apply spec mitigation (a): train a quick fuel-only model on the Karthik split, run it on Benjamin images to pseudo-label fuel (class 2), spot-review, and re-run the merge with those labels. (Add this as a follow-up task only if needed — do not pre-build it.)

- [ ] **Step 3: Commit**

```bash
git add scripts/inspect_fuel_labels.py
git commit -m "chore(training): fuel-label inspection for unlabeled-fuel risk; decision: <RECORD HERE>"
```

---

## Task 5: Train the model

**Files:**
- Create: `scripts/train_bumper_model.py`

Long-running; not TDD. Verification = `best.pt` exists with correct `names` and reasonable metrics.

- [ ] **Step 1: Write the training script**

```python
#!/usr/bin/env python3
"""Train YOLOv8m on the merged bumper dataset -> runs/train/frc_bumpers/.

Schema: 0=red_bumper, 1=blue_bumper, 2=fuel. RTX 3090 (24GB).
"""
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = PROJECT_ROOT / "data" / "bumper_merged" / "dataset.yaml"
RUNS = PROJECT_ROOT / "runs" / "train"

BASE_MODEL = "yolov8m.pt"
EPOCHS = 100
PATIENCE = 20
IMG_SIZE = 640
BATCH_SIZE = 16        # yolov8m@640 fits comfortably on 24GB; raise to 24/32 if VRAM allows
DEVICE = 0
WORKERS = 8


def main():
    if not DATASET_YAML.exists():
        raise SystemExit(f"Dataset not found: {DATASET_YAML} (run merge first)")
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS, patience=PATIENCE, imgsz=IMG_SIZE, batch=BATCH_SIZE,
        device=DEVICE, workers=WORKERS,
        project=str(RUNS), name="frc_bumpers", exist_ok=True,
        pretrained=True, optimizer="auto", seed=42, deterministic=True,
        amp=True, close_mosaic=10, val=True, plots=True,
        # Color is class-defining here — keep hue jitter modest so red/blue don't bleed.
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, flipud=0.0, mosaic=1.0, mixup=0.0,
    )
    best = RUNS / "frc_bumpers" / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    m = YOLO(str(best))
    print(f"Model names: {m.names}")
    metrics = m.val(data=str(DATASET_YAML))
    print(f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
```

> **Note on `hsv_h`:** keep hue augmentation small. Large hue shifts can turn a red bumper bluish in augmented samples and confuse the very distinction the model exists to make. `0.015` matches the existing v5 script and is safe.

- [ ] **Step 2: Run training**

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_bumper_model.py`
Expected: training runs to completion (or early-stops via patience), prints `Model names: {0: 'red_bumper', 1: 'blue_bumper', 2: 'fuel'}`, and final mAP50.

> This can run a while. If executing via subagents, run it in the background and poll; do not block a review checkpoint on it.

- [ ] **Step 3: Verify the contract artifact**

Run: `.venv/bin/python -c "from ultralytics import YOLO; print(YOLO('runs/train/frc_bumpers/weights/best.pt').names)"`
Expected: `{0: 'red_bumper', 1: 'blue_bumper', 2: 'fuel'}` — this is the exact contract SP-B depends on. If names differ, the merge `dataset.yaml` order is wrong; fix and retrain.

- [ ] **Step 4: Commit the script + weights**

```bash
git add scripts/train_bumper_model.py
# best.pt is a binary artifact; the existing hub_zone model IS committed under runs/,
# so follow that precedent and commit the new weights too.
git add -f runs/train/frc_bumpers/weights/best.pt
git commit -m "feat(training): train YOLOv8m bumper model (red/blue/fuel)"
```

> Check whether `runs/` is git-ignored first: `git check-ignore runs/train/frc_bumpers/weights/best.pt`. If ignored, `-f` is required (as above) and matches how the current `hub_zone_yolo26s/weights/best.pt` got committed.

---

## Task 6: Evaluate — per-class mAP + red↔blue confusion (acceptance gate)

**Files:**
- Create: `scripts/eval_bumper_model.py`

- [ ] **Step 1: Write the eval script**

```python
#!/usr/bin/env python3
"""Per-class metrics + red<->blue confusion for the trained bumper model."""
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = PROJECT_ROOT / "data" / "bumper_merged" / "dataset.yaml"
BEST = PROJECT_ROOT / "runs" / "train" / "frc_bumpers" / "weights" / "best.pt"


def main():
    model = YOLO(str(BEST))
    print(f"names={model.names}")
    metrics = model.val(data=str(DATASET_YAML), plots=True)
    # Per-class metrics. ap50 = AP@0.5 per class (the gate metric); maps = mAP@0.5:0.95.
    ap50 = metrics.box.ap50          # per-class AP@50
    maps = metrics.box.maps          # per-class mAP@50-95
    p = metrics.box.p; r = metrics.box.r
    for i, name in model.names.items():
        print(f"  class {i} {name:12s}  P={p[i]:.3f} R={r[i]:.3f} "
              f"mAP50={ap50[i]:.3f} mAP50-95={maps[i]:.3f}")
    # Confusion matrix: rows/cols include a background column at the end.
    cm = metrics.confusion_matrix.matrix
    print("\nConfusion matrix (rows=true, cols=pred; last index=background):")
    print(cm)
    # Red(0)<->Blue(1) off-diagonal as a fraction of true red/blue.
    red_total = cm[:, 0].sum() if cm.shape[1] > 0 else 0
    blue_total = cm[:, 1].sum() if cm.shape[1] > 1 else 0
    red_as_blue = cm[1, 0]; blue_as_red = cm[0, 1]
    print(f"\nred->blue confusions: {red_as_blue} (of {red_total} pred-red)")
    print(f"blue->red confusions: {blue_as_red} (of {blue_total} pred-blue)")


if __name__ == "__main__":
    main()
```

> The exact ultralytics `confusion_matrix.matrix` orientation (true vs pred on rows) can vary by version; the saved `runs/train/frc_bumpers/confusion_matrix_normalized.png` from training is the authoritative visual. Use the printout to locate the red/blue cells and the PNG to confirm orientation.

- [ ] **Step 2: Run eval**

Run: `.venv/bin/python scripts/eval_bumper_model.py`
Expected: per-class P/R/mAP for all three classes and the confusion matrix. Also open `runs/train/frc_bumpers/confusion_matrix_normalized.png`.

- [ ] **Step 3: Acceptance-footage sanity check**

Using the app's Python download API + a clip from the acceptance harness (see project memory `acceptance-harness-download`), extract a few frames and run:

```bash
.venv/bin/python -c "
from ultralytics import YOLO; import cv2, sys
m = YOLO('runs/train/frc_bumpers/weights/best.pt')
for f in sys.argv[1:]:
    r = m(f, conf=0.35)[0]
    print(f, [(m.names[int(c)], float(s)) for c, s in zip(r.boxes.cls, r.boxes.conf)])
    cv2.imwrite(f.replace('.jpg','_pred.jpg'), r.plot())
" /tmp/frame1.jpg /tmp/frame2.jpg
```
Eyeball the `_pred.jpg` outputs: red bumpers boxed red, blue boxed blue, fuel detected.

- [ ] **Step 4: Record the gate decision**

Write a short `runs/train/frc_bumpers/EVAL.md` capturing: per-class mAP50, red↔blue confusion %, sanity-check verdict, and GO / NO-GO for switching SP-B's default model. Gate (judgment, per spec): both bumper classes' mAP50 roughly comparable to current robot recall, red↔blue off-diagonal < ~5%.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_bumper_model.py runs/train/frc_bumpers/EVAL.md
git commit -m "feat(training): bumper model evaluation + red/blue confusion gate"
```

---

## Done when

- `runs/train/frc_bumpers/weights/best.pt` exists with `names == {0: red_bumper, 1: blue_bumper, 2: fuel}`.
- `EVAL.md` records a GO decision (or a NO-GO with the reason + next step).
- SP-B can now point its detector at this artifact.
