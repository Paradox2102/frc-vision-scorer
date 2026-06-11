# frc_bumpers — Evaluation & Gate Decision

**Date:** 2026-06-06
**Model:** YOLOv8m, 2 classes — `{0: blue_bumper, 1: red_bumper}`
**Dataset:** Benjamin (`frc-bumpers-h0p1g` v4) — 382 train / 109 val / 54 test
**Weights:** `runs/train/frc_bumpers/weights/best.pt`

## Decision: ✅ GO

Replaces the robot+color path: this model supplies robot detections **and** alliance color
directly (class = color). Fuel + hub stay with the existing model ("augment" integration
in SP-B); no hub class needed (hubs static per event).

## Validation metrics (109 val images, 748 instances)

| Class | P | R | mAP50 | mAP50-95 |
|-------|------|------|-------|----------|
| all | 0.947 | 0.949 | **0.977** | 0.770 |
| blue_bumper | 0.918 | 0.939 | 0.970 | 0.756 |
| red_bumper | 0.976 | 0.959 | 0.984 | 0.783 |

## Red↔blue confusion (the headline gate metric)

Confusion matrix (rows = true, cols = pred; last index = background):

```
            pred:  blue   red   bkgd
true blue        [ 341    0    35 ]
true red         [   0  379    20 ]
true bkgd        [  16   12     0 ]
```

- **blue→red swaps: 0**
- **red→blue swaps: 0**

**Zero cross-color errors** — perfect alliance discrimination, far under the ~5% gate.
Residuals are recall gaps (35 blue / 20 red missed → background, ≈9% / 5%) and 28 false
positives (background→bumper, likely unlabeled bumpers in the broadcast crowd/field).

## Gate check vs spec

- Per-class mAP50 (0.97 / 0.984) — strong, comparable-or-better than the legacy robot recall. ✅
- Red↔blue off-diagonal 0% « 5% target. ✅
- Inference ~6 ms/image on the 3090. ✅

## Notes / provenance

- Trained on the Benjamin dataset only. Karthik (`frc-bumpers-red-blue`) was evaluated and
  rejected as a bumper source: it is a fuel dataset (~18k fuel labels, ~10 robot labels).
- AutoBatch picked batch 14 (61% VRAM) on the exclusive 3090; GPU compute-bound at ~97%.
- Inspect `confusion_matrix_normalized.png` + `results.png` in this directory for curves.
- Next: SP-B integration (adapt to "augment": robots+color from this model; fuel from the
  existing model; no hub).
