# frc_bumpers_offpalette_v1 — Evaluation

**Date:** 2026-11-20
**Model:** YOLOv8m, 2 classes `{0: blue_bumper, 1: red_bumper}`. Fine-tuned from
`frc_bumpers_equipment_hardneg/weights/best.pt` (production) with the stock
`scripts/train_bumper_model.py` recipe (imgsz 640, AutoBatch, patience 20, hsv_h 0.015).
Early-stopped at epoch 85; best = epoch 65.
**Data:** `data/bumper_offpalette/v1/merged.yaml` = the old hard-negative set (512 train / 131 val)
plus the off-palette harvest (`scripts/harvest_offpalette_bumpers.py`). The harvest has 221 train
and 71 valid frames, all full MAIN views (1920x723) with every bumper labelled. They come from
cabl qm18/26/44/65 and cur qm22/39/52/63, with about 86 teal and 23 navy instances. 6 harvested frames with
bad teal pixel labels were moved to `v1/dropped_review/` after contact-sheet review.
**Weights:** `runs/train/frc_bumpers_offpalette_v1/weights/best.pt` (Git LFS)

## Decision: ✅ WIN on the target, recommend a live A/B before making it the default

Teal recall goes from **24% → 96%** and navy from **76% → 83%**. On the 2026cabl_qm5 navy bumper
(team 498) it goes **63% → 88%**. Royal-blue and red recall are flat or up, and colour calls are unchanged.
False boxes per frame fall when pooled (0.61 → 0.49). The one cost: about +0.1 false boxes per frame on cur_qm103,
mostly a static red ramp corner (see below). This is NOT wired into the pipeline.

## Held-out matches (never trained on)

Method (`scripts/eval_bumper_offpalette.py`): both models run on the same x-ray layer frames.
Each match uses every 3rd layer (5 fps), about 850 frames. The input is the main view, rows 0..722, at conf 0.35,
called exactly as the pipeline calls it. Each tracked robot's cached SAM mask is
grouped by the HSV colour of its lower 40% (label-free). The thresholds are the same as the Nov 20
baseline: teal H85-100, royal H101-127 with S>110, navy V16-50, red. A mask "has a bumper" if a box's centre
falls in the mask bbox's lower 65% (one-to-one, greedy by conf). The `before` arm re-run
reproduces the cached production detections 100% (8,119 samples, |Δconf| 0.000).
2026cur_qm85 is held out but has no x-ray cache, so it is not measured.

### Pooled by bumper colour (4 matches)

| group | n | det % before → after | conf med | conf p25 | colour ok |
|---|---:|---:|---:|---:|---:|
| **teal** (→ blue) | 227 | **24.2% → 96.0%** (+71.8) | 0.55 → 0.84 | 0.45 → 0.80 | 90.9% → 100% |
| **navy** (→ blue) | 476 | **75.6% → 83.2%** (+7.6) | 0.77 → 0.81 | 0.63 → 0.73 | 99.7% → 99.7% |
| royal blue | 3118 | 93.1% → 92.8% (−0.3) | 0.87 → 0.87 | 0.83 → 0.85 | 99.6% → 99.3% |
| red | 3554 | 91.3% → 92.9% (+1.7) | 0.90 → 0.88 | 0.84 → 0.86 | 100% → 99.9% |
| unclassified* | 744 | 34.1% → 50.1% (+16.0) | 0.70 → 0.76 | 0.58 → 0.65 | – |

\* Masks no colour group claims, either tiny or washed out by venue lighting. Most cabl robots fail the
strict royal/red saturation gates, so cabl_qm5 has 0 "red" and 17 "royal" samples. (The brief's qm5
royal/red baseline numbers came from different settings; teal and navy match it exactly.)

### Per match, off-palette groups

| match | group | n | det % before → after | conf p25 |
|---|---|---:|---:|---:|
| 2026cabl_qm5 | teal (4322) | 227 | 24.2% → 96.0% | 0.45 → 0.80 |
| 2026cabl_qm5 | navy (498) | 89 | 62.9% → 87.6% | 0.57 → 0.73 |
| 2026cur_qm103 | navy (6800) | 328 | 79.6% → 83.2% | 0.63 → 0.72 |
| 2026cur_qm112 | navy | 57 | 75.4% → 78.9% | 0.66 → 0.79 |
| 2026cabl_qm12 | red | 388 | 87.6% → 96.6% | 0.78 → 0.84 |

### Focus teams (team = the pipeline's slot label)

| team | match | n | det % before → after |
|---|---|---:|---:|
| 498 (navy) | cabl_qm5 | 365 | 38.9% → 68.8% |
| 4322 | cabl_qm12 (red alliance, orange-red bumper) | 401 | 85.8% → 95.0% |
| 6800 (navy) | cur_qm103 | 373 | 63.5% → 68.6% |
| 6800 (navy) | cur_qm112 | 446 | 93.7% → 95.3% |

4322 in qm5 has only 8 labelled samples, because the pipeline barely labelled it. The teal colour group above covers it label-free.

### False boxes per frame

A false box is matched to no robot mask, has its centre in no mask bbox, and has IoS < 0.5 with every cached whole-robot box.

| | cabl_qm5 | cabl_qm12 | cur_qm103 | cur_qm112 | pooled |
|---|---:|---:|---:|---:|---:|
| before | 1.353 | 0.421 | 0.219 | 0.456 | 0.610 |
| after | 0.793 | 0.486 | 0.316 | 0.367 | **0.490** |

From a visual check of the "after-only" false boxes:
- **cabl_qm12:** they are mostly real, untracked robots and bumpers (no SAM mask), so they are recall gains the metric can't credit. Only a few are true false positives: blue field structure and a referee.
- **cur_qm103 (+0.10/frame):** these are genuine. Most are a static red ramp corner (conf 0.4-0.7) and red objects at the frame edges. They are static, so the pipeline's static-equipment handling may absorb them; a live A/B will tell.

## Regression: standard bumper test split (`data/bumper_hardneg_equipment` test, 54 images)

`scripts/eval_bumper_model.py --split test`

| | mAP50 | mAP50-95 | blue P / R | red P / R | red→blue | blue→red |
|---|---:|---:|---:|---:|---:|---:|
| before | 0.951 | 0.697 | 0.917 / 0.924 | 0.934 / 0.900 | 0 | 0 |
| after | **0.967** | **0.721** | 0.936 / 0.905 | 0.951 / 0.913 | 1 | 0 |

There is one new red→blue swap among 190 red instances. Validation on the merged val set
(old val + harvest valid, 202 images) goes from mAP50 0.950 / mAP50-95 0.692 at epoch 1 to 0.966 / 0.734 at best.

## Caveats / next

- Navy gains are modest. The harvest could only label 6800 when the old model already found most of
  the bumper, because 6800's near-black bumper can't be separated by colour. The hardest 6800 frames are
  therefore not in the training set. Hand- or SAM-assisted navy boxes would be the next lever.
- Teal training labels come from a pixel labeller (about 10% questionable after the drop). About 0.5 occluded
  robots per frame are left unlabelled in harvested frames.
- Not wired in: `web/app.py` `BUMPER_MODEL_*_PATH` / `FRC_BUMPER_MODEL`. To try it, run a job with
  `FRC_BUMPER_MODEL=<abs path>/runs/train/frc_bumpers_offpalette_v1/weights/best.pt`.
  A full-match A/B on cabl_qm5 and cur_qm103 (teal and navy) plus one standard match is the next step,
  but only with Nick's go-ahead.

Full tables: `heldout_eval.md` / `heldout_eval.json` in this directory.

## Full-match A/B (Nov 20 evening): the model ALONE does not help, and needs the HSV-knob fix

The `after` arm ran on a second app instance on :8014, using the same code as :8004 (main-server 986b93df plus
its uncommitted HSV-knob patch), with `FRC_BUMPER_MODEL` set to these weights and default job config.
`before` = today's default runs on :8004. `hsv_wide` = the other session's arm: old model plus widened blue
HSV knobs (hue_min 85, sat_min 60, val_min 25). Each arm has one run per match. Scored by
`scripts/ab_replay_team_metrics.py`, using TBA rosters and no GT; full tables are in `ab_fullmatch.md`.

| arm | replays (cabl qm5, qm12, cur qm103, qm112) | focus-team label cov | focus slot share | focus wrong-alliance | other teams cov |
|---|---|---:|---:|---:|---:|
| before | 525ed57e 2dcf7631 1173562e 07f2b816 | 50.3% | 0.28 | 27.9% | 64.2% |
| **after (new model)** | ebfc1f1d 12c81643 107a0769 386b1529 | **41.9%** | **0.16** | **34.4%** | 65.2% |
| hsv_wide (old model) | 765906e3 1a224b1a 38f1cdc6 c27bcb80 | 53.9% | 0.48 | 1.8% | 65.7% |

Focus teams: 4322 (qm5/qm12), 498 (qm5), 6800 (qm103/qm112).
- **cabl_qm5:** blue-slot occupancy fell from 58% to 48%. Team 498's coverage fell from 56% to 14%, and 4322 is still unlabelled.
- **cur_qm103:** 6800 sits on a wrong-alliance slot 53% of the time, up from 34%.
- **Other teams:** flat (64% → 65% coverage).

**Why:** with `bumper_color_read=strict`, the HSV read replaces the YOLO colour on every main-view detection.
The default thresholds can't see teal, so the new model's extra teal detections get flipped or
cleared. On the after-arm cabl_qm5 run, 1,392 of 7,537 colours were flipped and 2,245 cleared. The hsv_wide run of the same match
flipped 184 and cleared 504. The extra detections therefore become colourless or wrong-colour tracks.

**Verdict:** don't ship the model alone. The next test is the combined arm: new model plus the widened HSV knobs.

### Combined arm: new model + widened HSV knobs (same :8014 setup)

Replays: a20f1b8b (cabl qm5), 8e901296 (cabl qm12), 27f4624b (cur qm103), 9c15a07a (cur qm112).
The colour read logged `blue_hue_min=85, blue_sat_min=60, blue_val_min=25`.

| arm | focus-team label cov | focus slot share | focus wrong-alliance | other teams cov | other share |
|---|---:|---:|---:|---:|---:|
| before | 50.3% | 0.28 | 27.9% | 64.2% | 0.54 |
| new model only | 41.9% | 0.16 | 34.4% | 65.2% | 0.52 |
| hsv_wide only | 53.9% | 0.48 | 1.8% | 65.7% | 0.54 |
| **combined** | 53.1% | **0.65** | **1.6%** | 66.2% | 0.50 |

- **Wrong-alliance:** the combined arm undoes the model-only regression, matching hsv_wide.
- **Slot naming:** combined names the focus teams' slots most firmly. 6800 on cur_qm103 reaches a 0.91 share (hsv_wide 0.48, before 0.22); 4322 on cabl_qm12 reaches 0.71 (hsv_wide 0.60).
- **Coverage:** focus-team coverage ties hsv_wide (53% vs 54%).
- **4322 on cabl_qm5** stays essentially unlabelled (≤ 9% cov) in every arm. Detection is no longer the bottleneck there; the reader/lineup path is the next place to look.
- **Other teams:** the slot share dip (0.54 → 0.50) comes from single-run swings of individual slots (e.g. 302 on qm112 at 0.27, against 0.65–0.88 elsewhere). One run per arm can't separate that from noise.

**Verdict:**
- The HSV-knob fix is the main win and should ship first.
- The new weights add firmer focus-team naming on top, but only alongside the HSV knobs. Never ship them with the default strict thresholds.
- Before making the weights default: run a second combined and hsv_wide run on these matches to measure noise, plus a standard-palette match to check for regressions.

## Repeats + standard-palette regression (Nov 20, late)

Each arm got a second run on the same :8014 setup. `combined_r2` used the new weights with the widened HSV knobs. `hsvwide_r2` used the production weights with the widened HSV knobs. Both arms also ran on 2026cur_qm8, which has standard bumpers and hand-labelled ground truth. All 10 jobs exited rc=0. Full tables are in `ab_repeats.md` / `ab_repeats.json`.

| arm | replays (cabl qm5, qm12, cur qm103, qm112) | focus cov | focus share | focus wrong-al | other cov | other share |
|---|---|---:|---:|---:|---:|---:|
| before | 525ed57e 2dcf7631 1173562e 07f2b816 | 50.3% | 0.28 | 27.9% | 64.2% | 0.54 |
| hsvwide_r1 | 765906e3 1a224b1a 38f1cdc6 c27bcb80 | 53.9% | 0.48 | 1.8% | 65.7% | 0.54 |
| hsvwide_r2 | 30e6bbee 80b4a2eb 6f2f0db8 9eede4e7 | 52.3% | 0.43 | 1.1% | 65.5% | 0.53 |
| combined_r1 | a20f1b8b 8e901296 27f4624b 9c15a07a | 53.1% | 0.65 | 1.6% | 66.2% | 0.50 |
| combined_r2 | 5b1cfd9b 60d3001e f9d72a47 22460399 | 49.8% | 0.65 | 1.5% | 67.0% | 0.50 |

**Noise:** run-to-run spread is small, about 2–3 pp on coverage and ≤ 0.05 on share. The arm differences below therefore hold up.
- **Firmer focus naming with combined, but uneven.** Focus share is 0.65 in both combined runs versus 0.43–0.48 for hsvwide. Most of the gain is 6800 on cur_qm103 (0.87–0.91 vs 0.40–0.48) and 4322 on cabl_qm12 (0.70 vs 0.48–0.60). Part of it is a degenerate cell: 4322 on cabl_qm5 gets share 1.00 at ≤ 1% coverage. Leave that cell out and focus share is about 0.56 (combined) vs 0.48 (hsvwide).
- **498 on cabl_qm5 gets worse with combined, and replicably.** No slot ends up named 498 (share 0.00 in both runs, vs 0.42 for hsvwide), and coverage is 42–57% vs 55–60%.
- **The "other share" dip with combined is real, not noise.** It is 0.50 in both runs vs 0.53–0.54. Per team it is mixed: 1197, 2724 and 302 lose share, while 59 and 1684 gain. Other-team coverage is flat to slightly up (66–67% vs 65.5%).
- **Wrong-alliance:** both HSV arms stay at 1–2%, against 28% before.

### 2026cur_qm8 regression (standard palette, GT, `score_replay(score_shift_m=0.475)`)

| arm | replay | cov | err m | label ok | wrong | label on wrong robot | misID s (cross) | wrong-colour slot | lost s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default ×3 | 485037fd bc3961f2 9b8bbb8f | 84.9% | 0.36 | 68.7–69.9% | 0.9% | 5.8–6.1% | 7.25 (3.00) | 4.5% | 117.2 |
| hsvwide_r2 | b76197c7 | 84.3% | 0.36 | 68.2% | 1.3% | 7.8% | 9.75 (4.25) | 6.8% | 121.8 |
| combined_r2 | fbc24932 | 85.0% | 0.35 | 68.8% | 1.1% | 6.6% | 8.50 (5.50) | 7.9% | 117.0 |

The three default runs produce identical tracking, and only the label stage varies. Both HSV arms show a small rise in cross-colour error on standard bumpers: misID +1.3 to +2.5 s, wrong-colour slot 4.5% → 6.8–7.9%. Both arms share the widened knobs, so the widened blue gate is the likely cause, not the weights. It is within the known ±4 s qm8 misID noise for single runs, so it needs watching rather than blocking. The new weights themselves cause no regression on qm8: combined is no worse than hsvwide, and matches default on coverage and label-correct.

### Updated verdict

1. **Ship the widened HSV knobs** (already the other session's patch). This is the big win: focus wrong-alliance falls from 28% to 1–2% and replicates. Before flipping the default, watch the small cross-colour rise on standard-palette matches with 1–2 more GT matches, e.g. qm11 or qm22.
2. **These weights are optional, on top of the HSV knobs only.** They replicably name 6800 and 4322 more firmly and do not regress qm8. However, they lose the 498 slot name on cabl_qm5 and shift naming among other teams, so they are a net positive but not a clear one. Do not make them default yet. The next step would be a GT-labelled off-palette match, which would turn "share" into right/wrong.
3. **4322 on cabl_qm5 stays unlabelled in every arm.** That is a reader/lineup problem, not detection.
