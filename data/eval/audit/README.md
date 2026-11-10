# Replay audits

A replay audit estimates how often a stored replay's team labels are wrong,
with no hand annotation. It samples labelled (frame, slot) moments, cuts crops
of the robot at each slot from every camera view, and lets a blind judge say
which team it is (or abstain):

**estimated mis-ID = judge-confident disagreements / judge-confident labelled
samples, weighted by slot-time.**

The tool is `scripts/audit_replay.py`. Tests: `tests/test_audit_replay.py`.

## Commands

```
# 1. sample: deterministic for a given replay, n and seed
.venv/bin/python scripts/audit_replay.py sample --replay a0bbc991 --n 150 --seed 0

# 2a. Jev as the judge: calls the Jev API, about 1 cent per 150 items
.venv/bin/python scripts/audit_replay.py jev --audit data/eval/audit/2026cur_qm8_a0bbc991 --mode views
#     --mode multi: one request with every view; --mode views: one request per view, fused

# 2b. Claude Code subagent judges: see JUDGE_BRIEF.md

# 3. score: GT section only when the match has hand GT (2026cur_qm8)
.venv/bin/python scripts/audit_replay.py score --audit data/eval/audit/2026cur_qm8_a0bbc991 \
    --answers data/eval/audit/2026cur_qm8_a0bbc991/answers_jev_views.jsonl [--min-conf 0.7] [--dump rows.jsonl]
```

## Audit directory layout

`<match_key>_<replay8>/`

| file | contents | blind? |
|---|---|---|
| `items.jsonl` | one line per judgeable item: `sample_id`, image paths (relative to the audit dir), roster red/blue | yes: give this to judges |
| `items/<sid>/{main,corner_left,corner_right}.jpg` | 512x512 crops, deinterlaced, tick-marked (gitignored; regenerate with `sample`) | yes |
| `items/<sid>/item.json` | sample record: frame, t_s, slot, alliance, **pipeline_label**, stratum, views, projection/crop details | **no** |
| `samples.jsonl` | every `item.json` in one file | **no** |
| `audit.json` | replay, strata sizes (`N`) and draws (`n`), frame period, labelled/unlabelled slot-seconds | **no** |
| `answers_*.jsonl` | judge answers `{sample_id, team, confidence, ...}` | n/a |

Sample ids are assigned in shuffled order, so an id says nothing about time,
slot or stratum.

## Sampling

- Units are the replay's (frame, slot) rows that have a field position and fall
  inside a processed period (auto or teleop).
- Labelled units are stratified by (period, slot). Auto gets at least 20% of
  the labelled draw (it is about 13% of the match). Every slot stratum gets at
  least one sample. Within a stratum, draws are systematic over time (equal
  bins, one random offset).
- Unlabelled units form a separate stratum per period, with 10% of n, for the
  coverage estimate.
- Weights: each item stands for `N_stratum / n_judgeable_in_stratum x frame
  period` slot-seconds. Items with no usable view drop out, and their weight
  is spread over the rest of their stratum.

## Crops

The geometry is the window-label stage's (`web/field/window_crops.py`). The
slot's `field_pos` is pushed half a robot away from the main camera (the
tracker follows the near bumper edge). The footprint centre at 0.10 m is then
projected into each view with the calibration versions in force when the job
ran (`scripts/build_track_truth.job_views`). The crop side is 150 px + 0.5 x
the projected width of a 0.95 m bumper. Each crop is cut at native resolution,
put through `web/bumper_register.deinterlace_crop`, upscaled to 512 px, and
tick-marked at the spot.

**Corner views are gated.** The corner calibrations (2026-06-30) are only good
near each camera. Against the job's own corner detections, the projected spot
lands within one bumper width of a detection for under 10% of targets whose
projected bumper is under 40 px, and for only about 50–75% of targets of
100 px or more. The main view manages about 80%. Far targets are off by 3–20
bumper widths, always in the same direction. So a corner view is kept only
when:

- its projected bumper is at least 60 px, and
- exactly one job detection lies within 1.5 bumper widths of the projection,
  with the runner-up at least 1.5x farther away.

The crop is then centred on that detection. Detection colour is not used,
because it would leak the slot's alliance to the judge. As a result, only
about a quarter of items get a corner view, and the main view does almost all
the work.

## Answers and scoring

Answer rows are `{"sample_id", "team": "<roster team>"|"none"|"unsure"|"no_robot",
"confidence": 0-1 | "high"|"medium"|"low"}`. `high` maps to 0.9, `medium` to
0.7 and `low` to 0.4. A team counts as confident at `--min-conf`, default 0.7.

`score` prints:

- judge coverage
- agreement with the pipeline
- the estimated mis-ID, with a stratified-bootstrap 95% CI, in seconds, and split into cross- and same-alliance
- the judge's `no_robot` rate on labelled time
- breakdowns by slot, by pipeline team and by period
- the unlabelled stratum: the share where the judge could name a team

With GT, each sample is matched to a GT robot the way `score_match` does it:

- pipeline positions shifted 0.475 m away from the main camera
- GT positions from `gt_scoring.gt_position_at`
- one Hungarian assignment at that moment, 1.0 m gate

`score` then prints pipeline vs GT, judge vs GT, the audit's estimate vs the
GT-measured mis-ID on the same items, and P(judge right) per confidence bin.

## Results (2026-11-10, Jev 1.14 judge)

| audit | items (of 150) | views per item | Jev mode, min-conf | coverage | est. mis-ID [95% CI] |
|---|---|---|---|---|---|
| `2026cur_qm8_a0bbc991` | 149 | main 149, corner_left 23, corner_right 15 | views, 0.7 | 22% | 21% [8–35] |
| | | | views, 0 | 37% | 17% [7–28] |
| | | | multi, 0 | 34% | 16% [7–26] |
| `2026cur_qm13_0baf4625` | 147 | main 147, corner_left 15, corner_right 21 | views, 0.7 | 25% | 67% [51–81] |
| | | | views, 0 | 35% | 67% [55–79] |
| | | | multi, 0 | 34% | 69% [57–80] |

### qm8 calibration against hand GT

114 of the 134 labelled items sit on a GT robot within 1 m. The rest are on
the unannotated robot 9066, on phantom slots, or in GT gaps.

- **The pipeline is wrong on 20.1% (22/114)** of GT-matched labelled items.
  That matches `score_ground_truth`: 9% wrong / (37% right + 9% wrong) = 20%.
- **Jev is almost never wrong when it names a team.** In multi mode it was
  44/44 right at any confidence. In views mode it was 43/44 right (the one
  miss is in the 0.5–0.7 bin) and 28/28 right at 0.7 or above.
- **On the same items, the audit and GT agree:**

  | mode | items | audit est. | GT-measured |
  |---|---|---|---|
  | multi | 44 | 14.3% | 14.3% (identical item by item) |
  | views | 44 | 18.9% | 16.5% (one false flag) |

- **Coverage bias.** Jev abstains more often where the pipeline is wrong. GT
  mis-ID on the items Jev answered is 14–16%, against 20% on all GT-matched
  items. So the audit estimate runs about 4–6 points low on qm8: it is a lower
  bound. The 70 items Jev abstained on hold 15–16 of the 22 wrong labels.
- **Jev's `no_robot` is not a phantom detector.** It says `no_robot` on about
  40 GT-matched items where the robot is plainly at the spot. It is too
  conservative to measure phantom labels.

qm13 (`0baf4625`, Nov 6, before the Nov 7 identity work): Jev flags about two
thirds of its confident items as mislabelled, half of them cross-alliance.
Spot checks confirm it: 8725 labelled 302, 589 labelled 8725, 6045 labelled 302.

## Claude-judge calibration (Nov 10 2026)

Claude Code subagents as blind judges (`JUDGE_BRIEF.md`, one item at a time), `--min-conf 0.7`:

| audit | judge | coverage (labelled time) | judge right vs GT | audit mis-ID | GT mis-ID (same items) |
|---|---|---|---|---|---|
| qm8 a0bbc991 | Fable (`answers_fable.jsonl`) | 41% | 54/54 | 15.4% | 15.4% |
| qm8 a0bbc991 | Sonnet (`answers_sonnet.jsonl`) | 39% | 50/51 | 14.3% | 16.3% |
| qm8 a0bbc991 | Jev multi | 34% | 44/44 | 14.3% | 14.3% |
| qm13 0baf4625 | Fable | 38% | (no GT) | 69.5% [58–80] | – |
| qm13 0baf4625 | Jev multi | 34% | (no GT) | 69% [57–80] | – |

- On qm13, Fable and Jev both named a team on 49 items and agree on 48, so the cheap Jev audit reproduces the Claude audit. Jev is the runtime auditor; Claude calibrates it.
- All judges abstain on the hardest items, where the labels are wrong more often. On qm8, GT mis-ID is 20.1% over all GT-matched items but 15–16% on the judge-confident ones, so treat audit mis-ID as a lower bound (about 5 points low on qm8).
- qm13 0baf4625 is a Nov 6 replay, before the Nov 7 identity defaults.
