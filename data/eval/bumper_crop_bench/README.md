# Bumper-crop bench

A persisted, labelled-in-place set of bumper crops for benchmarking the
team-number reader offline: the incumbent free-form reader against closed-set
"decision model" arms. It lives in git because the July 125-crop bench was lost
from a temp dir.

## Provenance

Built 2026-10-12 by `scripts/build_bumper_crop_bench.py --seed 0 --target 200`
from `web/uploads/diagnostic_<job_id>.jsonl` (gitignored, not reproducible
elsewhere):

- 384 diagnostic files; 258 jobs whose `replays.match_key` has a 3v3 roster in
  `match_robots` were scanned (126 skipped: job id not in `replays`, e.g. rerun
  slices). They held 85,497 candidate rows: 12,418 matched, 72,369 null, 710
  filtered.
- Rows are deduped on `(match_key, frame, robot_id)`, since qm8-qm16 were re-run
  many times.
- 200 crops from 26 matches and 65 jobs: 100 `matched` (the baseline matched a
  roster team), 70 `null` (the baseline read nothing or matched no roster team),
  30 `filtered` (`vlm_filtered`, rejected below the legibility floor and never
  read). Each stratum's quota is spread evenly over matches. Within a match,
  picks are spread over the video by quantile bins on `frame`, because `t_ms` is
  processing wall time.
- `crops/<id>.jpg` holds the decoded `crop_b64` bytes as written, never
  re-encoded. A `vlm_read` crop is the exact JPEG the reader received: the
  bumper band, already upscaled by the app to at least 384 px tall, JPEG q90. A
  `filtered` crop is the raw band crop (about 80x40, q60).
- The `baseline_*` fields are historical. They are whatever reader, model and
  config that job ran. The `freeread` arm re-runs today's code.

Rebuilding is deterministic for a fixed seed, target and set of diagnostics. It
keeps labels on re-selected ids and refuses to drop labelled rows without
`--force`.

## manifest.jsonl

| field | meaning |
|---|---|
| `id` | `<match_key>_f<frame>_r<robot_id>`, the dedupe key |
| `file` | `crops/<id>.jpg` |
| `job_id`, `match_key`, `frame`, `t_ms`, `robot_id` | provenance (`job_id == replays.id`) |
| `width`, `height`, `sha1` | crop JPEG dimensions and hash |
| `stratum` | `matched` / `null` / `filtered` |
| `red_teams`, `blue_teams` | roster from `match_robots`, in position order |
| `baseline_raw_read`, `baseline_matched_team`, `baseline_alliance_read`, `baseline_confidence`, `baseline_legibility`, `baseline_surface` | the incumbent reader's output (`null` for filtered rows) |
| `baseline_filter_reason` | for filtered rows (`below_legibility_floor`) |
| `label_status` | `unlabelled` (default), `legible`, `illegible`, `not_bumper` |
| `label_team` | the team digits as a string, set when `legible`. May be off-roster; scoring then expects "other". |
| `label_alliance` | the bumper colour (`red` / `blue`). Set it whenever the colour is visible, even on illegible crops, because it defines the decoys. |
| `label_note` | free text |

### Labelling

1. Render the sheets with
   `.venv/bin/python scripts/bench_contact_sheet.py --filter unlabelled`. The
   PNGs go to `sheets/`, which is gitignored.
2. Edit the `label_*` fields in `manifest.jsonl` in place. Keep one JSON object
   per line.

Use the statuses this way:

- `legible`: you can read the number on a robot bumper.
- `illegible`: it is a robot bumper, but the number can't be read.
- `not_bumper`: there is no robot bumper, e.g. ball piles, field structures, or
  a number printed on a crate.

## Running the bench

```bash
# Inspect the payloads; no network calls
.venv/bin/python scripts/bench_bumper_readers.py --dry-run --arms freeread,vllm-choice,clef-flash

# Local arms. vLLM must already be up and awake (web/start_vllm.sh); the bench
# never starts it. Don't run while a pipeline job is analysing.
.venv/bin/python scripts/bench_bumper_readers.py --arms freeread,vllm-choice --score
.venv/bin/python scripts/bench_bumper_readers.py --arms vllm-choice --blank-calibrate --score
.venv/bin/python scripts/bench_bumper_readers.py --arms vllm-choice --vllm-model Qwen/Qwen3.5-9B --no-thinking

# Workers AI arms. CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN come from the
# env or the repo-root .env.
.venv/bin/python scripts/bench_bumper_readers.py --arms clef-flash,clef --options decoy --score

# Roster-forced-misread stress test: the true team is removed from the options
.venv/bin/python scripts/bench_bumper_readers.py --arms freeread,vllm-choice,clef-flash --options missing-true --score

# Re-score the latest results file per arm (e.g. after adding labels)
.venv/bin/python scripts/bench_bumper_readers.py score
```

Results go to `results/<arm>-<timestamp>.jsonl` (one prediction per crop) and
`results/run-<timestamp>.json` (config and metrics). Both are gitignored.

Other flags: `--limit N` takes a stable-hash subset that spans matches,
`--only-labelled`, `--concurrency` (default 4), `--hint-source baseline|label|none`
and `--permutations K`.

`--labels manifest|tracklet|truth|<path>` (run and `score`) scores against
another label file instead of the manifest's crop labels: `tracklet` is
`tracklet_labels.jsonl`, `truth` is `track_truth.jsonl`. Crops the file does
not cover count as unlabelled. `--subset tracklet` restricts a run (or `score`)
to the crops whose tracklet has at least 3 frames.

**Frame modes (`--frames`).** Tracklets (`tracklets.jsonl`,
`tracklets/<id>/<offset>.jpg`, built by `scripts/build_tracklet_bench.py`)
follow each crop's bumper for +-1.5 s at native resolution. A crop without a
tracklet always falls back to the single bench crop.

- `single` (default): the bench crop.
- `multi4` / `multi8`: the bench crop plus 3 (7) tracklet frames as separate
  images. Clef arms and `freeread` take `multi4`; only `freeread` takes `multi8`.
- `video`: every tracklet frame as an H.264 clip (`clef-omni` only).
- `registered`: the tracklet registered and fused into ONE image by
  `web/bumper_register.py`, then prepped like the app's crops (integer upscale
  to at least 384 px tall, JPEG q90). Every arm reads it as its single image.
  The source video is interlaced, so combed frames are split into their two
  fields first; each field is a separate sample. Samples are aligned to the
  anchor (offset 0) with ECC (affine, phase-correlation init). A sample is
  dropped when its ECC correlation is low, its warp is implausible, or its
  post-warp NCC is low. Survivors are merged with a per-pixel anti-ghost
  weighted mean, then a mild unsharp mask is applied. Classical only: no
  learned super-resolution.
- `anchor`: the same deinterlace, upscale, unsharp and app prep applied to the
  anchor frame alone. This is the no-fusion control: `single` vs `anchor`
  measures the prep, and `anchor` vs `registered` measures the fusion.

Rows record `frames_used` (distinct source frames fused), `reg_fused`,
`reg_samples`, `reg_rejected` and `reg_fallback`. `--reg-opts '<json>'`
overrides `register_and_fuse` keyword arguments, e.g.
`{"deinterlace": "fields", "fuse": "median"}`.
`scripts/build_registered_bench.py` writes every fused image to
`registered/<id>.jpg` with `registered/diagnostics.jsonl`, and with
`--sheet DIR` it also renders contact sheets. `registered/` is gitignored.

```bash
.venv/bin/python scripts/bench_bumper_readers.py --arms jev --frames registered --subset tracklet --labels tracklet --score
.venv/bin/python scripts/build_registered_bench.py --sheet /tmp/reg_sheets
```

**Arms.** `freeread` calls `web.app._vllm_read_bumpers` itself: the real prompt,
the `_BUMPER_READ_SCHEMA` guided JSON, roster matching and blended confidence.
`vllm-choice` scores lettered options with the same model: `max_tokens=1`,
`top_logprobs=20`, letter probabilities renormalised. `clef-flash` and `clef`
call the Cloudflare decision models with three questions per request: `legible`
(noul), `surface` (choice) and `team` (choice).

**Option sets.**

- `alliance`: the hinted alliance's 3 teams, or all 6 when the alliance is
  unknown, plus two escapes, `none` (no legible number) and `other` (a number not
  listed).
- `decoy`: all 6 teams plus the two escapes, with no colour hint. The opposite
  alliance's teams are the decoys.
- `missing-true`: `alliance` with the true team removed. The true team is
  `label_team`, or the baseline's matched team if the crop isn't labelled.
  `freeread` gets the same reduced roster. The headline metric is the `assert`
  rate, i.e. how often the arm still picks a listed team. Lower is better.

Options are shuffled per crop, seeded by the crop id. `--permutations K` (3 for
`vllm-choice`, 1 for clef) averages K cyclic rotations, and `slotA` reports
first-slot bias.

`--blank-calibrate` (`--blank-calibrate-clef` for the clef arms) divides by the
distribution for a mid-gray crop and is scored as a separate `+cal` row. A
blank image *should* pick `none`, so calibration pushes mass away from the
escapes. Read `+cal` as a diagnostic, not a default.

**Cost.** This is the estimate for one permutation over the 200 crops, input
tokens only:

- clef-flash: about $0.005.
- clef: about $0.03.

Image tokens are `round(w/32)*round(h/32)+3`, capped at 1024. Crops are
upscaled so the short side is at least 256 px.

**Metrics.**

- On `legible` crops: correct / wrong / abstain.
- On `illegible` and `not_bumper` crops: the FP rate, i.e. any team asserted.
- For probabilistic arms:
  - ECE (10 bins) on the top choice.
  - Precision and coverage at confidence 0.5, 0.7 and 0.9.
  - The mean probability mass on decoys.
- Label-free proxies, usable before any labelling:
  - `agreeB`: agreement with the baseline team on `matched` crops.
  - `filtA`: how often a team is asserted on `filtered` crops.

**Caveat.** With `--hint-source baseline`, the hint for a `matched` crop comes
from the baseline's own read. Use `--hint-source label` once labels exist.
