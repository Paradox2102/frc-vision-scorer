# Max-quality live swarm — plan (Nov 13 2026, nothing rented yet)

**Goal.** At a live event, run every match at the highest-quality config we can find,
even if one match takes ~25 min, and keep results within **~30 min of the video
appearing** by staggering matches across a small swarm of pods.

**Decisions (Nick, Nov 13):** rank configs **team identity first** (labels right /
wrong, misID s; position + auto-path metrics break ties) · quality sweep budget
**≤ ~$75** · live latency target **≤ ~30 min** after the video appears.

## Why a swarm, not a bigger card

Nov 13 pod ae217wpsxp1lzd (RTX PRO 6000; data in `data/runpod_pull/ae217wpsxp1lzd/`): one qm8 job uses
~3.5 CPU cores and the GPU is ~27 % busy — the pipeline is serial-Python bound, so a
bigger GPU makes one match only ~1.3x faster than the local 3090, and 2 jobs in one
server gave 1.2x. One job per server, many servers, scales ~linearly.

## Phases

| phase | what | rents? | est. cost | est. wall |
|---|---|---|---|---|
| 0 | Tooling (offline): swarm (building now), quality-sweep driver, live dispatcher | no | $0 | — |
| 1 | **Quality sweep**: ~9 configs × 6 eval matches on ~9 big-card pods in parallel | yes | ~$50-70 | ~2.5-3 h |
| 2 | **GPU pick**: winning config, 2 matches each on 3-4 GPU types | yes | ~$10 | ~1 h |
| 3 | **Live rehearsal**: replay a past event's schedule at real cadence on the chosen swarm | yes | ~$20-30 | ~3 h |

Each phase needs Nick's go-ahead before anything is rented.

## Phase 1 — quality sweep

**Eval set — will grow before phase 1:** Nick is hand-tracking more GT matches at
other events with different camera setups (Nov 13). The sweep driver therefore reads
an eval-set file (match keys across events + video paths) and scores every match that
has GT under `data/eval/ground_truth/<match_key>/`, Manta truth where cached, and the
label audit everywhere. The starting set below is only 2026cur.

**Starting eval set (6 matches, 2026cur):** qm8 (5-robot hand GT), qm11 (digit-verified
truth), qm7, qm13, qm14, qm22 (the Nov 6 held-out set). Every match also has Manta
auto-path truth (`data/manta_cache`) and gets a label audit (`audit_replay.py`, Jev;
Claude only to calibrate the auditor — evals are allowed to use Claude, the runtime
is not).

**Noise:** labels swing ±5 pts and misID ±3 s per match between identical runs, so the
baseline runs twice and a config only "wins" if it beats baseline on the 6-match
aggregate by more than the baseline's own repeat spread.

**Ranking (identity first):** primary = aggregate labels-right minus 5× labels-wrong
(a wrong team is worse than an unlabelled one), then misID s; coverage, position
error and Manta auto meanNN only break ties or veto a config that regresses them
badly.

**Candidate configs** (one swarm pod each; final list confirmed against the code
before renting). Base = R1 (5 fps, corner stride 2) — it had the best labels on Nov 13.

| id | change vs base | why it might raise quality | prior evidence |
|---|---|---|---|
| Q0a/Q0b | base, run twice | noise floor | — |
| Q1 | `sam3_track_fps` 10, `corner_track_stride` 1, `sam3_chunk_frames` 1000 | denser tracking through crossings, fewer chunk boundaries (each drops appearance memory) | R3 on qm8: not better (66 % vs 70 %) — single match, inside noise |
| Q2 | Q1 + `selective_mask_prop` OFF | SAM3 on every object every frame instead of only ambiguous windows | selective was tuned for speed; qm16 suggested it may also *help* — needs the 6-match test |
| Q3 | `read_census_interval_s` 1.0 + `window_labels_reader` jev | twice the live census reads + Jev window reads instead of the local student | student_only was quality-neutral at 4x speed; more reads untested |
| Q4 | `team_reader` vllm + Qwen3.5-4B resident | free local reader, parity with Jev on qm8 Nov 13 | 71/0 vs Jev 59/0 on the 200-crop bench; live parity |
| Q5 | `event_lineup` ON | event-level identity from all prior matches | built Nov 10, never evaluated (eval was ~$7 of Jev) |
| Q6 | `window_labels_eliminate` ON | roster elimination fills the last unlabelled slot | held-out 14/20 precise — risky, wrong-label veto applies |
| Q7 | best tracking (Q1/Q2) + best identity knobs (Q3-Q6) combined | the actual max-quality candidate | built from Q1-Q6 results (round 2 if budget allows) |

**Local smoke (Nov 13, 3090, qm8 auto-only, one run each, NOT a verdict):** every
candidate ran without error. Auto-only job wall: Q0 82 s, Q1 131 s, Q2 303 s (SAM on
every frame ≈ 3.7x base), Q3 104 s, Q4 134 s (includes vLLM start), Q5 79 s, Q6 83 s.
Identity-only arms (same tracking, so comparable with each other): labels right Q3 63 %,
Q4 67 %, Q5 67 %, Q6 72 % (unlabelled 10 %), all 0 % wrong; Q1 had 6 % wrong /
5.5 s misID. **Caveat found:** the server leaks ~0.7 GB of live CUDA tensors per job;
from job 3 on SAM3 offloaded tracker state to CPU, and jobs 4-7 tracked differently
(cov 81 %) from job 1 with the same tracking config (cov 88 %). Until that is fixed,
sweep pods must restart the server between jobs (or the leak must be fixed first —
proposed as its own session), or config comparisons are confounded by job order.

Not on the list (already measured as losses): multiplex, thread-parallel views,
`jev_verify_lowconf`, starve windows, `identity_swap_resolve`, the gap-fill detector.
Bigger-detector / higher-resolution YOLO is the largest untested quality lever (the
Nov 11 study: missing YOLO boxes are most of the untracked time) but needs training,
so it is a separate project, not a sweep knob.

**Cost:** ~9 pods × ~2.5 h × ~$2.69/h (96 GB cards, so any config fits, incl. the
resident reader) ≈ $60. Matches run per pod in schedule order; results are scored
locally with the existing scorers (`score_pulled.py`, audit).

## Phase 2 — pick the swarm GPU

Run the winning config on 2 eval matches each on: L40S ($1.09/h), RTX A6000 /
community ($0.33-0.69/h), A100 80 GB ($2.09/h), RTX PRO 6000 ($2.69/h, already
measured in phase 1). Pick the card with the lowest **$ per hour of swarm** that
keeps one match ≤ ~25 min and fits the config's peak VRAM (+ resident reader if Q4
wins). CPU matters as much as GPU: record the host fingerprint (single-thread
speed) per card.

## Live staggered dispatch (phase 0 tooling, phase 3 rehearsal)

* **Pods needed** = ceil(T_match / cadence) + 1 spare. Quals run about every 7 min:
  T = 25 min → 4 + 1 = **5 pods**. Latency ≈ T_match once a pod is free.
* **Cost estimate:** 5 × L40S ≈ $5.5/h → ~$150 per 3-day event (~9 h/day); 5 × PRO
  6000 ≈ $13.5/h → ~$360.
* **Dispatch:** a poller watches TBA for new match videos; each new match goes to the
  idle pod (least recently busy). The video is downloaded at home and uploaded (avoids
  YouTube's datacenter bot block). Replays / pulls stream back as each match finishes.
* **Calibration across pods:** in live mode a pod sees every ~5th match, not
  consecutive ones. The dispatcher syncs new calibration / layout files from the pod
  that made them to all pods, so a camera move found on one pod is not re-solved (or
  missed) on the others.
* **Rehearsal (phase 3):** replay a past event's real match timestamps at 1x into the
  live dispatcher; measure latency per match, pod utilization, failures.

## Tooling status

* Swarm batch tooling (`runpod_cli.py swarm up/plan/run/status/pull/import/terminate`)
  — being built on branch `runpod-swarm` (offline, fakes only). Its dispatcher keeps
  the assignment policy pluggable for live mode and allows a per-pod `config_json`.
* To build next: (1) **quality-sweep driver** — one config per pod over the eval set,
  local scoring + audit, ranking table with noise bands; (2) **live mode** — TBA
  video poller, idle-pod policy, calibration sync, latency report.
