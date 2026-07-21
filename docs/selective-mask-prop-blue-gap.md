# Selective mask propagation — the same-alliance mask-convergence gap

**Status:** open problem, discovered Jul 19 2026 after the window merge/cap fix landed (`selective_sam_budget_frac`, commit a2f3a846). Selective mask propagation (`selective_mask_prop`, default OFF) is now cost-correct on qm11 but has a quality regression isolated to one mechanism. This doc scopes that mechanism and the candidate fixes so the next session can start from the diagnosis, not the symptom.

**Update Jul 20 2026:** fixes #1 and #2 were implemented (`selective_mask_overlap_exit`, `selective_kill_coast`, both default OFF) and A/B'd — **both are dead on qm11**. See "Blue-gap A/B results" below. The kills are total mask collapses (identical outcomes at 0.90/0.95/0.97), and 5 of 7 killed windows are *born converged* (no pre-convergence SAM state exists to coast from), which reframes the problem: the same-alliance confusion predates the windows. Fix #3 (appearance term in TBD) is the remaining candidate.

## Where we are

Two qm11 A/Bs against an exactly-reproduced baseline (`agg_meanNN` 0.816, blue 0.594, red 1.037):

| run | agg_meanNN | blue | red | wall | main SAM cost |
|-----|-----------|------|-----|------|---------------|
| baseline (uniform SAM3) | 0.816 | 0.594 | 1.037 | 298s | 100% (181 frames) |
| selective, pre-merge fix | 0.935 | 0.883 | 0.986 | ~232s | **213%** (17 windows) |
| selective, post-merge fix | 0.877 | **1.035** | **0.719** | **173s** | **60%** (6 episodes) |

(Lower `agg_meanNN` = closer to Manta ground truth = better.)

The merge/cap fix did its job: main-view SAM propagation fell 213% → 60%, wall clock is 42% below baseline, and the merged multi-frame-conditioning episodes run clean on the real SAM3 backend (no `FALLBACK`, no propagation error). **Cost is solved.**

But selective is still a net quality regression vs baseline (0.877 > 0.816), and the fix *changed the shape* of that regression in a diagnostic way:

- **Red got markedly better than baseline** (1.037 → 0.719).
- **Blue got markedly worse** (0.594 → 1.035).

## The mechanism: same-alliance mask convergence → `killed` → weak TBD fallback

The tell is the main-view outcome histogram, before vs after the merge fix:

| outcome | pre-fix (per-seed groups) | post-fix (merged episodes) |
|---------|---------------------------|-----------------------------|
| killed  | 2 | **7** |
| swap    | 5 | 2 |
| clean   | 2 | 2 |
| degraded| 4 | 3 |
| stale   | 2 | 1 |

`killed` is the `ExitMachine` abort that fires when two windows' masks reach pixel-IoU ≥ `mask_overlap_exit` (0.90, an internal `SelectiveConfig` constant) for `min_streak` consecutive frames — i.e. two tracked masks have collapsed onto the same pixels. When it fires, **both** windows abort and their objects revert to the cheap image-space TBD tracker's output (no SAM mask).

Merging windows into one session is *why* kills rose 2 → 7: a single `ExitMachine` now sees every concurrent mask in the episode at once, so genuine convergences that the old per-seed-frame grouping never co-observed now register. That is correct behavior — the kills are real mask collapses, not false positives.

The problem is what convergence *is* on this match. qm11's dense main view has same-alliance robots (two blues, plus reds) maneuvering close together. SAM3 masks for two same-color robots that pass within a bumper-width tend to merge — the model has no jersey/number signal at mask resolution, only color+shape, and same-alliance robots are near-identical. So:

1. Two blue masks converge → `killed` → both blue objects fall back to TBD.
2. The TBD tracker (constant-velocity + expanded-IoU Hungarian, `web/field/selective_prop.py`) has **no appearance term** — it separates tracks by motion and box overlap alone.
3. On two slow, close, same-alliance robots that is exactly the case motion+IoU can't resolve, so TBD mis-associates → blue coverage drops (3181 100→52, 2992 99→73) and `blue_meanNN` degrades.

Red improved because reds were more separated on this match — their windows settle `clean`/`swap` and SAM's foot-point mask is a better ground-contact estimate than the baseline box-bottom, so red actually beats baseline.

**In one sentence:** selective mode is now bottlenecked not by cost but by same-alliance appearance ambiguity — the exact P1 weakness the whole pipeline has (see the DINOv3 same-alliance re-ID gap), surfaced here as mask convergence on the dense view.

## Candidate fixes, cheapest first

1. **Tune `mask_overlap_exit` (kill threshold).** Currently 0.90, an internal constant. Raising it (e.g. 0.95–0.97) makes the kill fire only on near-total mask collapse, letting partially-overlapping same-alliance masks survive rather than dumping both to TBD. Cheapest probe: promote it to a knob (`selective_mask_overlap_exit`) and A/B 0.90/0.95/0.97 on qm11. Risk: too high and genuine hijack/merge cases stop being culled. Measures whether the kills are premature vs genuinely-collapsed.
2. **Better fallback than raw TBD on a killed pair.** When a window is killed, instead of reverting to the appearance-blind TBD association, keep the last-confident SAM box per object and coast it (the uniform path's coast-and-decay). Medium effort in `merge_selective_results` / the `killed` branch. Attacks the "TBD is weak exactly here" half rather than the "kills fire" half.
3. **DINOv3 appearance term in the TBD cost (the plan's deferred v2).** Add a gallery/embedding affinity term to `TbdTracker`'s expanded-IoU Hungarian cost so same-alliance robots are separated by appearance, not just motion. This is the principled fix — it makes the TBD fallback strong enough that kills reverting to it stop hurting — but it's the most work (per-frame embedding cost in the TBD pass, plumbing the embedder into `selective_prop.py`, which is currently torch-free) and overlaps the broader same-alliance re-ID program. Note it partly re-introduces the per-frame embedding cost the selective architecture was trying to avoid — measure the net speed.

## Blue-gap A/B results (Jul 20 2026)

Six-arm `--profile blue-gap` run on qm11. Baseline reproduced exactly
(0.816/0.594/1.037, third time); every selective arm came back **byte-identical**
to the selective control (agg 0.883, blue 1.033, red 0.733; blue_cov
4786=26 3181=52 2992=73). Config receipt verified per arm in the server log;
no `FALLBACK`, no `Error during propagation` in any job log.

**Fix #1 (kill threshold) is dead.** At 0.90, 0.95, and 0.97 the main view
produces the identical outcome histogram (`killed: 7`, windows=17,
sam_frames=130 = 72%). The convergences exceed 0.97 pixel-IoU — total mask
collapse, not partial overlap. No threshold value rescues these windows.

**Fix #2 (kill-coast) never engages.** A `coast=` diagnostic added to the
`[selective]` per-view log line (probe: `--profile coast-probe`) shows zero
coasted entries across all 7 kills:

```
coast={'kills': 7, 'no_seed': 5, 'entries': 0, 'per_kill': [
  (6, None, 0, 'no_seed'), (6, None, 0, 'no_seed'), (6, None, 0, 'no_seed'),
  (1, 95, 0, 'reclaim'), (1, None, 0, 'no_seed'),
  (16, 155, 0, 'reclaim'), (16, None, 0, 'no_seed')]}
```

Two distinct defects:

1. **Born-converged windows (5/7 kills, `no_seed`).** `_capture_coast` requires
   stored mask frames at/before `close_frame - min_streak`; masks are stored
   from `entry_frame` on. `no_seed` therefore means the overlap streak started
   at (or before) window entry — the window never had a confident
   pre-convergence SAM state *at all*. The same-alliance confusion exists at
   seed time (two windows seeded onto the same robot because TBD had already
   mis-associated), so there is nothing to coast from. Only 3 distinct oids
   (1, 6, 16) account for all 7 kills — the same objects re-seed, re-converge,
   and re-die.
2. **Instant reclaim (2/7 kills, `reclaim` at step 1, 0 entries written).**
   `_coast_reclaim_margin` returns `inf` when exactly one TBD detection
   overlaps the coasted box — and the killed object's *own* TBD track is in
   `frame_obs`, sitting within a frame of the coast seed, so it always
   overlaps. `inf >= margin_exit (0.10)` ends the coast before the first entry
   is written. The test asks "does some TBD box own this position?" when it
   should ask "has the killed *pair* separated?"

**Implication.** The doc's original causal story ("kills → weak TBD fallback")
is at best half the picture: for 5 of 7 kills the TBD mis-association
*precedes* the window, so no post-kill fallback — however good — can recover
identity that was already lost at seeding. Patching kill-coast (seed from the
window's seed box, fix the reclaim test) would engage it, but it would coast
from possibly-already-wrong positions. The remaining principled fix is #3
(appearance term in the TBD cost), which attacks the mis-association itself.
Also note baseline blue_cov 4786=1 vs selective 26 — selective *helps* the
never-seeded robot; the blue regression is entirely 3181 (100→52) and
2992 (99→73).

## How to evaluate

Server launch: `web/start_all.sh` (vLLM reader, job-managed). qm11 A/B:

```
.venv/bin/python scripts/ab_qm8.py --profile selective \
    --match-key 2026cur_qm11 --match 11 \
    --video web/uploads/video_85447fcf12e7e4689ebde452b498a32d.mp4 \
    --results /tmp/frc-ab-selective-qm11.json
```

Watch, per view, in the job's `web/logs/*.log`:
- `[selective] view=main: ... outcomes={...}` — the `killed` count is the primary signal for fix #1/#2; want it down without `swap`/`clean` collapsing.
- `blue_meanNN` and blue coverage in the harness summary — the metric that must recover toward 0.594.
- `sam_frames (pct% of uniform)` — must stay ≤ 100% (don't regress the cost win).

Baseline must reproduce `agg_meanNN` 0.816 each run (it has, twice) — if it doesn't, the setup drifted and the selective numbers are untrustworthy.

**Do not treat parity with baseline as the only bar.** Selective already wins on red and on speed; the goal is recovering blue without losing those. A config that lands `agg_meanNN` at/below 0.816 *and* keeps main `sam_frames` under 100% is the flip-the-default candidate — validate it across qm8–qm12, not just qm11, before any default change.

## Pointers

- Algorithm + knobs: `docs/SAM3.md` (selective section), `web/field/selective_prop.py`, `web/field/sam2_tracker.py::track_selective`.
- Kill logic: `ExitMachine._check_killed` in `web/field/selective_prop.py` (`mask_overlap_exit`).
- The broader same-alliance re-ID problem this is a facet of: `docs/tech-upgrade-review-2026-07.md` §8.
