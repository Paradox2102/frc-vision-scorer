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

## Appearance-term results (Jul 20 2026, fix #3) — works, but the blue diagnosis was WRONG

`selective_app_weight` (DINOv3 term in the TBD cost, spec
`docs/superpowers/specs/2026-07-20-selective-appearance-term-design.md`,
merged on this branch) swept 0.3/0.6/1.0 on qm11 auto:

| arm | agg | blue | red | main windows | killed | sam_frames |
|-----|-----|------|-----|--------------|--------|-----------|
| baseline | 0.816 | 0.594 | 1.037 | — | — | 100% |
| selective (w=0) | 0.883 | 1.033 | 0.733 | 17 | 7 | 72% |
| sel-app03 | 0.837 | 1.034 | **0.639** | **10** | **2** | **56%** |
| sel-app06 | 0.837 | 1.034 | 0.639 | (byte-identical to 0.3) | | |
| sel-app10 | 0.834 | 1.034 | 0.635 | (~same) | | |

The term works for what embeddings can separate: red 0.733→0.639 (38% better
than baseline), windows −41%, kills 7→2, SAM cost 72%→56%, and the effect
saturates at weight 0.3. **Blue did not move at all** — meanNN 1.033→1.034 and
coverage byte-identical (4786=26, 3181=52, 2992=73) across all four selective
variants.

**This falsifies the original causal story.** Kills collapsed 7→2 with zero
blue movement, so the killed window pairs were evidently RED convergences
(red is what improved as they resolved), not blue. Blue coverage is
deterministically pinned across wildly different window/kill/appearance
structures, which means the blue gap lives OUTSIDE the window mechanism
entirely: either the pure-TBD stream for the blue robots is qualitatively
worse than uniform SAM tracking regardless of association correctness, or a
downstream layer (identity binding / fusion) treats mask-less TBD boxes
differently. Blue-gap v2 needs a fresh root-cause pass: diff baseline-vs-
selective per-frame blue behavior (where exactly does 3181 lose its 48%?)
before proposing any further fix.

**Full-match validation (same day) — the framing flips.** `--full-match
--restart-between`, 3 arms, auto-window scored:

| arm (full-match processing) | agg | blue | red |
|-----------------------------|-----|------|-----|
| baseline | 0.983 | 1.062 | 0.903 |
| selective | 0.883 | 1.034 | 0.733 |
| sel-app03 | **0.837** | **1.034** | **0.639** |

On like-for-like full-match processing — the production case — selective+
appearance **beats baseline by 15% agg, red by 29%, and even blue slightly**.
The "selective is a net regression" story was an artifact of comparing
against the auto-only baseline (0.816), a mode full production wouldn't run.
Two more full-match facts: (1) selective's auto-window output is byte-identical
between auto-only and full-match runs (per-frame TBD is causal), while uniform
baseline degrades 0.816→0.983 from whole-match state pollution; (2) the SAM
cost win shrinks at full scale (56%→88% of uniform; teleop density opens
230–272 windows, killed 71–73, stale 95–121) — window mechanics get turbulent
in teleop and are the next cost lever if it matters. Blue-gap v2 remains real
against full-match baseline coverage too (3181: 52 vs 100).

Incidental find (fixed, commit 4b5e040e): the analysis-loop relink
`embed_batch` raced the local reader's in-process VLM load (transformers
temporarily flips torch's global default dtype during `from_pretrained`) →
"Half input / float bias" job kill. Input dtype now pinned to the model's
parameter dtype. Separately, the vLLM engine failing to come up mid-sweep
traced to the unmerged VRAM ratchet (branch `vram-leak-fix`) — sweeps now run
`--restart-between`.

## Blue-gap v2 root cause (Jul 21 2026) — detector blind spots, not association

Per-second sample diff of the three full-match replays (baseline f4ee7cd5,
selective 3d80fc70, sel-app03 dbdb0aea) shows the blue loss is two **hard,
permanent cliffs**, identical with/without the appearance term:

- **3181 (B2): samples stop at t=37s** and never return (52% coverage).
- **2992 (B3): samples stop at t=45s** and never return (73% coverage).

The detection archive pins the mechanism: main-view blue detections drop
2/frame → 1/frame at exactly 37s (and 0/frame during 44–45s); corner views
see essentially no blue robots the whole time. The HUD frames
(`docs/assets/hud_{baseline,selective}_t40.png`) show why: **3181 parks
behind the blue hub with only a bumper sliver visible.** YOLO never fires on
it again. Baseline's uniform SAM3 keeps a mask latched to the sliver's
pixels — detector-free persistence — while selective's TBD tracker starves:
a coasting TBD track emits no observations at all (`TbdTracker.step`,
unmatched → `missed += 1` only) and dies permanently at `max_missed`.
2992's cliff is the 2-second full occlusion at 44–45s: reacquisition
requires expanded-IoU overlap with a constant-velocity coast; a failed
re-match spawns a fresh identity-less oid.

**Why every prior fix was a no-op for blue:** windows trigger on *ambiguity*
(margin conflict, gap reappearance, witness). **Disappearance is not a
trigger class.** A robot that vanishes from the detector never creates
ambiguity — its stream just ends. The appearance term can't engage either:
`eiou <= 0` returns FORBIDDEN before appearance is consulted. The selective
architecture inherited the paper's assumption that the detector is reliable
and SAM exists to disambiguate; on this footage SAM's real value is
*persistence through detector blind spots* — an orthogonal capability the
window system doesn't know to buy.

Candidate fixes (fix #1's design approved and built; #2/#3 not yet approved):
1. **Starvation-triggered windows** — third trigger: track unmatched >= N
   frames, last box not near frame border → SAM window seeded from the last
   confident box; existing exit machine + budget cap apply. Mirrors exactly
   what baseline does right. **Implemented** (`selective_starve_windows`,
   default OFF; `selective_starve_trigger`, `selective_starve_max_frames`) —
   spec `docs/superpowers/specs/2026-07-21-selective-starvation-windows-design.md`,
   `--profile starve` in `scripts/ab_qm8.py`. A/B'd Jul 21 — see
   "Starvation-window A/B results" below.
2. **Zero-velocity coast-emit + appearance reacquisition** — emit coasted
   boxes as `tbd_coast` evidence (accurate for parked robots) + embed-vs-EMA
   oid merge on respawn. No GPU cost; dishonest for moving robots.
3. **Detector-side** (hub-sliver hard negatives / occlusion-aware training)
   — the bigger program (see equipment-hijack memory), out of selective scope.

## Starvation-window A/B results (Jul 21 2026, fix #1)

Three-arm `--profile starve --full-match --restart-between` on qm11
(/tmp/frc-ab-starve-qm11.json; jobs c40f277f / b89bc7b1 / ddf07026). Both
controls reproduced exactly (baseline 0.983/1.062/0.903 with blue cov
100/100/26; sel-app03 0.837/1.034/0.639 with the 52/73/26 cliffs). No
`FALLBACK`, no propagation errors.

Aggregates hide the result — read it per team (per-slot `mean_nn_m`):

| team (blue) | baseline | sel-app03 | sel-app03-starve |
|-------------|----------|-----------|------------------|
| 3181 (hub sliver) | 0.614 @ 100% | 0.575 @ 52% | **0.455 @ 99%** |
| 2992 (occlusion)  | 0.259 @ 100% | 0.214 @ 73% | 0.225 @ 73% |
| 4786 (never-seeded) | ~2.31 @ 26% | ~2.31 @ 26% | **3.11 @ 28%, lag 9.7s** |
| red meanNN | 0.903 | 0.639 | 0.642 (cov identical) |

**The thesis is proven on its target.** 3181's starve window carried the
hub-sliver blind spot detector-free: coverage 52→99 with positions *better*
than baseline's uniform SAM (0.455 vs 0.614) — selective's clean seed +
sliver mask beats baseline's whole-match-polluted track. Red held.

**Three problems, all mechanical:**

1. **Uncapped starve windows blew the corner budgets.** Corner junk tracks
   (transient detections that vanish forever — starve trig 15 and 22 on the
   corners) open windows that run to sequence end: corner sam_frames went
   26%→100% and 42%→100% of uniform, wall clock +51% (1232s→1864s), and
   main dropped 34 windows on budget (control dropped 0). Main itself stayed
   at 92%.
2. **4786 regressed (2.31→3.11, start-lag 9.7s).** Likely budget
   cannibalization (the dropped-34 windows) and/or a junk starve window
   swapping onto 4786's stream (main renames 7→17). Not yet attributed
   per-frame.
3. **2992 untouched (73%).** Its cliff is a ~2s FULL occlusion; a starve
   window's mask goes dark ≥ `min_streak` (5) frames → closes `degraded`
   before the robot reappears, so the swap re-binding never happens. Fixing
   this needs a starve-specific dark tolerance (design question, not a knob).

**Probe in flight:** `--profile starve-cap` — `selective_starve_max_frames`
240 (~28s at full-match fps), single arm. Should keep 3181's recovery (the
scored window ends well within 240 frames of the park) while stopping corner
junk windows from eating the budget; watch whether 4786 and the corner
percentages recover toward control.

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
