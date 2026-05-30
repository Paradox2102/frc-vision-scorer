# Tracking & Identity Upgrade — Design

**Date:** 2026-05-29
**Status:** Draft (pending spec review)
**Component:** new `web/field/sam2_tracker.py`; `web/field/tracker.py` (Kalman → thin `SlotMapper`); SP3 fusion re-point in `web/field/` + `web/app.py`; SP4 identity (detector `bumper_number` class, IoS matching, consecutive-agreement lock, appearance embeddings); spike harness `scripts/sam2_spike.py`
**Part of:** "Field-coordinate rework" — unifies the remaining roadmap (replaces SP2 tracking approach, re-points SP3, defines SP4). See [SP1 roadmap](2026-05-28-field-coordinate-foundation-design.md#roadmap-where-this-fits).

## SP-T.0 Result (2026-05-29, preliminary)

First spike run on a 6-robot window of `qual1_first_champs_2026.mp4` (frames
2130–2280, stride 2, prompt 2160), SAM2.1-hiera-large on the RTX 3090.

| Signal | FieldTracker (Kalman) | SAM2 | Trust |
|---|---|---|---|
| Track births/deaths (fragmentation) | 31 / 31 | **7 / 7** | anchor-free — trustworthy |
| Tracks maintained | 6 | 6 | trustworthy |
| Mean frames/track | 57.3 | 58.7 | trustworthy |
| Latency | — | **151 ms/frame** (GPU, 0.5× crop) | trustworthy |
| Identity swaps | pending | pending | needs manual count |

**Continuity strongly favors SAM2** (~77% less track fragmentation: 7 vs 31
births/deaths over 76 frames), which is the mechanism behind fewer identity
errors. Latency (151 ms/frame) is fine for offline batch (~11 min/match), too slow
for naive 30 fps live — a known later concern. **The swap-based gate criterion is
NOT yet decided:** the run used an empty anchor set, so swaps must be counted
manually from `data/test_footage/qual1_spike_compare.mp4` (the spec's intended
method) or via human-placed anchors. **Verdict: PASS-leaning on the continuity
proxy, pending the manual swap count.** Harness/metrics/checkpoint all validated.

## Problem

Sub-projects 1–3 are merged: the main camera is calibrated (SP1), a field-space
constant-velocity Kalman tracker assigns detections to 6 fixed alliance slots
(SP2), and the two corner cameras are fused as additive field-space sensors (SP3).
Detection, calibration, and alliance-by-side are working.

The remaining failures are concentrated in **tracking continuity** and
**identity**:

- **Same-alliance slot swaps.** The Kalman tracker associates detections to slots
  by minimum-cost distance matching in field meters. When two same-colored robots
  cross or pile up in a hub scrum, distance matching has no appearance signal to
  keep them apart, so slot A's filter can be handed robot B. Same-alliance
  crossings are the worst case (no color penalty separates them).
- **Identity locks that freeze wrong.** Team-number assignment locks on cumulative
  OCR votes and can stick to a wrong value with no reliable escape (see
  `reread-audit-deadlock`).
- **Viewpoint-fragile bumper crops.** The number crop fed to the reader is derived
  from fixed box ratios (`bumper_band_box`); under perspective (corner cameras,
  angled or tipped robots) the crop misframes the number, degrading reads.

The Roboflow basketball notebook the team validated demonstrates a stronger
tracking-and-identity stack on the same pipeline shape (detect → track → identify
→ map-to-field): **SAM2 mask-propagation tracking**, **number as a detection
class** matched to players by **Intersection-over-Smaller (IoS)**, and
**consecutive-agreement number validation**. This upgrade adopts those methods.

## Scope

A single, sequenced upgrade that folds the notebook's tracking and identity
methods into the remaining roadmap. Processing is **offline/batch** (process a
match, then review); the development target is the **local RTX 3090**. Cloud and
live operation are explicitly later.

### In scope

- **SP-T.0 — SAM2 tracking spike (validation gate).** Prove SAM2 reduces
  identity swaps through occlusion versus the current Kalman tracker, at acceptable
  offline latency, before retiring working SP2 code.
- **SP-T.1 — SAM2-as-tracker integration.** A `sam2_tracker` that emits stable
  per-camera track IDs by mask propagation; demote the Kalman `FieldTracker` to a
  thin field-space `SlotMapper`.
- **SP-T.2 — Re-point SP3 fusion** to fuse per-camera SAM2 *tracklets* rather than
  raw per-frame detections.
- **SP4 — Identity rebuild** on stable SAM2 tracks: (a) `bumper_number` detection
  class + IoS matching, (b) consecutive-agreement number lock, (c) appearance
  embeddings for re-identification.

### Out of scope

- Live / real-time operation and cloud deployment (later milestone).
- **SP5 — polish / scoring / analytics** (trajectory smoothing, scoring-from-event,
  keypoint auto-homography). Unchanged and deferred.
- Team assignment by appearance clustering. Alliance stays field-side geometry +
  bumper color, as established in SP2.
- Detector architecture change (RF-DETR vs YOLO). The current YOLO detector
  (`hub_zone_yolo26s`) remains the prompt/region source; swapping it is a separate
  question.

## Roadmap revision

This upgrade revises the SP1 roadmap. The roadmap table in
`2026-05-28-field-coordinate-foundation-design.md` should be updated to reflect:

| # | Sub-project | Status after this upgrade |
|---|---|---|
| 1 | Field-coordinate foundation | Done (unchanged) |
| 2 | Field-space tracking + alliance | **Superseded.** The Kalman `FieldTracker`'s image-space association role moves to SAM2; it is demoted to a thin field-space `SlotMapper`. |
| **T** | **Tracking replacement (this upgrade)** | SAM2 mask-propagation tracker, spike-gated. New. |
| 3 | Multi-camera fusion | **Re-pointed** to fuse per-camera SAM2 tracklets. Fusion-evidence model and corner calibration unchanged. |
| 4 | Identity rebuild | Defined by this upgrade: `bumper_number` class + IoS, consecutive-agreement lock, appearance embeddings. |
| 5 | Polish / scoring / analytics | Unchanged, deferred. |

## Architecture decisions

- **SAM2 replaces the Kalman tracker** (user-selected). SAM2 mask propagation
  becomes the source of frame-to-frame identity continuity; the field layer keeps
  only the FRC-specific slot logic.
- **A validation gate precedes the switchover.** SP-T.0 must pass before SP-T.1
  retires SP2 behavior. If the gate fails, fall back to *augment* (SAM2 IDs as an
  association cue into the existing Kalman matching) without discarding SP2.
- **Offline `SAM2VideoPredictor`.** Use the official `facebookresearch/sam2`
  `SAM2VideoPredictor`, which supports adding prompts at any frame — strictly
  better for offline use than the notebook's `segment-anything-2-real-time` fork,
  which prompts only the first frame and cannot pick up robots occluded at clip
  start. The real-time fork is reserved for the eventual live path.
- **The SlotMapper keeps the FRC domain knowledge** that has no analog in the
  basketball notebook: 6 fixed slots, alliance-by-field-side, occlusion coast
  budgets, and the static-structure-zones seeding gate.

## Components

### SP-T.0 — SAM2 tracking spike (gate)

**Purpose.** Decide, on real FRC footage, whether SAM2 tracking beats the Kalman
tracker on the failure that matters (identity swaps through occlusion), at a
latency the project can afford.

**Harness** (`scripts/sam2_spike.py`):
1. Load a match clip and the saved main-camera calibration.
2. On a clean frame, detect robots with the current YOLO detector
   (`hub_zone_yolo26s`, `class_id == 1`) and prompt `SAM2VideoPredictor` with those
   boxes.
3. Every *N* frames (or when the active-track count drops below the expected
   roster), re-run the detector, IoS-match new detections against current masks,
   and add prompts for unmatched (newly visible) robots.
4. Propagate masks across the clip; convert each mask to a box
   (`sv.mask_to_xyxy`), take its bottom-center, and project to field with
   `image_to_field`.
5. Run the current `FieldTracker` over the same clip.
6. Render a side-by-side annotated video and emit metrics.

**Metrics** (on a hard segment — hub scrum with same-alliance crossings, ~15–30 s
of `data/videos/qual37.mp4`):
- **Identity swaps** (primary): manual ground-truth count per tracker.
- **Track continuity:** track births/deaths and frames-per-ID (fragmentation).
- **Occlusion survival:** for hand-tagged occlusion events, was the ID preserved.
- **Latency:** ms/frame for SAM2 on the 3090 (informs the live/cloud question).

The manual ground-truth lives in a JSON sidecar beside the clip
(`data/videos/qual37.spike_labels.json`): the segment's frame range, and a list of
swap/occlusion events as `{frame, robot_a_slot, robot_b_slot, kind}`. The metric
computations read this sidecar; their unit tests run against a tiny hand-built
fixture of the same shape.

**Gate (objective):** SAM2 reduces identity swaps on the hard segment by **≥ 50 %**
relative to the Kalman tracker **and** offline latency is acceptable for batch
review. Fail → fall back to *augment*; SP2 stays primary.

**Dependencies.** `facebookresearch/sam2` + a checkpoint (`sam2.1_hiera_large`,
fallback small/base), existing detector, SP1 calibration, `data/videos/qual37.mp4`.

**Testing.** The harness is an experiment, not production; correctness is checked
by the rendered video and the metrics table. The metric computations
(swap-counting from a labeled segment, continuity stats) get unit tests against a
small hand-built fixture.

### SP-T.1 — SAM2-as-tracker integration

**Purpose.** Make SAM2 the tracker behind a stable interface, and reduce the
Kalman tracker to slot bookkeeping.

**`web/field/sam2_tracker.py`.** Wraps `SAM2VideoPredictor`. Responsibilities:
prompt management (initial + periodic re-prompt with IoS-matched new detections),
mask propagation, and mask→box conversion. Emits, per frame, a list of
`(track_id, box, mask)` in one camera's crop-local pixels.
- *Interface:* `prompt(frame, boxes)`; `propagate(frame) -> list[Track]`;
  `reprompt(frame, boxes)`. Stateful over a clip; `reset()` clears it.
- *Depends on:* SAM2 predictor, the detector (for re-prompts), `supervision`.
- *Testable in isolation* with a stub predictor returning canned masks (verify
  mask→box, re-prompt matching, ID stability).

**`SlotMapper` (revised `web/field/tracker.py`).** Inputs are now per-object field
positions carrying a *stable upstream ID* (from SAM2), not raw detections needing
distance association. Keep: 6 fixed slots, alliance-by-field-side, occlusion coast
budget, static-structure-zones seeding gate, the `TrackOut` emission contract.
Drop: the per-frame distance min-cost permutation matching — a stable SAM2 ID
binds to a slot on first sight and stays. The `update(...) -> list[TrackOut]`
signature is preserved so downstream consumers are unaffected.

Coast-budget semantics change under stable IDs: today a slot coasts because no
detection matched it; with SAM2, a slot coasts only while its bound SAM2 ID is
absent (occluded or lost). The `view_occluded` input and `max_coast_occluded`
budget are retained for the "ID temporarily gone" case, but they no longer guard
against mis-association (SAM2 ID identity does that). SP-T.1 must define what
happens when a coasting slot's SAM2 ID never returns but a *new* ID appears in its
field region — resolved by SP4c appearance re-id, not by distance.
- *Testable in isolation*, mirroring the existing tracker tests, plus new cases:
  a stable ID keeps its slot across an occlusion coast; a never-before-seen ID in a
  static zone is not seeded.

**Offline-first.** Process the whole clip, store per-frame tracks, then review.

### SP-T.2 — Re-point SP3 multi-camera fusion

**Purpose.** Feed the SlotMapper fused, cross-camera tracklets instead of raw
detections, without rebuilding SP3's fusion model.

**Change.** Run `sam2_tracker` per sub-view (main + two corners). Project each
view's mask bottom-centers to shared field coordinates. Fuse per-camera tracklets
in field space using SP3's existing additive-evidence cross-camera association
(field-proximity). Feed the result to the SlotMapper. Reuse SP3's
`composite_layout`, per-camera calibration, and fusion-evidence code unchanged;
only the per-camera detection source changes (raw detections → SAM2 tracklets).

**Dependencies.** SP-T.1, merged SP3 fusion infrastructure, all cameras calibrated.

**Testing.** Extend existing fusion tests: same fused field position from two
views now arrives as two tracklets and resolves to one slot; a corner-only
tracklet (main-camera-occluded robot) still maps to its slot.

### SP4 — Identity rebuild

Identity rides on stable SAM2 track IDs. Three isolated pieces.

**4a — `bumper_number` detection class + IoS matching.**
Add a `bumper_number` class to the detector training set. Per frame, detect number
regions; rasterize each number box to a mask (`sv.xyxy_to_mask`) and match it to the
robot mask it sits inside via Intersection-over-Smaller
(`sv.mask_iou_batch(robot_masks, number_masks, overlap_metric=sv.OverlapMetric.IOS)`,
threshold ≈ 0.9 — the number lies fully inside the robot), exactly as the notebook
does. Both arguments to `mask_iou_batch` are mask arrays, so the box-to-mask
rasterization is required. Output: a clean, viewpoint-correct number crop keyed to a
SAM2 track ID. Replaces `bumper_band_box` geometry.
- *Dependency / long pole:* labeled `bumper_number` training data. Labeling runs in
  parallel with SP-T.
- *Testable in isolation:* IoS matching given masks + number boxes (containment,
  ambiguous overlap, no-match).

**4b — Consecutive-agreement number lock.**
Adopt the notebook's `ConsecutiveValueTracker` pattern: lock a number to a track
only after *N* consecutive consistent reads (resets on disagreement), combined with
the existing high-confidence direct-contradiction unlock
(`VLM_DIRECT_CONTRADICTION_CONF`) so a wrong lock can still be overturned. Replaces
the cumulative-vote lock and directly addresses `reread-audit-deadlock`.
- *Testable in isolation:* state-machine tests including lock-after-N,
  reset-on-disagree, and wrong-lock recovery via contradiction.

**4c — Appearance embeddings for re-identification.**
Per-track appearance embedding (SigLIP or DINOv2 crop embedding) as a *secondary*
re-identification signal: when SAM2 loses a track across a long occlusion and a new
ID appears, match it to a recently-lost track by embedding similarity before
treating it as new. Builds on the existing embedding-reid draft. The number reader
(Claude vision) remains the primary identity source; embeddings are continuity
glue, not the team or alliance decision.
- *Testable in isolation:* embedding match/no-match given stored track embeddings
  and a re-entry crop.

**Alliance** continues to come from field side + bumper color. No appearance
clustering.

## Data flow

```
composite frame
  └─ per sub-view crop (composite_layout, masks applied)
       └─ SAM2 tracker  → (track_id, box, mask) per camera          [SP-T.1]
            └─ bottom-center → image_to_field → field (x, y)
                 └─ cross-camera tracklet fusion (additive evidence) [SP-T.2]
                      └─ SlotMapper → 6 alliance slots               [SP-T.1]
                           └─ identity:
                                bumper_number det + IoS → crop       [SP4a]
                                  → reader → consecutive lock        [SP4b]
                                appearance embedding (re-id glue)    [SP4c]
                                     → team number on slot
```

## Sequencing and gates

1. **SP-T.0 spike** — gate on swaps (≥ 50 % fewer) and latency.
2. **SP-T.1** — `sam2_tracker` + `SlotMapper` (only after the gate passes).
3. **SP-T.2** — re-point SP3 fusion.
4. **SP4a → 4b → 4c** — identity, in order.

`bumper_number` labeling for SP4a proceeds in parallel with SP-T so training data
is ready when identity work starts.

## Testing strategy

- **SP-T.0:** rendered comparison video + metrics; unit tests for the metric
  computations on a labeled fixture.
- **SP-T.1:** `sam2_tracker` unit tests with a stub predictor (mask→box, re-prompt
  matching, ID stability); `SlotMapper` unit tests mirroring existing tracker tests
  plus stable-ID-keeps-slot and zone-gate cases.
- **SP-T.2:** extended fusion tests (two-view → one slot; corner-only → slot).
- **SP4:** IoS-matching tests; consecutive-lock state-machine tests including
  wrong-lock recovery; embedding re-id match tests.
- Reuse the patterns in `tests/field/`.

## Risks and mitigations

- **SAM2 latency / GPU budget.** Large + per-camera + per-frame is heavy. Mitigate
  with the spike's latency measurement; fall back to `sam2_hiera_small/base`; keep
  scope offline for now.
- **Gate fails.** Documented fallback to *augment* (SAM2 IDs as a cue into the
  existing Kalman matching) keeps SP2 working rather than leaving a gap.
- **Re-prompt strategy misses robots.** Periodic detector re-prompt with IoS
  matching is the mitigation for the frame-0-only limitation; the spike validates
  it on a real clip.
- **SP4a training data is the schedule long pole.** Parallelize labeling with SP-T.
- **SP3 rework risk.** Re-pointing recently-merged fusion code; mitigated by
  reusing the fusion-evidence model unchanged and extending its existing tests.

## Open questions

- Spike re-prompt interval *N* and the exact "expected roster" trigger — to be
  tuned empirically in SP-T.0.
- Embedding backbone for 4c (SigLIP vs DINOv2) — decide during SP4c against the
  older embedding-reid draft.
- Whether SAM2 masks are reliable for small/distant robots on the corner cameras —
  the spike should report this as a secondary observation.
