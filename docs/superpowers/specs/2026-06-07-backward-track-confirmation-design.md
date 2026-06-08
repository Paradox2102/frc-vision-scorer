# Backward-in-Time Track Confirmation

**Date:** 2026-06-07
**Status:** Design — pending spec review
**Branch context:** `feat/bumper-model`

## Problem

The tracker decides "is this a robot?" **forward and online**, at the moment a
detection seeds a track. That decision is unanswerable for two real cases:

1. **A robot parked through autonomous.** It has no motion early and may carry no
   bumper read yet. Any forward motion-gate would wrongly reject it.
2. **Refs / field structures that look robot-ish.** Refs stand at the field edge,
   move, and occlude robots (observed: a ref tracked as `R1 S0`). Static structures
   (the field radio, scoring structure) get segmented as robots (observed: a SAM2
   mask leaking across the red scoring structure into a too-wide box). The
   mask-shape (`plausible_robot_mask`) and forward-projected width
   (`_expected_robot_pixel_width`) filters added on this branch catch some, but not
   structures that are robot-shaped and robot-sized.

Motion is the strongest positive signal for "robot", but it cannot be a forward
gate (case 1) and cannot exclude refs (refs move — case 2).

## Core Principle

The pipeline is **offline**: the whole match is available. So invert the question.

> Online/forward: "Is this a robot *right now*?" — unanswerable for a parked robot.
> Offline/whole-match: "Did this track *ever* prove it was a robot?" — moved at any
> point, got a bumper read, or matched appearance. Once true, that verdict
> propagates **backward** to every frame the track existed, including stationary ones.

Refs are handled **orthogonally** by person-detector overlap (they move, so motion
can't exclude them; but they overlap `person` detections, which robots do not).

### Classification rules (confirmed)

For each track over its whole-match lifetime:

- **robot** — net field displacement exceeds `MOTION_CONFIRM_M` at any point in the
  match, **OR** ever confirmed by a bumper-number read / appearance-gallery match.
- **human** — overlaps `person` detections in ≥ `PERSON_OVERLAP_FRAC` of its frames
  **AND** never robot-confirmed.
- **static** — never moved (≤ `MOTION_CONFIRM_M` for the whole match) **AND** no
  robot evidence (no read, no appearance, not even bumper color). **Pruned.**
  Rationale (user): "if it never moves, there isn't really anything we need to
  track about it anyway." A robot parked only during auto still moves in teleop, so
  it fails *whole-match* immobility and is kept, with its auto span back-filled.

Bumper **color** alone (no number) is treated as weak: it keeps a track only if the
track also moved; a never-moving color-only track is still pruned (it is almost
always red/blue field structure — see [[hsv-alliance-color-rejected]],
[[bumper-model-false-positives]]).

## Target Architecture

```
Phase 2 (exists):  SAM2 tracks the whole clip → sam2_raw[view][frame][obj_id]
        │
Pass 1 (forward):  analysis loop as today, but ACCUMULATE per-track whole-match evidence:
        │            ├─ motion trajectory (field bottom-center per frame)
        │            ├─ confirmation log (read / color / appearance) + first-confident frame
        │            └─ person-overlap count
        ▼
Classify:          each track → robot | human | static   (rules above)
        ▼
Pass 2 (backward): from each robot's first-confident frame, propagate identity BACKWARD
        │            over earlier/stationary spans; re-resolve the fixed 6-slot roster
        │            across the whole timeline; prune human/static tracks;
        │            (final slice) re-run scoring attribution on the corrected timeline.
        ▼
Output:            corrected _replay.frames + summary (+ corrected attribution).
```

The whole match is already retained in memory at end-of-loop (`self._replay.frames`,
`sam2_raw`), and SAM2 already runs over the full clip before analysis — so this fits
the existing shape. There is **no** post-pass today; each frame is finalized inline
(`app.py:4715`), and per-track velocity (`TrackOut.vel`) is computed and discarded.

## Decomposition (4 slices — each its own spec → plan → build)

1. **Whole-match evidence layer** — *this spec details slice 1.* Pure instrumentation,
   no behavior change. Everything else consumes it.
2. **Track classifier + retroactive output cleanup** — robot/human/static verdict
   applied to prune `_replay.frames` and set `confirmed`. First visible payoff:
   refs/structures stop appearing. (≈ the lightweight "Approach A".)
3. **Backward identity propagation** — re-resolve slot/identity assignment over the
   full timeline from first-confident frames; back-fill stationary spans so a
   parked-in-auto robot is present and correctly identified from frame 0.
4. **Attribution re-run on corrected timeline** — make scoring attribution replayable
   and re-run it against back-filled identities (the fuel-scoring payoff, flagged by
   the user as "later down the line"; ties into the existing ambiguity-deferral at
   `app.py:3340/3374/3410`).

Slice 2 already fixes the current false-positive pain; each later slice is
independently testable.

---

## Slice 1 — Whole-Match Evidence Layer (detailed)

### Goal

Collect, in one place, the per-track whole-match evidence the classifier (slice 2)
and the backward pass (slice 3) need — without changing any current behavior or
output. Shippable as a no-op-to-output instrumentation change, verifiable by unit
tests on the evidence structure.

### New unit: `TrackEvidence` and `MatchEvidence`

A new module `web/field/track_evidence.py`, pure (no torch / no app deps), testable
in isolation:

```python
@dataclass
class TrackEvidence:
    track_id: int                       # fused track id (stable across the match)
    frames_seen: list[int]              # processed-frame indices where it appeared
    field_positions: list[tuple]        # (frame, x, y) bottom-center in field meters
    person_overlap_frames: int          # frames whose box overlapped a person box
    ever_team_read: bool                # a bumper-number read landed on this track
    ever_appearance_match: bool         # DINOv2 gallery match landed on this track
    ever_bumper_color: bool             # red/blue bumper detection on this track
    first_confident_frame: int | None   # earliest frame with ANY robot confirmation

    def net_displacement_m(self) -> float: ...   # robust (percentile-based) span
    def person_overlap_frac(self) -> float: ...
```

`net_displacement_m` is **robust**: it does not use instantaneous velocity (noisy at
14 m range — see the leaked-box jitter of ~1.5 m). It clusters the position samples
into an early window and a late window, takes the median of each, and also the
max pairwise distance between per-second medians; "moved" = that exceeds
`MOTION_CONFIRM_M`. This rejects per-frame jitter while still catching a genuine
drive across the field.

```python
class MatchEvidence:
    def __init__(self): self.by_track: dict[int, TrackEvidence] = {}
    def observe(self, track_id, frame, field_xy, *, person_overlap, team_read,
                appearance_match, bumper_color): ...
```

### Where it hooks (Pass 1, inside the existing analysis loop)

`MatchEvidence` is a member of `InferenceJob`, updated once per processed frame from
data already computed there:

- **Field position & track_id** — from the fused observations / `track_outs`
  already produced at `app.py:4401-4403`. Each fused track has a stable `track_id`
  and a field position.
- **person_overlap** — for each track's box, test overlap against
  `person_boxes_by_cam` (already computed at `app.py:4381-4382`). *New, small:*
  compute per-track overlap there (the boxes are already in hand).
- **team_read / appearance_match / bumper_color** — tap the existing confirmation
  sites: OCR consensus (`app.py:948`), VLM read (`app.py:1514`), gallery match
  (`app.py:1296`), bumper-color split (`detect_schema.split`). Each already knows
  which track/slot it applied to; we additionally record it on `MatchEvidence`.

`first_confident_frame` is set the first frame any robot confirmation fires.

### Retain per-frame track→slot binding

Slice 3 needs to know which `track_id` occupied which slot at each frame (today only
the slot summary is persisted to `_replay.frames`). Slice 1 adds a compact
per-frame binding log: `self._binding_log: list[dict] = []` with
`{frame_num, bindings: {slot_id: track_id}}`, populated from
`field_tracker._id_to_slot` each frame. Kept in memory; **not** yet written to
meta.json (slice 2/3 decide what surfaces).

### Data flow

```
analysis loop frame N:
   fused, track_outs  ──►  for each track:
                              field_xy        (from track_out.pos)
                              person_overlap  (box vs person_boxes_by_cam[view])
                           MatchEvidence.observe(track_id, N, field_xy, person_overlap=…)
   confirmation sites  ──►  MatchEvidence.by_track[tid].ever_* = True ; set first_confident_frame
   binding            ──►  self._binding_log.append({frame_num, dict(_id_to_slot)})
end of loop:
   self._match_evidence is fully populated and available to slice 2+.
```

### Error handling / invariants

- `observe` tolerates missing `field_xy` (None) — records the frame-seen but no
  position (a coasting/occluded frame contributes no motion evidence).
- A track that never appears in `track_outs` produces no `TrackEvidence` (nothing to
  classify).
- No exceptions escape into the loop: evidence collection is wrapped so a bug here
  can never break live processing (consistent with `IdentityDiagnosticRecorder`).

### Config knobs (env, slice 1 introduces but does not yet act on)

- `MOTION_CONFIRM_M` (default ~0.75) — net displacement to count as "moved".
- `PERSON_OVERLAP_FRAC` (default ~0.5) — fraction of frames overlapping a person.
- `PERSON_OVERLAP_IOS` (default ~0.3) — per-frame box/person IoS to count as overlap.

### Testing (slice 1)

Unit tests on `track_evidence.py` (no app/torch):

- `net_displacement_m`: a static jittery track (±0.2 m noise, 14 m range) → below
  threshold; a track that drives 5 m → above. Robust to a single outlier frame.
- `person_overlap_frac`: counts only frames flagged overlapping.
- `first_confident_frame`: set to the earliest confirming frame, stable under later
  confirmations.
- `MatchEvidence.observe`: aggregates multiple frames/tracks correctly; tolerates
  None positions.

Integration smoke: run the existing SAM2 streaming test path and assert
`self._match_evidence` is populated with the expected track ids and that **no
existing output changed** (meta.json frames identical to pre-change for a fixture).

### Non-Goals (slice 1)

- No pruning, no classification, no output change, no attribution change. Those are
  slices 2–4.
- No new ML. Motion/overlap are geometric; reads/appearance reuse existing signals.

## Risks / Open Questions

- **Motion threshold robustness at long range.** 14 m projections are noisy;
  `net_displacement_m` must use robust statistics, validated by the jitter test.
  `MOTION_CONFIRM_M` may need per-event tuning.
- **Read-but-never-moved robot.** Current rule keeps it (ever_team_read ⇒ robot).
  A disabled robot read once but immobile is kept — acceptable (positive identity).
  Flag for confirmation.
- **Memory.** Per-frame positions for ≤ ~20 tracks × ~1500 frames is trivial.
- **Slice 3 assignment re-solve** (fixed 6-slot roster over the whole timeline) is
  the hard part; slice 1/2 are low-risk and deliver the immediate false-positive fix.
