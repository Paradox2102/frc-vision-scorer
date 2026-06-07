# Whole-Robot Gap-Fill for Bumper-Obscured Robots

**Date:** 2026-06-07
**Status:** Design — approved, pending spec review
**Branch context:** `feat/bumper-model`

## Problem

The bumper detection model (`runs/train/frc_bumpers*/weights/best.pt`) only fires
when a robot's red/blue bumper is visible. When a bumper is occluded — by another
robot, or by a pile of fuel balls on the floor — no detection is produced, so no
SAM2 object is seeded and the robot goes untracked for the duration of the
occlusion.

This is a regression from the previous whole-robot detector (the hub_zone model,
`runs/train/hub_zone_yolo26s/weights/best.pt`, class `1 = robot`), which detected
the chassis regardless of bumper visibility. That model is **still loaded** in
`InferenceJob` (`self.model`) but is currently used only as an either/or fallback
when the bumper model is absent — never alongside it.

## Goal

Recover tracking of bumper-obscured robots by running the whole-robot model as a
**gap-fill** pass at SAM2 seeding points, without reintroducing uncontrolled false
positives. The bumper model remains the high-trust source of alliance color (and,
downstream, team number); the whole-robot model only speaks for robots the bumper
model cannot see.

## Non-Goals

- Retraining either model.
- Changing the per-frame, non-SAM2 detection path (`_detect_view`,
  `app.py` ~984–1020). The ask is specifically about SAM2 seeding.
- Resolving identity (team number / alliance) of gap-filled robots at seed time —
  they enter anonymous and are resolved later by the existing OCR / appearance-gallery
  systems (see "Identity resolution" below).

## Key Enablers Already Present

The codebase already has every primitive this needs; the change is composition,
not new ML or new tracker plumbing:

1. **Whole-robot model already loaded.** `self.model` (hub_zone, class `1=robot`)
   is instantiated in `InferenceJob.__init__` regardless of the bumper model.
2. **`detect_fn` already carries per-detection metadata.** The SAM2 re-prompt
   contract (`sam2_tracker.py:366–407`) accepts either a plain box or a
   `(box, color)` tuple and stores the color in `object_metadata[obj_id]`.
3. **Colorless detections are coerced and routed.**
   `detect_bridge.py:97` maps any falsy/missing color to the literal `"unknown"`.
   `FieldTracker.update` (`tracker.py:113–123`) routes an `"unknown"` detection to
   the alliance whose field half it occupies (`x < field_length/2 → blue`, else
   `red`), and `_assign` (`tracker.py:226–227`) lets it match/seed with a small
   `color_penalty`.
4. **SAM2 re-prompt already suppresses re-seeding.** `match_new_detections`
   (`sam2_tracker.py:394`) only seeds boxes that don't overlap a currently-tracked
   mask (IoS gate).
5. **Static-zone and per-alliance caps already exist.** `FieldTracker` seeds new
   slots only outside static-structure zones (`tracker.py:184,263`) and provides
   exactly 3 slots per alliance (`tracker.py:92–96`).

## Design

### Component 1 — Shared merge helper (new, pure function)

`web/field/detect_schema.py`:

```
def merge_robot_detections(bumper_robots, robot_boxes, frame_shape, ios_thresh):
    """Combine bumper-derived robot boxes (with color) and whole-robot boxes
    (no color), suppressing any whole-robot box that overlaps a bumper box.

    bumper_robots: list[(box_xyxy, color)]   # color in {"red","blue"}
    robot_boxes:   list[box_xyxy]            # whole-robot, colorless
    returns:       list[(box_xyxy, color)]   # bumper entries unchanged,
                                             # survivors tagged color="unknown"
    """
```

Rule: a whole-robot box is dropped when its Intersection-over-Smaller with **any**
bumper box exceeds `ios_thresh` (already covered → redundant). Survivors are
tagged `color="unknown"`. Pure and unit-testable in isolation.

### Component 2 — Whole-robot detect wrapper

A small helper on `InferenceJob` that runs `self.model` on a frame, keeps
`class_id == 1`, applies `ROBOT_FILL_CONF`, and applies an **image-fraction size
guard** of its own. Returns plain boxes (crop space; Hook A multiplies by
`SAM2_SCALE` at append time, matching the existing bumper call at `app.py:3173`).

**Important — size filtering is not free here.** The existing `_too_large`
footprint/image-fraction filter lives inside `to_field_detections`
(`detect_bridge.py:55-99`), which runs only on the per-frame non-SAM2 path
(`app.py:1017`) and the early-overhead path. The two hooks this design touches
feed boxes **directly to SAM2 as prompts** with no `to_field_detections` call, and
the downstream SAM2-mask→field conversion (`app.py:3478-3499`) is hand-inlined
(`_on_field` + bottom-center projection + `sam2_obj_color or "unknown"`) and also
never calls `_too_large`. So no size filter currently runs on the seeding path.
The wrapper therefore applies the calibration-independent image-fraction backstop
itself (the `max_box_frac` branch of `_too_large`, `detect_bridge.py:65-69`:
reject if `(x2-x1)/fw` or `(y2-y1)/fh` exceeds the fraction) — extract that branch
into a shared helper so the wrapper and `_too_large` stay consistent.

### Component 3 — Hook A: SAM2 prompt-frame seeding (`app.py` ~3159–3182)

After collecting bumper-derived prompt boxes for the prompt frame:

1. Run the whole-robot wrapper on the same crop.
2. `merge_robot_detections(...)` to suppress boxes overlapping bumper boxes.
3. **Territory gap-fill gate:** for each surviving `"unknown"` box, project its
   bottom-center to field coords (`view.calibration`, available inside the
   `for view in self.views` loop), determine its half
   (`x < field_length/2 → blue`, else `red`, matching `FieldTracker._territory`).
   Count the **bumper-color boxes in this same view's prompt frame** whose
   projected bottom-center falls in that half. Append the unknown box to
   `view_prompt_boxes` / `view_prompt_colors` **only if** that half has fewer than
   3 such bumper boxes. Note: there is no `FieldTracker` state at seed time
   (tracking hasn't run) and no cross-view fusion yet — "confirmed" here means
   strictly "bumper detections present in this prompt frame," not tracker-level
   confirmation. This implements "gap-fill toward 6" at the prompt frame and
   avoids handing SAM2 surplus objects to track.
4. Unknown boxes are appended with color `"unknown"` so `sam2_obj_color` carries
   it forward.

### Component 4 — Hook B: periodic re-prompt (`app.py` `make_detect_fn` ~3234–3265)

The detect_fn returns bumper-expanded boxes (with their color) **plus**
non-overlapping whole-robot boxes tagged `(box, "unknown")`:

1. Run bumper model → expanded `(box, color)` list (unchanged).
2. Run whole-robot wrapper → boxes.
3. `merge_robot_detections(...)` to drop boxes overlapping bumper boxes.
4. Return the combined list; unknown survivors as `(box, "unknown")` tuples.

Per-alliance/territory capping is **not** re-derived inside the detect_fn here.
The re-prompt path delegates capping to the two guards that already run at this
point: SAM2's `match_new_detections` (won't re-seed an already-tracked robot) and
`FieldTracker`'s hard 3-slots-per-alliance cap downstream. This is a deliberate
scope decision to keep the re-prompt path simple; deriving live per-territory
counts from existing masks inside the detect_fn would add projection logic for
marginal benefit.

### Component 5 — False-positive gating (layered, mostly pre-existing)

The "gap-fill toward 6" posture emerges from stacked guards rather than a single
check, so the only *new* FP exposure is whatever survives all layers:

| Layer | Where | Effect |
|-------|-------|--------|
| Overlap suppression | merge helper (new) | drops whole-robot boxes already covered by a bumper |
| Territory gap-fill cap | Hook A (new) | seed-frame unknown boxes only when that half has < 3 bumper boxes in this prompt frame |
| Confidence threshold | wrapper (new) | `ROBOT_FILL_CONF` filters low-confidence chassis boxes |
| Image-fraction size guard | wrapper (new) | reject implausibly large boxes; `_too_large` does **not** run on the SAM2 seed path (see Component 2) |
| Static-structure zones | `tracker.py` (existing) | never seed a track inside a field-element zone |
| SAM2 mask overlap | `sam2_tracker.py` (existing) | re-prompt won't re-seed a tracked robot |
| Per-alliance slot cap | `tracker.py` (existing) | an unknown box binds only if its half has a genuinely free slot |

## Data Flow

```
bumper model ─┐
              ├─ merge_robot_detections ─► boxes (color: red/blue/unknown)
whole-robot ──┘            │
   model                   ▼
        SAM2 seed (Hook A) / SAM2 re-prompt (Hook B)
                           │  object_metadata[obj_id] = color
                           ▼
        SAM2 masks ─► boxes ─► inlined projection (app.py:3478-3499:
                           │     _on_field + bottom-center;
                           │     color = sam2_obj_color[obj_id] or "unknown")
                           ▼
        FieldTracker.update  ── routes "unknown" by field half + color_penalty
                           ▼
        full track in a slot  (anonymous but tracked)
                           ▼
        OCR / appearance gallery later assigns team number + confirms alliance
```

### Identity resolution

A gap-filled robot is a full-fledged track that simply starts anonymous
(`color="unknown"`, no team number). The existing OCR re-read and DINOv2
appearance-gallery systems remain free to assign it a team number and confirm its
alliance once a bumper becomes visible again — no extra wiring. This is a benefit
of reusing the `"unknown"` pathway rather than inventing a parallel one.

## Configuration (env vars, matching existing style)

| Var | Default | Meaning |
|-----|---------|---------|
| `ROBOT_FILL_ENABLED` | `1` (on) | master switch for the gap-fill pass |
| `ROBOT_FILL_CONF` | `DET_CONF_THRESHOLD` | confidence threshold for the whole-robot model |
| `ROBOT_FILL_IOS` | `0.5` | overlap threshold above which a whole-robot box is suppressed by a bumper box |

When `ROBOT_FILL_ENABLED=0`, both hooks behave exactly as today (bumper-only).

Note: the bumper-model SAM2 calls use a hardcoded `conf=0.4` (`app.py:3161,3243`),
*not* `DET_CONF_THRESHOLD` (0.35). `ROBOT_FILL_CONF` is an independent threshold
for the whole-robot model; defaulting it to `DET_CONF_THRESHOLD` is a starting
point, not a shared constant with the bumper path.

## Testing

**Unit — `merge_robot_detections`:**
- No bumpers → all whole-robot boxes survive, all tagged `"unknown"`.
- Whole-robot box fully overlapping a bumper box → suppressed.
- Partial overlap below `ios_thresh` → survives.
- Bumper entries pass through unchanged (color preserved).

**Unit — territory gap-fill cap (Hook A logic, extracted to a testable fn):**
- 3 confirmed blue + 1 unknown box in blue half → unknown dropped.
- 2 confirmed red + 1 unknown box in red half → unknown admitted.
- Unknown box projected near the half-line uses the same boundary as
  `FieldTracker._territory`.

**Integration:**
- Synthetic prompt frame with one bumper-visible robot and one bumper-occluded
  (chassis-only) robot → two SAM2 objects seeded, the second with color
  `"unknown"`.
- `ROBOT_FILL_ENABLED=0` reproduces current bumper-only seeding exactly.

## Files Touched

- `web/field/detect_schema.py` — add `merge_robot_detections` (+ IoS helper if not already present).
- `web/app.py` — add whole-robot wrapper; wire Hook A (prompt seeding) and Hook B (re-prompt detect_fn); add config constants.
- Tests under the project's existing test location (run via `.venv/bin/python -m pytest`).

## Risks & Mitigations

- **Whole-robot model FPs (documented pain point).** Mitigated by the layered
  gating table; the whole-robot model's FP profile differs from the bumper model's,
  so this is new exposure, bounded by overlap suppression + zones + size +
  per-alliance cap. `ROBOT_FILL_CONF` and `ROBOT_FILL_IOS` give runtime control,
  and `ROBOT_FILL_ENABLED=0` is a clean kill switch.
- **Extra SAM2 objects = more tracking compute.** Hook A's territory cap prevents
  seeding beyond the 6-robot roster; Hook B is bounded by `MAX_TRACKED_ROBOTS` /
  `max_objects` already passed to the tracker.
- **An unknown box straddling the half-line could route to the wrong alliance.**
  Acceptable: it still produces a track, and OCR/gallery corrects alliance on
  identity resolution; matches existing `"unknown"` behavior, not a new failure mode.
