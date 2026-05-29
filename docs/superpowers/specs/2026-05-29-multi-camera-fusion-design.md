# Multi-Camera Fusion (Corner Cameras) — Design

**Date:** 2026-05-29
**Status:** Draft (pending spec review)
**Component:** new `web/field/` fusion + per-camera ingest, corner-camera homography calibration, calibrate UI extension, `web/app.py` inference loop
**Part of:** "Field-coordinate rework" — Sub-project 3 of 5 (see [SP1 roadmap](2026-05-28-field-coordinate-foundation-design.md))

## Problem

The main broadcast camera is an elevated wide overhead view. The hub structure in
the center of the field **occludes parts of the field** from it, and its distance
makes **bumper team numbers hard to read**. The footage already includes two
additional **corner cameras** (composited into the bottom half of the video) that
look across the field at a low, oblique angle. They:

1. **See field regions the hub occludes from the main view** — occlusion fill.
2. **Read bumper numbers more clearly** — they are closer and lower.

The requirement: corner cameras are **additive information, not a failover.** They
must contribute every frame — especially their clearer identity reads — even when
the main camera sees a robot perfectly well. The existing pipeline is single-camera
(`self.camera = "main"`, one crop, one calibration, one detection stream).

## Key findings from the footage

Reconnaissance on `data/test_footage/qual1_first_champs_2026.mp4` (1920×1080, 60fps)
drove the central design decisions:

- **Composite layout:** main camera occupies the top region; the bottom region is
  an **equal left/right split** — corner-left and corner-right. Graphics overlays
  cover field area and must be masked: "Hopper Division" (left, bottom-right
  corner), "Presented By GE Aerospace" (right, bottom-left), "molex" watermark
  (right, bottom-right). The seam between main and corner regions sits slightly
  below the main-camera crop and must be measured precisely, not assumed.
- **The corner lenses are *moderate* wide-angle, not extreme fisheye** — straight
  structures stay mostly straight with visible-but-mild barrel bow.
- **Tags are unusable for resection on the corner cameras.** Accumulating
  AprilTag detections across the entire match (with 2× upscaling) yields only
  **3–4 distinct field tags per corner**, all on the central hub structure, jammed
  into a ~130 px-wide, near-collinear cluster in the center of a 960-wide crop:

  | Camera | Distinct tags | Image cluster |
  |---|---|---|
  | Left | 18, 25, 26, 27 | x≈560–690, y≈240–253 |
  | Right | 8, 9, 10 | x≈313–381, y≈234–240 |

  A wide-FOV intrinsic/distortion model is unconstrained where there are no
  points. Fitting it to a central cluster would **extrapolate wildly into the frame
  periphery — exactly the occluded-field region these cameras exist to recover** —
  and the all-central tags make it impossible to even *measure* that edge error.
  **Tag-anchored pinhole/fisheye resection (right for the main cam) is the wrong
  tool here.**

## Approach

### Calibration: ground-plane homography (decided: loose homography only)

Sidestep intrinsics entirely. The corner cameras are oblique views of the flat
field — the textbook homography case. Reuse the existing
`Calibrator.solve_ground_homography()` / `mode="homography"`:

- The operator **manually clicks known field landmarks** (from
  `data/field/2026_landmarks.json` plus field-perimeter / hub-footprint / field-line
  points) **spread across the whole view, including the edges**, mapping each image
  point to its field `(x, y)`. Distributing control points across the area of
  interest means the fit *covers* that area instead of extrapolating into it.
- **No lens-distortion model.** For these moderate-wide lenses and the tolerance
  required (~0.3–0.5 m), the pure-homography residual is expected to be acceptable.
- **Validate honestly:** hold out a subset of clicked landmarks distributed toward
  the edges; report their field-position error in meters. This is the acceptance
  gate.

Rationale: identity — the corner cameras' headline value — only needs a read to
associate to the *correct slot*, and the tracker's association gate is 2.5 m. So
±0.5 m calibration is more than enough for the identity benefit, and adequate for
occlusion-fill placement. Tighter accuracy (plumb-line radial-distortion estimation
from straight field lines) is a **documented future escalation**, triggered only if
held-out validation comes back poor — it is *out of scope* for this spec.

### Per-camera ingest

A `CameraView` describes each physical camera and is the single source of
single-camera assumptions that get lifted:

- `name` (`main` / `corner_left` / `corner_right`)
- `crop_region` (x, y, w, h) into the composite frame — populates the currently
  unused `CameraCalibration.crop_region`
- `masks` — rectangles/polygons of graphics overlays to blank before detection
- `calibration` — loaded from `field_store` by name (already keyed by camera)

Every frame, for **every** view: extract crop → blank masks → YOLO robot detection
→ project each box's ground-contact point (bottom-center) to field `(x, y)` via that
view's `calibration.image_to_field()` (main: pinhole; corners: homography) → bumper
color + bumper crop.

### Cross-camera fusion (new unit, before the tracker)

A new, isolated `fuse_detections(per_camera_detections)` unit keeps the
`FieldTracker` untouched:

- **Input:** for each camera, a list of `FieldDetection` (field pos, color, box,
  bumper crop, source camera).
- **Cluster** detections within ~0.5–0.75 m across cameras as one physical robot.
- **Merge** each cluster to a single `FieldDetection`: quality-weighted average
  position, pooled color votes, and **pooled identity evidence** (the bumper crops +
  source-camera weights, carried forward to the reader).
- **Output:** a merged `FieldDetection` list → existing `FieldTracker.update()`,
  unchanged. Still 6 slots (3 blue + 3 red).

### Identity: additive, corner-favored (decided: weighted votes, corner favored)

Reuse the existing roster-constrained reader and per-slot vote tally — corner
cameras add evidence, they don't get a separate identity path:

- Match roster (`match_blue_teams` / `match_red_teams`, from TBA) makes reading a
  **closed-set** problem ("which of these 6 numbers?"). Bumper crops from **all**
  cameras go through `_vision_read_bumpers(crops, match_teams)`.
- Each read attaches to a slot via the fused field position and casts **votes in the
  existing per-slot tally** (`team_ocr_votes`). **Corner-camera reads are weighted
  higher** (e.g. 2–3× a main-cam vote).
- **No hard override.** A confirmed identity can still flip if corner cameras
  consistently disagree — robust to one-off misreads. This is the explicit conflict
  policy.
- Corner reads are **not gated on `bumper_occluded`** — they run every frame. (The
  existing main-cam VLM gate stays as-is.) This is the mechanism that makes the
  corner cameras additive rather than failover.

### Occlusion

Corner cameras have **no 3D pose** (homography is pose-free), so they do not
participate in ray-cast occlusion (`resolve_occlusion`). They fill occluded zones
simply by contributing detections there. The main camera keeps its hub/person
occlusion logic unchanged. The current field-tracking gate that **rejects pose-free
calibrations** must be relaxed to admit homography mode *for corner cameras*, while
still requiring a posed calibration for the main camera (occlusion depends on it).

## Components & data flow

```
composite frame
   ├─ CameraView[main]        crop+mask → YOLO → image_to_field(pinhole) ─┐
   ├─ CameraView[corner_left] crop+mask → YOLO → image_to_field(homog.) ──┤
   └─ CameraView[corner_right]crop+mask → YOLO → image_to_field(homog.) ──┤
                                                                          ▼
                                              fuse_detections() ── cluster+merge
                                                                          ▼
                                              FieldTracker.update()  (6 slots, unchanged)
                                                                          ▼
   bumper crops (all cams, corner-weighted) → _vision_read_bumpers(roster)
                                            → per-slot vote tally → identities
```

New/changed units, each independently testable:

| Unit | Location | Responsibility |
|---|---|---|
| `CameraView` config | `web/field/` (new) | crop rect + masks + calibration per camera |
| corner homography calibration | existing `solve_ground_homography` + UI | image↔landmark clicks → `mode="homography"` |
| `fuse_detections()` | `web/field/` (new) | cross-camera cluster + merge → merged detections |
| identity vote weighting | `web/app.py` reader path | corner reads weighted, ungated, into existing tally |
| ingest loop | `web/app.py` `process_video()` | iterate views instead of single main crop |
| tracking gate | `web/app.py` `_init_field_tracking()` | allow homography for corner cams only |
| calibrate UI | `web/static/calibrate.html` + endpoints | camera picker, per-camera crop, held-out error |

## Error handling

- **Uncalibrated corner camera:** that view is skipped (detections from it dropped);
  main-camera tracking proceeds. Logged, surfaced in the calibrate UI.
- **Failed/garbage homography (high held-out error):** calibration is rejected at
  solve time with the measured error; not saved as usable.
- **No tags / no landmarks clickable in a corner view:** that camera simply cannot
  be calibrated; system degrades to main + the other corner.
- **Conflicting reads:** resolved by the weighted vote tally (no exception path).

## Testing

- **Unit:** `fuse_detections()` clustering/merge (synthetic multi-cam detections);
  `CameraView` crop+mask extraction; homography held-out error computation.
- **Calibration acceptance:** held-out-landmark field error per corner camera, in
  meters, under threshold (target ≤0.5 m; gate value set during implementation).
- **End-to-end smoke (mirrors the SP2 smoke test):** on
  `qual1_first_champs_2026.mp4`, confirm (a) corner detections populate
  hub-occluded field regions, and (b) corner-camera reads lock/confirm identities
  the main camera reads poorly — the additive-identity requirement, demonstrated.

## Out of scope

- Plumb-line / radial-distortion refinement of corner calibration (future
  escalation if held-out error is poor).
- Corner cameras participating in occlusion ray-casting (requires 3D pose they
  don't have).
- Auto-detection of crop seams / overlay masks (measured once, configured).
- Triangulating robot position from multiple cameras for sub-decimeter accuracy.
