# Multi-Camera Fusion (Corner Cameras) — Design

**Date:** 2026-05-29
**Status:** Draft (pending spec review)
**Component:** new `web/field/` fusion + per-camera ingest, corner-camera tag-anchored calibration (homography fallback), calibrate UI extension, `web/app.py` inference loop
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
- **Auto-detection alone under-performs, but the tags are spread.** An aggressive
  detection sweep (`quad_decimate=1.0`, 2–3× upscale, accumulated across the whole
  match) only decodes the **bright central hub cluster** — left: 18/25/26/27 at
  x≈560–690; right: 5/8/9/10 at x≈277–381 — a ~130 px near-collinear blob. But
  visual inspection confirms **at least ~7 tags per corner on structures spread
  across the frame**: a near pillar, the central hub, a far goal, and wall panels.
  Their published 3D positions span **~4.7 m of depth, nearly the full ~7 m field
  width (y≈0.6 → 7.4), and two heights** — a genuinely non-coplanar, non-collinear
  constellation. The detector simply can't decode the small/oblique off-hub tags at
  this resolution; a human can locate them.

  | Tag | x (m) | y (m) | z (m) | structure |
  |---|---|---|---|---|
  | 31, 32 | 0.01 | 3.75, 4.18 | 0.55 | far wall |
  | 25, 26 | 4.02 | 4.39, 4.03 | 1.12 | central hub |
  | 17 | 4.66 | 0.64 | 0.89 | near rail |
  | 22 | 4.66 | 7.42 | 0.89 | far rail |

  So tag-anchored **pinhole resection — the same method as the main camera — is the
  right primary tool**, provided the spread off-hub tags are supplied by manual
  click (auto-detect gets the hub cluster's corners; the operator adds the rest).

## Approach

### Calibration: tag-anchored pinhole resection (primary), homography (fallback)

Use the **same method as the main camera** —
`Calibrator.solve_camera_from_tags()` / `mode="pinhole"` (intrinsics + 2-radial
distortion + pose from AprilTag correspondences). The corner-camera difference is
purely in how correspondences are *gathered*, because the detector only decodes the
central hub cluster:

- **Auto-detect** supplies the hub cluster's tag **corners** (4 points/tag).
- **Manual locate → auto refine** supplies the spread off-hub tags. The operator
  clicks roughly on a tag the detector missed; the system runs AprilTag detection in
  a **small upscaled ROI around that click**. On success it snaps to the tag's
  sub-pixel corners (full 4-point constraint, detector-grade precision); on failure
  it falls back to the single clicked center point. This combines human tag-spotting
  with detector precision and is the key addition over the main-camera flow. Two
  details the implementation must handle: (a) corners decoded in the upscaled ROI are
  **mapped back to full-frame pixels** (divide by upscale factor, add ROI offset) —
  `detect_tags` currently returns corners in the frame it is handed; (b) on ROI
  success the **decoded `tag_id` must match the field tag the operator is mapping
  to** — if a different tag decodes (or the id isn't in the field layout), reject the
  refine and keep the manual center, so a stray decode can't create a mislabeled
  correspondence.
- The spread tags (far wall, near/far rails) are what make the resection
  well-conditioned and constrain distortion across the frame, not just the center.

**Validate honestly:** hold out a subset of tags distributed toward the frame edges;
report their reprojection error (px) and field-position error (m). The acceptance
gate is **both** that held-out error (target ≤0.5 m) **and a spread requirement** —
at least N successfully-located off-hub tags spanning the frame (not just the
collinear hub blob), so a solve can't clear `MIN_CALIB_POINTS` on the central
cluster alone and still validate poorly at the edges. (N set during implementation.)

**If validation fails** (manual-click noise or wide-lens distortion exceeding two
radial terms), escalate in order: (1) relax the solver's `CALIB_FIX_K3` /
`CALIB_ZERO_TANGENT_DIST` flags to add a third radial / tangential term, given the
points now span the frame; (2) fall back to a ground-plane homography
(`solve_ground_homography`, `mode="homography"`) from the same clicked points
projected to the floor plane — pose-free, ~0.3–0.5 m, sufficient for identity
association (the tracker gate is 2.5 m) but with no occlusion ray-casting. The
homography fallback is the contingency, not the plan.

Rationale: a posed pinhole calibration gives corner cams a real 3D camera center —
so they use the same model as the main cam, **keep occlusion ray-casting**, and need
no relaxation of the pose-free tracking gate. Identity tolerates loose positioning
regardless, so even the fallback preserves the headline benefit.

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
view's `calibration.image_to_field()` (all three `mode="pinhole"` on success; a
corner that fell back to homography projects the same way via its `H`) → bumper
color + bumper crop.

### Cross-camera fusion (new unit, before the tracker)

A new, isolated `fuse_detections(per_camera_detections)` unit keeps the
`FieldTracker` untouched:

- **Input:** for each camera, a list of detections. Today `FieldDetection`
  (tracker.py) is only `pos, color, box, score` — it carries no bumper crop or
  source camera. Identity evidence must therefore ride a **side channel**, not the
  `FieldDetection` the tracker consumes. Define a small `FusedObservation` that
  pairs the merged `FieldDetection` (what the tracker sees) with its pooled
  identity evidence: `[(source_camera, bumper_crop, weight), ...]`. The tracker
  contract is unchanged; the evidence travels alongside.
- **Cluster** detections within ~0.5–0.75 m across cameras as one physical robot.
- **Merge** each cluster to a single `FieldDetection` (quality-weighted average
  position, pooled color votes) plus its pooled identity evidence.
- **Output:** the merged `FieldDetection` list → existing `FieldTracker.update()`,
  unchanged (still 6 slots, 3 blue + 3 red) — *and* the parallel
  `FusedObservation` list, retained for the identity step below.

**Read → slot attachment (the fusion↔identity seam).** `FieldTracker.update()`
assigns each merged detection to a slot and returns that mapping (it already
associates detections to slots internally). Carry the fused detection's index
through `update()`'s assignment so each `FusedObservation`'s pooled bumper crops
attach to the **slot the tracker chose** for that detection — not a re-derived
nearest-slot guess. This keeps fusion, tracking, and voting consistent on one
association decision per frame.

### Identity: additive, corner-favored (decided: weighted votes, corner favored)

Reuse the existing roster-constrained reader and per-slot vote tally — corner
cameras add evidence, they don't get a separate identity path:

- Match roster (`match_blue_teams` / `match_red_teams`, from TBA) makes reading a
  **closed-set** problem ("which of these 6 numbers?"). Bumper crops from **all**
  cameras go through the existing reader
  `_vision_read_bumpers(crops, match_teams, blue_teams, red_teams)`.
- Each read attaches to a slot via the fusion↔identity seam above (the tracker's own
  assignment) and casts **votes in the existing per-slot tally**
  (`team_ocr_votes`). **Corner-camera reads are weighted higher**, using the
  per-source weight carried on the `FusedObservation`. The corner weight is chosen
  *relative to the existing weights* on the vote path (`EASYOCR_VOTE_WEIGHT` and the
  reader's `weight` param for VLM vs OCR), not as a number picked in isolation — so
  a corner read out-weights a same-modality main read without distorting the
  established VLM/OCR balance.
- **No hard override.** A confirmed identity can still flip if corner cameras
  consistently disagree — robust to one-off misreads. This is the explicit conflict
  policy.
- Corner reads are **not gated on `bumper_occluded`** — they run every frame. (The
  existing main-cam VLM gate stays as-is.) This is the mechanism that makes the
  corner cameras additive rather than failover.

### Occlusion

With the primary pinhole resection, corner cameras **have a 3D pose**, so they
*can* participate in ray-cast occlusion (`resolve_occlusion`) from their own camera
centers — more viewpoints make hub/person occlusion inference more robust. They also
fill occluded zones simply by contributing detections there. The main camera keeps
its logic unchanged. No change to the pose-free rejection gate is needed for the
primary path.

Only if a corner camera lands on the **homography fallback** is it pose-free; that
camera then skips ray-cast occlusion, and the field-tracking gate is relaxed to
admit homography mode **for corner cameras only** (the main camera still requires a
posed calibration, since its occlusion depends on it).

## Components & data flow

```
composite frame
   ├─ CameraView[main]        crop+mask → YOLO → image_to_field(pinhole) ─┐
   ├─ CameraView[corner_left] crop+mask → YOLO → image_to_field(pinhole) ─┤
   └─ CameraView[corner_right]crop+mask → YOLO → image_to_field(pinhole) ─┤
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
| corner tag resection | existing `solve_camera_from_tags` + UI | auto hub corners + manual-locate→auto-refine spread tags → `mode="pinhole"` |
| ROI tag refine | `web/field/apriltag_detect.py` (extend) | detect in small upscaled ROI around a manual click; snap to corners or fall back to center |
| homography fallback | existing `solve_ground_homography` | contingency when resection won't validate → `mode="homography"` |
| `fuse_detections()` | `web/field/` (new) | cross-camera cluster + merge → merged detections |
| identity vote weighting | `web/app.py` reader path | corner reads weighted, ungated, into existing tally |
| ingest loop | `web/app.py` `process_video()` | iterate views instead of single main crop |
| tracking gate | `web/app.py` `_init_field_tracking()` | unchanged for posed corner cams; relax to allow homography only for a fallback corner cam |
| calibrate UI | `web/static/calibrate.html` + endpoints | camera picker, per-camera crop, manual-locate refine, held-out error |

## Error handling

- **Uncalibrated corner camera:** that view is skipped (detections from it dropped);
  main-camera tracking proceeds. Logged, surfaced in the calibrate UI.
- **Resection fails validation (high held-out error):** escalate per the
  calibration section (relax distortion flags, then homography fallback); a
  calibration that still exceeds the gate is rejected at solve time and not saved.
- **Too few tags locatable in a corner view:** that camera cannot be calibrated;
  system degrades to main + the other corner.
- **Conflicting reads:** resolved by the weighted vote tally (no exception path).

## Testing

- **Unit:** `fuse_detections()` clustering/merge (synthetic multi-cam detections);
  `CameraView` crop+mask extraction; ROI tag-refine (snap vs center-fallback);
  held-out error computation.
- **Calibration acceptance:** held-out-tag reprojection error (px) and
  field-position error (m) per corner camera, under threshold (target ≤0.5 m; gate
  value set during implementation).
- **End-to-end smoke (mirrors the SP2 smoke test):** on
  `qual1_first_champs_2026.mp4`, confirm (a) corner detections populate
  hub-occluded field regions, and (b) corner-camera reads lock/confirm identities
  the main camera reads poorly. **Pass signal:** for at least one slot the main cam
  leaves unconfirmed or mislabeled, the final voted slot identity matches the TBA
  roster ground truth once corner reads are included — the additive-identity
  requirement, demonstrated against an objective reference.

## Out of scope

- Plumb-line / straight-line radial-distortion refinement (only the in-solver flag
  relaxation is in scope; a dedicated distortion estimator is a future escalation).
- Auto-detection of crop seams / overlay masks (measured once, configured).
- Triangulating robot position from multiple cameras for sub-decimeter accuracy.
