# Zone seed-only semantics + corner-camera weight falloff

**Date:** 2026-06-09
**Context:** Follow-ups to the qm24 "robot never detected" investigation (commit
4e6ac85d). Both items are policy filters that discard good evidence at the
detection level. Validated against job d6ce1999 (2026hop qm24, robot 3506).

## Item 1 — static zones must gate *seeding only*, not detections

### Problem

`zones.py`'s own docstring states the rule: "a NEW slot is never seeded from a
detection inside a zone." `FieldTracker._assign` implements exactly that. But
the detection layer drops zone hits outright before the tracker ever sees them:

- `to_field_detections` (live Kalman path, app.py:1202): zone detections
  removed with reason `static_structure` → an existing track that drives into
  a zone coasts ≤8 frames and dies.
- `filter_field_robot_boxes(blocked_zones=…)` in the SAM2 per-frame consumption
  (app.py:4790): a *tracked mask* over a zone is discarded every frame.
- Same filter inside `detect_fn` (app.py:4434): a robot parked in a zone can't
  corroborate its own SAM2 mask. **Post-eviction (4e6ac85d) this is dangerous:**
  zone-parked robot → detector-absent + stationary → evictable.

### Change

1. `to_field_detections`: stop dropping zone detections (and stop logging them
   to `rejected_out` — they are no longer rejections, and a parked robot logged
   as a hard negative would poison the mining set). The FieldTracker's existing
   seed gate (which uses the *full* zone set: drawn zones + hub zones) is the
   sole enforcement point on this path.
2. SAM2 per-frame consumption: drop the `blocked_zones` argument — tracked
   masks are post-seeding evidence.
3. `detect_fn`: drop `blocked_zones` from its field filter so zone detections
   corroborate masks; move the seed gate into `Sam2FieldTracker.track()` via a
   new optional `seed_gate_fn(box) -> bool` parameter, applied only in the
   new-object seeding loop (rejection reason `static_structure`). app.py builds
   the closure: project box bottom-center → field → `not point_in_zones(...)`.
4. Prompt-frame seeding (app.py:4263) keeps `blocked_zones` — it *is* seeding.

Scope note: SAM2 seed gating keeps today's zone set (hub zones only); drawn
static zones continue to gate only FieldTracker slot seeding, as today.

### Tests

- `to_field_detections` keeps a zone detection; nothing appended to rejected_out
  (updates `test_detection_inside_blocked_structure_zone_is_dropped`).
- FieldTracker: track seeded outside a zone keeps updating (not coasting) from
  detections inside the zone; fresh detection inside a zone still never seeds.
- `Sam2FieldTracker`: `seed_gate_fn` blocks seeding with a `static_structure`
  rejection event; a gate-blocked box still corroborates an existing mask
  (no absence accrued → not evictable).

## Item 2 — corner-camera hard cutoff → weight falloff

### Problem

`corner_detection_in_alliance_zone` hard-drops corner-camera evidence beyond
4.8 m from the camera's end wall, in both the live path (app.py:1211) and SAM2
consumption (app.py:4800). Center field is therefore covered by exactly one
camera (main), which the hub partially occludes — zero redundancy exactly where
robots cross (the 3506 scenario). The corner SAM2 trackers happily *track*
robots past 4.8 m; their observations are then discarded at consumption.

### Change

Replace the binary gate with a confidence weight:

- New `corner_detection_weight(view_name, xy, field_dims, zone_depth,
  falloff_m, midline_margin_m)` in detect_bridge:
  - 1.0 within `zone_depth` of the camera's end (current trusted band);
  - `exp(-(d - zone_depth) / falloff_m)` beyond it (oblique projection trusted
    progressively less; `CORNER_ZONE_FALLOFF_M`, default 2.0);
  - 0.0 past the field midline + `CORNER_MIDLINE_MARGIN_M` (default 1.0) —
    the far half stays out of bounds (extreme range + hub occlusion);
  - 1.0 always for non-corner views.
- `corner_detection_in_alliance_zone` becomes `weight > 0` (kept for callers/
  tests; its meaning widens from "within 4.8 m" to "within midline + margin").
- Both consumption sites multiply `FieldDetection.score` by the weight instead
  of dropping (drop only at weight 0).
- `fusion.fuse_detections`: member weight becomes `camera_weight * det.score`
  (score defaults to 1.0, so all other paths are unchanged). Position
  averaging, color votes, and embedding averaging inherit the falloff
  automatically; main-camera positions remain authoritative when present.

### Accepted risk

A corner-only detection in the decayed band may now seed a tracker slot. The
protections that remain: main-priority positioning in fusion, the half-field
territory rule for unknown colors, tracker distance gating, static zones, and
the bumper model's hard-negative retrain. If phantom seeds reappear in
re-analysis, the follow-up is a `min score to seed` gate in FieldTracker.

### Tests

- Weight function shape: 1.0 inside depth; monotone decay beyond; 0 past
  midline+margin for both corners; 1.0 for `main` everywhere.
- Fusion respects `det.score` (a half-score corner member pulls the weighted
  position/color vote half as hard).
- Consumption-site behavior covered via updated corner-zone tests.

## Validation

Re-analyze qm24 (job 2fff2a03 baseline) and diff: detector coverage stats,
slots identified, field-map continuity of 3506 across the midline.
