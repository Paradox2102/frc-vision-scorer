# Field Map v2 — accurate geometry, track-quality visibility, stable positions

**Date:** 2026-06-12
**Status:** approved (option B)
**Primary use:** post-match replay review

## Problem

The 2D field map (web/field/radar.py, composited into the HUD below the main
view) has three defects:

1. **Crude geometry.** Field structures are hand-approximated at made-up
   fractional positions, while exact CAD-derived geometry already exists in
   `web/field/overlay_geometry.py` (it drives the /calibrate overlay).
2. **No track information.** Each robot is an anonymous alliance-colored dot:
   no team number, no tracking state, no history.
3. **Bad location data.** Dots teleport (corner-only depth error passes
   through the Kalman filter at full trust, `_KF` has one fixed `r=0.05`),
   and occluded robots wander (coasting tracks keep their last velocity
   forever under the constant-velocity predict).

## Design

### 1. Renderer (web/field/radar.py)

- `_draw_field` rebuilt from `overlay_geometry` + `FieldModel` dimensions:
  perimeter, midline, hub footprints (ground vertices of `hub_prisms`),
  bump rectangles, tower and trench footprints. One source of truth shared
  with the calibrate overlay; all hand-tuned fractions deleted.
- Render API: `render(tracks)` where each track carries
  `{pos, alliance, team, slot_id, state, corner_only, trail}`.
  - solid dot = visible; hollow/faded dot = coasting (fade with miss count)
  - dashed ring = position currently sourced from corner cameras only
  - team number label above the dot (slot id fallback)
  - trail: fading polyline of recent filtered positions (~2 s)
- `px_per_m` 40 → 56 for label legibility (HUD `paste_fit` caps upscale).

### 2. Trails (web/app.py)

Per-slot `deque(maxlen≈16)` of filtered positions appended each analysis
frame; cleared when the slot empties. Passed into `render`. No tracker state.

### 3. Tracker data quality (web/field/tracker.py)

Three small changes, each independently testable:

- **Corner-only downweight.** `_KF.update(z, r_scale=1.0)`; matched updates
  pass `r_scale = corner_only_r_scale` (default 8.0) when the detection has
  `box is None` (fusion already sets main-view boxes only). Positions lean
  on the motion model and are gently corrected instead of teleporting in
  depth. This is the corner-cams-identity-first doctrine expressed as
  measurement noise.
- **Coast velocity decay.** Each missed frame multiplies KF velocity by
  `coast_vel_decay` (default 0.85): an occluded robot eases to a stop at its
  last seen position instead of sailing away.
- **Teleport suppression.** If an accepted measurement implies speed above
  `TRACKER_MAX_SPEED_M_S` (5.5), inflate its `r_scale` by the squared excess
  ratio rather than snapping to it.

`corner_only_r_scale` and `coast_vel_decay` become job-config knobs
(`CORNER_ONLY_R_SCALE`, `COAST_VEL_DECAY`) wired through `FieldTracker`
construction; the speed cap is a module constant.

`TrackOut` gains a `corner_only: bool` field so the renderer doesn't have to
infer it from `box is None` at a distance.

### 4. Validation

- Unit tests: corner-only update moves the state less than a main-backed
  update of the same innovation; coasting velocity decays toward zero;
  an implausible-speed measurement does not yank the position; structure
  footprints land at exact field coordinates; render API accepts the new
  track struct (labels, trails smoke-checked via pixel sampling).
- Full suite must stay green.
- qm24 re-run (start_all.sh profile): eyeball the replay map; confirmed-lock
  count must not regress vs the 2026-06-12 baseline (3 confirmed).

## Out of scope

Client-side interactive map, fuel/ball rendering, ray-aligned anisotropic
measurement noise (revisit only if scalar downweight proves insufficient).
