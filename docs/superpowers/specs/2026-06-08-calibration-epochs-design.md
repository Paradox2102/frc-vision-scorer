# Match-Ordered Calibration Epochs with AprilTag Validation (Spec 1: Core Guardrail)

**Date:** 2026-06-08
**Status:** Design — pending spec review
**Branch context:** `feat/bumper-model`

## Problem

A single-view camera calibration is only valid while the camera stays put. At a
multi-day event the broadcast/static cameras get moved or re-zoomed (observed:
calibrated on an early qual, then by `2026hop_qm72` on day 2 the main camera had
moved — the field projected ~40% too small, so real robots inverse-projected
off-field and were silently dropped, leaving 2-3 of 6 robots tracked). Today the
store keeps exactly one calibration per `(event, camera)` and overwriting it loses
the earlier-match calibration; nothing checks that the calibration still fits the
match being analyzed.

## Goal

Keep **multiple calibrations per `(event, camera)`**, each anchored to the match it
was created from, forming contiguous **match-ordered epochs**. At analysis time,
select the epoch covering the match, **validate** it against the match's
auto-detected AprilTags, and — on drift — **cancel** the run with a clear "needs
calibration for `<match>`" status rather than producing silently-wrong results.
Recalibrating creates a **new** epoch and never erases earlier ones.

## The Epoch Model

- Each stored calibration carries a **`start_match`** (a match key like
  `2026hop_qm72`) and a **camera** (`main` / `corner_left` / `corner_right`).
- A calibration applies from its `start_match` **forward in match order** until the
  next calibration's `start_match` for that camera. Match order is the FRC
  convention parsed from the key — qual before playoff, then ascending
  set/match number — **not** wall-clock time, and requiring **no TBA call**.
- The **earliest** epoch for a camera extends backward to the start of the event,
  so a single calibration always provides coverage.
- Epochs are **independent per camera**: a moved corner gets a new epoch for that
  camera only; the others keep theirs.
- A legacy single-file calibration (no `start_match`) is treated as one epoch
  covering the whole event (start_match = "beginning").

## Non-Goals (Spec 2 — separate)

- `/calibrate` match selector, listing event matches **from TBA**, and
  **downloading** missing footage.
- Per-camera + frame selectors and save-with-label UI; the guided
  "needs-calibration" recalibration UX.
- Rich version management (relabel / list / delete).

Spec 1 makes the *existing* save path write a new epoch (so versions can be created
via the current calibrate flow / API), and exposes the structured
needs-calibration status that Spec 2's UI will consume.

## Components

### 1. Epoch store (`web/field/store.py`)

Today: one file `web/uploads/calibration_{event}_{camera}.json` holding a
`CameraCalibration` dict; `save()` overwrites, `load()` returns it.

Add epoch metadata at the **store layer** (keep `CameraCalibration` a pure
projection object — don't bloat it):

- On disk, each version is `calibration_{event}_{camera}__{version_id}.json`
  (`version_id` = a sortable timestamp string) containing:
  ```json
  { "calibration": { <existing CameraCalibration.to_dict> },
    "start_match": "2026hop_qm72", "label": "day2", "created_at": "..." }
  ```
- `save_version(event, camera, cal, start_match, label=None) -> version_id` — writes
  a NEW file; never overwrites an existing version.
- `load_versions(event, camera) -> list[CalibrationVersion]` where
  `CalibrationVersion = {calibration: CameraCalibration, start_match, label,
  created_at, version_id}`. Globs `calibration_{event}_{camera}__*.json` **and**
  the legacy `calibration_{event}_{camera}.json` (→ a version with
  `start_match=None`, meaning "covers from the beginning").
- `update_version(event, camera, version_id, cal) -> None` — overwrite an
  **existing** version's calibration in place (for property edits like masks /
  top-crop), without changing its `start_match`/`created_at` or creating a new
  epoch.
- Keep `load(event, camera)` returning the most-recent version's calibration for
  any caller that doesn't need epoch selection (back-compat).

Sanitization of `event`/`camera` into the filename follows the existing `_path`
helper.

### 2. Match-order utility (`web/field/match_order.py`, new, pure)

- `match_sort_key(match_key: str) -> tuple` — parse `"{event}_{level}{nums}"`
  (e.g. `qm72`, `sf1m1`, `f1m1`) into a sortable tuple
  `(level_rank, set_number, match_number)`. `level_rank`: qm < ef < qf < sf < f.
  Robust to the keys this codebase already produces (the `matches` table stores
  `comp_level`, `match_number`, `set_number` from TBA — use the same parsing so DB
  and key agree).
- `select_epoch(versions, match_key) -> CalibrationVersion | None` — among a
  camera's versions, return the one whose `start_match` is the greatest that still
  sorts `<=` `match_key`; a `start_match=None` (legacy) version sorts first
  (covers from the beginning). Returns `None` only if there are no versions.

Pure and unit-testable with no DB/IO.

### 3. Tag-alignment scorer (`web/field/calibration.py`, pure)

```
def tag_alignment_error(cal, detected_tags, field_model) -> tuple[float, int]:
    """(mean field error in meters, n_tags) between auto-detected AprilTags and
    their calibrated positions. detected_tags: {tag_id: corners(4,2)} from
    detect_tags(). Builds correspondences from tags known to field_model and reuses
    holdout_field_error (height-aware). (inf, 0) if no usable tags."""
```

Reuses the existing `holdout_field_error` and `field_model.tag_corners_3d`. A moved
camera yields meters of error even though pixel-reprojection RMS would stay low.

### 4. Match-time validation + epoch selection (`InferenceJob`, `web/app.py`)

`self.match_key` is already set on the job (line 762, from the TBA roster) — so
epoch **selection** can happen at init; only **validation** (which needs frames)
waits for the timer phase.

`_init_field_tracking` currently does `cal = field_store.load(self.event_key,
self.camera)` for the single main camera, sets `self.calib = cal`, applies a
pose-free rejection guard (`cal.camera_center is None` → homography/no-pose →
"re-calibrate" error, lines ~3679-3694), then builds `self.views` via `load_views`.
The rework:
- Load **versions** per camera (`load_versions`) and choose each camera's epoch
  with `select_epoch(versions, self.match_key)`.
- Set `self.calib` to the **selected main-camera epoch's** calibration (not the raw
  `load()`), and apply the *same* pose-free guard to it — so the calibration the
  occlusion/radar path uses is the same one validation will check.
- Build `self.views` so each `CameraView.calibration` is its camera's selected
  epoch. (Either extend `load_views` to take a per-camera version map, or build the
  views from the selected versions directly.)
- If a camera has no versions, behave as today (field tracking disabled with the
  existing clear error). If `self.match_key` is None, fall back to the most-recent
  version per camera (+ warn).

New step in `process_video`, **after** the timer/clock phase
(`compute_active_windows` → `self.processing_ranges`, ~line 3876) and **before** the
SAM2 pre-computation pass (~line 3905), so we sample frames from inside an active
match window and abort before any expensive compute. It validates the **already
selected** `self.calib` (the main-camera epoch chosen at init):

```
verify calibration alignment:
  sample up to CALIB_CHECK_SAMPLE_FRAMES frames spread across processing_ranges[0]
  for the MAIN view (tags are static field structures, visible during play):
    extract crop -> detect_tags -> tag_alignment_error(selected_epoch_cal, ...)
    keep the best (most tags / lowest error) across sampled frames
  decision (main view):
    - tags found (n >= CALIB_CHECK_MIN_TAGS) and error <= CALIB_DRIFT_MAX_M -> OK
    - tags found and error > CALIB_DRIFT_MAX_M -> DRIFT: set self.error +
      self.status="error" + a structured needs_calibration payload
      {event, camera:"main", match_key, error_m, epoch:version_id}; return before SAM2
    - no tags across all sampled frames -> WARN (log) and proceed with the
      selected epoch (cannot validate)
  corner views: cannot auto-validate (tags not machine-decodable) -> use their
  selected epoch + a one-line warning; never block on corners.
```

Config (env knobs, matching existing style):

| Var | Default | Meaning |
|-----|---------|---------|
| `CALIB_CHECK_ENABLED` | `1` | master switch for the match-start validation |
| `CALIB_DRIFT_MAX_M` | `0.5` | max mean tag field-error (m) before a match is rejected |
| `CALIB_CHECK_MIN_TAGS` | `2` | min auto-detected tags needed to validate (else "no tags") |
| `CALIB_CHECK_SAMPLE_FRAMES` | `5` | frames sampled across the active window before giving up on tags |

### 5. Recalibration save (existing `/api/calibration/*` endpoints)

Six endpoints currently call `field_store.save(event, camera, cal)` (overwrite).
They split into two kinds and must be handled differently:

**Geometry solves → create a new epoch** (`save_version`): these produce a fresh
camera calibration and are the real recalibration paths —
- `POST /api/calibration/solve-tags` (line ~5917, `solve_camera_from_tags` — the
  **primary** pinhole resection for the broadcast/main camera, the one this whole
  feature targets),
- `POST /api/calibration/solve-homography` (~5848),
- `POST /api/calibration/manual` (~5640) and `POST /api/calibration/auto` (~5681)
  (`solve_pose` against existing intrinsics).

Change these to `save_version(event, camera, cal, start_match, label=None)`:
- `start_match` comes from the request (the match the user is calibrating against);
  if absent, default to `None` (legacy/whole-event epoch).
- `label` optional. No required input — recalibrating just adds a version.

**Property edits → update an existing version in place** (`update_version`, NOT a
new epoch): these `load → modify one field → save` an existing calibration —
- `POST /api/calibration/masks` (~5582, sets `cal.masks`),
- `POST /api/calibration/robot-track-top-crop` (~5604, sets
  `cal.robot_track_top_crop`).

These must read the target version (default the latest; accept an optional
`version_id` in the body so Spec 2's UI can target a specific epoch), modify the
field, and `update_version(...)` it **in place**. They must operate on the
**versioned store**, not the legacy single file — otherwise an edited mask/top-crop
would desync from the versioned pose (inference loads versions; a legacy-file edit
would be invisible). If no version exists yet, create one (`save_version` with
`start_match=None`) as today's "no calibration → default" branch does.

This is the minimal authoring hook so epochs can be created and edited now (Spec 2
adds the match selector / download / richer per-version UI around it).

## Data Flow

```
/calibrate solve  ->  save_version(event, camera, cal, start_match)  ->  new file (never overwrites)

inference (match_key known):
  _init_field_tracking: load_versions per camera -> select_epoch(.., match_key) -> provisional views
  process_video: timer scan -> processing_ranges
     -> [verify_calibration_alignment on MAIN over sampled active-window frames]
          fits  -> proceed to SAM2
          drift -> status=error + needs_calibration{match,camera,error_m}; STOP (no SAM2)
          no tags -> warn + proceed
     -> SAM2 pass -> tracking -> ...
```

## Error Handling / Edge Cases

- **No match_key on the job:** can't do epoch lookup → use the most-recent version +
  warn (don't block). (Normal runs set match_key from the TBA roster.)
- **No calibration at all for a camera:** existing "field tracking disabled" error
  (unchanged).
- **Earliest epoch later than the match:** the earliest version (smallest
  `start_match`, or a legacy `None`) covers backward to the start, so coverage is
  guaranteed when ≥1 version exists; the tag check then decides if it actually fits.
- **`CALIB_CHECK_ENABLED=0`:** skip validation entirely (epoch selection still
  applies); full revert of the guardrail.

## Backward Compatibility

- Legacy `calibration_{event}_{camera}.json` loads as a single epoch
  (`start_match=None`) covering the whole event. Existing runs behave as before
  until a second version is added.
- `load()` semantics preserved for non-epoch callers.

## Testing

**Pure unit:**
- `match_sort_key`: `qm2 < qm10 < qm120 < sf1m1 < f1m1`; set/match ordering; bad key
  handled.
- `select_epoch`: picks the greatest `start_match <= match`; legacy `None` covers
  from the start; later-only versions still selected for earlier matches (then the
  tag check decides fit); empty → None.
- `tag_alignment_error`: synthetic detected tags = `field_to_image(known_3d)` →
  error ≈ 0; shifted/scaled tags → large error; no usable tags → `(inf, 0)`.
- `save_version` / `load_versions`: two saves → two files, neither overwritten;
  legacy file loads as a `None`-start version; versions sorted.

**Integration (`InferenceJob` with stubbed detect/cameras or a synthetic frame):**
- Aligned tags → validation passes, run proceeds (status not error).
- Drifted calibration (shifted tags) → `status=="error"`, `needs_calibration`
  payload names the match + main camera + error; SAM2 not invoked.
- No tags across sampled frames → warning logged, run proceeds.
- `CALIB_CHECK_ENABLED=0` → no validation regardless of tags.

## Files Touched

- `web/field/store.py` — `save_version`, `load_versions`, `update_version`,
  `CalibrationVersion`; keep `save`/`load`.
- `web/field/match_order.py` (new) — `match_sort_key`, `select_epoch`.
- `web/field/calibration.py` — `tag_alignment_error`.
- `web/field/camera_view.py` — `load_views` (or a new `load_view_candidates`) to
  expose per-camera versions for selection / accept a per-camera version map.
- `web/app.py` —
  - `_init_field_tracking`: load versions per camera, select epoch via
    `self.match_key`, set `self.calib` to the selected main epoch (+ existing
    pose-free guard), build views from selected epochs.
  - `process_video`: post-timer `verify_calibration_alignment` (validate
    `self.calib`, abort on drift).
  - Geometry-solve endpoints (`solve-tags`, `solve-homography`, `manual`, `auto`)
    → `save_version`; property-edit endpoints (`masks`, `robot-track-top-crop`)
    → `update_version` on the versioned store.
  - Config constants.
- Tests under `tests/field/` and `tests/` (run via `.venv/bin/python -m pytest`).

## Implementation notes

- `tag_alignment_error` must pair detected tag corners to `tag_corners_3d` the
  **same way `solve_camera_from_tags` does** (the field model flags its 2D↔3D
  corner ordering as provisional/PnP-reconciled) — reuse the same pairing so the
  scorer and the solver agree; don't invent a different one.
- `holdout_field_error` already returns `inf` when there are no usable points, which
  gives the `(inf, 0)` contract for free — `tag_alignment_error` is essentially a
  correspondence-builder around it.
- Keep AprilTag detection on a single sequential path on the inference thread (the
  detector is a non-thread-safe singleton); the existing `_scan_match_clock` runs
  via `asyncio.to_thread`, so run the sampling/detection similarly off the loop.
- `load_views(event_key)` takes only `event_key` today (`camera_view.py:27`). The
  per-camera epoch selection is a **real signature/behavior change** — pick ONE
  path (extend `load_views` to accept a `{camera: version_id}` map, or add
  `load_view_candidates` and build views in `_init_field_tracking`) and use it
  everywhere, so no two callers diverge.
- The geometry-solve endpoints currently copy `existing.masks` /
  `robot_track_top_crop` forward onto the freshly-solved `cal` (so a re-solve keeps
  authored masks/crop). When these become `save_version` (a NEW epoch), source that
  carry-forward from the **selected/latest version** (via `load_versions` /
  `select_epoch`), not the legacy `load()`, or a new epoch silently drops
  previously-authored masks/top-crop.

## Risks & Mitigations

- **Corner cams can't auto-validate** (non-decodable tags). Accepted: the guardrail
  is a main-camera feature; corners use their selected epoch + warn. The versioned
  store still serves corners (manual per-camera recalibration).
- **Sampling lands on tag-occluded frames** → false "no tags". Mitigated by sampling
  `CALIB_CHECK_SAMPLE_FRAMES` across the active window and taking the best.
- **Match-key parsing drift** (unusual keys). Mitigated by mirroring the existing
  `matches`-table comp_level/number parsing and unit tests; unknown keys fall back
  to most-recent + warn rather than crashing.
- **A genuinely-correct calibration near 0.5 m** (e.g. poor tag geometry) could
  false-cancel. `CALIB_DRIFT_MAX_M` is env-tunable; default 0.5 m is 5× the 10 cm
  tracking budget and well below the meters-scale error a real camera move produces.
