# Detector-Absence Cull for Hijacked SAM2 Masks

**Date:** 2026-06-08
**Status:** Design — pending spec review
**Branch context:** `feat/bumper-model`

## Problem

A tracked robot drove past a static field structure (the trench). During SAM2
mask propagation its mask **transferred** onto the trench and stayed glued there
for the rest of the match — a "mask hijack." The hijacked object keeps emitting a
plausible robot box at a fixed field location, so:

1. It is a **persistent false positive** on the field map / radar.
2. It **consumes one of the `MAX_TRACKED_ROBOTS = 6` active SAM2 object slots**,
   so when the roster is capped the real robot (which kept moving — clearly
   visible in the footage for the rest of the match) cannot be re-seeded.

This is **drift, not seeding**, so the static-structure zone gate
(`tracker.py`, `point_in_zones`) does **not** fix it: zones block *new* track
seeding inside a field element, but the hijacked mask was already an active object
that drifted in. It is never re-seeded; it is continuously propagated.

### Why the existing retirement never fires

`sam2_tracker.py` already has an object-retirement mechanism (`retired_set`,
~line 566): an object retires when its mask has been **absent** for
`retirement_frames`. The hijacked mask is never absent — it is permanently visible
on the trench — so it never retires. The trench is, to SAM2, a perfectly stable
object. We need a **second retirement trigger** for masks that stay visible but
have stopped corresponding to a real robot.

## Goal

Detect and cull a hijacked SAM2 object using a **detector-absence** signal
(no fresh bumper *or* whole-robot detection backs the mask) gated by
**stationarity** (the mask's field position is not moving), so that:

- the phantom track disappears, and
- its freed object slot lets the still-moving real robot re-seed via the existing
  re-prompt path within the same tracking pass.

A genuinely parked or briefly-occluded robot must **not** be culled.

## Non-Goals

- Retraining any model.
- Touching the per-frame, non-SAM2 detection path.
- Appearance-discontinuity detection at the transfer frame (a viable alternative
  signal, deferred; see Alternatives Considered).
- Any change to `FieldTracker` slot logic — the downstream slot expires on its own
  once its mask source stops (existing coast/expire, `tracker.py:135-156`).

## Cull Predicate (decided)

Cull an active SAM2 object when **both** hold:

1. **Detector-absence:** the object's current mask has had **no** corroborating
   fresh detection for `CULL_UNCORROBORATED_REPROMPTS = 3` consecutive re-prompts
   (~6 s at `REPROMPT_SECONDS = 2.0`). "Corroborating" = best Intersection-over-
   Smaller between the mask and any fresh detector box ≥ `CULL_CORROB_IOS = 0.3`.
   The re-prompt detector already merges **bumper** *and* **whole-robot** boxes
   (`app.py:4154-4155`), so absence means *neither* model fired on it.
2. **Stationary:** the object's projected field position has moved less than
   `CULL_STATIONARY_M = 0.5` m across that window.

Detector-absence is the discriminator; stationarity is the safety gate. Velocity
≈ 0 alone never culls (would drop parked robots); detector-absence alone is the
near-sufficient signal but stationarity protects a *moving* robot that a detector
momentarily loses (a mover is self-evidently a robot — never cull it).

### Why stationarity is measured in **field** space, not pixels

This is the payoff of the field-coordinate rework. A broadcast camera pans and
zooms while following the action; over a ~6 s window the static trench's mask
*moves in pixels* even though it is bolted to the floor. Pixel-space stationarity
would therefore read the trench as "moving" during a pan and refuse to cull it —
weakening the cull exactly when the camera is busy. Projecting the mask's
bottom-center to field coordinates removes camera ego-motion, so a static trench
reads as stationary regardless of camera motion, and a real driving robot reads as
moving regardless of camera motion. Bottom-center projection matches the
convention used everywhere else (`detect_bridge._bottom_center`).

## Key Enablers Already Present

The change is composition; the primitives exist:

1. **Re-prompt detector already runs every ~2 s** and merges bumper + whole-robot
   boxes (`make_detect_fn`, `app.py:4137-4191`). This is the natural cadence and
   the source of "fresh detections."
2. **IoS overlap math exists** (`_ios`, used for birth evidence at
   `sam2_tracker.py:666`) and `match_new_detections` already computes the
   *complement* (detections not overlapping any mask, line 617). Corroboration of
   existing masks is the same computation, iterated objects × fresh boxes.
3. **Birth-time corroboration is already tracked** (`birth_detector_overlap`,
   `detector_overlap` evidence at line 510/527) — we extend the idea to ongoing
   corroboration.
4. **Object retirement & active-set recomputation already exist** (`retired_set`,
   `active_objects`, lines 563-577). Culling reuses this shape — it adds a
   `culled_set` checked the same way.
5. **Mask skipping on a per-object basis already exists**: implausible masks are
   dropped with `continue` in `_record_chunk` (line 493). Culled objects skip the
   same way.
6. **Calibration is available where `detect_fn` is built** (inside
   `for view in self.views`, `app.py`), so a field-projection callable can be
   injected into `track()` exactly as `detect_fn` is.

## Design

### Load-bearing change: run the detector even when capped

Today `detect_fn` runs **only when `not capped`** (`sam2_tracker.py:579`). But the
phantom hurts most precisely when it fills the 6th active slot → roster capped →
detector skipped → no corroboration data → the phantom can never be evaluated and
the cull **deadlocks in the exact case it exists to fix**.

Fix: at each re-prompt boundary, **always** run `detect_fn` (when provided) to
obtain `new_boxes`, and use it for two independent purposes:

- **Corroboration of existing objects** — always (this is the new use).
- **Seeding new objects** — only when `not capped` (unchanged behavior).

So `capped` no longer gates *whether the detector runs*; it gates only *new-object
addition*.

**Caveat on what culling relieves.** The real `capped` condition is two-termed:
`len(active_objects) >= max_active_objects OR (next_obj_id - 1) >=
max_lifetime_objects` (lines 577/644). A cull removes an object from
`active_objects`, relieving only the **first** term; the lifetime cap
(`max_lifetime_objects = MAX_TRACKED_ROBOTS * 3 = 18` in production) is never
relieved (`next_obj_id` only increments). This is acceptable: in a normal match
the lifetime cap does not bind, so freeing an active slot is what unblocks
re-seeding. The slot-freeing claim below is precise for the active-roster cap.

### Component 1 — Per-object corroboration & stationarity state (in `track`)

Maintain, keyed by object id, across re-prompts:

- `uncorroborated_reprompts: dict[int, int]` — consecutive re-prompts with no
  corroborating detection.
- `field_pos_window: dict[int, list[tuple]]` — recent projected field positions
  (bounded to the last `CULL_UNCORROBORATED_REPROMPTS + 1` entries).
- `culled_set: set[int]` — objects culled by this mechanism.

### Component 2 — Corroboration check at the re-prompt boundary (in `track`)

After the load-bearing decouple above makes `new_boxes` available at every
re-prompt boundary (it is the `_normalize_detections` output, currently produced
only inside the `if not capped:` block at line 587 — the decouple lifts it out so
it is in scope unconditionally), for each currently-active object with a latest
mask in `existing`:

1. Build the object's mask (already in `existing[oid]`).
2. Compute `best_ios = max(_ios(existing[oid], box_mask(b)) for b in new_boxes)`
   (0.0 if no boxes). Respect the same color-conflict skip already used at
   lines 661-665 (a red mask is not corroborated by a blue detection).
3. If `best_ios >= CULL_CORROB_IOS`: reset `uncorroborated_reprompts[oid] = 0`.
   Else: `uncorroborated_reprompts[oid] += 1`.

This reuses `_ios` and the box-mask construction already present at lines 656-668.

### Component 3 — Stationarity via injected projector (in `track`)

Add an optional `project_fn` parameter to `track()` mirroring the `detect_fn`
injection. `project_fn(box) -> (x, y) | None` projects a mask box's bottom-center
to field meters. Supplied by `app.py`, bound to `view.calibration`
(`image_to_field` ∘ `_bottom_center`), scaled out of SAM2 (0.5×) coords the same
way the re-prompt boxes are.

At each boundary, for each active object, append `project_fn(latest_box)` to
`field_pos_window[oid]` (drop the oldest beyond the window). The object is
**stationary** if the span of its window positions is < `CULL_STATIONARY_M`
(euclidean max pairwise, or `max−min` per axis). If `project_fn` is `None` (tests,
simple callers) or returns `None`, fall back to pixel-space bottom-center
displacement scaled by a conservative constant, and document that this is the
ego-motion-confounded fallback (safe direction: it under-culls).

### Component 4 — Cull decision & effect (in `track`)

At the boundary, after updating corroboration + stationarity, for each active
object:

```
if (uncorroborated_reprompts[oid] >= CULL_UNCORROBORATED_REPROMPTS
        and is_stationary(oid)):
    culled_set.add(oid)
    diagnostics.record("sam2.object_culled",
        details={object_id, frame_idx, reason: "detector_absent_stationary",
                 uncorroborated_reprompts, field_span})
```

Effects, all via existing machinery:

- **Stop emitting the mask.** In `_record_chunk`, add `if oid_int in culled_set:
  continue` alongside the implausible-mask skip (line 493). SAM2 keeps propagating
  the object internally (no clean per-object delete in the predictor), but we stop
  recording it into `results`, so it vanishes from output. (Minor wasted compute,
  documented.)
- **Free the slot.** Exclude `culled_set` from `active_objects` (lines 564-567,
  638-642), so `capped` drops and the next re-prompt's fresh detection on the real
  moving robot is added as a new object through the *unchanged* seeding path.
- **Downstream slot expires.** The phantom's `FieldTracker` slot stops receiving a
  detection and coasts → expires via existing logic (`tracker.py:135-156`). No
  tracker change.

### Data Flow

```
re-prompt boundary (every ~REPROMPT_SECONDS)
        │
        ├─ detect_fn(last_idx) ──► new_boxes  (bumper ⊕ whole-robot, ALWAYS run)
        │                              │
        │            ┌─────────────────┴───────────────────┐
        │            ▼                                      ▼
        │   corroboration check                    new-object seeding
        │   (existing masks × new_boxes,           (only if not capped —
        │    _ios ≥ CULL_CORROB_IOS?)               unchanged path)
        │            │
        │            ▼
        │   uncorroborated_reprompts[oid] ++ / reset
        │            │
        ├─ project_fn(latest_box) ──► field_pos_window[oid]  ──► stationary?
        │            │
        ▼            ▼
   cull if uncorroborated ≥ K AND stationary
        │
        ├─ culled_set.add(oid)
        ├─ _record_chunk skips oid  ──► mask stops being emitted
        └─ active_objects excludes oid ──► slot freed ──► real robot re-seeds
                                                           next re-prompt
        ▼
   phantom's FieldTracker slot loses source ──► coasts ──► expires (existing)
```

## Configuration (env vars, matching existing style)

| Var | Default | Meaning |
|-----|---------|---------|
| `CULL_HIJACK_ENABLED` | `1` (on) | master switch for the detector-absence cull |
| `CULL_UNCORROBORATED_REPROMPTS` | `3` | consecutive uncorroborated re-prompts (~6 s) before a stationary object is cullable |
| `CULL_CORROB_IOS` | `0.3` | min IoS of a fresh detection vs the mask to count as corroboration (lenient → conservative culling) |
| `CULL_STATIONARY_M` | `0.5` | max field-position span over the window to count as stationary |

When `CULL_HIJACK_ENABLED=0`, `track()` behaves exactly as today **except** that
`detect_fn` may now run at boundaries where it previously didn't (capped) — guard
the extra detector call behind the enable flag too, so the kill switch fully
restores prior behavior and compute.

## Testing

**Unit — corroboration counter (extract to a pure helper):**
- Mask with an overlapping fresh box (IoS ≥ thresh) → counter resets to 0.
- Mask with no overlapping box for 3 boundaries → counter reaches 3.
- Color-conflicting fresh box (blue box vs red mask) → does **not** corroborate.

**Unit — stationarity over window:**
- Fixed field position across window → stationary.
- Position drifting > `CULL_STATIONARY_M` → not stationary.
- `project_fn=None` → pixel-space fallback path exercised.

**Integration — `sam2_tracker.track` with stubbed predictor + detect_fn:**
- An object whose mask stays fixed and is never returned by `detect_fn` for 3
  re-prompts, with a stationary projection → `culled_set` contains it; its masks
  stop appearing in `results` afterward; `sam2.object_culled` recorded.
- Same object but **moving** (projection drifts) → not culled (mover protection).
- Same object but `detect_fn` keeps returning an overlapping box → not culled
  (corroborated; e.g. a real parked robot the detector still sees).
- **Deadlock regression:** roster capped (6 active incl. the phantom), phantom
  uncorroborated + stationary → it is still evaluated and culled (proves
  detect_fn runs when capped), and a new detection then seeds the freed slot.
- `CULL_HIJACK_ENABLED=0` → no culling, and detector call cadence matches prior
  behavior.

## Files Touched

- `web/field/sam2_tracker.py` — add `project_fn` param; always-run-detect_fn
  decouple; per-object corroboration + stationarity state; `culled_set` +
  `_record_chunk` skip; cull decision + diagnostics. Extract corroboration and
  stationarity checks as pure helpers for unit testing.
- `web/app.py` — pass a `project_fn` bound to `view.calibration` into
  `sam2_tracker.track`; add the four config constants.
- Tests under `tests/field/` (run via `.venv/bin/python -m pytest`).

## Risks & Mitigations

- **Culling a real robot the detector lost while parked.** Mitigated by the
  whole-robot detector fallback (a parked robot with occluded bumpers still trips
  the chassis model → corroborated) and the 6 s / 3-reprompt window. Residual:
  a parked robot in a genuine detector blind spot for 6 s. Even then the cull
  **self-heals** — the moment it is re-detected it re-seeds; whereas the trench is
  never re-detected, so a correct cull sticks. This asymmetry favors culling.
- **Camera pan defeats pixel-space stationarity.** Resolved by measuring
  stationarity in field space via `project_fn`; the pixel fallback errs toward
  under-culling (safe).
- **False corroboration when the real robot drives back across the trench.** A
  fresh detection on the robot then overlaps the trench mask and resets the
  counter — delaying, never wrongly causing, a cull. Acceptable.
- **Wasted SAM2 compute** for culled-but-still-propagated objects. Bounded (≤ a
  few objects per match); a clean predictor-side delete is out of scope.
- **Extra detector calls when capped.** Previously skipped; now run for
  corroboration. Bounded to one detect_fn call per re-prompt (already the cadence
  when not capped) and fully reverted by `CULL_HIJACK_ENABLED=0`.

## Alternatives Considered

- **Appearance discontinuity at the transfer frame** (position jump → embedding
  drop → stationary): precise and uses the per-frame embeddings already being
  logged, but requires reconstructing the transfer event and a per-object
  embedding-history comparison. Deferred — detector-absence is cheaper, online,
  and reuses existing re-prompt machinery. Could be layered later as a second
  trigger.
- **Cull at the `FieldTracker` layer:** rejected — the corroboration signal only
  exists at the SAM2 layer (the tracker sees a provenance-free projected point),
  and culling there would not free the SAM2 object budget that re-acquisition
  needs.
- **Mask-shape filter only:** the trench mask is plausibly robot-shaped enough to
  pass `plausible_robot_mask` (it already survives — that's why it persists), so
  shape alone is insufficient here.
