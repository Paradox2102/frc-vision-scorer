# Track-Identity Persistence & Confidence-Driven VLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Keep team identity, votes, appearance, trajectory, and scoring attribution
attached to the same physical robot across slot expiry and ambiguous re-emergence, while
moving VLM reads from pass-count scheduling to confidence-driven targeted reads with a
hard match-clock audit floor.

**Architecture:** Add a pure `IdentityRegistry` above the six `FieldTracker` slots.
`FieldTracker` continues to own short-lived visibility/Kalman state, but reports binding
events and assignment evidence. The registry owns match-durable `RobotIdentity` objects
and the current `slot_id <-> identity_id` bindings. `InferenceJob` consumes identities
instead of treating `slot_id` as the robot id. A separate pure confidence module turns
continuity, appearance, assignment margin, roster state, and VLM agreement into read
priority. A final joint-occlusion resolver compares the best and swapped assignments and
requests targeted VLM reads when the margin is genuinely ambiguous.

**Tech Stack:** Python 3.12, dataclasses, NumPy, existing DINOv2 embeddings and
`AppearanceGallery`, existing `FieldTracker`/`FusedTrackRegistry`, pytest. Run tests with
`.venv/bin/python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-06-06-track-identity-persistence-design.md`

---

## Non-Negotiable Invariants

These are stronger than implementation details. Add assertion helpers for them in tests
and optionally in diagnostic builds.

1. **No silent identity transfer:** assigning a slot never copies, swaps, or resets
   durable state. It only changes a binding.
2. **Atomic association:** all observations for one alliance/frame are associated as one
   constrained assignment. Input iteration order cannot change the result.
3. **No evidence poisoning:** an observation may update an identity's EMA/anchors only
   after the assignment clears the confidence threshold. Ambiguous observations remain
   quarantined and cannot modify durable appearance state.
4. **Confirmed identity protection:** a low-margin tracker/appearance assignment cannot
   move a confirmed team identity. It must become ambiguous and request verification.
5. **Roster uniqueness:** one roster team can belong to at most one live identity, and
   inactive confirmed identities continue to reserve it.
6. **Alliance consistency:** an identity's alliance is fixed once supported by a
   confirmed roster team. A contradictory slot/color observation lowers confidence; it
   does not silently flip the identity.
7. **Temporal-source correctness:** identity cadence uses source-video/match elapsed
   seconds, not processing wall time. Results must be invariant to inference speed,
   pause/resume, and skipped non-match ranges.
8. **Audit liveness:** after the floor expires, an eligible visible identity remains
   audit-due until a usable result is obtained. Occlusion/unreadability may prevent a
   successful audit, but cannot mark the obligation satisfied.
9. **Atomic ambiguity resolution:** a group assignment is committed all at once or not
   at all. Partial VLM results cannot move one participant and strand the other.
10. **Stable external identity:** persistence, corrections, replay, scoring, and crops
    use `identity_id`. Slot is nullable presentation/history metadata.
11. **Explainable operation:** every identity-binding change, ambiguity transition, VLM
    scheduling decision, and dropped observation records a structured reason and its
    relevant evidence. Performance timings use stable stage names and cannot silently
    disappear on exceptions.

## Current-State Constraints

- `FieldTracker` expires a slot after its coast budget and drops the stale
  `track_id -> slot_id` mapping. The next detection seeds the first free slot.
- `InferenceJob.robot_tracks` is keyed by slot and `RobotTrack.robot_id == slot_id`.
  Votes, locks, gallery state, trajectories, scores, diagnostics, replay events, crop
  review, and DB records therefore all inherit slot reassignment.
- `FusedTrackRegistry` already preserves an appearance EMA and can re-link a new camera
  member key to an inactive fused id, but it does not own team/vote state or roster caps.
- `FusedTrackRegistry.assign()` performs appearance re-link greedily in observation
  order. Two similar re-emergent robots can therefore produce different assignments when
  input order changes.
- SAM2 has temporal video memory, but re-prompted detections receive new object ids.
  Its current `max_objects=6` cap counts ids ever created, not currently visible robots;
  when six objects were prompted initially, re-prompting cannot add a replacement id.
- SAM2's re-prompt dedupe only compares against masks visible at that boundary. A
  temporarily empty/drifted mask can cause a duplicate object id for the same robot.
- `_is_reread_pass()` is pass-count based. Once confidence stretches the normal VLM
  interval, it cannot enforce a 30-second maximum between audits.
- Existing VLM timing uses `time.time()`, so read cadence changes with processing speed
  rather than match-video time.
- SQLite identity/crop tables are keyed by `(job_id, slot)`. Adding an `identity_id`
  column without changing the primary key is insufficient because slots can be reused
  and inactive identities may have no final slot.
- The working tree currently contains in-flight bumper-model and timer-gate changes in
  `web/app.py`. Implement against the current file and preserve those paths; do not
  restore an older app snapshot.

## Target Data Ownership

`RobotIdentity` is the durable match object. At minimum it owns:

```python
@dataclass
class RobotIdentity:
    identity_id: int
    alliance: str
    team_number: str | None = None
    team_number_confirmed: bool = False
    team_ocr_votes: dict[str, float] = field(default_factory=dict)
    blue_votes: int = 0
    red_votes: int = 0
    scores: int = 0
    appearance_embedding: np.ndarray | None = None
    anchor_embeddings: list[np.ndarray] = field(default_factory=list)
    appearance_crop_b64: str | None = None
    consecutive_tracker: ConsecutiveValueTracker | None = None
    last_position: tuple[float, float] | None = None       # image coordinates
    field_position: tuple[float, float] | None = None      # field meters
    last_velocity: tuple[float, float] | None = None
    positions_history: deque = field(default_factory=lambda: deque(maxlen=30))
    last_seen_frame: int | None = None
    last_seen_match_sec: float | None = None
    identity_confidence: float = 0.0
    ambiguity_state: str = "clear"  # clear | pending_vlm | unresolved
    last_vlm_result_match_sec: float | None = None
    last_audit_success_match_sec: float | None = None
    last_audit_attempt_match_sec: float | None = None
    pending_read_reasons: set[str] = field(default_factory=set)
```

The registry owns:

- `identities: dict[int, RobotIdentity]`
- `slot_to_identity: dict[int, int]`
- `identity_to_slot: dict[int, int]`
- inactive identities, which remain roster-reserving match state
- creation/re-attachment policy and hard three-per-alliance cap

`slot_id` remains valid for HUD labels (`B1`..`R3`), current crop lookup, and tracker
occlusion input. `identity_id` becomes the key for durable state, scoring events,
diagnostics, VLM cadence, and team assignment.

---

## Diagnostic and Performance Logging Contract

Implement this contract alongside the phases below, not as an afterthought. Logs must be
machine-readable enough to compare runs and concise enough to leave enabled during real
matches.

### Structured event shape

Every diagnostic event includes:

```python
{
    "schema_version": 1,
    "event": "identity.assignment",
    "job_id": "...",
    "source_frame": 1234,
    "processed_frame": 456,
    "match_sec": 37.25,
    "camera": "main",             # nullable
    "slot_id": 1,                 # nullable
    "identity_id": 7,             # nullable
    "fused_track_id": 12,         # nullable
    "reason": "reattach_confident",
    "details": {...},
}
```

Use enums/constants for event and reason names. Do not parse human log strings to
reconstruct decisions.

### Required event families

- `sam2.object_created`, `sam2.object_retired`, `sam2.reprompt_candidate`,
  `sam2.reprompt_rejected`, `sam2.mask_gap`, `sam2.chunk_complete`
- `fusion.assignment`, `fusion.relink`, `fusion.low_margin`,
  `fusion.embedding_update_skipped`
- `tracker.assignment`, `tracker.slot_expired`, `tracker.gate_rejected`
- `identity.created`, `identity.promoted`, `identity.reattached`,
  `identity.detached`, `identity.blocked`, `identity.tombstoned`,
  `identity.assignment_ambiguous`, `identity.binding_committed`
- `identity.confidence_changed`, including component values, prior/new score, and reason
- `occlusion.group_opened`, `occlusion.group_reemerged`,
  `occlusion.group_resolved`, `occlusion.group_unresolved`
- `vlm.scheduled`, `vlm.deferred`, `vlm.crop_unavailable`, `vlm.result`,
  `vlm.audit_overdue`, `vlm.contradiction`
- `scoring.attributed`, `scoring.deferred`, `scoring.backfilled`
- `invariant.violation` with enough state to reproduce the failure

Every rejection/defer event records a reason such as `below_score_floor`,
`below_margin`, `alliance_conflict`, `roster_cap`, `identity_already_active`,
`no_usable_crop`, `vlm_top_k_starvation`, or `unreadable_result`.

### Performance timings

Measure per-frame and aggregate durations with `time.perf_counter()` while scheduling
still uses `match_sec`. Required stages:

- detector inference by model/camera
- SAM2 propagation and re-prompt detector work
- embedding extraction by camera and crop count
- multi-camera fusion
- fused-id assignment
- field tracker update
- occlusion resolution
- identity score-matrix construction
- assignment solver
- confidence update
- VLM candidate/crop preparation
- VLM model inference
- VLM result application
- scoring attribution
- HUD/rendering, websocket serialization, replay/DB writes
- total processed-frame latency

Record count, total, mean, max, and p50/p90/p95/p99 where sample volume permits. Also
record queue/backlog metrics: identities due for reads, overdue audits, pending
ambiguities, deferred scoring events, and VLM candidates dropped by `VLM_TOP_K`.

### Storage and volume control

- Keep a bounded in-memory ring buffer for recent detailed events.
- Persist the full structured log as JSONL in the job/replay artifacts.
- Put aggregate counters/timings in the replay summary for quick comparison.
- Sample routine high-frequency success events, but never sample state transitions,
  ambiguity, errors, invariant failures, audit overdue events, or slow-stage outliers.
- Add configurable slow thresholds per stage; emit `performance.slow_stage` with stage,
  duration, frame, counts, and active configuration.
- Never place crop bytes, full embeddings, API secrets, or unbounded matrices in logs.
  Log embedding dimensions, similarities, score components, and compact candidate ids.
- Logging failures must not fail inference; increment a dropped-log counter.

### Diagnostic tests

Add tests asserting:

- identical saved observations produce identical decision-event sequences
- every binding change has a preceding decision event and reason
- ambiguous observations emit no anchor-update event
- VLM attempt/success/defer/overdue states are distinguishable
- stage timers close and record durations even when the wrapped operation raises
- JSONL events validate against the schema and remain bounded in size
- aggregate counts reconcile with detailed transition events

---

## Phase 0: Temporal Observation Contract

This phase is required. Persistent identity cannot compensate for an observation layer
that silently stops re-prompting or mutates ids order-dependently.

### Task 0A: Add the structured diagnostic recorder

**Files:**
- Create: `web/field/identity_diagnostics.py`
- Create: `tests/field/test_identity_diagnostics.py`
- Modify: `web/replay_recorder.py`
- Modify: `tests/test_replay_recorder.py`

- [ ] **Step 1: Write failing recorder/schema/timer tests**

Test schema validation, enum reason names, bounded ring-buffer behavior, JSONL
serialization, dropped-log accounting, exception-safe stage timers, slow-stage events,
and aggregate p50/p90/p95/p99 summaries.

- [ ] **Step 2: Implement `IdentityDiagnosticRecorder`**

Provide:

- `record(event, **fields)`
- `stage_timer(stage, **context)` context manager
- `increment(counter, amount=1)`
- `gauge(name, value)`
- `summary()`
- `flush_jsonl(path)`

Use a pluggable clock in tests. Normalize NumPy scalar values before serialization.

- [ ] **Step 3: Integrate with replay artifacts**

Store the JSONL path and aggregate diagnostic summary in replay metadata. Ensure
finalization flushes diagnostics in `finally` paths even when inference fails.

- [ ] **Step 4: Run and commit**

```bash
.venv/bin/python -m pytest \
  tests/field/test_identity_diagnostics.py \
  tests/test_replay_recorder.py -v
git add web/field/identity_diagnostics.py web/replay_recorder.py tests/field/test_identity_diagnostics.py tests/test_replay_recorder.py
git commit -m "feat(diagnostics): add structured identity and performance recorder"
```

### Task 0: Fix SAM2 re-prompt lifecycle and expose continuity evidence

**Files:**
- Modify: `web/field/sam2_tracker.py`
- Modify: `web/app.py`
- Modify: `tests/field/test_sam2_tracker.py`
- Modify: `tests/field/test_sam2_streaming.py`

- [ ] **Step 1: Add failing recovery tests**

Cover:

1. Six initial SAM2 objects do not prevent a genuinely new re-prompt id after one object
   has been absent beyond a configurable retirement window.
2. A one-boundary empty mask does not immediately create a duplicate id.
3. Repeated detector boxes for a healthy propagated mask do not create duplicates.
4. A re-prompted id is marked as `new_after_gap`; it is not presented as uninterrupted
   temporal continuity.
5. Object-id creation remains bounded over a full synthetic match with repeated
   occlusion/reappearance.

- [ ] **Step 2: Separate active and lifetime limits**

Replace the single lifetime `max_objects` interpretation with:

- `max_active_objects=6`
- a bounded `max_lifetime_objects` safety limit
- `last_mask_frame`/absence age per SAM2 object
- a retirement window used for active-cap accounting

SAM2 predictor state may retain old ids internally; the wrapper must stop treating every
historical id as an active robot. If the predictor cannot retire objects safely, allow a
bounded number of lifetime ids and rely on the downstream registry to canonicalize them.

- [ ] **Step 3: Strengthen re-prompt dedupe**

Compare new detections against:

- current masks
- recent boxes/masks for temporarily absent objects
- detector color metadata when available

Return continuity evidence per object: `initial`, `propagated`, `reappeared_same_id`,
`reprompted_new_id`, mask-gap length, and detector overlap. This is a confidence input,
not an identity decision.

- [ ] **Step 4: Update app wiring and tests**

Pass active/lifetime limits explicitly. Confirm the six-robot initial prompt still works
and a later replacement id can be introduced without exceeding six active candidates.
Emit the required SAM2 events and stage timings, including rejection reasons and active
vs lifetime object counts at each re-prompt boundary.

- [ ] **Step 5: Run and commit**

```bash
.venv/bin/python -m pytest \
  tests/field/test_sam2_tracker.py \
  tests/field/test_sam2_streaming.py -v
git add web/field/sam2_tracker.py web/app.py tests/field/test_sam2_tracker.py tests/field/test_sam2_streaming.py
git commit -m "fix(sam2): preserve bounded temporal recovery across reprompts"
```

### Task 0B: Make fused-track re-link batch-global and non-poisoning

**Files:**
- Modify: `web/field/fused_track_id.py`
- Modify: `tests/field/test_fused_relink.py`
- Modify: `tests/field/test_fused_track_id.py`

- [ ] **Step 1: Add permutation-invariance tests**

Present two similar inactive fused tracks and two re-emergent observations in both input
orders. Assert identical optimal assignments. Add a low-margin case that creates new
provisional fused ids instead of greedily stealing an old id.

- [ ] **Step 2: Replace greedy inactive matching**

Build one appearance-cost matrix for all unmatched observations vs eligible inactive
fused tracks and solve the complete assignment (exhaustive permutations are acceptable
at six robots). Enforce one-to-one matching and expose best-vs-runner-up global margin.

- [ ] **Step 3: Quarantine uncertain embedding updates**

Do not blend a low-margin observation into an old fused embedding. Update EMA only for
continued/member-key matches or confident batch re-links. Preserve immutable/recent
exemplars so one bad EMA update cannot erase the prior appearance.

Emit complete assignment summaries: candidate fused ids, a compact score matrix,
best/runner-up totals, margin, chosen mapping, and skipped embedding-update reason.

- [ ] **Step 4: Run and commit**

```bash
.venv/bin/python -m pytest \
  tests/field/test_fused_relink.py \
  tests/field/test_fused_track_id.py \
  tests/field/test_tracklet_fusion_to_slot.py -v
git add web/field/fused_track_id.py tests/field/test_fused_relink.py tests/field/test_fused_track_id.py
git commit -m "fix(fusion): make fused re-link global and order-independent"
```

---

## Phase 1: Persistent Identity

### Task 1: Add the pure identity model and registry

**Files:**
- Create: `web/field/identity.py`
- Create: `tests/field/test_identity.py`

- [ ] **Step 1: Write failing tests for lifecycle and invariants**

Cover:

1. Creating a detection creates one identity and binds it to a slot.
2. Detaching an expired slot leaves the identity and all votes/team fields intact.
3. Re-attaching the same robot to another free slot returns the original identity.
4. One identity cannot bind to two slots; one slot cannot bind to two identities.
5. Confirmed teams remain reserved while identities are inactive.
6. A fourth identity for one alliance is rejected or forced through re-attachment.
7. Permuting observation order produces the same identity assignment.
8. A low-margin observation does not alter an identity's embedding or anchors.
9. Binding updates are transactional: a raised exception leaves both binding maps valid.
10. A provisional false track can be tombstoned/replaced without deleting a real
    confirmed identity.

Use synthetic normalized embeddings and field positions. Do not import `web.app`.

- [ ] **Step 2: Run the tests and verify the module is missing**

```bash
.venv/bin/python -m pytest tests/field/test_identity.py -v
```

- [ ] **Step 3: Implement registry types and explicit result objects**

Add:

- `RobotIdentity`
- `IdentityObservation` containing `slot_id`, `alliance`, `field_pos`, `velocity`,
  `embedding`, `gallery_scores`, `fused_track_id`, and timestamp/frame
- `IdentityMatch` containing `identity_id`, combined score, component scores,
  runner-up score, margin, and action (`keep`, `reattach`, `create`, `blocked`)
- `IdentityRegistry.bind_active(...)`
- `IdentityRegistry.detach_slot(...)`
- `IdentityRegistry.associate_frame(...)`
- `IdentityRegistry.identity_for_slot(...)`
- `IdentityRegistry.slot_for_identity(...)`

Keep the registry storage-agnostic and independent of VLM/model imports.

- [ ] **Step 4: Implement batch-global identity association**

For each alliance/frame, build a complete observation-to-identity score matrix using
configurable normalized components:

- appearance: best cosine against the identity EMA, anchors, and team gallery exemplars
- trajectory: distance from predicted position divided by a reachable-distance envelope
- roster: hard reject alliance mismatch and confirmed-team collision; positive weight
  when a confirmed inactive identity is the only identity for that roster team
- temporal continuity: uninterrupted SAM2/fused membership is useful evidence, while a
  re-prompted id after a gap is not treated as identity proof
- tracker assignment: slot/sticky assignment is a prior, not a hard identity constraint

Keep appearance evidence camera-aware. Prefer comparisons against anchors from the same
camera/viewpoint; combine per-camera scores only after normalization. Do not average
embeddings from different simultaneous camera crops into one vector without testing that
behavior. Trajectory plausibility must use elapsed source time, a configurable physical
speed/acceleration envelope, field boundaries, and uncertainty growth; a long gap should
be weak evidence rather than an automatic rejection.

Roster evidence is a constraint/tiebreaker, not independent proof of physical identity.
Cap its positive contribution so a wrong confirmed lock cannot make its own attachment
maximally confident through circular reasoning.

Solve the complete one-to-one assignment, including `create provisional` and `unassigned`
options. Compare the best complete assignment against the runner-up complete assignment.
Require both a minimum score and global margin before normal re-attachment. Return all
component scores for diagnostics and Phase 2. Add property-style tests over every
permutation of up to three observations.

Emit one `identity.assignment` decision event before applying a binding transaction and
one `identity.binding_committed` event afterward. Include candidate ids, component
scores, global totals/margin, action, and threshold/config values.

- [ ] **Step 5: Enforce the persistent roster cap**

Count all active and inactive identities for the alliance. At three:

- never create a fourth identity
- prefer the best inactive candidate even if it only clears a lower recovery threshold
- return `blocked` when no defensible candidate exists, so integration can hold/drop the
  observation without contaminating an existing identity

Distinguish real persistent identities from erroneous provisional candidates:

- newly created, unconfirmed identities begin `provisional`
- a provisional identity becomes durable after continuity/evidence thresholds or a
  confirmed team read
- at the cap, an inactive low-evidence provisional identity may be tombstoned and
  replaced atomically
- a confirmed identity is never tombstoned automatically
- tombstones remain in diagnostics/binding history but do not reserve roster capacity

- [ ] **Step 6: Run tests**

```bash
.venv/bin/python -m pytest tests/field/test_identity.py -v
```

- [ ] **Step 7: Commit**

```bash
git add web/field/identity.py tests/field/test_identity.py
git commit -m "feat(identity): add persistent robot identity registry"
```

### Task 2: Make tracker binding transitions observable

**Files:**
- Modify: `web/field/tracker.py`
- Modify: `tests/field/test_tracker_trackid.py`
- Modify: `tests/field/test_tracker_assignment.py`

- [ ] **Step 1: Add failing tests for expiration events**

Test that after `max_coast_*` is exceeded:

- the slot becomes inactive
- every fused-id mapping that points to that slot is removed
- the update exposes a `slot_expired` event with slot, alliance, last position,
  velocity, and former fused id

Also test `detection_assigned` evidence includes assignment source
(`sticky`, `distance`, `seed`) and best/runner-up costs where available.

- [ ] **Step 2: Add tracker event/result dataclasses**

Preserve the existing `update(...) -> list[TrackOut]` contract to limit blast radius.
Publish per-update metadata on explicit fields such as:

```python
self.binding_events: list[BindingEvent]
self.assignment_evidence: dict[int, AssignmentEvidence]  # detection index -> evidence
```

Do not make the identity registry a dependency of `FieldTracker`.

- [ ] **Step 3: Remove all stale fused-id mappings on expiry**

The current expiry clears the slot but leaves `_id_to_slot` entries until a matching id
returns. Delete mappings pointing to the expired slot immediately and include their ids
in the expiration event.

- [ ] **Step 4: Expose assignment margins**

For distance matching, record the chosen cost and the best alternative finite cost. For
sticky assignment, mark the margin as decisive unless the sticky id was recovered after
expiry. For seed assignment, leave the margin unknown rather than inventing certainty.

- [ ] **Step 5: Run tracker tests**

```bash
.venv/bin/python -m pytest \
  tests/field/test_tracker_trackid.py \
  tests/field/test_tracker_assignment.py \
  tests/field/test_tracklet_fusion_to_slot.py -v
```

- [ ] **Step 6: Commit**

```bash
git add web/field/tracker.py tests/field/test_tracker_trackid.py tests/field/test_tracker_assignment.py
git commit -m "feat(tracker): expose slot lifecycle and assignment evidence"
```

### Task 3: Feed fused re-link evidence into identity matching

**Files:**
- Modify: `web/field/fused_track_id.py`
- Modify: `tests/field/test_fused_relink.py`
- Modify: `tests/field/test_fused_track_id.py`

- [ ] **Step 1: Add failing tests for explicit re-link results**

Verify each fused observation exposes:

- whether the id was continued, appearance-relinked, merged, or newly created
- best and second-best appearance similarities
- the appearance margin
- the prior fused id when ids collapse

- [ ] **Step 2: Add a pure `FusedAssignmentEvidence` result**

Keep stamping `obs.track_id` for compatibility, but store evidence by observation index
or return it from `assign()`. Identity matching must not inspect private
`_id_to_embedding` or infer confidence from ids alone.

- [ ] **Step 3: Preserve embeddings when fused ids collapse**

Merge the winner/loser appearance state instead of simply dropping the loser embedding.
This avoids discarding a useful anchor during a cross-camera bridge.

This task consumes the batch-global fused assignment added in Phase 0; do not reintroduce
per-observation greedy matching here.

- [ ] **Step 4: Run fused-id tests**

```bash
.venv/bin/python -m pytest \
  tests/field/test_fused_relink.py \
  tests/field/test_fused_track_id.py \
  tests/field/test_tracklet_fusion_to_slot.py -v
```

- [ ] **Step 5: Commit**

```bash
git add web/field/fused_track_id.py tests/field/test_fused_relink.py tests/field/test_fused_track_id.py
git commit -m "feat(identity): expose fused track re-link evidence"
```

### Task 4: Integrate identity bindings into `InferenceJob`

**Files:**
- Modify: `web/app.py`
- Create: `tests/test_identity_integration.py`
- Modify: focused tests that construct `RobotTrack` directly

- [ ] **Step 1: Add a failing long-occlusion integration test**

Build an `InferenceJob` with `__new__` to avoid model loading. Simulate:

1. slot 0 bound to identity A
2. A gains a confirmed team and votes
3. slot 0 expires after more than 90 occluded frames
4. A reappears and tracker seeds slot 1
5. the registry reattaches identity A to slot 1

Assert the team, votes, confirmation, score, and identity id survive, and slot 1's old
identity state is not inherited.

Also cover:

- the early OCR pass creates/binds bootstrap identities that the first SAM2 frame adopts
  instead of duplicating
- `use_sam2=False` still creates persistent identities using tracker/trajectory/color
  evidence, with appropriately lower confidence
- an auto-to-teleop processing gap preserves identity but does not count as uninterrupted
  continuity or extrapolate a precise trajectory across the gap

- [ ] **Step 2: Add registry state to `InferenceJob`**

Initialize:

```python
self.identity_registry = IdentityRegistry(...)
self.robot_identities = self.identity_registry.identities
```

During migration, retain `robot_tracks` only as a compatibility alias if needed by
small helper tests. Do not maintain two independently mutable copies of durable state.

- [ ] **Step 3: Replace `_get_or_create_slot_track`**

Introduce helpers with explicit semantics:

- `_identity_for_slot(slot_id)`
- `_bind_detection_identity(slot_id, fused_obs, tracker_evidence, match_sec)`
- `_detach_expired_slots()`
- `_visible_identities(track_outs)`

Process tracker expiration events before assigning current observations. Use
`field_tracker.last_det_to_slot` to pair each fused observation with its assigned slot,
then call `IdentityRegistry.associate_frame(...)` once per alliance. Apply the returned
binding transaction atomically. Never call identity matching once per observation.

Add a bootstrap reconciliation step for identities created by the early OCR frame before
`FusedTrackRegistry` has produced ids. First-frame SAM2 observations must batch-match
those identities using slot, alliance, position, and any available appearance.

- [ ] **Step 4: Migrate durable readers/writers to identities**

Update these paths to iterate identities or resolve slot to identity:

- remaining roster calculation and team-vote assignment
- gallery vote/deposit and consecutive lock
- contradiction and overturn handling
- scoring attribution and score totals
- diagnostic snapshots
- visible HUD labels and websocket robot payloads
- replay event labels
- autonomous path keys

Every new `ScoringEvent` stores `identity_id` as `robot_id` plus the current `slot_id`
separately. Historical attribution then follows the physical identity even if its slot
changes later.

Also migrate `BallTrajectory.shooter_robot_id`: it must store `identity_id`, not the slot
that happened to be nearest when the ball was observed. Add a regression where the
shooter changes slots before the score is finalized.

- [ ] **Step 5: Re-key crop/source state**

Store review crops and confirmation source by `identity_id`, while including the current
slot label as presentation metadata. A crop collected before slot expiry must remain
associated with the same identity after re-attachment.

- [ ] **Step 6: Replace slot-keyed persistence with identity-keyed persistence**

An additive `identity_id` column is not enough because the current primary key is
`(job_id, slot)`. Implement an explicit SQLite migration:

- rebuild `match_robot_identity` with primary key `(job_id, identity_id)`
- store `current_slot INTEGER NULL`, `slot_label`, identity status, confidence, and team
- add `match_robot_binding(job_id, identity_id, slot, start_frame, end_frame, reason,
  score, margin)` for slot history
- rebuild `match_robot_crop` to include `identity_id`; retain slot-at-capture as metadata
- migrate legacy rows with `identity_id = slot` and `current_slot = slot`
- update queries to order by alliance/team/current slot rather than identity id alone
- add migration idempotence and legacy-data preservation tests

Update correction APIs and frontend payloads to use `identity_id`. Accept legacy
`slot`-only corrections only when that slot resolves to exactly one identity for the
job; otherwise return a conflict instead of editing the wrong robot.

- [ ] **Step 6B: Define restart/resume behavior**

Inference currently keeps live identity state in memory. Explicitly choose and test one:

- jobs are non-resumable and partial identity state is discarded/recomputed, or
- checkpoint the registry, bindings, pending ambiguity, audit timestamps, and confidence
  using source frame as the recovery point

Do not accidentally reload final review rows as live tracker state.

- [ ] **Step 7: Delete slot-keyed durable object creation**

Once all call sites resolve through the registry, remove `RobotTrack` or rename/move it
to `RobotIdentity`. No code should create durable team state merely because a slot
became visible.

- [ ] **Step 7B: Add runtime invariant validation**

At the end of each processed frame in tests/diagnostic mode, validate:

- binding maps are exact inverses
- no slot or identity appears twice
- confirmed teams are unique
- alliance/team roster consistency holds
- ambiguous observations have not updated identity anchors
- scoring/VLM/crop references point to existing identity ids

Fail tests immediately on violation; production diagnostics should record and quarantine
the affected assignment rather than continuing with corrupt state.

Include a compact snapshot of both binding maps, confirmed-team ownership, ambiguity
event ids, and the operation that triggered the violation.

- [ ] **Step 8: Run focused integration tests**

```bash
.venv/bin/python -m pytest \
  tests/test_identity_integration.py \
  tests/test_alliance_consistency.py \
  tests/test_reid_assignment.py \
  tests/test_reid_identity.py \
  tests/test_gallery_match_logic.py \
  tests/test_gallery_manager_api.py \
  tests/test_replay_db.py \
  tests/test_gallery_correction_db.py \
  tests/test_correction_api.py \
  tests/test_slot_crop_capture.py \
  tests/test_replay_recorder.py -v
```

- [ ] **Step 9: Commit**

```bash
git add web/app.py web/database.py tests
git commit -m "feat(identity): bind durable robot state above tracker slots"
```

### Task 5: Phase 1 regression and acceptance gate

- [ ] **Step 1: Run the complete field/identity suite**

```bash
.venv/bin/python -m pytest tests/field tests/test_reid_*.py tests/test_alliance_consistency.py -v
```

- [ ] **Step 2: Run the complete test suite**

```bash
.venv/bin/python -m pytest
```

- [ ] **Step 3: Acceptance check**

Phase 1 is shippable only when:

- a greater-than-90-frame occlusion cannot transfer one robot's team/votes to another
- inactive confirmed identities still reserve their roster teams
- no alliance can accumulate a fourth persistent identity
- inconclusive matching blocks/defers instead of cross-contaminating identity state

---

## Phase 2: Confidence-Driven VLM

### Task 6: Add pure confidence calculation

**Files:**
- Create: `web/field/identity_confidence.py`
- Create: `tests/field/test_identity_confidence.py`

- [ ] **Step 1: Write failing confidence transition tests**

Cover:

- clean continuous tracking raises confidence gradually
- missed/occluded frames decay it
- post-occlusion re-emergence starts below the prior confidence
- high appearance agreement raises it; divergence lowers it
- a low assignment margin lowers it
- roster uniqueness contributes only when the roster state is valid
- matching VLM raises confidence
- a strong VLM contradiction drops confidence sharply and emits an immediate-read reason
- output is always clamped to `[0, 1]`
- missing/unknown signals do not accidentally behave like strong negative evidence
- threshold hysteresis prevents read/no-read oscillation on adjacent frames

- [ ] **Step 2: Implement immutable input/output dataclasses**

Use `ConfidenceSignals`, `ConfidenceState`, and `ConfidenceUpdate`. The pure update
returns the new score plus reason flags; it must not mutate `InferenceJob`.

Document that this is an operational score, not a calibrated probability. Keep component
values and reason flags in diagnostics so threshold tuning is explainable. Apply
hysteresis/debounce to confidence-drop events so one noisy embedding does not trigger a
read every frame, while hard events (re-attachment, contradiction, ambiguity) bypass the
debounce.

- [ ] **Step 3: Define named thresholds/config**

Add environment-backed app constants only at integration time. The pure module accepts a
`ConfidenceConfig` so tests do not depend on environment variables.

- [ ] **Step 4: Run tests and commit**

```bash
.venv/bin/python -m pytest tests/field/test_identity_confidence.py -v
git add web/field/identity_confidence.py tests/field/test_identity_confidence.py
git commit -m "feat(identity): calculate per-robot identity confidence"
```

### Task 7: Replace pass-count audits with per-identity match-clock scheduling

**Files:**
- Modify: `web/app.py`
- Replace/modify: `tests/test_reread_audit_deadlock.py`
- Create: `tests/test_identity_vlm_schedule.py`

- [ ] **Step 1: Write the match-clock audit regression tests**

Use explicit source-video/match elapsed seconds. Verify:

1. every confirmed identity becomes audit-due at `AUDIT_FLOOR_SECONDS`
2. it is due even while another identity remains perpetually unconfirmed
3. a confidence of `1.0` cannot delay the audit past the floor
4. normal VLM pass frequency has no effect on the audit deadline
5. processing the same frames at 0.5x, 1x, or 10x wall-clock speed gives the same reads
6. a usable audit result updates `last_audit_success_match_sec`
7. failed/skipped/unreadable crops do not falsely satisfy the audit
8. an unavailable overdue identity stays due but respects a short attempt backoff
9. pause/resume and skipped pre-match/inter-period ranges do not age the audit clock

- [ ] **Step 2: Add a pure schedule decision**

Create a helper returning:

```python
ReadDecision(
    due: bool,
    reason: str,       # seed | low_confidence | confidence_drop | audit | none
    priority: float,
    is_audit: bool,
)
```

Inputs include identity confirmation, confidence, last successful VLM/audit timestamps,
last attempt timestamp, pending reason flags, crop availability, and
`current_match_sec`.

- [ ] **Step 3: Replace global pass-count selection**

Remove `REREAD_EVERY_N_PASSES` and `_is_reread_pass()` after all callers migrate.
Keep the unconfirmed fast/medium/slow intervals as a seed retry policy, but evaluate
confirmed identities independently. Set:

```python
AUDIT_FLOOR_SECONDS = float(os.environ.get("AUDIT_FLOOR_SECONDS", "30"))
```

The floor means maximum allowed time between successful audit reads, not a target global
batch interval. `current_match_sec` is derived from source frame/video FPS within active
match periods. Do not use `time.time()` or processing duration.

- [ ] **Step 4: Make fused VLM reads targeted**

Change `read_fused_bumper_numbers` to accept requested identity ids/reasons. Resolve each
identity's current slot and available camera crops, rank only those candidates, and keep
the `VLM_TOP_K` batch cap.

Unconfirmed identities may still be batched. A confidence-drop or overdue-audit identity
must outrank routine retries.

Define fair scheduling when more candidates are due than `VLM_TOP_K`: oldest overdue
audit first, then ambiguity/contradiction, then low confidence, then routine seeding.
Carry unsent obligations forward. Add starvation tests.

- [ ] **Step 5: Preserve contradiction behavior**

The existing direct contradiction and vote-margin overturn paths remain active. Feed
their result into confidence:

- agreement: raise confidence and clear the triggering reason
- contradiction: unlock as today, sharply lower confidence, and retain a pending read
- unreadable/error: leave the audit due

Track attempt and success separately. An unreadable attempt updates
`last_audit_attempt_match_sec` for retry backoff but never updates the success timestamp.
An identity with no usable crop remains visibly overdue in status/diagnostics.

Audit reads must match against the full alliance roster, including the identity's
currently locked team and teams locked to other identities. Using only the "remaining"
roster would make contradictions impossible to observe.

- [ ] **Step 5B: Distinguish label contradiction from binding contradiction**

A crop reading another identity's confirmed team is not just a vote update. It can mean
the physical identities were swapped. Add a conflict resolver:

- freeze the affected bindings and lower confidence
- gather the conflicting identities/observations
- evaluate an atomic re-assignment before changing team ownership
- preserve both identities' vote/score histories
- only unlock/reset a label directly when evidence indicates the label itself was wrong
- if evidence is insufficient, mark the group ambiguous rather than moving a confirmed
  team between identities

Add tests for both cases: wrong OCR lock on the correct physical identity, and correct
team locks attached to swapped physical observations.

- [ ] **Step 6: Run schedule tests**

```bash
.venv/bin/python -m pytest \
  tests/test_reread_audit_deadlock.py \
  tests/test_identity_vlm_schedule.py \
  tests/test_bumper_vlm_crop_mode.py -v
```

- [ ] **Step 7: Commit**

```bash
git add web/app.py tests/test_reread_audit_deadlock.py tests/test_identity_vlm_schedule.py
git commit -m "feat(vlm): schedule targeted reads from identity confidence"
```

### Task 8: Wire per-frame confidence signals

**Files:**
- Modify: `web/app.py`
- Modify: `web/field/identity.py`
- Modify: `tests/test_identity_integration.py`

- [ ] **Step 1: Build signals after tracker/identity assignment**

For every active identity, derive:

- continuity/missed state from `TrackOut` and binding events
- appearance agreement from current observation vs identity anchors/gallery
- assignment margin from tracker and fused assignment evidence
- roster uniqueness from the persistent registry
- VLM signal from the latest completed read

For inactive identities, apply time-based occlusion decay without requiring a visible
crop.

Update appearance anchors only after the frame's assignment is committed and confidence
clears an anchor-update threshold. Keep at least one immutable high-confidence anchor;
EMA drift alone must never replace all historical evidence.

- [ ] **Step 2: Queue immediate read reasons**

Add reasons on:

- re-attachment after occlusion
- low assignment margin
- appearance divergence
- roster conflict
- swap suspicion
- VLM contradiction

Reasons remain pending until a usable VLM result is obtained or the identity is detached
again. Do not clear them because a global VLM pass happened.

- [ ] **Step 3: Add diagnostics/status output**

Expose current confidence, pending reasons, last read time, last audit time, and next
audit deadline per identity in diagnostic/replay payloads. Keep the UI additive; do not
make frontend rendering a prerequisite for scheduler correctness.

Log every confidence change that crosses a scheduling or anchor-update threshold, plus
sampled routine updates. Include each component, missing-signal flags, hysteresis state,
and the exact pending reasons added or cleared.

- [ ] **Step 4: Run Phase 2 tests and full suite**

```bash
.venv/bin/python -m pytest \
  tests/field/test_identity_confidence.py \
  tests/test_identity_integration.py \
  tests/test_identity_vlm_schedule.py \
  tests/test_reread_audit_deadlock.py -v
.venv/bin/python -m pytest
```

- [ ] **Step 5: Commit**

```bash
git add web/app.py web/field/identity.py tests
git commit -m "feat(identity): drive VLM reads from live confidence signals"
```

---

## Phase 3: Joint Same-Alliance Occlusion Tiebreak

### Task 9: Add pure co-occlusion event tracking and assignment comparison

**Files:**
- Create: `web/field/identity_ambiguity.py`
- Create: `tests/field/test_identity_ambiguity.py`

- [ ] **Step 1: Write failing event tests**

Cover:

- two same-alliance identities becoming occluded in the same window creates one event
- two same-alliance identities disappearing/missing together creates an event even when
  geometric occlusion classification is unavailable or wrong
- unrelated single occlusions do not
- re-emergence outside the configured window expires the event
- opposite-alliance identities are never grouped
- resolved events cannot trigger twice

- [ ] **Step 2: Write failing assignment-margin tests**

For two identities and two observations:

- separable appearance/trajectory yields a decisive direct assignment
- the swapped assignment within `IDENTITY_AMBIGUITY_MARGIN` yields `ambiguous`
- roster/alliance-invalid assignments are rejected
- three-identity events compare the best two complete permutations, not pairwise greedy
  choices

- [ ] **Step 3: Implement pure event and resolver types**

Add:

- `CoOcclusionEvent`
- `CoOcclusionTracker`
- `AssignmentCandidate`
- `AmbiguityResolution`
- exhaustive permutation scoring for at most three same-alliance identities

Return the best assignment, runner-up assignment, margin, and component breakdown.

- [ ] **Step 4: Run tests and commit**

```bash
.venv/bin/python -m pytest tests/field/test_identity_ambiguity.py -v
git add web/field/identity_ambiguity.py tests/field/test_identity_ambiguity.py
git commit -m "feat(identity): detect and score joint occlusion ambiguity"
```

### Task 10: Integrate ambiguity handling and targeted tiebreak reads

**Files:**
- Modify: `web/app.py`
- Modify: `web/field/identity.py`
- Create: `tests/test_joint_occlusion_integration.py`

- [ ] **Step 1: Add three end-to-end synthetic tests**

1. Two blue identities co-occlude, re-emerge with separable evidence, and bind without
   VLM.
2. Evidence is inseparable, so both identities get immediate targeted VLM requests; VLM
   team results resolve the permutation correctly.
3. Evidence is inseparable and crops are unreadable, so both remain ambiguous and no
   identity/team state is swapped.

- [ ] **Step 2: Feed merged occlusion state into the event tracker**

Translate current slot occlusion to identity ids before updating the co-occlusion
tracker. Combine geometric occlusion with simultaneous missed-mask/detection transitions;
the hard case often has no reliable current box with which to prove geometry. Detaching
a slot must not lose membership in an open event.

- [ ] **Step 3: Resolve the group before committing bindings**

When two or three observations reappear for an open event:

- build a group assignment matrix from trajectory and appearance
- commit immediately only when the best permutation clears the ambiguity margin
- otherwise mark all participants `pending_vlm` and queue targeted reads

Do not let normal first-free-slot seeding commit durable identity assignments first and
attempt to repair them afterward.

Capture the candidate observation crops and their observation ids at this point. The VLM
result must map candidate observation -> roster team -> identity permutation; it must not
read through a slot binding that has already changed.

- [ ] **Step 4: Resolve from VLM results**

Use roster team numbers as the tiebreak only when reads are valid and one-to-one. Apply
the permutation atomically. If results are duplicated, contradictory, or unreadable,
leave the event unresolved.

Require all identities in a VLM tiebreak to have confirmed, distinct roster teams. If an
identity is still unconfirmed, VLM results may improve evidence but cannot claim that the
permutation is resolved solely through roster exclusion.

- [ ] **Step 5: Define scoring behavior while unresolved**

During ambiguity:

- retain alliance-level score attribution
- do not increment an individual identity's score from proximity/trajectory evidence
  that cannot distinguish the participants
- record a deferred attribution with candidate identity ids
- once resolved, backfill the team/identity when evidence supports one candidate;
  otherwise keep the event alliance-only

Store deferred attribution records by immutable event id and candidate identity ids.
Test resolution after multiple slot changes and test that unresolved records survive
finalization/replay serialization.

- [ ] **Step 6: Add HUD/diagnostic state**

Show `AMBIGUOUS` for participating slots and include event id, candidate identities, and
assignment margin in diagnostics. Do not display a guessed team number as confirmed.

Persist the full ambiguity lifecycle in structured events so a replay can show why the
group opened, every attempted assignment/VLM result, and why it resolved or remained
unresolved.

- [ ] **Step 7: Run tests**

```bash
.venv/bin/python -m pytest \
  tests/field/test_identity_ambiguity.py \
  tests/test_joint_occlusion_integration.py \
  tests/test_identity_integration.py \
  tests/test_identity_vlm_schedule.py -v
.venv/bin/python -m pytest
```

- [ ] **Step 8: Commit**

```bash
git add web/app.py web/field/identity.py tests/test_joint_occlusion_integration.py
git commit -m "feat(identity): resolve same-alliance co-occlusion with targeted VLM"
```

---

## Final Verification

### Task 11: Real-match comparison and rollout gate

- [ ] **Step 1: Add counters to the replay/diagnostic summary**

Record:

- identities created, reattached, blocked, and left ambiguous
- slot expirations
- VLM reads by reason (`seed`, `low_confidence`, `confidence_drop`, `audit`, `tiebreak`)
- successful audit interval per confirmed identity
- identity binding changes and their score/margin
- per-stage latency percentiles and slow-stage counts
- SAM2 active/lifetime ids, re-prompt acceptance/rejection reasons, and mask-gap lengths
- assignment score/margin distributions and threshold rejection counts
- VLM queue depth, starvation/defer counts, unavailable-crop counts, and overdue duration
- dropped diagnostic event count and structured-log artifact path

- [ ] **Step 2: Run a known-occlusion match before and after**

Use the same video, calibration, roster, model weights, frame skip, and SAM2 settings.
Create a small ground-truth sidecar with physical robot labels at frames before and after
each selected occlusion/crossing; visual impression alone is not an acceptance metric.
Verify:

- the physical robot keeps one `identity_id` across the occlusion
- slot changes, if any, do not transfer votes or scores
- every eligible visible confirmed identity is attempted by `AUDIT_FLOOR_SECONDS`
- failed/unavailable audits remain overdue and their overdue duration is reported
- calm confirmed robots generate materially fewer VLM reads
- the known same-alliance ambiguity either resolves correctly or remains explicitly
  unresolved; it never silently swaps identities

- [ ] **Step 2B: Run deterministic replay and fault injection**

Run the same saved detections/embeddings twice and require identical identity/binding
history. Then inject:

- shuffled observation order
- dropped masks/detections
- new SAM2 ids after occlusion
- one corrupted appearance embedding
- one wrong high-confidence VLM read
- unreadable crops across an audit deadline
- process pause/resume and auto-to-teleop gap

Assert invariants hold and no durable state crosses identities.

- [ ] **Step 2C: Validate on more than one clip**

Tune on one development clip, then hold thresholds fixed and evaluate at least:

- one calm/well-separated match
- one hub-scrum match
- one match with camera dropout or poor bumper visibility

Report identity switches per physical robot, unresolved ambiguity duration, false
re-attachments, VLM reads per robot-minute, and overdue-audit duration. Do not tune on
the final validation clips.

Also compare stage p50/p95/p99 and total runtime before/after. Any material slowdown must
be attributable to a named stage and justified against the identity-quality gain.

- [ ] **Step 3: Run all tests**

```bash
.venv/bin/python -m pytest
```

- [ ] **Step 4: Document tuned thresholds**

Record the selected defaults and real-match evidence for:

- inactive match minimum score
- recovery threshold at roster cap
- re-attachment margin
- confidence decay/rise rates
- low-confidence immediate-read threshold
- `AUDIT_FLOOR_SECONDS`
- `IDENTITY_AMBIGUITY_MARGIN`
- co-occlusion/re-emergence window
- SAM2 absence-retirement window and lifetime object cap
- anchor update threshold and confidence hysteresis

- [ ] **Step 5: Final commit**

```bash
git add docs web tests
git commit -m "docs(identity): record persistence and confidence rollout results"
```

## Rollback Boundaries

- Phase 1 can ship with the existing VLM cadence; identity persistence is independently
  useful.
- Phase 2 can be disabled behind a scheduler flag while retaining the identity registry.
  The fallback must still use the new match-clock audit floor, not restore pass counting.
- Phase 3 can be disabled by treating ambiguous group assignments as unresolved. Do not
  fall back to committing the best low-margin guess.

## Definition of Done

- Durable robot state is keyed by `identity_id`, never by a reassignable slot.
- Slot expiry detaches visibility without deleting or transferring identity state.
- Persistent alliance roster caps include inactive identities.
- Confidence-triggered reads are per identity and targeted.
- Every eligible visible confirmed identity is attempted by
  `AUDIT_FLOOR_SECONDS`; unsuccessful/unavailable audits remain overdue until a usable
  result is obtained, independent of global pass cadence.
- Same-alliance joint occlusion either resolves with decisive evidence/VLM or remains
  explicitly ambiguous.
- Association and output are deterministic under observation-order permutation.
- SAM2 re-prompt recovery still works after six initial objects and remains bounded over
  a full match.
- Appearance anchors, votes, scores, crops, and corrections cannot be updated through an
  uncommitted ambiguous binding.
- Every binding/ambiguity/VLM transition has a structured reason event, and replay
  artifacts contain bounded JSONL diagnostics plus aggregate latency/counter summaries.
- Slowdowns and backlogs are attributable to named stages with p50/p90/p95/p99 metrics.
- Existing contradiction/overturn, gallery, replay, correction, and scoring paths pass
  their regression tests.
