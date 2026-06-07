# Staged Track-Identity Persistence & Whole-Robot Gap-Fill Master Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Combine track-identity persistence, confidence-driven VLM scheduling, joint occlusion resolution, and whole-robot gap-fill into a single, unified execution plan. This avoids redundant modifications to overlapping core files (`web/app.py`, `web/field/sam2_tracker.py`, etc.) and ensures that new "unknown" color detections introduced by the gap-fill pass are correctly handled by the persistent identity registry.

---

## Architecture Overview

1. **Detection & Seeding Gap-Fill:** The bumper model provides high-trust alliance colors. The whole-robot model (hub_zone) fills gaps for bumper-occluded robots, seeding "unknown" color detections.
2. **SAM2 Tracking & Evidence:** SAM2 tracklet generation is updated to respect active/lifetime object caps, manage mask gaps, and output rich temporal-continuity evidence.
3. **Multi-Camera Fusion:** Fused track relinking is upgraded to be batch-global and non-greedy, preventing evidence poisoning and exposing relink margins.
4. **Identity Registry:** A pure `IdentityRegistry` sits above tracker slots, mapping slots to durable `RobotIdentity` records. It performs atomic, batch-global association and caps identities per alliance.
5. **Confidence-Driven VLM:** A confidence module calculates active scores, driving targeted VLM audits based on match-clock elapsed seconds (replacing pass-count loops).
6. **Ambiguity Resolution:** A joint co-occlusion tracker identifies same-alliance ambiguity groups, holding updates and triggering VLM-based tiebreak queries before committing bindings.

---

## Non-Negotiable Invariants

1. **No silent identity transfer:** assigning a slot never copies, swaps, or resets durable state. It only changes a binding.
2. **Atomic association:** all observations for one alliance/frame are associated as one constrained assignment. Input iteration order cannot change the result.
3. **No evidence poisoning:** an observation may update an identity's EMA/anchors only after the assignment clears the confidence threshold. Ambiguous observations remain quarantined and cannot modify durable appearance state.
4. **Confirmed identity protection:** a low-margin tracker/appearance assignment cannot move a confirmed team identity. It must become ambiguous and request verification.
5. **Roster uniqueness:** one roster team can belong to at most one live identity, and inactive confirmed identities continue to reserve it.
6. **Alliance consistency:** an identity's alliance is fixed once supported by a confirmed roster team. A contradictory slot/color observation lowers confidence; it does not silently flip the identity.
7. **Temporal-source correctness:** identity cadence uses source-video/match elapsed seconds, not processing wall time. Results must be invariant to inference speed, pause/resume, and skipped non-match ranges.
8. **Audit liveness:** after the floor expires, an eligible visible identity remains audit-due until a usable result is obtained. Occlusion/unreadability may prevent a successful audit, but cannot mark the obligation satisfied.
9. **Atomic ambiguity resolution:** a group assignment is committed all at once or not at all. Partial VLM results cannot move one participant and strand the other.
10. **Stable external identity:** persistence, corrections, replay, scoring, and crops use `identity_id`. Slot is nullable presentation/history metadata.
11. **Explainable operation:** every identity-binding change, ambiguity transition, VLM scheduling decision, and dropped observation records a structured reason and its relevant evidence. Performance timings use stable stage names and cannot silently disappear on exceptions.

---

## File Structure & Touchpoints

| File | Primary Plan Responsibility |
| :--- | :--- |
| `web/field/identity_diagnostics.py` | (New) Pure structured diagnostic logging |
| `web/field/detect_schema.py` | Pure box IoS, merge, and gap-fill helpers |
| `web/field/detect_bridge.py` | Reusable size filters and image-fraction checks |
| `web/field/sam2_tracker.py` | Active/lifetime caps, re-prompt deduplication, continuity evidence |
| `web/field/fused_track_id.py` | Batch-global relink solver, non-greedy matching |
| `web/field/identity.py` | (New) Pure registry, durable `RobotIdentity`, association solver |
| `web/field/tracker.py` | Expose slot expiration events and assignment evidence |
| `web/field/identity_confidence.py` | (New) Pure operational confidence and hysteresis |
| `web/field/identity_ambiguity.py` | (New) Co-occlusion tracking and group ambiguity evaluation |
| `web/app.py` | Orchestration, gap-fill wrapper/hooks, VLM scheduling, database migrations |
| `web/database.py` | Migrate robot tables to be identity-keyed |

---

## Phase 0: Pure Utilities & Seams

### Task 0A: Add the structured diagnostic recorder
- **Files:**
  - Create: [identity_diagnostics.py](file:///home/nick/frc-vision-scorer/web/field/identity_diagnostics.py)
  - Create: [test_identity_diagnostics.py](file:///home/nick/frc-vision-scorer/tests/field/test_identity_diagnostics.py)
  - Modify: [replay_recorder.py](file:///home/nick/frc-vision-scorer/web/replay_recorder.py)
  - Modify: [test_replay_recorder.py](file:///home/nick/frc-vision-scorer/tests/test_replay_recorder.py)

- [ ] **Step 1: Write failing recorder/schema/timer tests**
  Test schema validation, enum reason names, bounded ring-buffer behavior, JSONL serialization, dropped-log accounting, exception-safe stage timers, slow-stage events, and aggregate p50/p90/p95/p99 summaries.
- [ ] **Step 2: Implement `IdentityDiagnosticRecorder`**
  Provide `record(event, **fields)`, `stage_timer(stage, **context)`, `increment(counter, amount=1)`, `gauge(name, value)`, `summary()`, and `flush_jsonl(path)`. Normalize NumPy scalars before serialization.
- [ ] **Step 3: Integrate with replay artifacts**
  Store the JSONL path and aggregate diagnostic summary in replay metadata. Ensure finalization flushes diagnostics in `finally` paths even when inference fails.
- [ ] **Step 4: Run tests and commit**
  ```bash
  .venv/bin/python -m pytest \
    tests/field/test_identity_diagnostics.py \
    tests/test_replay_recorder.py -v
  git add web/field/identity_diagnostics.py web/replay_recorder.py tests/field/test_identity_diagnostics.py tests/test_replay_recorder.py
  git commit -m "feat(diagnostics): add structured identity and performance recorder"
  ```

### Task 0B: Pure gap-fill helpers
- **Files:**
  - Modify: [detect_schema.py](file:///home/nick/frc-vision-scorer/web/field/detect_schema.py)
  - Modify: [test_detect_schema.py](file:///home/nick/frc-vision-scorer/tests/test_detect_schema.py)

- [ ] **Step 1: Write the failing tests**
  Add unit tests for `box_ios`, `merge_robot_detections`, and `select_gap_fills` (testing full containment, disjoint boxes, partial overlap, suppression of fills overlapping bumpers, and per-half capping).
- [ ] **Step 2: Implement the helpers**
  Append `box_ios`, `merge_robot_detections`, and `select_gap_fills` to `detect_schema.py`.
- [ ] **Step 3: Verify tests and commit**
  ```bash
  .venv/bin/python -m pytest tests/test_detect_schema.py -v
  git add web/field/detect_schema.py tests/test_detect_schema.py
  git commit -m "feat(detect): add box IoS and gap-fill merge helpers"
  ```

### Task 0C: Reusable image-fraction size guard
- **Files:**
  - Modify: [detect_bridge.py](file:///home/nick/frc-vision-scorer/web/field/detect_bridge.py)
  - Modify: [test_detect_bridge.py](file:///home/nick/frc-vision-scorer/tests/test_detect_bridge.py)

- [ ] **Step 1: Write the failing test**
  Add unit tests for `box_exceeds_frac` (covering wide boxes, small boxes, and None frame handling).
- [ ] **Step 2: Extract the helper**
  Define `box_exceeds_frac` in `detect_bridge.py` and refactor the `_too_large` method to delegate to it.
- [ ] **Step 3: Verify tests and commit**
  ```bash
  .venv/bin/python -m pytest tests/test_detect_bridge.py -v
  git add web/field/detect_bridge.py tests/test_detect_bridge.py
  git commit -m "refactor(detect): extract box_exceeds_frac size guard"
  ```

---

## Phase 1: Temporal Observation & Detection Seams

### Task 1A: Fix SAM2 re-prompt lifecycle and expose continuity evidence
- **Files:**
  - Modify: [sam2_tracker.py](file:///home/nick/frc-vision-scorer/web/field/sam2_tracker.py)
  - Modify: [app.py](file:///home/nick/frc-vision-scorer/web/app.py)
  - Modify: [test_sam2_tracker.py](file:///home/nick/frc-vision-scorer/tests/field/test_sam2_tracker.py)
  - Modify: [test_sam2_streaming.py](file:///home/nick/frc-vision-scorer/tests/field/test_sam2_streaming.py)

- [ ] **Step 1: Add failing recovery tests**
  Cover:
  1. Six initial SAM2 objects do not prevent a new re-prompt ID after one object has retired.
  2. A single-boundary empty mask does not trigger a duplicate ID.
  3. Propagated masks don't get duplicates from repeated detector boxes.
  4. Exposing `new_after_gap` flag on re-prompted IDs.
  5. Bounded object-id counts over full synthetic matches.
- [ ] **Step 2: Separate active and lifetime limits**
  Implement `max_active_objects=6`, a `max_lifetime_objects` safety cap, `last_mask_frame`/absence age per object, and a retirement window.
- [ ] **Step 3: Strengthen re-prompt dedupe**
  Deduplicate candidate detections against current masks and recent boxes/masks. Return rich continuity evidence: `initial`, `propagated`, `reappeared_same_id`, `reprompted_new_id`, mask-gap length, and overlap.
- [ ] **Step 4: Update app wiring and tests**
  Pass limits explicitly. Emit SAM2 events and stage timings (rejection reasons, active/lifetime counts).
- [ ] **Step 5: Run tests and commit**
  ```bash
  .venv/bin/python -m pytest \
    tests/field/test_sam2_tracker.py \
    tests/field/test_sam2_streaming.py -v
  git add web/field/sam2_tracker.py web/app.py tests/field/test_sam2_tracker.py tests/field/test_sam2_streaming.py
  git commit -m "fix(sam2): preserve bounded temporal recovery across reprompts"
  ```

### Task 1B: Integrate Whole-Robot Gap-Fill Hooks (A & B)
- **Files:**
  - Modify: [app.py](file:///home/nick/frc-vision-scorer/web/app.py)
  - Create: [test_robot_gap_fill.py](file:///home/nick/frc-vision-scorer/tests/test_robot_gap_fill.py)

- [ ] **Step 1: Add config constants**
  Append config constants to `web/app.py`:
  ```python
  ROBOT_FILL_ENABLED = os.environ.get("ROBOT_FILL_ENABLED", "1") == "1"
  ROBOT_FILL_CONF = float(os.environ.get("ROBOT_FILL_CONF", str(DET_CONF_THRESHOLD)))
  ROBOT_FILL_IOS = float(os.environ.get("ROBOT_FILL_IOS", "0.5"))
  ROBOT_FILL_MAX_BOX_FRAC = float(os.environ.get("ROBOT_FILL_MAX_BOX_FRAC", "0.5"))
  ```
- [ ] **Step 2: Resolve the whole-robot class ID once in `InferenceJob.__init__`**
  Find the class in the model schema where the color is `None` (fallback to `1`).
- [ ] **Step 3: Add `_detect_whole_robots` wrapper on `InferenceJob`**
  YOLO pass on the frame selecting only the whole-robot class, filtering with `box_exceeds_frac`.
- [ ] **Step 4: Wire Hook A (SAM2 Prompt Seeding)**
  Compute `top_crop_px` unconditionally. Expand bumper boxes, run `_detect_whole_robots`, run `merge_robot_detections` with `ROBOT_FILL_IOS`, map boxes to field halves via `view.calibration.image_to_field`, cap fills per half using `select_gap_fills`, and seed them with `"unknown"` color.
- [ ] **Step 5: Wire Hook B (Re-prompt `detect_fn`)**
  Update the `detect_fn` block inside `make_detect_fn` to detect bumpers, expand them, detect whole robots, and return merged detections (whole-robot gap-fills are tagged `"unknown"`).
- [ ] **Step 6: Write and run integration tests**
  Create `tests/test_robot_gap_fill.py` and run the suite:
  ```bash
  .venv/bin/python -m pytest tests/test_robot_gap_fill.py tests/test_detect_schema.py -v
  git add web/app.py tests/test_robot_gap_fill.py
  git commit -m "feat(app): wire whole-robot gap-fill seeding Hooks A and B"
  ```

### Task 1C: Make fused-track re-link batch-global and non-poisoning
- **Files:**
  - Modify: [fused_track_id.py](file:///home/nick/frc-vision-scorer/web/field/fused_track_id.py)
  - Modify: [test_fused_relink.py](file:///home/nick/frc-vision-scorer/tests/field/test_fused_relink.py)
  - Modify: [test_fused_track_id.py](file:///home/nick/frc-vision-scorer/tests/field/test_fused_track_id.py)

- [ ] **Step 1: Add permutation-invariance tests**
  Assert identical optimal assignments regardless of input order. Test low-margin cases.
- [ ] **Step 2: Replace greedy inactive matching**
  Build an appearance-cost matrix for all unmatched observations vs eligible inactive fused tracks and solve using a global assignment solver. Expose margins.
- [ ] **Step 3: Quarantine uncertain embedding updates**
  Update EMA only for confident matches. Save recent exemplars to prevent EMA drift from wiping out historic anchors.
- [ ] **Step 4: Run tests and commit**
  ```bash
  .venv/bin/python -m pytest tests/field/test_fused_relink.py tests/field/test_fused_track_id.py -v
  git add web/field/fused_track_id.py tests/field/test_fused_relink.py tests/field/test_fused_track_id.py
  git commit -m "fix(fusion): make fused re-link global and order-independent"
  ```

---

## Phase 2: Persistent Identity

### Task 2A: Add the pure identity model and registry
- **Files:**
  - Create: [identity.py](file:///home/nick/frc-vision-scorer/web/field/identity.py)
  - Create: [test_identity.py](file:///home/nick/frc-vision-scorer/tests/field/test_identity.py)

- [ ] **Step 1: Write failing tests for lifecycle and invariants**
  Assert slot detachments, re-attachments, single-slot-single-identity mappings, roster reservations, cap enforcement, and permutation invariance.
- [ ] **Step 2: Implement registry types and explicit result objects**
  Implement `RobotIdentity`, `IdentityObservation`, `IdentityMatch`, and `IdentityRegistry` containing active/inactive dictionaries.
- [ ] **Step 3: Implement batch-global identity association**
  For each frame, build a cost matrix (appearance similarity, trajectory reachable envelope, roster consistency, continuity evidence) and solve globally. Expose margins and emit structured `identity.assignment` events.
- [ ] **Step 4: Enforce the persistent roster cap**
  Count active and inactive identities. Hard-reject a 4th identity. Distinguish `provisional` identities (tombstoneable at cap) from `confirmed` identities (protected).
- [ ] **Step 5: Run tests and commit**
  ```bash
  .venv/bin/python -m pytest tests/field/test_identity.py -v
  git add web/field/identity.py tests/field/test_identity.py
  git commit -m "feat(identity): add persistent robot identity registry"
  ```

### Task 2B: Make tracker binding transitions observable
- **Files:**
  - Modify: [tracker.py](file:///home/nick/frc-vision-scorer/web/field/tracker.py)
  - Modify: [test_tracker_trackid.py](file:///home/nick/frc-vision-scorer/tests/field/test_tracker_trackid.py)
  - Modify: [test_tracker_assignment.py](file:///home/nick/frc-vision-scorer/tests/field/test_tracker_assignment.py)

- [ ] **Step 1: Add failing tests for expiration events**
  Ensure slot expiry deletes fused mappings and fires `slot_expired` event details.
- [ ] **Step 2: Add tracker event/result dataclasses**
  Publish `self.binding_events` and `self.assignment_evidence` structures without importing the identity registry.
- [ ] **Step 3: Clear stale fused mappings and expose margins**
  Clean up mappings immediately on slot expiry. Calculate and return distance margin values.
- [ ] **Step 4: Run tests and commit**
  ```bash
  .venv/bin/python -m pytest tests/field/test_tracker_trackid.py tests/field/test_tracker_assignment.py -v
  git add web/field/tracker.py tests/field/test_tracker_trackid.py tests/field/test_tracker_assignment.py
  git commit -m "feat(tracker): expose slot lifecycle and assignment evidence"
  ```

### Task 2C: Feed fused re-link evidence into identity matching
- **Files:**
  - Modify: [fused_track_id.py](file:///home/nick/frc-vision-scorer/web/field/fused_track_id.py)
  - Modify: [test_fused_relink.py](file:///home/nick/frc-vision-scorer/tests/field/test_fused_relink.py)
  - Modify: [test_fused_track_id.py](file:///home/nick/frc-vision-scorer/tests/field/test_fused_track_id.py)

- [ ] **Step 1: Add failing tests for explicit re-link results**
  Ensure results report whether an ID was continued, appearance-relinked, merged, or newly created, along with margins.
- [ ] **Step 2: Add `FusedAssignmentEvidence` structures**
  Store and return evidence structures alongside raw observations.
- [ ] **Step 3: Merge appearance state on collapse**
  Combine winner/loser embeddings rather than discarding historical appearance on cameras during merges.
- [ ] **Step 4: Run tests and commit**
  ```bash
  .venv/bin/python -m pytest tests/field/test_fused_relink.py tests/field/test_fused_track_id.py -v
  git add web/field/fused_track_id.py tests/field/test_fused_relink.py tests/field/test_fused_track_id.py
  git commit -m "feat(identity): expose fused track re-link evidence"
  ```

### Task 2D: Integrate identity bindings into `InferenceJob`
- **Files:**
  - Modify: [app.py](file:///home/nick/frc-vision-scorer/web/app.py)
  - Create: [test_identity_integration.py](file:///home/nick/frc-vision-scorer/tests/test_identity_integration.py)
  - Modify: [database.py](file:///home/nick/frc-vision-scorer/web/database.py)
  - Modify: related tests that mock or construct `RobotTrack`

- [ ] **Step 1: Add a failing long-occlusion integration test**
  Verify team and vote persistence across slot expiry (e.g. slot 0 -> occluded -> expires -> slot 1 -> re-binds to original identity).
- [ ] **Step 2: Add registry state to `InferenceJob`**
  Initialize `self.identity_registry` and alias `self.robot_identities`.
- [ ] **Step 3: Replace `_get_or_create_slot_track`**
  Implement `_identity_for_slot`, `_bind_detection_identity`, `_detach_expired_slots`, and `_visible_identities`. Process expiration events, pair fused observations with tracker assignments, and apply binding transactions atomically once per alliance frame. Add first-frame SAM2 bootstrap logic.
- [ ] **Step 4: Migrate durable paths to `identity_id`**
  Update roster evaluation, gallery voting, scoring attribution, diagnostics, websockets, and autonomous paths. Ensure `BallTrajectory.shooter_robot_id` uses `identity_id`.
- [ ] **Step 5: Re-key crop/source state**
  Save review crops and crop selections by `identity_id` rather than slot.
- [ ] **Step 6: Re-key database schema**
  Migrate database tables: rebuild `match_robot_identity` to use `(job_id, identity_id)` as the primary key. Add `match_robot_binding` for history and update `match_robot_crop`.
- [ ] **Step 7: Define restart/resume behavior**
  Add checkpoint storage for the registry, bindings, VLM timestamps, and confidence metrics.
- [ ] **Step 8: Retain runtime invariants**
  Add end-of-frame validations asserting binding inversions, unique team ownerships, and crop/scoring references.
- [ ] **Step 9: Run tests and commit**
  ```bash
  .venv/bin/python -m pytest tests/test_identity_integration.py tests/test_replay_db.py -v
  git add web/app.py web/database.py tests
  git commit -m "feat(identity): bind durable robot state above tracker slots"
  ```

### Task 2E: Phase 2 regression and acceptance gate
- [ ] **Step 1: Run complete field and identity tests**
  ```bash
  .venv/bin/python -m pytest tests/field tests/test_reid_*.py tests/test_alliance_consistency.py -v
  ```
- [ ] **Step 2: Run complete project test suite**
  ```bash
  .venv/bin/python -m pytest
  ```
- [ ] **Step 3: Acceptance check**
  Confirm:
  - 90+ frame occlusion doesn't transfer team/vote data.
  - Inactive identities reserve their team roster number.
  - No alliance exceeds a 3-identity cap.
  - Ambiguous observations are quarantined.

---

## Phase 3: Confidence-Driven VLM

### Task 3A: Add pure confidence calculation
- **Files:**
  - Create: [identity_confidence.py](file:///home/nick/frc-vision-scorer/web/field/identity_confidence.py)
  - Create: [test_identity_confidence.py](file:///home/nick/frc-vision-scorer/tests/field/test_identity_confidence.py)

- [ ] **Step 1: Write failing confidence transition tests**
  Cover gradual rising, decay on occlusion, post-occlusion starting margins, appearance gallery similarity feedback, low-margin drops, roster uniqueness contribution, VLM result reinforcement, VLM contradiction drops, hysteresis limits, and clamping.
- [ ] **Step 2: Implement dataclasses**
  Define `ConfidenceSignals`, `ConfidenceState`, and `ConfidenceUpdate`. The pure helper must return the new confidence and reasons without mutating the global state.
- [ ] **Step 3: Define named thresholds/config**
  Expose a `ConfidenceConfig` object so parameters can be parameterized and tested without environment vars.
- [ ] **Step 4: Run tests and commit**
  ```bash
  .venv/bin/python -m pytest tests/field/test_identity_confidence.py -v
  git add web/field/identity_confidence.py tests/field/test_identity_confidence.py
  git commit -m "feat(identity): calculate per-robot identity confidence"
  ```

### Task 3B: Replace pass-count audits with per-identity match-clock scheduling
- **Files:**
  - Modify: [app.py](file:///home/nick/frc-vision-scorer/web/app.py)
  - Replace/Modify: [test_reread_audit_deadlock.py](file:///home/nick/frc-vision-scorer/tests/test_reread_audit_deadlock.py)
  - Create: [test_identity_vlm_schedule.py](file:///home/nick/frc-vision-scorer/tests/test_identity_vlm_schedule.py)

- [ ] **Step 1: Write match-clock audit regression tests**
  Ensure audits are scheduled using match seconds (not processing wall time), unconfirmed identities aren't starved, VLM results update successful audit timestamps, and unreadable crops backoff correctly.
- [ ] **Step 2: Add a pure schedule decision helper**
  Implement `ReadDecision` returning `due` (bool), `reason` (str), `priority` (float), and `is_audit` (bool).
- [ ] **Step 3: Replace global pass-count selection**
  Delete `REREAD_EVERY_N_PASSES` and `_is_reread_pass()`. Set `AUDIT_FLOOR_SECONDS` (default 30.0s).
- [ ] **Step 4: Make fused VLM reads targeted**
  Update `read_fused_bumper_numbers` to accept requested IDs. Solve camera-crop resolution for requested IDs and enforce a prioritized scheduling strategy under `VLM_TOP_K` caps.
- [ ] **Step 5: Handle contradiction and conflicts**
  Lower confidence and trigger immediate-read flags on VLM contradictions. Build conflict resolution when a crop matches a team confirmed on another identity (freeze, evaluate global assignment, mark ambiguous).
- [ ] **Step 6: Run schedule tests**
  ```bash
  .venv/bin/python -m pytest \
    tests/test_reread_audit_deadlock.py \
    tests/test_identity_vlm_schedule.py \
    tests/test_bumper_vlm_crop_mode.py -v
  git add web/app.py tests/test_reread_audit_deadlock.py tests/test_identity_vlm_schedule.py
  git commit -m "feat(vlm): schedule targeted reads from identity confidence"
  ```

### Task 3C: Wire per-frame confidence signals
- **Files:**
  - Modify: [app.py](file:///home/nick/frc-vision-scorer/web/app.py)
  - Modify: [identity.py](file:///home/nick/frc-vision-scorer/web/field/identity.py)
  - Modify: [test_identity_integration.py](file:///home/nick/frc-vision-scorer/tests/test_identity_integration.py)

- [ ] **Step 1: Build signals after tracker/identity assignment**
  Derive continuity, appearance agreement, assignment margins, roster uniqueness, and VLM signals. Apply decay to inactive identities. Update appearance anchors only above the anchor-update confidence floor.
- [ ] **Step 2: Queue immediate read reasons**
  Queue reads for re-attachment, low margins, appearance divergence, roster conflict, swap suspicion, and VLM contradiction.
- [ ] **Step 3: Add diagnostics and status output**
  Publish confidence, pending reasons, and next deadline in status payloads. Log confidence changes crossing thresholds.
- [ ] **Step 4: Run tests and commit**
  ```bash
  .venv/bin/python -m pytest \
    tests/field/test_identity_confidence.py \
    tests/test_identity_integration.py \
    tests/test_identity_vlm_schedule.py \
    tests/test_reread_audit_deadlock.py -v
  git add web/app.py web/field/identity.py tests
  git commit -m "feat(identity): drive VLM reads from live confidence signals"
  ```

---

## Phase 4: Joint Same-Alliance Occlusion Tiebreak

### Task 4A: Add pure co-occlusion event tracking and assignment comparison
- **Files:**
  - Create: [identity_ambiguity.py](file:///home/nick/frc-vision-scorer/web/field/identity_ambiguity.py)
  - Create: [test_identity_ambiguity.py](file:///home/nick/frc-vision-scorer/tests/field/test_identity_ambiguity.py)

- [ ] **Step 1: Write failing event tests**
  Verify co-occlusion grouping for same-alliance identities, re-emergence windows, exclusion of opposite alliances, and resolution lifecycles.
- [ ] **Step 2: Write failing assignment-margin tests**
  Check that low-margin direct assignments yield `ambiguous` status, while high-margin ones yield decisive direct assignments.
- [ ] **Step 3: Implement pure event and resolver types**
  Add `CoOcclusionEvent`, `CoOcclusionTracker`, `AssignmentCandidate`, and `AmbiguityResolution` (scoring permutations for up to 3 same-alliance identities).
- [ ] **Step 4: Run tests and commit**
  ```bash
  .venv/bin/python -m pytest tests/field/test_identity_ambiguity.py -v
  git add web/field/identity_ambiguity.py tests/field/test_identity_ambiguity.py
  git commit -m "feat(identity): detect and score joint occlusion ambiguity"
  ```

### Task 4B: Integrate ambiguity handling and targeted tiebreak reads
- **Files:**
  - Modify: [app.py](file:///home/nick/frc-vision-scorer/web/app.py)
  - Modify: [identity.py](file:///home/nick/frc-vision-scorer/web/field/identity.py)
  - Create: [test_joint_occlusion_integration.py](file:///home/nick/frc-vision-scorer/tests/test_joint_occlusion_integration.py)

- [ ] **Step 1: Add end-to-end synthetic tests**
  Test:
  1. Two identities co-occlude, re-emerge with separable evidence, and bind without VLM.
  2. Evidence is inseparable: trigger targeted VLM, resolve permutation atomically from results.
  3. Evidence is inseparable, crops unreadable: remain ambiguous, don't swap bindings.
- [ ] **Step 2: Feed merged occlusion state into the tracker**
  Update the co-occlusion tracker with identity IDs on slot occlusion/detections missing.
- [ ] **Step 3: Resolve the group before committing bindings**
  For co-occluded re-emergences: run group assignment, commit if above margin, else mark `pending_vlm` and queue targeted reads.
- [ ] **Step 4: Resolve from VLM results**
  Map crop -> team -> identity permutation. If results are invalid, contradictory, or duplicated, keep the event unresolved.
- [ ] **Step 5: Define scoring behavior while unresolved**
  Defer individual identity score attribution; increment only the alliance-level score. Record deferred attributions.
- [ ] **Step 6: Expose HUD and diagnostics**
  Display `AMBIGUOUS` labels and record the full co-occlusion resolution history in structured events.
- [ ] **Step 7: Run tests and commit**
  ```bash
  .venv/bin/python -m pytest \
    tests/field/test_identity_ambiguity.py \
    tests/test_joint_occlusion_integration.py \
    tests/test_identity_integration.py -v
  git add web/app.py web/field/identity.py tests/test_joint_occlusion_integration.py
  git commit -m "feat(identity): resolve same-alliance co-occlusion with targeted VLM"
  ```

---

## Phase 5: Rollout Verification and Acceptance Gate

### Task 5A: Real-match comparison and rollout gate
- **Files:**
  - Modify: [app.py](file:///home/nick/frc-vision-scorer/web/app.py)
  - Modify: [docs/superpowers/plans/2026-06-07-staged-identity-and-gap-fill.md](file:///home/nick/frc-vision-scorer/docs/superpowers/plans/2026-06-07-staged-identity-and-gap-fill.md)

- [ ] **Step 1: Add counters to the replay/diagnostic summary**
  Track identity creations, expirations, VLM reads by reason type, audit intervals, latency stages, queue/starvation counts.
- [ ] **Step 2: Run a known-occlusion match before and after**
  Verify physical identity persistence across occlusions, score/vote protection, and VLM read volume reduction.
- [ ] **Step 3: Run deterministic replay and fault injection**
  Test shuffled inputs, dropped masks, and injected faults. Assert invariants hold.
- [ ] **Step 4: Validate on multiple clips**
  Evaluate on calm, hub-scrum, and camera-dropout clips. Compare before/after latency and quality metrics.
- [ ] **Step 5: Run all tests**
  ```bash
  .venv/bin/python -m pytest
  ```
- [ ] **Step 6: Document tuned thresholds**
  Document scores, decay/rise rates, and windows.
- [ ] **Step 7: Final commit**
  ```bash
  git add docs web tests
  git commit -m "docs(identity): record persistence and confidence rollout results"
  ```

---

## Rollback Boundaries

- **Phase 1 Gap-Fill** can be completely disabled via the `ROBOT_FILL_ENABLED=0` environment flag, restoring the original bumper-only seeding/tracking behavior exactly.
- **Phase 2 Identity Registry** can ship with the existing VLM cadence if needed.
- **Phase 3 confidence scheduling** can be disabled behind a scheduler flag, falling back to match-clock audits instead of pass-counting.
- **Phase 4 joint occlusion** can be disabled by treating ambiguous group assignments as unresolved (without guess-committing).
