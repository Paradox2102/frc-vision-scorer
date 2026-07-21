# Default-ON candidacy A/B — rts_smooth, global_identity_resolve, sam3_multiplex (2026-07-20)

Question: the three merged-but-default-OFF tech-upgrade features — do they improve
results enough to flip on by default? Harness: `scripts/ab_qm8.py --profile
default-on` (six arms on qm11: per-feature isolation + the stacked proposed
default twice for rerun stability), then `--profile default-on-cross` on qm8.
`score()` gained identity metrics for this run: `mislabel` (slot label vs Manta
DS position under geometric pairing) and `slot_map`, plus per-arm `job_id` /
`job_s`.

**Outcome: none of the three flips today.** Validation instead surfaced two real
bugs (both fixed, tested, default-safe) and one metric-semantics conflict. All
knobs stay default OFF; the concrete re-entry criteria are listed per feature.

## qm11 six-arm sweep (video_85447fcf, auto period)

| arm | agg | blue | red | mislabel | job_s | note |
|---|---|---|---|---|---|---|
| baseline | 0.816 | 0.594 | 1.037 | 0 | 580 | 4th exact reproduction of 0.816 |
| rts | 0.890 | 0.589 | 1.192 | 0 | 683 | fired: 681 replaced, 82 inserted (37 survive reresolve) |
| gir | 0.816 | 0.594 | 1.037 | 0 | 850 | geometry-neutral; **collision bug found** (see below) |
| mux | 0.836 | 0.984 | 0.687 | 2 | 355 | **INVALID: all 3 views crashed at init_state; ran no-SAM3 fallback** |
| all3 | 0.934 | 1.022 | 0.846 | 2 | 315 | mux crash + rts insertions compounded |
| all3-b | 0.934 | 1.022 | 0.846 | 2 | 304 | bit-identical to all3 → stack deterministic |

qm8 cross-match (ff4be1d9): baseline 0.414/0.370/0.443; rts 0.477/0.337/0.570;
gir 0.414/0.370/0.443 (bit-identical to baseline). The qm8 mux arm ran pre-fix
server code (same crash) and is discarded.

## rts_smooth — keep OFF; gap-fill needs a cap before its own gate can pass

- Fires correctly; rewrites exported `autonomous_paths` and fills gaps
  (`interpolated: true`, `tracker_prob: null`).
- Decomposition (scratch `rts_decompose.py`, both metrics): on **common frames**
  vs a densified truth trajectory, smoothing improved EVERY qm11 robot (pooled
  ours→truth 0.726→0.692). The headline `compare_manta` metric is symmetric NN
  against **raw sparse Manta waypoints**, and under it even the no-insertion
  variant regresses (0.885 vs 0.816) — waypoint sparsity + the truth→ours term
  reward causal coast-and-stall clusters over kinematic interpolation.
- The big driver is inserted samples inside long occlusion gaps: qm11 R1's 18
  fills average 2.9 m off; qm8 2724's coverage jumped 24→80% while red meanNN
  went 0.443→0.570. Short-gap fills are excellent (qm11 B3 0.14 m, R2 0.32 m).
- Same sign on both matches → verdict is stable, not match noise.
- **Re-entry criteria:** add a max-gap knob (only interpolate gaps ≤ ~1.5 s),
  re-run this profile; separately decide whether scoring should exclude
  `interpolated` samples from meanNN (metric-definition change — do NOT slip it
  in silently; it moves every historical baseline).

## global_identity_resolve — keep OFF; closest to flip-ready

- Geometry-neutral everywhere (qm11 and qm8 metrics bit-identical to baseline;
  the export rebuild only rewrites team labels), deterministic given evidence.
- **Bug found on its debut run:** the exact assignment moved 1477 onto qm11 R2
  while R3's assignment came back None; `merge_global_assignment`'s v1
  keep-on-None rule preserved R3's stale greedy 1477 → **41 frames with two
  1477s on the overlay**, voiding the feature's collision-free contract.
  Fixed with a collision sweep (strip any kept label whose team was assigned
  elsewhere; recorded as a change row) —
  `web/field/identity_resolve.py::merge_global_assignment`, tests
  `test_merge_strips_kept_label_colliding_with_assignment` /
  `test_merge_strips_collision_when_slot_absent_from_assignment`. Live re-run
  confirmed: strip fires (`R3 1477->None`), 0 duplicate-team frames.
- qm8 true positive: greedy accepted zero teams; gir stamped R1=2724
  **confirmed** (score 5.0, margin 2.0) — matches geometric truth.
- qm11 caveat: with thin auto-only evidence (5–7 reads), it shuffles red labels
  that geometry can't adjudicate; and the read layer itself varies run to run
  (read sets differ across identical jobs), so label outcomes aren't
  reproducible regardless of gir.
- **Re-entry criteria:** full-match runs (the knob comment's own protocol:
  confirmed-team accuracy, `compare_manta --assign` MISLABEL, rerun stability)
  on qm8–qm12 with a healthy vLLM reader, now that the collision fix is in.

## sam3_multiplex — keep OFF; was never actually exercised until tonight's fix

- **The knob has been completely broken in live runs:** upstream
  `Sam3BasePredictor.start_session` unconditionally forwards
  `offload_state_to_cpu` into `model.init_state(**kwargs)`, and the SAM 3.1
  multiplex model's `init_state` doesn't accept it. Every view raised
  `TypeError` at session start, `[DEGRADED] sam3: view produced no tracks`,
  and the job silently ran the **no-SAM3 detector+fusion fallback**. The
  sweep's "mux" numbers measured that fallback, not multiplex (hence the fake
  −40% wall-clock: propagation was simply skipped).
- Fixed with a signature-guarded shim in the multiplex branch
  (`web/field/sam3_tracker.py`): wrap the built model's `init_state` to
  accept-and-drop the kwarg; auto-disables if upstream adds the param. Tests
  `test_sam3_multiplex_shims_init_state_without_state_offload` /
  `test_sam3_multiplex_leaves_native_init_state_unshimmed`.
- Post-fix numbers: see `mux-single` runs below.
- Reminder from the knob comment: multiplex is NOT a pure perf toggle (different
  detection thresholds, boundary suppression, masklet confirmation), so quality
  parity must be re-established from scratch before any flip.
- Accidental ablation worth remembering: the fallback arm (detector+fusion
  only, no SAM3) scored red 0.687 vs baseline 1.037 on qm11 — propagation is
  actively hurting qm11 red. Grist for the selective-propagation work.

## mux-single re-validation (post-shim)

The shim clears the session-start crash — and exposes the next incompatibility
layer. qm11 (`fd57f69f`):

- **Prompted views crash mid-propagation**, and the error value is a
  deterministic function of PROMPT COUNT, not of match, view, or frame:
  3 prompted objects → `KeyError: 76`, 1 prompted object → `KeyError: 34`,
  reproduced identically on qm11 and qm8 across different views. 0-prompt
  views never crash. That correlation points at the multiplex bucket
  architecture emitting/expecting object ids that don't match the
  instance-prompt obj_ids our adapter assigns. Root-causing is its own
  project — the 3.1 "WithInteractivity" stack is demo-grade and our chunked
  instance-prompt sessions exercise paths Meta's demos don't.
- **The unprompted view "succeeded"**: corner_right (no prompt-frame robots)
  propagated 181 frames at 315 ms/frame with an empty roster — proving the
  shim works and multiplex propagation itself runs, but nothing was tracked.
- **Net job output: zero autonomous paths** — and that surfaced a harness
  footgun: `load_our_auto_paths` keeps the newest replay per key THAT HAS
  PATHS, so the arm silently scored the previous run's replay bit-for-bit.
  `score()` now takes `expect_replay=job_id` and errors on a stale read
  instead. (Retro-checked: all ten sweep arms above exported their own 6-slot
  paths; only this post-shim mux run hit the hazard.)

Bottom line for `sam3_multiplex`: two integration bugs deep, still not
exercisable end-to-end on prompted objects. Keep OFF; treat "adopt SAM 3.1
multiplex" as a standalone integration task (obj-id mapping audit against the
bucket controller, then the qm8–qm12 quality-parity A/B this doc was meant to
run), not a knob flip.

## VRAM / runtime

Peak VRAM is chunk-dominated: every valid arm peaked 22.5–24 GB, so multiplex's
"~half VRAM per object" claim doesn't move job-level peaks on this card. Job
wall time drifted upward across back-to-back arms in one server session
(baseline 580 s → gir 850 s for functionally identical propagation) — restart
between arms before quoting per-arm speedups.
