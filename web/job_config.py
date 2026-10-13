"""Typed registry of per-job inference knobs.

One place defining every job-scoped A/B knob: its override key, env var,
type, default, and valid range. JobConfig resolves
override > runtime setting > env > default at job creation and validates
everything up front, so a bad knob fails the submit with a clear message
instead of failing minutes into inference. Runtime settings come from the
settings UI (/api/settings) and live only until restart.

Adding a knob is one Knob line here plus reading `job.config.<name>` at the
consumption site. Module-level constants consumed at import time by helper
functions are intentionally NOT registered: listing them here would falsely
advertise them as per-job overridable.

`non_defaults()` is the A/B provenance record — everything that differs from
the code default (whether set by env or per-job override) — logged at job
start and echoed in the `complete` payload.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from web import clef_reader, vlm_models

# Matches the truthiness rule used by the app.py env constants.
_FALSY = ("0", "false", "False")


@dataclass(frozen=True)
class Knob:
    name: str                       # override key / config attribute
    env: str                        # environment variable for the default
    type: type                      # bool | int | float | str
    default: Any
    choices: Optional[tuple] = None
    min: Optional[float] = None
    max: Optional[float] = None
    section: str = "misc"           # settings-UI grouping; see KNOBS comments


KNOBS = [
    # --- tracking backend ---
    Knob("tracking_backend", "TRACKING_BACKEND", str, "sam3",
         choices=("sam2", "sam3"), section="Tracking backend"),
    Knob("sam2_model_size", "SAM2_MODEL_SIZE", str, "small",
         choices=("tiny", "small", "base_plus", "large"), section="Tracking backend"),
    Knob("sam2_track_fps", "SAM2_TRACK_FPS", float, 8.0, min=0.5, max=60.0,
         section="Tracking backend"),
    # Floor 0.25: below that SAM2 is starved of pixels; above 1.0 only
    # upsamples (no information) and inflates mask RAM.
    Knob("sam2_export_scale", "SAM2_EXPORT_SCALE", float, 0.5, min=0.25, max=1.0,
         section="Tracking backend"),
    Knob("sam2_max_gpu_frames", "SAM2_MAX_GPU_FRAMES", int, 0, min=0,
         section="Tracking backend"),
    # Seed at the active-period open (0.0) rather than +3s: the +3s single seed
    # left clock 0:20->0:17 untracked and missed robots still en route. Validated
    # 2026-07-06 as lever 2 across qm8/10/11/12 (agg meanNN improved on every
    # match; 6800/870 late-start robots now seed at ~0 lag).
    Knob("sam2_prompt_offset_seconds", "SAM2_PROMPT_OFFSET_SECONDS", float, 0.0,
         min=0.0, section="Seeding & re-seeding"),
    Knob("reprompt_seconds", "REPROMPT_SECONDS", float, 2.0, min=0.1,
         section="Seeding & re-seeding"),
    # qm12 0:20-0:16 — during the first `opening_window_seconds` of active
    # tracking, re-prompt every `opening_reprompt_seconds` (denser than
    # reprompt_seconds) so robots streaming into center seed within a fraction
    # of a second instead of waiting a full ~2s boundary; 0 on either disables.
    # DEFAULT ON (0.5s / 6s) — validated 2026-07-06, no cross-match regression.
    Knob("opening_reprompt_seconds", "OPENING_REPROMPT_SECONDS", float, 0.5, min=0.0,
         section="Seeding & re-seeding"),
    Knob("opening_window_seconds", "OPENING_WINDOW_SECONDS", float, 6.0, min=0.0,
         section="Seeding & re-seeding"),
    # Motion override: frame-diff pixel motion as positive robot evidence at
    # re-prompt boundaries — a moving detection claimed by a stationary mask
    # seeds anyway, and may evict a stationary once-absent object from a
    # capped roster (qm11: sprinting robot starved by already_tracked/
    # roster_cap against equipment-parked masks). DEFAULT OFF pending A/B.
    Knob("motion_override", "SAM2_MOTION_OVERRIDE_ENABLED", bool, False,
         section="Seeding & re-seeding"),
    # Min mean-abs-gray-diff (normalized /255) inside the candidate box, one
    # track_fps step apart, to count as "moving". Static-scene JPEG noise is
    # <0.01; a robot sweeping its own box scores >0.06.
    Knob("motion_energy_thresh", "SAM2_MOTION_ENERGY_THRESH", float, 0.04,
         min=0.0, max=1.0, section="Seeding & re-seeding"),
    # Motion-gated low-confidence admission: run the bumper detector down to
    # det_conf_floor at seed/re-prompt frames and admit the sub-threshold band
    # only when the box shows pixel motion (qm11 6800: dark blurred bumper
    # read 0.26-0.34 vs the 0.4 seed conf — blur lowers confidence, and the
    # blur IS the motion evidence). DEFAULT OFF pending A/B.
    Knob("motion_gated_conf", "MOTION_GATED_CONF_ENABLED", bool, False,
         section="Seeding & re-seeding"),
    Knob("det_conf_floor", "DET_CONF_FLOOR", float, 0.2, min=0.01, max=1.0,
         section="Seeding & re-seeding"),
    # Detector-independent seed candidates from frame-diff blobs, fed into
    # the colorless gap-fill path (color backfilled later). Rescues a robot
    # YOLO whiffs on entirely (heavy blur, dark bumpers). DEFAULT OFF
    # pending A/B.
    Knob("motion_seed_blobs", "MOTION_SEED_BLOBS_ENABLED", bool, False,
         section="Seeding & re-seeding"),
    # Blob seeding is scoped to the first N seconds of the active period
    # (the auto-start sprint problem); match-long admission tripled roster
    # churn on qm11. 0 disables the scope (blobs whole match).
    Knob("motion_blob_window_s", "MOTION_BLOB_WINDOW_S", float, 6.0, min=0.0,
         section="Seeding & re-seeding"),
    # Autonomous starting-zone seed priors: during the opening window, admit
    # sub-threshold bumper reads whose ground contact projects into a legal
    # starting band (within start_zone_depth_m of an alliance wall, full field
    # width) as SUBORDINATE (strong=False) candidates — robots sit stationary
    # and rule-constrained at t=0, so grab them before they sprint (qm11 fast
    # robot never seeded). Reuses det_conf_floor. DEFAULT OFF pending A/B.
    Knob("start_zone_seed", "START_ZONE_SEED_ENABLED", bool, False,
         section="Seeding & re-seeding"),
    Knob("start_zone_depth_m", "START_ZONE_DEPTH_M", float, 2.0,
         min=0.5, max=8.0, section="Seeding & re-seeding"),
    Knob("start_zone_window_s", "START_ZONE_WINDOW_S", float, 6.0, min=0.0,
         section="Seeding & re-seeding"),
    # Seed candidates adopt the union of the bumper-expanded box and the
    # overlapping whole-robot detection (bumper color kept). A blurred
    # bumper expands to a narrow sliver — a poor SAM prompt; the whole-robot
    # detector knows the true extent. DEFAULT OFF pending A/B.
    Knob("seed_union_extent", "SEED_UNION_EXTENT_ENABLED", bool, False,
         section="Seeding & re-seeding"),
    # Negative-point prompting: seed/re-anchor prompts carry label-0 points at
    # the centers of OTHER concurrent detections near the prompt box, so SAM
    # conditions against bleeding onto a neighboring robot / adjacent
    # structure from birth. DEFAULT ON — validated 2026-07-08 (qm8/9/10 auto
    # A/B): neutral on the easy matches, big qm10 win (aggNN 0.644->0.509,
    # rescued blue 59 coverage 24%->94%), no regressions. Module env default
    # (SAM2_NEG_POINT_PROMPTS) stays off for direct/library callers; the
    # pipeline passes this Knob value into track().
    Knob("neg_point_prompts", "SAM2_NEG_POINT_PROMPTS", bool, True,
         section="Seeding & re-seeding"),
    Knob("sam2_vos_optimized", "SAM2_VOS_OPTIMIZED", bool, False,
         section="Tracking backend"),
    # SAM3 chunked sessions: frames per GPU-resident window (~14 MB/frame).
    # Bigger = fewer identity-risky chunk boundaries; cap is VRAM, not speed.
    # Default lowered 400->200 for 24GB cards: 400 peaked the whole card and OOM'd
    # when other state was co-resident (see web/field/sam3_tracker.py header).
    Knob("sam3_chunk_frames", "SAM3_CHUNK_FRAMES", int, 200, min=16, max=2000,
         section="Tracking backend"),
    # SAM 3.1 Object Multiplex builder — joint multi-object propagation
    # (~2x propagation throughput, ~half VRAM per Meta); NOT a pure perf
    # toggle: the multiplex builder bakes in different tracking heuristics
    # (new_det_thresh 0.65 vs 0.7, score_threshold_detection 0.4 vs 0.5,
    # boundary suppression on, masklet confirmation on), so mask/re-seed/
    # occlusion behavior shifts. DEFAULT OFF. Until 2026-07-20 the knob was
    # broken outright — upstream start_session forwards offload_state_to_cpu
    # into the multiplex model's init_state, which rejects it, so every view
    # crashed and jobs ran the no-SAM3 fallback (now shimmed in
    # sam3_tracker.py; any earlier "multiplex" numbers measured the
    # fallback). Post-shim A/B: docs/default-on-ab-2026-07-20.md.
    Knob("sam3_multiplex", "SAM3_MULTIPLEX", bool, False,
         section="Tracking backend"),
    # Selective SAM3 mask propagation (arXiv 2606.13033 re-implementation):
    # skip full-frame mask propagation for a tracked object while its match is
    # comfortably inside selective_margin_entry (in 1 - expanded-IoU cost
    # units, the paper's cost metric) and only resume full propagation once
    # the margin degrades past selective_margin_exit, a propagation-free gap
    # reaches selective_gap_trigger frames, or the skipped window would exceed
    # selective_window_max_frames. Margin/gap/streak values are the paper's
    # defaults and are A/B-tunable. selective_margin_exit doubles as the seed
    # margin and selective_min_streak doubles as SEED/EXIT_CONSECUTIVE.
    # DEFAULT OFF pending qm8-qm12 A/B.
    Knob("selective_mask_prop", "SELECTIVE_MASK_PROP", bool, False,
         section="Tracking backend"),
    Knob("selective_margin_entry", "SELECTIVE_MARGIN_ENTRY", float, 0.05,
         min=0.0, max=1.0, section="Tracking backend"),
    Knob("selective_margin_exit", "SELECTIVE_MARGIN_EXIT", float, 0.10,
         min=0.0, max=1.0, section="Tracking backend"),
    Knob("selective_gap_trigger", "SELECTIVE_GAP_TRIGGER", int, 7,
         min=1, section="Tracking backend"),
    Knob("selective_min_streak", "SELECTIVE_MIN_STREAK", int, 5,
         min=1, section="Tracking backend"),
    Knob("selective_window_max_frames", "SELECTIVE_WINDOW_MAX_FRAMES", int, 120,
         min=8, section="Tracking backend"),
    # Pixel-IoU at which two windows' masks count as converged and both are
    # killed to TBD (ExitMachine._check_killed). 0.90 (the knob default) fires
    # on partially-overlapping same-alliance robots -- the qm11 blue gap, see
    # docs/selective-mask-prop-blue-gap.md -- not just genuine mask collapse.
    # Higher = only near-total collapse kills. Was an internal SelectiveConfig
    # constant; promoted to a knob so it can be A/B'd (scripts/ab_qm8.py
    # "blue-gap" profile) without editing source.
    Knob("selective_mask_overlap_exit", "SELECTIVE_MASK_OVERLAP_EXIT", float, 0.90,
         min=0.5, max=1.0, section="Tracking backend"),
    # Kill-coast fallback (docs/selective-mask-prop-blue-gap.md fix #2). When a
    # window pair is killed (masks converged), the two objects normally revert
    # to the appearance-blind TBD tracker, which mis-links close same-alliance
    # robots (the qm11 blue regression). ON instead pins each killed object to
    # its last confident SAM box, coasted at constant velocity for up to
    # selective_kill_coast_frames frames, until a TBD detection unambiguously
    # re-claims it (expanded-IoU margin >= selective_margin_exit) or the budget
    # expires. Default OFF -> byte-identical to today; A/B'd via the "blue-gap"
    # profile in scripts/ab_qm8.py.
    Knob("selective_kill_coast", "SELECTIVE_KILL_COAST", bool, False,
         section="Tracking backend"),
    Knob("selective_kill_coast_frames", "SELECTIVE_KILL_COAST_FRAMES", int, 24,
         min=1, max=240, section="Tracking backend"),
    # Cross-episode SAM-frame budget for selective mode: caps total SAM
    # propagation frames across a view's episodes at sam_budget_frac * n_frames.
    # Motivated by the qm11 A/B (web/logs/20260719_222247_ec06450f*.log): the
    # main view's 17 overlapping windows drove sam_frames to 213% of uniform
    # propagation cost before episode merging existed. Default 1.0 means
    # "never more SAM work than uniform" -- episodes that don't fit the
    # remaining budget are skipped (dropped_windows stats, TBD stands for
    # those frames); a later, smaller episode may still fit.
    Knob("selective_sam_budget_frac", "SELECTIVE_SAM_BUDGET_FRAC", float, 1.0,
         min=0.1, max=10.0, section="Tracking backend"),
    # DINOv3 appearance term in the selective TBD tracker's assignment cost
    # (blue-gap fix #3, docs/selective-mask-prop-blue-gap.md; spec in
    # docs/superpowers/specs/2026-07-20-selective-appearance-term-design.md).
    # Same-color pairs gain app_weight * (1 - cosine(track EMA, det emb)) so
    # same-alliance robots separate by appearance where motion+IoU cannot.
    # 0.0 = OFF (exact no-op). Track gallery EMA refreshes only on confident
    # matches (assignment margin >= selective_margin_exit).
    Knob("selective_app_weight", "SELECTIVE_APP_WEIGHT", float, 0.0,
         min=0.0, max=5.0, section="Tracking backend"),
    Knob("selective_app_ema", "SELECTIVE_APP_EMA", float, 0.9,
         min=0.0, max=1.0, section="Tracking backend"),
    # Starvation-triggered windows (blue-gap v2 fix #1,
    # docs/selective-mask-prop-blue-gap.md; spec docs/superpowers/specs/
    # 2026-07-21-selective-starvation-windows-design.md). Margin/gap/witness
    # all fire on AMBIGUITY; disappearance is not a trigger class, so a robot
    # that vanishes from the detector (3181 parks behind the hub at t=37s,
    # sliver only -- YOLO never re-fires) just ends its TBD stream and
    # selective loses it while baseline's uniform SAM stays latched to the
    # sliver. ON: a track never observed again (dark tail >=
    # selective_starve_trigger frames to sequence end) whose last box is off-
    # border opens a SAM window from its last confident box. Starve windows
    # skip the area-degradation abort (slivers shrink by design) and run
    # uncapped by default (selective_starve_max_frames 0; the cross-episode
    # selective_sam_budget_frac budget is the cost governor). DEFAULT OFF ->
    # byte-identical no-op.
    Knob("selective_starve_windows", "SELECTIVE_STARVE_WINDOWS", bool, False,
         section="Tracking backend"),
    Knob("selective_starve_trigger", "SELECTIVE_STARVE_TRIGGER", int, 7,
         min=1, section="Tracking backend"),
    Knob("selective_starve_max_frames", "SELECTIVE_STARVE_MAX_FRAMES", int, 0,
         min=0, section="Tracking backend"),
    # Starve-trigger junk gate: a starve candidate must have been OBSERVED in
    # at least this many frames (count, not lifetime span -- flickery junk
    # stretches span with few actual observations). Transient YOLO
    # false-fires otherwise satisfy the starve trigger and flood the SAM
    # budget with uncapped windows (qm11 Jul 21: corner sam_frames 26%/42% ->
    # 100% of uniform, main dropped 34 windows, 4786 2.31 -> 3.11). A real
    # disappearing robot has hundreds of observations first (3181: ~37s
    # tracked before parking). 0 disables the gate (v1 behavior).
    Knob("selective_starve_min_obs", "SELECTIVE_STARVE_MIN_OBS", int, 30,
         min=0, section="Tracking backend"),
    # torch.compile for SAM3 mask propagation. First run pays inductor
    # compile warm-up (cached on disk thereafter); A/B knob for the
    # propagation stage that dominates tracking wall time.
    Knob("sam3_compile", "SAM3_COMPILE", bool, False, section="Tracking backend"),
    # Corner-cam decimation: track corner views at track_fps/stride. SAM
    # propagation cost scales with frames/view (export resolution is NOT a
    # lever — the model resizes everything to a fixed square). Stride 2 cut
    # corner propagation ~39% on the qm24 smoke, BUT cost confirmed locks
    # (5/6 at stride 1 vs 3/6 and 1/6 across two stride-2 runs) — corner
    # reads carry 2.5x vote weight and halved crop opportunities starve
    # confirmation. Default OFF; opt in per job when speed outweighs
    # identity. Wall-clock-denominated tracker knobs (reprompt_seconds,
    # evict_dark_frames) are rescaled per view automatically.
    Knob("corner_track_stride", "CORNER_TRACK_STRIDE", int, 1, min=1, max=8,
         section="Tracking backend"),
    # Dense head for corner decimation: corners run full-rate for the first
    # N seconds of tracked footage (the lock-formation window), then pay the
    # stride. Only meaningful when corner_track_stride > 1. Smoke A/B
    # (qm24 45s, June 12 2026): stride 2 + 20s head got 2/6 locks vs 3/6
    # same-day baseline — softer than pure stride 2 (1/6) but still never
    # matched baseline, so decimation stays opt-in.
    Knob("corner_dense_seconds", "CORNER_DENSE_SECONDS", float, 25.0, min=0.0,
         section="Tracking backend"),
    Knob("cull_hijack_enabled", "CULL_HIJACK_ENABLED", bool, False,
         section="Track hygiene"),
    # Scope the absence-based cull to the main view: mid-field equipment is
    # fully visible to main-view YOLO (absence there is strong evidence),
    # while corner views are where qm24's false culls killed real robots
    # parked in detector blind spots. Only meaningful with cull_hijack_enabled.
    Knob("cull_main_view_only", "CULL_MAIN_VIEW_ONLY", bool, True,
         section="Track hygiene"),
    # Person-overlap veto: cull a stationary mask that spends consecutive
    # boundaries mostly inside a person detection — a positive hijack signal,
    # independent of the absence cull (qm11: R3 spent all of auto on a ref).
    Knob("person_veto_enabled", "SAM2_PERSON_VETO_ENABLED", bool, False,
         section="Track hygiene"),
    # Zone-blind corroboration: detections whose ground contact lies inside a
    # static-structure zone (UNinflated rects) never count as mask
    # corroboration, so a bumper-lookalike FP on equipment (2026cur OVERFLOW
    # bin base reads "1477") can't vouch for its own hijacked mask and the
    # absence cull can fire. Robots can't park inside an equipment footprint,
    # so blinding corroboration there is safe. Pair with cull_hijack_enabled.
    Knob("zone_blind_corroboration", "ZONE_BLIND_CORROB", bool, False,
         section="Track hygiene"),
    # Corroboration confidence floor (seed/corroborate threshold split): at
    # re-prompt boundaries the bumper detector runs down to this floor, and
    # detections in [corrob_conf_floor, seed conf) that the motion/zone
    # probes did NOT admit ride a corroborate-only band — they count as
    # presence evidence (a parked robot rendering weak detections must not
    # accrue absences toward cull/eviction) but can never seed, match,
    # re-anchor, or evict. 0 disables (single-threshold behavior).
    # SHELVED 2026-07-08: A/B'd qm8/9/10 auto (byte-identical to baseline —
    # the band fires but flips no downstream decision) and qm11 full (inert).
    # No benefit shown; kept default-off for a future match where roster
    # contention actually turns on a weak detection.
    Knob("corrob_conf_floor", "CORROB_CONF_FLOOR", float, 0.0, min=0.0, max=1.0,
         section="Track hygiene"),
    # Zone-blind VOTING (identity layer): a VLM read taken while the track's
    # field position is inside a raw static zone is printed equipment text
    # (2026cur OVERFLOW crate reads "1477" at digit_prob 1.0), never a
    # bumper — discard the vote. Default ON: it only acts inside explicit
    # equipment footprints, where a robot cannot be.
    Knob("zone_blind_votes", "VLM_ZONE_BLIND_VOTES", bool, True,
         section="Identity & votes"),
    # Vote ray-consistency gates (identity layer, geometry-only): a bumper
    # crop's own camera saw it on a specific ground ray, and votes are
    # applied later by a background task. Two independent checks —
    # (1) candidate time: reject a crop whose source-camera ray misses the
    # fused slot position by more than vote_ray_cross_m (wrong merge, never
    # send to the VLM); (2) vote time: drop a vote if the slot's live
    # position has since moved farther than vote_stale_max_m from the
    # crop-time fused position (re-bound between crop and vote). DEFAULT OFF
    # pending A/B.
    Knob("vote_ray_gate", "VOTE_RAY_GATE_ENABLED", bool, False,
         section="Identity & votes"),
    Knob("vote_ray_cross_m", "VOTE_RAY_CROSS_M", float, 1.5, min=0.1,
         section="Identity & votes"),
    Knob("vote_stale_max_m", "VOTE_STALE_MAX_M", float, 2.0, min=0.1,
         section="Identity & votes"),
    # Weight VLM team votes by the SAM tracker's per-object confidence in the
    # slot being voted into (sqrt damped, floored at 0.25). Votes cast while
    # the tracker itself is unsure of the mask (mid-hijack, mid-occlusion)
    # are exactly the misattributed votes that flip identity locks.
    # SHELVED 2026-07-08 (UNPROVEN, not disproven): the Manta A/B harness
    # scores auto-period POSITION via geometric pairing and is blind to
    # team-label features, so it can't measure this. On qm11 full it left
    # identity counters + the visible misattribution unchanged vs baseline.
    # Needs a team-label-accuracy metric (our team per robot vs roster) to
    # judge; revisit then.
    Knob("vote_prob_weight", "VLM_VOTE_PROB_WEIGHT", bool, False,
         section="Identity & votes"),
    Knob("reanchor_enabled", "SAM2_REANCHOR_ENABLED", bool, False,
         section="Track hygiene"),
    # Evidence-ranked roster: re-prompt a dark object claimed by a fresh
    # detection in place, and let weak-evidence roster members (dark or
    # repeatedly detector-orphaned) lose their slot to detector-backed
    # newcomers. The roster holds the 6 objects most likely to be robots.
    Knob("dark_reanchor_enabled", "SAM2_DARK_REANCHOR_ENABLED", bool, True,
         section="Track hygiene"),
    Knob("evict_dark_frames", "SAM2_EVICT_DARK_FRAMES", int, 8, min=1,
         section="Track hygiene"),
    Knob("evict_min_absences", "SAM2_EVICT_MIN_ABSENCES", int, 2, min=1,
         section="Track hygiene"),
    # Misaligned-mask re-anchor: the common drift case between dark (total
    # mask loss) and sliver (area collapse) re-anchor — a healthy-size,
    # color-consistent mask that slid off its robot until detection overlap
    # fell below the match threshold. When a partial-overlap claimant
    # persists for misalign_boundaries CONSECUTIVE boundaries, re-prompt that
    # SAME id at the detection box instead of churning a new id or hitting
    # the roster cap. DEFAULT ON — validated 2026-07-08 (qm8/9/10 auto A/B):
    # neutral on the easy matches, big qm10 win (aggNN 0.644->0.520), no
    # regressions. Module env default (SAM2_MISALIGN_REANCHOR) stays off for
    # direct/library callers; the pipeline passes this Knob value into track().
    Knob("misalign_reanchor", "SAM2_MISALIGN_REANCHOR", bool, True,
         section="Track hygiene"),
    Knob("misalign_ios_min", "SAM2_MISALIGN_IOS_MIN", float, 0.2, min=0.05, max=0.5,
         section="Track hygiene"),
    Knob("misalign_boundaries", "SAM2_MISALIGN_BOUNDARIES", int, 2, min=1, max=10,
         section="Track hygiene"),
    # Seed-box plausibility gate: a detection box taller than this aspect
    # (h/w) may not seed, re-anchor, or evict at SAM reprompt boundaries; 0
    # disables. qm12 hub-face sliver (aspect 4.47) evicted a real track.
    # DEFAULT ON (3.0) — validated 2026-07-06; blocked the sliver hijack with
    # no cross-match regression (rejects only geometrically-impossible boxes).
    Knob("seed_box_max_aspect", "SEED_BOX_MAX_ASPECT", float, 3.0, min=0.0, max=20.0,
         section="Track hygiene"),
    # Capped-roster eviction requires the candidate detection to prove pixel
    # motion when a motion probe is available (motion_override); static
    # candidates fall through to roster_cap rejection. No-op without
    # motion_fn. qm12: a static hub-face sliver evicted a real robot.
    # DEFAULT ON — validated 2026-07-06 (no-op unless motion_override is also on).
    Knob("evict_requires_motion", "EVICT_REQUIRES_MOTION", bool, True,
         section="Track hygiene"),
    # qm11/qm10/qm12 gated leak cleanup: a robot's SAM mask sometimes leaks onto
    # an adjacent bump/ramp, so mask_to_box returns the union bbox spanning both
    # and the projected field position lands between them. Drop a mask component
    # only when it is BOTH small AND detached — a legitimate secondary component
    # (a robot arm, or a robot split by hub/ball occlusion) is preserved.
    # Unconditional largest-component regressed qm10/qm12 by dropping those
    # halves. DEFAULT OFF pending re-validation.
    Knob("mask_leak_cleanup", "SAM2_MASK_LEAK_CLEANUP", bool, False,
         section="Track hygiene"),
    # Derive a robot's field position from the centroid of its SAM mask's
    # carpet-contact band instead of the tracked box bottom-center. An extended
    # intake/arm stretches the box, shifting the box bottom-center off the
    # robot's actual carpet contact; the mask footprint is a truer ray origin.
    # Falls back to the box bottom-center when no mask is available.
    # DEFAULT OFF (byte-identical default path) pending A/B.
    Knob("mask_footprint_pos", "MASK_FOOTPRINT_POS", bool, False,
         section="Track hygiene"),
    # Size-prior hijack cull: a tracked mask whose pixel width, at its
    # projected field position, stays far wider (or narrower) than a robot
    # could appear for size_prior_boundaries consecutive boundaries has latched
    # onto structure/a pile. Equipment hijacks are YOLO-corroborated, so the
    # absence cull can't catch them — geometry is the only clean signal.
    # Independent of cull_hijack_enabled. DEFAULT OFF pending A/B.
    Knob("size_prior_cull", "SIZE_PRIOR_CULL_ENABLED", bool, False,
         section="Track hygiene"),
    Knob("size_prior_hi", "SIZE_PRIOR_HI", float, 3.0, min=1.0,
         section="Track hygiene"),
    Knob("size_prior_lo", "SIZE_PRIOR_LO", float, 0.25, min=0.0, max=1.0,
         section="Track hygiene"),
    Knob("size_prior_boundaries", "SIZE_PRIOR_BOUNDARIES", int, 3, min=1,
         section="Track hygiene"),
    # tracker_prob gates (SAM3 only — SAM2 emits no per-object confidence, so
    # both stay inert there). cull_low_prob_max requires LOW mean tracker
    # confidence before the absence cull may fire — the positive
    # discriminator the qm24 false-cull failure lacked (absence+stationarity
    # can't tell static junk from a robot parked in a corner blind spot).
    # evict_prob_immune grants HIGH mean confidence immunity from the weak
    # capped-roster eviction tiers. Both default 0.0 = OFF.
    # SHELVED 2026-07-08: cull_low_prob_max is inert unless cull_hijack_enabled
    # (itself off), so it was untestable standalone. evict_prob_immune A/B'd
    # qm8/9/10 auto + qm11 full — tiny mixed deltas with small coverage
    # regressions (qm8 2724 41->37, qm9 1024 81->72), no net benefit. Kept off.
    Knob("cull_low_prob_max", "SAM2_CULL_LOW_PROB_MAX", float, 0.0, min=0.0, max=1.0,
         section="Track hygiene"),
    Knob("evict_prob_immune", "SAM2_EVICT_PROB_IMMUNE", float, 0.0, min=0.0, max=1.0,
         section="Track hygiene"),
    # Per-boundary detection archive (pure telemetry, no behavior change):
    # every re-prompt boundary's detections, all views, full-res view coords,
    # to detections.jsonl beside diagnostics. Feeds det-gated VLM reads and
    # hard-negative mining; ~KBs per match.
    Knob("detection_archive", "DETECTION_ARCHIVE", bool, True,
         section="Track hygiene"),
    # Detection-gated VLM reads: only read a bumper crop whose track box is
    # backed by an archived detection (same view, IoS >= 0.3) within
    # det_gate_max_age_s. A hijacked track YOLO no longer fires on stops
    # earning reads — and stops casting misattributed votes. Fails open when
    # the archive has no boundary coverage for that view/time.
    # SHELVED 2026-07-08 (UNPROVEN + too narrow): on qm11 full it fired only
    # 2 gate-skips of ~3950 scheduled reads — it currently gates only the
    # one-shot early-OCR pass (_vision_identify_robots_impl), not the teleop
    # VLM scheduler (_schedule_vlm_pass). Extend its scope to teleop reads and
    # evaluate with a label-accuracy metric before enabling.
    Knob("det_gated_vlm_reads", "DET_GATED_VLM_READS", bool, False,
         section="Track hygiene"),
    Knob("det_gate_max_age_s", "DET_GATE_MAX_AGE_S", float, 4.0, min=0.5, max=30.0,
         section="Track hygiene"),
    # Hard-negative mining: post-job, decode the frames named by cull/reject
    # diagnostics (size-prior, person-overlap, static-structure,
    # implausible-seed-box) and export HardNegativeLogger artifacts for
    # scripts/build_hardneg_dataset.py. Best-effort and non-fatal; OFF by
    # default since it's a retrain-loop aid, not a runtime behavior change.
    Knob("hardneg_mine", "HARDNEG_MINE", bool, False,
         section="Track hygiene"),
    # Late-seed backward backfill: a robot first detected at a LATE reprompt
    # boundary (e.g. +6s) was forward-only, permanently losing its earlier
    # history even though SAM2/SAM3 can propagate backward from the seed
    # frame just as well as forward. When True, after the forward tracking
    # loop finishes, each surviving late-seeded object is re-prompted at its
    # own seed frame and propagated backward a bounded span
    # (backfill_seconds), merging recovered masks into the results dict
    # (never clobbering a frame an earlier object already occupies).
    # DEFAULT ON — validated 2026-07-18 (qm11 auto A/B, 3 runs/arm on the
    # sprint stack): agg meanNN 1.105 -> 0.743/0.743/0.913 (baseline
    # byte-identical across runs; the backfill arm has residual
    # nondeterminism), red meanNN halved (0.972 -> ~0.45-0.50), late-seeded
    # teams recovered (6800 21->85/85/58%, 1477 43->99%, 2992 7->100%,
    # start-lags -> 0), ~+20% auto-job wall time. The apparent 4786 coverage
    # drop is a Hungarian pairing flip in compare_manta over identical
    # physical tracks, not lost tracking. Module env default
    # (SAM2_LATE_SEED_BACKFILL) stays off for direct/library callers; the
    # pipeline passes this Knob value into track().
    Knob("late_seed_backfill", "SAM2_LATE_SEED_BACKFILL", bool, True,
         section="Track hygiene"),
    # How far back to recover, in wall-clock seconds (converted to frames the
    # same way reprompt_seconds is, and rescaled for decimated corner views).
    Knob("backfill_seconds", "SAM2_BACKFILL_SECONDS", float, 4.0,
         min=0.5, max=20.0, section="Track hygiene"),
    # Offline RTS smoothing of exported 2D paths (web/field/smoother.py):
    # re-filter each slot's recorded measurements forward (no coast decay)
    # and smooth backward, then rewrite autonomous_paths field_pos and fill
    # in-gap samples (flagged interpolated=true, tracker_prob null). Live
    # HUD/radar and all tracking decisions stay causal. DEFAULT OFF —
    # 2026-07-20 qm11+qm8 A/B (docs/default-on-ab-2026-07-20.md): measured
    # samples improve on every robot, but insertions into LONG occlusion gaps
    # regress the compare_manta gate (qm11 agg 0.816->0.890, qm8 0.414->0.477;
    # qm11 R1 fills avg 2.9 m off). Re-A/B after adding a max-gap cap on
    # insertion (short-gap fills measured excellent: 0.14-0.32 m).
    Knob("rts_smooth", "RTS_SMOOTH", bool, False,
         section="Track hygiene"),
    # --- identity / VLM ---
    # "clef": Cloudflare Workers AI decision models (web/clef_reader.py) —
    # a closed-set choice over the full 6-team roster + none/other escapes.
    # No GPU: vLLM is never started for a clef job and there is no mid-job
    # fallback to the in-process model (quota/auth failure disables reads).
    Knob("team_reader", "TEAM_READER", str, "local",
         choices=("local", "deepseek", "vllm", "clef"), section="Identity & votes"),
    # Surface-aware vote gate: the bumper readers also report whether the
    # digits sit on a robot bumper band ("bumper") or are PRINTED on
    # equipment ("printed") — qm11's OVERFLOW crate base read "1477" at
    # digit_prob 1.0, a perfect read of the wrong object. The surface value
    # is always parsed and recorded in diagnostics; when this gate is ON a
    # "printed" read's vote is dropped. Complements zone_blind_votes, which
    # only acts inside explicitly mapped equipment zones (2026cur had none).
    # DEFAULT OFF pending A/B (observe-only via diagnostics until then).
    Knob("surface_gate", "VLM_SURFACE_GATE", bool, False,
         section="Identity & votes"),
    # VLM equipment adjudication (observe-only): a track static for
    # adjudicate_static_sec gets its whole-robot crop sent to the reader
    # with an "is this a robot / equipment / person?" prompt. Verdicts and
    # per-identity streaks are recorded to diagnostics ONLY — no eviction,
    # no confidence change — so the signal can be graded against replays
    # before any action is wired to it. Independent modality: equipment
    # hijacks are YOLO-corroborated, so detector-based culls can't win
    # (see size_prior_cull / cull_hijack_enabled notes above).
    Knob("equipment_adjudicate", "VLM_EQUIPMENT_ADJUDICATE", bool, False,
         section="VLM reader"),
    Knob("adjudicate_static_sec", "VLM_ADJUDICATE_STATIC_SEC", float, 6.0, min=1.0,
         section="VLM reader"),
    Knob("adjudicate_static_move_m", "VLM_ADJUDICATE_STATIC_MOVE_M", float, 0.5, min=0.0,
         section="VLM reader"),
    Knob("adjudicate_cooldown_sec", "VLM_ADJUDICATE_COOLDOWN_SEC", float, 10.0, min=1.0,
         section="VLM reader"),
    # Phase-exclusive vLLM: stop the server during tracking (its ~11GB pool
    # plus SAM3's ~11GB peak OOMs the 24GB card; it is idle in that phase
    # anyway), start it for the analysis loop's reads, stop it after the job.
    Knob("vllm_phase_exclusive", "VLLM_PHASE_EXCLUSIVE", bool, True,
         section="VLM reader"),
    # Which model serves bumper reads. Default "qwen3-vl-4b" is byte-for-byte
    # today's behavior: vLLM serving Qwen3-VL-4B fp16 at gpu-util 0.45
    # (web/start_vllm.sh). The other keys are alternative vision models served by
    # the SAME vLLM server in HF format (a different VLM_MODEL_NAME /
    # VLM_SERVER_GPU_UTIL at launch) — see web/vlm_models.py for the registry.
    # The lifecycle (web/vllm_lifecycle.py) verifies the right model is serving
    # and cold-restarts with the requested model on a mismatch. vLLM fetches an
    # uncached model from the HF hub on first serve, so "needs download" models
    # are selectable (the first run just pays the download).
    Knob("vlm_model", "VLM_MODEL", str, vlm_models.DEFAULT_KEY,
         choices=vlm_models.model_keys(), section="VLM reader"),
    # Which Workers AI decision model serves team_reader="clef" reads.
    # clef-flash ($0.038/M input tokens) tied local Qwen3-VL-8B on the
    # 200-crop bench (68 vs 69/90 legible, 0 wrong); clef ($0.24/M) and
    # clef-omni ($0.15/M) are the larger alternatives. Ignored by the other
    # readers.
    Knob("clef_model", "CLEF_MODEL", str, clef_reader.DEFAULT_MODEL,
         choices=tuple(clef_reader.CLEF_MODELS), section="VLM reader"),
    Knob("gallery_enabled", "GALLERY_ENABLED", bool, True,
         section="Identity & votes"),
    # Trust model: deposits are saved as unverified CANDIDATES; only
    # human-verified exemplars (gallery UI checkbox, or minted from identity
    # corrections) vote at match time. Unverified deposits were poisoning
    # identity — one bad row votes wrong in every later match, and DINOv2
    # can't tell same-alliance robots apart to catch it.
    Knob("gallery_verified_only", "GALLERY_VERIFIED_ONLY", bool, True,
         section="Identity & votes"),
    # Global exact slot->team assignment (identity layer): the end-of-job
    # labels (replay overlay, match_robot_identity, replay team lists) come
    # from whole-match PER-SLOT evidence — every read credited to the slot
    # its identity held at the read's frame — via an exact, collision-free,
    # deterministic assignment; ties leave the slot unassigned. Live
    # mid-match voting is untouched. DEFAULT ON since identity-binding-fix
    # (2026-10-13): on qm11 full match (80a1cd4c) the last-identity labels
    # were 0/6 (5 of 6 final identities born in the last ~10 s with no
    # votes -> NULL everywhere); the per-slot resolve replayed offline gives
    # 4/6 correct. GLOBAL_IDENTITY_RESOLVE=0 restores the last-identity
    # labels. History: 2026-07-20 qm11+qm8 A/B (docs/default-on-ab-2026-07-
    # 20.md) geometry-neutral, collision sweep fix, qm8 R1=2724 true positive.
    Knob("global_identity_resolve", "GLOBAL_IDENTITY_RESOLVE", bool, True,
         section="Identity & votes"),
    # Identity continuity (identity layer, web/field/identity.py): an
    # observation on the SAME slot its identity held last frame, with no
    # tracker binding disruption (seeded/expired/kinematic_reject/
    # occlusion_rebind_held or a feeder-track change), moved < max_m and no
    # team conflict, is not split by the appearance veto (soft 0.1 instead
    # of +0.5); young identities update appearance for their first
    # identity_warmup_frames observations. qm11 full match: 575 identities /
    # 168 s, 88% of teleop births with the slot moved < 1 m, veto-driven.
    # DEFAULT OFF pending A/B — the always-on [identity] summary line reports
    # how many veto births were continuity-eligible. max_m is capped at 1.5:
    # with the 0.1 soft penalty the association cost 0.6 + 0.1*d crosses the
    # 0.75 threshold at d = 1.5 m anyway.
    Knob("identity_continuity", "IDENTITY_CONTINUITY", bool, False,
         section="Identity & votes"),
    Knob("identity_continuity_max_m", "IDENTITY_CONTINUITY_MAX_M", float, 1.0,
         min=0.1, max=1.5, section="Identity & votes"),
    Knob("identity_warmup_frames", "IDENTITY_WARMUP_FRAMES", int, 5,
         min=0, max=30, section="Identity & votes"),
    # --- cross-camera fusion ---
    # Ray-aware cross-camera merge gate: corner cameras sit near carpet level,
    # so their range estimate is nearly meaningless (measured 1.1-3.8 m off on
    # 2026hop qm24 vs the 0.6 m isotropic radius) while bearing stays useful —
    # same-robot views were splitting into separate fused identities. The gate
    # merges on loose bearing agreement (fuse_ray_cross_m) with generous
    # range-proportional along-ray slack; contradicting per-frame bumper
    # colors veto a merge. Corners are identity evidence first, geometry
    # second. See web/field/fusion.py.
    Knob("fuse_ray_gate", "FUSE_RAY_GATE", bool, True,
         section="Fusion & field position"),
    Knob("fuse_ray_cross_m", "FUSE_RAY_CROSS_M", float, 1.0, min=0.1, max=2.0,
         section="Fusion & field position"),
    # Appearance-relink spatial gate: DINOv2 can't tell same-alliance robots
    # apart, so the fused-track registry was relinking one red robot onto
    # another red robot's inactive id (distinct robots collapse to one track).
    # When > 0, an unmatched observation may only relink to an inactive fused id
    # whose last-known position is within this many meters — a robot can't
    # teleport across the field. 0 = off (no spatial gate). See fused_track_id.py.
    Knob("relink_max_dist_m", "RELINK_MAX_DIST_M", float, 0.0, min=0.0, max=20.0,
         section="Fusion & field position"),
    # Fusion same-id merge guard: don't collapse a same-track_id group that has
    # >=2 independent main-camera detections (distinct robots the registry
    # mis-identified) — only collapse genuine corner-camera duplicates. Lets the
    # tracker's sticky gate give each robot its own slot. See fusion.py.
    # DEFAULT ON: A/B on 2026cur qm8 (red coverage 58->86%) and qm2 (aggregate
    # mean-NN 1.52->0.82) both improved with this + sticky_max_dist_m.
    Knob("merge_main_guard", "MERGE_MAIN_GUARD", bool, True,
         section="Fusion & field position"),
    # --- field tracker position stability (field-map-v2 spec) ---
    # Corner-only measurements (no main-view box) get inflated KF noise —
    # low cameras have poor range; 1.0 restores full-trust updates.
    Knob("corner_only_r_scale", "CORNER_ONLY_R_SCALE", float, 8.0,
         min=1.0, max=100.0, section="Fusion & field position"),
    # Per-missed-frame velocity decay while coasting: occluded robots ease
    # to a stop instead of sailing away; 1.0 restores constant-velocity.
    Knob("coast_vel_decay", "COAST_VEL_DECAY", float, 0.85,
         min=0.0, max=1.0, section="Fusion & field position"),
    # Sticky-rebind distance gate: a detection carrying a known SAM3 track_id
    # re-binds to that id's slot even when the registry mis-assigned the same id
    # to a different robot far away (distinct robots collapse into one slot).
    # When > 0, a sticky re-bind is only honored if the detection is within this
    # many meters of the slot's predicted position; otherwise it falls through to
    # distance matching and can claim its own slot. 0 = off. See tracker.py.
    # DEFAULT 2.5 m: A/B-validated on qm8 + qm2 with merge_main_guard (above).
    Knob("sticky_max_dist_m", "STICKY_MAX_DIST_M", float, 2.5, min=0.0, max=20.0,
         section="Fusion & field position"),
    # Occlusion re-acquisition guard: while a slot coasts hub-occluded, a
    # detection sitting on a static-structure footprint (bump/hub/trench) may
    # not re-bind that slot or seed a new one until its track_id proves motion
    # over recent frames. Targets the bump/depot mask-hijack after a robot
    # drives behind the hub (qm10 blue side). DEFAULT OFF pending A/B.
    Knob("occlusion_rebind_guard", "OCCLUSION_REBIND_GUARD", bool, False,
         section="Fusion & field position"),
    # Kinematic plausibility gate: hard-reject a tracker association that
    # implies a robot speed beyond the physical cap (FRC robots sustain
    # <= ~5.5-6 m/s) — a track "moving" several meters in one frame got
    # stolen, not driven. Additional to (not a replacement for) the KF's
    # existing r_scale downweighting of fast measurements. DEFAULT OFF
    # pending A/B. See tracker.py FieldTracker._kin_violates.
    Knob("kin_gate", "KIN_GATE_ENABLED", bool, False,
         section="Fusion & field position"),
    Knob("kin_max_speed_mps", "KIN_MAX_SPEED_MPS", float, 6.5, min=1.0,
         section="Fusion & field position"),
    # Occlusion exit portals: when a slot goes into coast behind a hub prism,
    # predict where its velocity carries it out of the hub shadow and let the
    # tracker re-acquire a detection near that exit even outside the plain
    # distance gate (a robot reappearing on the far side of the hub otherwise
    # waits for the coast-gate to widen, or seeds a duplicate under a new id).
    # Portal radius grows with coast uncertainty; teleport-through-hub paths
    # are logged only. DEFAULT OFF pending A/B. See web/field/occlusion_portal.py.
    Knob("occlusion_portal", "OCCLUSION_PORTAL_ENABLED", bool, False,
         section="Fusion & field position"),
    # Scope of the too-large image-fraction backstop in detect_bridge.
    # "always" (legacy): drop any box spanning >50% of the frame in either
    # dimension. "uncalibrated": apply only when the pose-based width checks
    # could not run — a close robot legitimately fills most of a short corner
    # panel's height (qm10: 4027 invisible to corner_left at close range).
    Knob("det_frac_backstop", "DET_FRAC_BACKSTOP", str, "always",
         choices=("always", "uncalibrated"), section="Fusion & field position"),
    # --- match-period gating ---
    # Process ONLY the autonomous window (clock-locked) instead of the whole
    # match. Used to A/B against Manta human-scouted auto paths, which only
    # cover auto. Needs a clean clock lock on the auto countdown; if the scan
    # can't find auto, the job logs a warning and runs the legacy full range.
    Knob("auto_period_only", "AUTO_PERIOD_ONLY", bool, False,
         section="Match scope"),
    # --- live-view streaming ---
    Knob("frame_stream_interval_sec", "FRAME_STREAM_INTERVAL_SEC", float, 0.1,
         min=0.0, max=5.0, section="Streaming & output"),
    Knob("frame_stream_jpeg_quality", "FRAME_STREAM_JPEG_QUALITY", int, 70,
         min=1, max=100, section="Streaming & output"),
    # --- verification overlays ---
    # Draw the reprojected field wireframe (perimeter, grid, hubs, bumps, towers,
    # trenches) onto the main view. Baked into both the live stream and the recorded
    # replay so the operator can confirm the calibration/results line up with the
    # real field. Off by default (it's a visual-check pass, not a scoring change).
    Knob("field_wireframe_overlay", "FIELD_WIREFRAME_OVERLAY", bool, False,
         section="Streaming & output"),
]

_BY_NAME = {k.name: k for k in KNOBS}

# Runtime overrides set from the settings UI. They sit between per-job
# overrides and the environment: job override > runtime > env > default.
# Deliberately in-memory: a server restart returns to the launch profile
# (match nights run a known-good env profile; a persisted stale override
# silently shadowing it would be a footgun). Values stored here are already
# coerced and validated.
_runtime_overrides: dict = {}


def set_runtime_override(name: str, value: Any) -> Any:
    """Validate and set a runtime override; ``None`` clears it.

    Returns the coerced value (or None when cleared). Raises ValueError on an
    unknown knob or invalid value.
    """
    knob = _BY_NAME.get(name)
    if knob is None:
        raise ValueError(
            f"unknown config knob: {name}; valid knobs: {sorted(_BY_NAME)}")
    if value is None:
        _runtime_overrides.pop(name, None)
        return None
    coerced = _coerce(knob, value)
    _validate(knob, coerced)
    _runtime_overrides[name] = coerced
    return coerced


def reset_runtime_overrides() -> None:
    _runtime_overrides.clear()


def _resolve_launch(knob: Knob) -> tuple[Any, str]:
    """Resolve a knob ignoring runtime overrides: (value, 'env'|'default')."""
    env_val = os.environ.get(knob.env)
    if env_val not in (None, ""):
        return env_val, "env"
    return knob.default, "default"


def describe_knobs() -> list[dict]:
    """Full per-knob metadata + current resolution, for the settings UI."""
    out = []
    for knob in KNOBS:
        launch_raw, launch_source = _resolve_launch(knob)
        try:
            launch_value = _coerce(knob, launch_raw)
        except ValueError:
            launch_value = knob.default
        if knob.name in _runtime_overrides:
            effective, source = _runtime_overrides[knob.name], "runtime"
        else:
            effective, source = launch_value, launch_source
        out.append({
            "name": knob.name,
            "env": knob.env,
            "type": knob.type.__name__,
            "section": knob.section,
            "default": knob.default,
            "choices": list(knob.choices) if knob.choices else None,
            "min": knob.min,
            "max": knob.max,
            "launch_value": launch_value,   # what a restart returns to
            "effective": effective,
            "source": source,               # runtime | env | default
        })
    return out


def _coerce(knob: Knob, value: Any) -> Any:
    if knob.type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip() not in ("", *_FALSY)
        raise ValueError(f"{knob.name}: cannot interpret {value!r} as bool")
    try:
        return knob.type(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{knob.name}: expected {knob.type.__name__}, got {value!r}")


def _validate(knob: Knob, value: Any) -> None:
    if knob.choices is not None and value not in knob.choices:
        raise ValueError(
            f"{knob.name}: {value!r} is not one of {list(knob.choices)}")
    if knob.min is not None and value < knob.min:
        raise ValueError(f"{knob.name}: {value!r} is below minimum {knob.min}")
    if knob.max is not None and value > knob.max:
        raise ValueError(f"{knob.name}: {value!r} is above maximum {knob.max}")


class JobConfig:
    """Resolved knob values for one inference job: override > env > default."""

    def __init__(self, overrides: Optional[dict] = None):
        overrides = dict(overrides or {})
        unknown = sorted(set(overrides) - set(_BY_NAME))
        if unknown:
            raise ValueError(
                f"unknown config knob(s): {unknown}; "
                f"valid knobs: {sorted(_BY_NAME)}")
        errors = []
        for knob in KNOBS:
            if knob.name in overrides:
                raw = overrides[knob.name]
            elif knob.name in _runtime_overrides:
                raw = _runtime_overrides[knob.name]
            else:
                env_val = os.environ.get(knob.env)
                raw = env_val if env_val not in (None, "") else knob.default
            try:
                value = _coerce(knob, raw)
                _validate(knob, value)
            except ValueError as e:
                errors.append(str(e))
                continue
            setattr(self, knob.name, value)
        if errors:
            raise ValueError("; ".join(errors))

    def to_dict(self) -> dict:
        return {knob.name: getattr(self, knob.name) for knob in KNOBS}

    def non_defaults(self) -> dict:
        """Knobs that differ from the code default — the A/B provenance record."""
        return {knob.name: getattr(self, knob.name) for knob in KNOBS
                if getattr(self, knob.name) != knob.default}

    def __repr__(self) -> str:
        nd = self.non_defaults()
        return f"JobConfig({nd if nd else 'all defaults'})"
