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

from web import clef_reader, jev_reader, vlm_models

# Matches the truthiness rule used by the app.py env constants.
_FALSY = ("0", "false", "False")

# Frame-denominated tracking knobs (selective_*_frames/trigger/streak,
# evict_dark_frames, dup_reap_updates, ...) and the tracker's hardcoded frame
# budgets were all tuned at the default sam3_track_fps of 5.0. Their values are
# read as "frames at this reference rate"; with frame_knobs_fps_relative ON the
# runtime objects get them rescaled to the job's actual tracking fps so each
# keeps its wall-clock duration (a 7-frame gap = 1.4 s at 5 fps AND at 10 fps).
FRAME_KNOB_REF_FPS = 5.0


def scale_frames(n_ref: int, fps: float, ref_fps: float = FRAME_KNOB_REF_FPS,
                 *, min_value: int = 1) -> int:
    """Rescale a frame count tuned at ``ref_fps`` to ``fps``, keeping its
    duration in seconds: ``max(min_value, round(n_ref * fps / ref_fps))``.

    ``n_ref <= 0`` is returned unchanged (0 conventionally means disabled /
    uncapped; a negative env sentinel is never turned into a threshold).
    At ``fps == ref_fps`` the result is exactly ``n_ref`` for any
    ``n_ref >= min_value`` (integer * 5.0 / 5.0 is exact in float). Python's
    ``round`` is round-half-to-even, matching view_stride.scale_frame_knob.
    """
    n_ref = int(n_ref)
    if n_ref <= 0:
        return n_ref
    return max(int(min_value), int(round(n_ref * float(fps) / float(ref_fps))))


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
    # Legacy names still accepted (Nov 2026 sam2_* -> sam3_* rename): old
    # override keys (persisted config_json / replay configs, A/B profiles,
    # runtime overrides) and old env vars (operator .env files). The canonical
    # name wins when both are given. Never listed by describe_knobs().
    aliases: tuple = ()
    env_aliases: tuple = ()


KNOBS = [
    # --- tracking backend ---
    Knob("tracking_backend", "TRACKING_BACKEND", str, "sam3",
         choices=("sam2", "sam3"), section="Tracking backend"),
    Knob("sam2_model_size", "SAM2_MODEL_SIZE", str, "small",
         choices=("tiny", "small", "base_plus", "large"), section="Tracking backend"),
    # Nov 6 2026 interim live-event defaults ("B3"): sam3_track_fps 5,
    # corner_track_stride 3, selective_mask_prop + selective_app_weight 0.3,
    # stream_analysis, team_reader jev + read_census + read_keep_evicted.
    # Full match 8.9 min (qm8) / 11.3 min (qm11) vs ~25 min; costs some auto
    # coverage on a few robots. Superseded once SAM 3.1 multiplex works.
    Knob("sam3_track_fps", "SAM3_TRACK_FPS", float, 5.0, min=0.5, max=60.0,
         section="Tracking backend",
         aliases=("sam2_track_fps",), env_aliases=("SAM2_TRACK_FPS",)),
    # Frame-denominated tracking knobs are expressed "in frames at 5 fps"
    # (FRAME_KNOB_REF_FPS): every one was tuned at the 5 fps default, so at
    # another sam3_track_fps they silently changed wall-clock meaning
    # (selective_gap_trigger 7 = 1.4 s at 5 fps but 0.7 s at 10 fps). ON: the
    # runtime tracker objects get each duration-meaning frame count (selective
    # gap/streak/window/kill-coast/starve frames, evict_dark_frames,
    # dup_reap_updates, blind_zone_max_coast and the trackers' hardcoded coast
    # / retirement / reseed / dark-stop budgets) rescaled by
    # track_fps / 5, so durations stay the same in seconds. The stored knob
    # values (UI, persisted config) are unchanged. At 5 fps the scale is 1.0:
    # byte-identical to OFF. OFF = raw frame counts at any fps (A/B arm).
    # Counts of things (chunk sizes, corner_track_stride, re-prompt
    # boundaries, VLM image counts) are never scaled.
    Knob("frame_knobs_fps_relative", "FRAME_KNOBS_FPS_RELATIVE", bool, True,
         section="Tracking backend"),
    # Floor 0.25: below that SAM is starved of pixels; above 1.0 only
    # upsamples (no information) and inflates mask RAM.
    Knob("sam3_export_scale", "SAM3_EXPORT_SCALE", float, 0.5, min=0.25, max=1.0,
         section="Tracking backend",
         aliases=("sam2_export_scale",), env_aliases=("SAM2_EXPORT_SCALE",)),
    Knob("sam3_max_gpu_frames", "SAM3_MAX_GPU_FRAMES", int, 0, min=0,
         section="Tracking backend",
         aliases=("sam2_max_gpu_frames",), env_aliases=("SAM2_MAX_GPU_FRAMES",)),
    # Seed at the active-period open (0.0) rather than +3s: the +3s single seed
    # left clock 0:20->0:17 untracked and missed robots still en route. Validated
    # 2026-07-06 as lever 2 across qm8/10/11/12 (agg meanNN improved on every
    # match; 6800/870 late-start robots now seed at ~0 lag).
    Knob("sam3_prompt_offset_seconds", "SAM3_PROMPT_OFFSET_SECONDS", float, 0.0,
         min=0.0, section="Seeding & re-seeding",
         aliases=("sam2_prompt_offset_seconds",), env_aliases=("SAM2_PROMPT_OFFSET_SECONDS",)),
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
    # Processing lead before the auto clock start_frame (teleop keeps 0.5 s).
    # The broadcast clock holds 0:20 for 1.0-1.8 s after the real start, so
    # the legacy 0.5 s lead missed starting positions and first motion;
    # stops at the clock run's first sample. 0.5 reproduces the legacy lead.
    Knob("auto_start_lead_seconds", "AUTO_START_LEAD_SECONDS", float, 2.5, min=0.0,
         section="Seeding & re-seeding"),
    # Motion override: frame-diff pixel motion as positive robot evidence at
    # re-prompt boundaries — a moving detection claimed by a stationary mask
    # seeds anyway, and may evict a stationary once-absent object from a
    # capped roster (qm11: sprinting robot starved by already_tracked/
    # roster_cap against equipment-parked masks). DEFAULT OFF pending A/B.
    Knob("motion_override", "SAM3_MOTION_OVERRIDE_ENABLED", bool, False,
         section="Seeding & re-seeding",
         env_aliases=("SAM2_MOTION_OVERRIDE_ENABLED",)),
    # Min mean-abs-gray-diff (normalized /255) inside the candidate box, one
    # track_fps step apart, to count as "moving". Static-scene JPEG noise is
    # <0.01; a robot sweeping its own box scores >0.06.
    Knob("motion_energy_thresh", "SAM3_MOTION_ENERGY_THRESH", float, 0.04,
         min=0.0, max=1.0, section="Seeding & re-seeding",
         env_aliases=("SAM2_MOTION_ENERGY_THRESH",)),
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
    # (SAM3_NEG_POINT_PROMPTS) stays off for direct/library callers; the
    # pipeline passes this Knob value into track().
    Knob("neg_point_prompts", "SAM3_NEG_POINT_PROMPTS", bool, True,
         section="Seeding & re-seeding",
         env_aliases=("SAM2_NEG_POINT_PROMPTS",)),
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
    # Base SAM3 only (ignored under sam3_multiplex): on the instance-prompt
    # propagation_partial path, upstream _prepare_backbone_feats runs the full
    # detector (DETR encoder/decoder/seg heads/NMS, ~33-38 ms/frame) and
    # discards the detections; only the tracker backbone features are used.
    # ON computes backbone + tracker neck only. Offline 2026-11-06 (qm11 v_main,
    # 9 objs): byte-identical masks, -36 ms/frame
    # (docs/sam3-per-frame-profile-2026-11-06.md). DEFAULT ON since 2026-11-12
    # (live A/B: qm8 GT tracking identical, see selective_retire_closed).
    Knob("sam3_skip_detector_heads", "SAM3_SKIP_DETECTOR_HEADS", bool, True,
         section="Tracking backend"),
    # Retire culled/evicted tracks from the SAM3 session (remove_object)
    # instead of only filtering their outputs. Today a culled hijack keeps its
    # last_box fresh, so the adapter re-anchors it every window and re-seeds
    # it at every chunk boundary: tracked forever at ~42 ms/frame. Offline
    # 2026-11-06: -~42 ms/frame per retired object, survivors' masks
    # byte-identical. Bookkeeping-only under sam3_multiplex. DEFAULT ON since
    # 2026-11-12 (live A/B, see selective_retire_closed).
    Knob("sam3_retire_culled", "SAM3_RETIRE_CULLED", bool, True,
         section="Tracking backend"),
    # Threads for SAM3 chunk-session frame loading. Upstream decodes+resizes
    # every chunk JPEG one at a time with PIL (the session-open bottleneck on
    # many-core / slow-single-thread hosts); this fans upstream's own
    # per-frame loader over a shared thread pool (PIL releases the GIL).
    # 0 = auto (min(16, effective CPUs), <=1 -> off), 1 = upstream
    # single-thread loader, N>1 = N threads. Byte-identical masks, so DEFAULT
    # ON (auto): Nov 12 RunPod microbench session open 8.3 s -> 1.2 s at 16
    # threads, masks byte-identical (est. -100-130 s/job on the pod).
    Knob("sam3_frame_load_workers", "SAM3_FRAME_LOAD_WORKERS", int, 0,
         min=0, max=64, section="Tracking backend"),
    # Propagate the SAM3 camera views (main + corners) CONCURRENTLY instead of
    # one after another: one predictor per in-flight view (~3.5 GB weights +
    # its chunk session each) on worker threads, each on its own CUDA stream
    # (sam3_parallel_streams), shared YOLO/embedder calls serialised, results
    # merged in view order (same output as sequential). Batch-1 SAM3 is
    # overhead-bound (2026-11-12 RTX PRO 6000 96 GB pod, qm8: util 0-77%,
    # 9-12 GB peak; main 170 s + corners 33 s + 24 s run back to back), so SAM
    # wall should drop from sum(views) towards max(view). For >=48 GB cards:
    # a free-VRAM check after the first predictor loads caps the concurrency
    # (sam3_parallel_max is the ceiling) and falls back to sequential with a
    # log line when even 2 don't fit (a 24 GB card always does at chunk 200).
    # DEFAULT OFF until validated on the pod. See web/field/parallel_views.py.
    Knob("sam3_parallel_views", "SAM3_PARALLEL_VIEWS", bool, False,
         section="Tracking backend"),
    Knob("sam3_parallel_max", "SAM3_PARALLEL_MAX", int, 3, min=1, max=8,
         section="Tracking backend"),
    # Per-view CUDA streams under sam3_parallel_views (no effect when it is
    # off). OFF = all views share the default stream: CPU-side overhead still
    # overlaps but kernels serialise — the fallback if streams misbehave.
    Knob("sam3_parallel_streams", "SAM3_PARALLEL_STREAMS", bool, True,
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
    Knob("selective_mask_prop", "SELECTIVE_MASK_PROP", bool, True,
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
    Knob("selective_app_weight", "SELECTIVE_APP_WEIGHT", float, 0.3,
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
    # Segmented episode propagation (docs/sam3-per-frame-profile-2026-11-06.md
    # lever 1, selective half). A selective episode is ONE SAM session that
    # propagates every window's oid until ALL its windows close; on main that
    # is 4-8 episodes covering 95-99% of frames, so a finished oid keeps
    # costing ~42 ms/frame to the episode end and selective saves nothing vs
    # uniform (main 425-596 vs 417-550 ms/frame). ON: as soon as every window
    # of an oid has closed, break the propagate, remove_object it from the
    # SAM3 session and resume at the next frame (survivors resume with a bare
    # `refine` action, no new prompt). Outcome-preserving: the ExitMachine
    # never reads a fully-closed oid's masks (kill pairs and settles only
    # involve open windows). SAM3 base model only (SAM2 / multiplex keep the
    # single-pass episode). DEFAULT ON since 2026-11-12: with the two SAM3 levers
    # above, live A/B cc qm1 14.6 -> 9.1 min, hop qm45 main view 753 -> 423
    # ms/frame (busy matches chain windows into 1-2 episodes), qm8 9.5 -> 8.0
    # min with GT tracking identical and labels within run-to-run noise.
    Knob("selective_retire_closed", "SELECTIVE_RETIRE_CLOSED", bool, True,
         section="Tracking backend"),
    # Lazy seeding for selective episodes. Today every oid is prompted up
    # front at its own seed frame and propagated from the EPISODE start, so a
    # late-seeded oid is paid for (and conditioned on its future prompt) on
    # frames before its window even exists. ON: hold the seed prompt back
    # until propagation reaches the seed frame, then add it and resume.
    # Changes the late oid's masks somewhat (no future-conditioned warm-up;
    # windows never read pre-seed frames, so what they see is otherwise the
    # same), hence its own knob. SAM3 base model only. DEFAULT OFF pending
    # live A/B.
    Knob("selective_lazy_seed", "SELECTIVE_LAZY_SEED", bool, False,
         section="Tracking backend"),
    # torch.compile for SAM3 mask propagation. First run pays inductor
    # compile warm-up (cached on disk thereafter); A/B knob for the
    # propagation stage that dominates tracking wall time.
    Knob("sam3_compile", "SAM3_COMPILE", bool, False, section="Tracking backend"),
    # Corner-cam decimation: track corner views at track_fps/stride. SAM
    # propagation cost scales with frames/view (export resolution is NOT a
    # lever — the model resizes everything to a fixed square). The June qm24
    # 45 s smoke saw stride 2 cost confirmed locks (5/6 -> 3/6, 1/6), but the
    # Oct 31 2026 full-match A/B (qm8 + qm11, Clef reader, census) overturned
    # it: corner propagation -45..-53%, job wall -27..-35%, identical per-slot
    # identity results (gir/swap/segment) and equal-or-better held-out read
    # agreement; per-slot measured frames and longest unmeasured gap
    # unchanged (a corner frame every 0.27 s is well inside the ~1 s coast
    # budget). Default 2. Wall-clock-denominated tracker knobs
    # (reprompt_seconds, evict_dark_frames) are rescaled per view
    # automatically.
    Knob("corner_track_stride", "CORNER_TRACK_STRIDE", int, 3, min=1, max=8,
         section="Tracking backend"),
    # Dense head for corner decimation: corners run full-rate for the first
    # N seconds of tracked footage (the lock-formation window), then pay the
    # stride. Only meaningful when corner_track_stride > 1. Smoke A/B
    # (qm24 45s, June 12 2026): stride 2 + 20s head got 2/6 locks vs 3/6
    # same-day baseline — softer than pure stride 2 (1/6). 25 s is the
    # head used in the Oct 31 full-match A/B that made stride 2 the default.
    Knob("corner_dense_seconds", "CORNER_DENSE_SECONDS", float, 25.0, min=0.0,
         section="Tracking backend"),
    # Overlap the analysis loop's SAM-independent per-frame work with SAM
    # propagation (web/analysis_prefetch.py): while the views propagate, a
    # worker thread decodes the analysis frames and runs the scoreboard OCR
    # (tesseract, CPU) plus the main 3-class YOLO and per-view person
    # detector on private model instances; the loop then takes those results
    # from a cache and computes anything missing inline. The full "loop trails
    # a SAM ready-horizon" design is NOT safe: late_seed_backfill (default on)
    # and the tracker's retroactive color backfill keep a view's frames
    # mutable until that view finishes (qm8 corner_right: first backfill seed
    # at local frame 40/727 pins the safe horizon at ~3% of the match), so the
    # loop still starts after the last view. GPU prefetch is skipped (OCR
    # only) without CUDA, for vLLM-reader jobs (VRAM handoff) and under the
    # sam3_max_gpu_frames VRAM-fit mode. DEFAULT OFF -> unchanged path.
    Knob("stream_analysis", "STREAM_ANALYSIS", bool, True,
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
    Knob("person_veto_enabled", "SAM3_PERSON_VETO_ENABLED", bool, False,
         section="Track hygiene",
         env_aliases=("SAM2_PERSON_VETO_ENABLED",)),
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
    Knob("reanchor_enabled", "SAM3_REANCHOR_ENABLED", bool, False,
         section="Track hygiene",
         env_aliases=("SAM2_REANCHOR_ENABLED",)),
    # Evidence-ranked roster: re-prompt a dark object claimed by a fresh
    # detection in place, and let weak-evidence roster members (dark or
    # repeatedly detector-orphaned) lose their slot to detector-backed
    # newcomers. The roster holds the 6 objects most likely to be robots.
    Knob("dark_reanchor_enabled", "SAM3_DARK_REANCHOR_ENABLED", bool, True,
         section="Track hygiene",
         env_aliases=("SAM2_DARK_REANCHOR_ENABLED",)),
    Knob("evict_dark_frames", "SAM3_EVICT_DARK_FRAMES", int, 8, min=1,
         section="Track hygiene",
         env_aliases=("SAM2_EVICT_DARK_FRAMES",)),
    Knob("evict_min_absences", "SAM3_EVICT_MIN_ABSENCES", int, 2, min=1,
         section="Track hygiene",
         env_aliases=("SAM2_EVICT_MIN_ABSENCES",)),
    # Misaligned-mask re-anchor: the common drift case between dark (total
    # mask loss) and sliver (area collapse) re-anchor — a healthy-size,
    # color-consistent mask that slid off its robot until detection overlap
    # fell below the match threshold. When a partial-overlap claimant
    # persists for misalign_boundaries CONSECUTIVE boundaries, re-prompt that
    # SAME id at the detection box instead of churning a new id or hitting
    # the roster cap. DEFAULT ON — validated 2026-07-08 (qm8/9/10 auto A/B):
    # neutral on the easy matches, big qm10 win (aggNN 0.644->0.520), no
    # regressions. Module env default (SAM3_MISALIGN_REANCHOR) stays off for
    # direct/library callers; the pipeline passes this Knob value into track().
    Knob("misalign_reanchor", "SAM3_MISALIGN_REANCHOR", bool, True,
         section="Track hygiene",
         env_aliases=("SAM2_MISALIGN_REANCHOR",)),
    Knob("misalign_ios_min", "SAM3_MISALIGN_IOS_MIN", float, 0.2, min=0.05, max=0.5,
         section="Track hygiene",
         env_aliases=("SAM2_MISALIGN_IOS_MIN",)),
    Knob("misalign_boundaries", "SAM3_MISALIGN_BOUNDARIES", int, 2, min=1, max=10,
         section="Track hygiene",
         env_aliases=("SAM2_MISALIGN_BOUNDARIES",)),
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
    Knob("mask_leak_cleanup", "SAM3_MASK_LEAK_CLEANUP", bool, False,
         section="Track hygiene",
         env_aliases=("SAM2_MASK_LEAK_CLEANUP",)),
    # Derive a robot's field position from the centroid of its SAM mask's
    # carpet-contact band instead of the tracked box bottom-center. An extended
    # intake/arm stretches the box, shifting the box bottom-center off the
    # robot's actual carpet contact; the mask footprint is a truer ray origin.
    # Falls back to the box bottom-center when no mask is available.
    # DEFAULT OFF (byte-identical default path) pending A/B.
    Knob("mask_footprint_pos", "MASK_FOOTPRINT_POS", bool, False,
         section="Track hygiene"),
    # Ground-centre offset (m): a robot's lowest image point (box bottom-centre
    # OR mask footprint) is its NEAR bumper edge as seen from an elevated
    # camera, so every projected robot position lands ~half a robot too close
    # to that camera (qm8 hand GT: ~0.5-0.6 m; post-hoc +0.475 m push took
    # coverage 58->74%, median err 0.66->0.39 m). When > 0, every robot
    # ground-point projection (detections, seeding, SAM tracklets, tracker
    # project_fn) is pushed this far AWAY from its own view's camera centre
    # along the horizontal ground ray. Pose-free (homography) views: no shift.
    # DEFAULT 0.0 = OFF (byte-identical) pending A/B (ab_qm8 ground-offset).
    Knob("ground_center_offset_m", "GROUND_CENTER_OFFSET_M", float, 0.0,
         min=0.0, max=2.0, section="Track hygiene"),
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
    Knob("cull_low_prob_max", "SAM3_CULL_LOW_PROB_MAX", float, 0.0, min=0.0, max=1.0,
         section="Track hygiene",
         env_aliases=("SAM2_CULL_LOW_PROB_MAX",)),
    Knob("evict_prob_immune", "SAM3_EVICT_PROB_IMMUNE", float, 0.0, min=0.0, max=1.0,
         section="Track hygiene",
         env_aliases=("SAM2_EVICT_PROB_IMMUNE",)),
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
    # (SAM3_LATE_SEED_BACKFILL) stays off for direct/library callers; the
    # pipeline passes this Knob value into track().
    Knob("late_seed_backfill", "SAM3_LATE_SEED_BACKFILL", bool, True,
         section="Track hygiene",
         env_aliases=("SAM2_LATE_SEED_BACKFILL",)),
    # How far back to recover, in wall-clock seconds (converted to frames the
    # same way reprompt_seconds is, and rescaled for decimated corner views).
    Knob("backfill_seconds", "SAM3_BACKFILL_SECONDS", float, 4.0,
         min=0.5, max=20.0, section="Track hygiene",
         env_aliases=("SAM2_BACKFILL_SECONDS",)),
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
    # "jev": TypeSafe Jev decision model (web/jev_reader.py) — same closed-set
    # read with four escapes and "tentative" low-probability reads (see
    # jev_tentative_min_p); same cloud-reader rules as clef.
    Knob("team_reader", "TEAM_READER", str, "jev",
         choices=("local", "deepseek", "vllm", "clef", "jev"), section="Identity & votes"),
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
    # Which TypeSafe Jev model serves team_reader="jev" reads. Pinned to
    # jev-1.14.0 ($0.042/M input tokens, ~0.2 s/read): the 200-crop bench
    # (59/90 legible, 0 wrong, 2% FP) and the tentative floor below were
    # measured on it; "jev-latest" floats. Ignored by the other readers.
    Knob("jev_model", "JEV_MODEL", str, jev_reader.DEFAULT_MODEL,
         choices=tuple(jev_reader.JEV_MODELS), section="VLM reader"),
    # Jev "tentative" reads (team_reader="jev" only): when the top option is
    # "unreadable" but the best TEAM option has p >= this, the read names that
    # team as tentative. Bench: the true team was the best team option in
    # 23/31 legible abstains; p >= 0.17 (fitted to the bench) -> ~73/90 with
    # 0 wrong. Tentative reads are WEAK evidence (qm8: Clef reads < 0.65 were
    # 3/4 wrong on red): they vote p x camera weight (below a normal read's
    # 0.5) and never drive a consecutive / contradiction / co-occlusion lock.
    # 0 disables tentative reads.
    Knob("jev_tentative_min_p", "JEV_TENTATIVE_MIN_P", float,
         jev_reader.DEFAULT_TENTATIVE_MIN_P, min=0.0, max=1.0, section="VLM reader"),
    # Multi-frame Clef reads (team_reader="clef" only; ignored otherwise): send
    # the newest bumper crop plus the sharpest earlier crops of the SAME
    # identity (a per-identity rolling buffer of the last clef_mf_window_s
    # source seconds, flushed whenever the object behind the identity may
    # have changed) in ONE request, up to clef_mf_frames images (the API max
    # is 4). Tracklet bench: clef-flash 31/48 -> 39-41/48 legible, 0 wrong,
    # 2% FP. A "same_robot" question rides along; below
    # clef_mf_same_robot_min the read is discarded as tracklet_inconsistent
    # (a hijack alarm). DEFAULT OFF pending a full-match A/B.
    Knob("clef_multiframe", "CLEF_MULTIFRAME", bool, False, section="VLM reader"),
    Knob("clef_mf_frames", "CLEF_MF_FRAMES", int, 4,
         min=2, max=clef_reader.CLEF_MAX_IMAGES, section="VLM reader"),
    Knob("clef_mf_window_s", "CLEF_MF_WINDOW_S", float, 3.0, min=0.5, max=10.0,
         section="VLM reader"),
    Knob("clef_mf_same_robot_min", "CLEF_MF_SAME_ROBOT_MIN", float,
         clef_reader.DEFAULT_SAME_ROBOT_MIN, min=0.0, max=1.0, section="VLM reader"),
    # Per-slot read census: every live, bound slot's current identity becomes
    # read-due once its last read ATTEMPT (on that slot or by that identity)
    # is read_census_interval_s old — regardless of confirmation, identity
    # births or audits — weakest-evidence slot first (per-slot top-minus-
    # runner-up vote margin). Decouples read supply from identity churn: qm11
    # full match read 107-170 times with blue slots attempted every 7-10 s
    # (median) and 4786 named <=1x, and identity_continuity cut births
    # 575->326, reads 170->145 and binding 4/6->1/6.
    # ~3 reads/s target (6 slots / 2 s, ~500/match, ~$0.02 on clef-flash).
    # Works with any team_reader (local vLLM: GPU time, warned once).
    # Census reads honour the single-flight pass and the all-camera occlusion
    # gate, and skip the (position-only) novelty gate. The failure backoff
    # (3-4 s after a failed read) applies only with read_census_honor_backoff:
    # ~65% of reads name no team, so honouring it caps the illegible (mostly
    # blue) slots at ~1 read / 4.5 s — scripts/sim_read_census.py on the qm11
    # timeline: OFF 265 reads, ON 455 (2.7/s, p90 gap <= 3.5 s), ON+honour
    # 330 (p90 4.5-6 s). DEFAULT OFF pending a full-match A/B.
    Knob("read_census", "READ_CENSUS", bool, True, section="VLM reader"),
    Knob("read_census_interval_s", "READ_CENSUS_INTERVAL_S", float, 2.0,
         min=0.5, max=30.0, section="VLM reader"),
    Knob("read_census_honor_backoff", "READ_CENSUS_HONOR_BACKOFF", bool, False,
         section="VLM reader"),
    # Keep reads whose identity was evicted (registry caps) while the read was
    # in flight: the paid read still writes its vlm_read row (crop frame +
    # the identity that held the crop's slot at that frame, flagged
    # "evicted": true) so the finalize slot resolvers (global_identity_resolve
    # / identity_swap_resolve), which credit rows to slots via the identity
    # log at the crop frame, can use it. No live vote / identity / backoff /
    # census state changes: live behaviour is identical to OFF. qm11 full
    # match lost 84 of 328 paid reads (26%) to in-flight eviction with
    # read_census on (reads.dropped_evicted; reads.kept_evicted counts the
    # rows kept). DEFAULT OFF pending a full-match A/B.
    Knob("read_keep_evicted", "READ_KEEP_EVICTED", bool, True, section="VLM reader"),
    # Clef evidence floor (finalize only): team-naming Clef reads whose
    # p(team) is below this are ignored as slot evidence by the finalize
    # resolvers (global_identity_resolve, identity_swap_resolve,
    # identity_segment_resolve). Live voting is untouched. Non-Clef reads
    # (vLLM freeread confidences are on another scale) pass through. qm8
    # 0f632e0a: all 3 wrong red reads were blurry-bumper "6497" at p 0.58/
    # 0.59/0.64; below 0.65 3/4 red reads were wrong, >= 0.65 32/32 right.
    # 200-crop bench (clef-flash decoy, single frame): 0.65 keeps 61/67
    # right team reads and drops 1 of 2 wrong (precision 97.1% -> 98.4%).
    # A p(team) - p(none) margin rule (0.4) separates no better (scripts/
    # identity_offline_check.py --min-margin). Held-out agreement with the
    # floor on the training reads: qm8 +0.9 pts, qm11 -0.2 (neutral: 0-3
    # low-p reads per job). 0 = off. DEFAULT OFF pending a live A/B.
    Knob("clef_min_evidence_conf", "CLEF_MIN_EVIDENCE_CONF", float, 0.0,
         min=0.0, max=1.0, section="VLM reader"),
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
    # Swap-aware, time-varying slot labels (web/field/identity_swap.py;
    # requires global_identity_resolve): slots swap robots mid-match, so one
    # whole-match label per slot is wrong for part of the match. A Viterbi
    # over the 6 per-alliance slot->team permutations (reads = emissions, a
    # transposition costs identity_swap_lambda vote-weight units and is only
    # possible where the two slots came within identity_swap_prox_m or both
    # lost their robot) splits each slot into team segments; labels keep gir
    # semantics. Replay per-frame labels + replay events use the segment at
    # their frame; segments persist to match_robot_identity_segment; the
    # whole-match match_robot_identity label is unchanged. qm11 18a8fd23
    # offline: R1 6800->4256 / R3 4256->6800 @f6128, auto window 4/6 -> 6/6,
    # held-out read agreement 66% -> 86%; never worse on 3 other jobs.
    # lambda 3.0 > the heaviest single read (2.5): one stray read never
    # flips a segment. DEFAULT OFF pending a live full-match run.
    Knob("identity_swap_resolve", "IDENTITY_SWAP_RESOLVE", bool, False,
         section="Identity & votes"),
    Knob("identity_swap_lambda", "IDENTITY_SWAP_LAMBDA", float, 3.0,
         min=0.5, max=100.0, section="Identity & votes"),
    Knob("identity_swap_prox_m", "IDENTITY_SWAP_PROX_M", float, 1.5,
         min=0.3, max=5.0, section="Identity & votes"),
    # Segment-level slot labels (web/field/identity_segments.py; requires
    # global_identity_resolve): a slot changes physical robot ~17x per
    # alliance per match, only at tracker discontinuities (expiry+reseed,
    # wide-gate coasting re-acquisition, disrupted identity births), and
    # holds duplicates/junk -- the swap HMM's 3-slot permutation state does
    # not hold. Each slot is cut at its discontinuities, every segment is
    # labelled by its OWN reads, unread segments inherit labels along
    # physically feasible end->start links (Hungarian, <= 5 m/s + 1.5 m,
    # <= 2 s) with team exclusivity at each instant; unreachable segments
    # stay unlabelled. Replay per-frame labels, replay events and
    # match_robot_identity_segment use the segments; the whole-match
    # match_robot_identity label stays gir's. Wins over identity_swap_resolve
    # when both are on. Collects slot_boundary rows (slot_boundary_log).
    # Offline (scripts/identity_offline_check.py --segments, 50 x 30% held-
    # out reads, unlabelled = miss): qm11 7 jobs pooled gir 55.2% / swap
    # 76.2% / segments 78.3% (labels 96.6% precise, 81% of reads labelled);
    # qm8 2 jobs gir 48.4% / swap 51.1% / segments 61.8% (92.9% precise).
    # Slot-time coverage 31-69%. DEFAULT OFF pending a live full-match A/B.
    # DEFAULT FLIPPED Nov 7 2026 (Nick) with the identity combo below — qm8
    # full match vs hand GT, job ecc13f9d: mis-ID 199-274 s -> 38 s, wrong
    # labels 31-41% -> 5%, cross-alliance mis-ID 62-84 s -> 4 s (30% of time
    # unlabelled instead of guessed). One match, one run: re-check on held-out
    # matches (qm22/14/13/7 vs Manta auto).
    Knob("identity_segment_resolve", "IDENTITY_SEGMENT_RESOLVE", bool, True,
         section="Identity & votes"),
    # Deinterlace bumper crops before the reader upscale: crops of moving robots
    # from interlaced source show comb teeth that blur the digits. When a crop
    # combs, ECC-align and merge its two fields (web/bumper_register.deinterlace_crop);
    # un-combed crops are untouched. Clef-flash bench: 30 -> 35 / 48, 0 wrong.
    # Live qm8 full-match A/B (Nov 10, 2 runs/arm): NEUTRAL (misID 72/54 s vs
    # 76/61 s baseline; only ~10% of reader crops comb). DEFAULT OFF.
    Knob("bumper_deinterlace", "BUMPER_DEINTERLACE", bool, False, section="VLM reader"),
    # Window labels: finalize-stage relabel of the replay from Jev reads on
    # short track windows (digit pass, then a lineup pass on unread windows;
    # one team per frame; unread windows stay unlabelled). Spec:
    # docs/superpowers/specs/2026-11-09-jev-window-labels-design.md. qm8 GT,
    # replay a0bbc991, offline: right 37% -> 46%, wrong 9% -> 2%, misID 69 s
    # -> 17 s at lineup 0.8. Adds ~2-4 min and ~$0.26 of Jev calls per match.
    # Live qm8 A/B: misID 40-68 s -> 12-20 s. DEFAULT ON since Nov 11 (Nick),
    # with window_labels_reader=student_only (local S1 digit reads, ~40 s and
    # ~4 cents of Jev lineup calls per match). WINDOW_LABELS=0 to disable.
    Knob("window_labels", "WINDOW_LABELS", bool, True, section="Identity & votes"),
    # Lowest probability at which a lineup answer names a window. 0.8 is the
    # conservative setting from qm8; 0.65 is the A/B arm (53% right / 3% wrong
    # / 21 s misID offline). A value above 1 disables the lineup pass.
    Knob("window_labels_lineup_min_p", "WINDOW_LABELS_LINEUP_MIN_P", float, 0.8,
         min=0.0, max=1.01, section="Identity & votes"),
    # Deinterlace window-label tiles: a native-resolution tile crop that combs
    # is ECC-merged from its two fields (web/bumper_register.deinterlace_crop)
    # before the resize, so the Jev digit reads see whole robots. Progressive
    # crops are untouched; the stage stats count tiles_deinterlaced. Cached
    # digit calls are not shared between the two settings. Needs window_labels.
    # Offline qm8 (relabel_replay a0bbc991, lineup 0.8, 2 runs per arm, Nov 10):
    # 701/964 tiles comb; digit-named windows 93-94 -> 111-114, full right
    # 46-47% -> 51-52%, wrong 1-2% -> 2%, misID 10-13 s -> 12-14 s, same cost.
    # Live qm8 full match x2 (ab_qm8 "wl-deinterlace", identical tracks): right
    # 47-50% -> 53%, wrong 1-2%, misID 11-16 s either way. DEFAULT ON (Nov 10);
    # only acts when window_labels is on. WINDOW_LABELS_DEINTERLACE=0 to disable.
    Knob("window_labels_deinterlace", "WINDOW_LABELS_DEINTERLACE", bool, True,
         section="Identity & votes"),
    # Window-label run fill (window_labels.fill_runs): a named window's team
    # spreads over the unread windows of the same track chain (one slot's
    # runs, joined across gaps <= 2 s when the slot reappears within
    # 1 m + 3 m/s * gap), between agreeing reads and out to the chain's ends,
    # stopping at a conflicting single-sample read or where another track
    # starts / ends within 3 m (where the tracker swaps robots). Fills are
    # unconfirmed (HUD "1234?", segment source window_fill) and never become
    # lineup / event anchors; one team per frame still holds. Offline, qm8
    # hand GT + Jev audit on 6 held-out matches: plain whole-run fill 85%
    # precise, this guarded rule ~98%, about +4-6 points of correctly
    # labelled robot time. DEFAULT ON (Nov 11, user request: a robot named at
    # t=15 is named back to t=0). Needs window_labels. WINDOW_LABELS_FILL=0
    # to disable.
    Knob("window_labels_fill", "WINDOW_LABELS_FILL", bool, True,
         section="Identity & votes"),
    # Window-label chain roster elimination (window_labels.eliminate_chains),
    # EXPERIMENTAL: after the fill, an unlabelled frame of a track chain takes
    # the third team of its slot's alliance when the alliance's other two
    # teams are read (digits / lineup) on its other slots in that frame and
    # the third team is on no other slot. Gated on the chain's S1 bumper
    # colours (every coloured sample must be the slot's alliance colour), so
    # it only acts with the student readers (Jev-only runs have no colours).
    # Labels are unconfirmed (segment source window_elim). Offline qm8 hand
    # GT WITHOUT the colour gate: ~+2-3 pts correct but ~1/3 wrong (all
    # cross-alliance slots). WITH the gate (Nov 11): qm8 GT correct 59.9 ->
    # 61.7%, wrong flat 1.7%, misID flat 12.8 s (54 GT-checked frames right,
    # 0 wrong); held-out Jev audit of the added frames (casnd qm14/26/30)
    # only 14 right / 6 wrong of 20 confidently judged - 4 of the 6 are a
    # duplicate slot sitting on a partner's robot (partner 1.2-1.4 m away
    # holding the team the judge saw). Stays OFF until a duplicate-slot guard
    # is validated on more matches (s1_colours are now stored in every
    # replay's window_labels_digit_calls for offline re-evaluation). Needs
    # window_labels. WINDOW_LABELS_ELIMINATE=1 to enable.
    Knob("window_labels_eliminate", "WINDOW_LABELS_ELIMINATE", bool, False,
         section="Identity & votes"),
    # Window-label digit reader (teacher-distillation spec section 14): "jev"
    # sends every digit request to Jev (unchanged); "student" reads every
    # window sample first with the local S1 student (bumper detector box near
    # the slot's projected target, padded like the harvester, PARSeq digit
    # reader, GPU) and asks Jev only for windows S1 leaves undecided;
    # "student_only" never asks Jev in the digit pass. The lineup pass is Jev
    # in every mode. No CUDA / missing models: falls back to "jev" with a
    # warning. Digit-call caches are not shared between modes. Needs
    # window_labels. DEFAULT "student_only" since Nov 11: live qm8 A/B x2
    # (spec section 15) quality-neutral vs jev (misID 11-16 vs 10-15 s, wrong
    # 1.4-2.1 vs 1.3-2.0%) at ~40 s vs 163 s stage time and ~4 vs ~20 cents.
    Knob("window_labels_reader", "WINDOW_LABELS_READER", str, "student_only",
         choices=("jev", "student", "student_only"), section="Identity & votes"),
    # Per-match corner calibration (web/field/corner_per_match.py). Broadcast
    # corner cameras are re-aimed/zoomed between matches (2026cur June corner
    # versions are 1.5-2.1 m off on every match) and the drift check covers only
    # main. Before tracking, each corner view's covering epoch is checked on ~6
    # in-match frames (<= corner_check_max_m with >= 3 tags, else >= 2 if that is
    # all the corner shows); failing that it is re-solved from THIS match's
    # footage (tags + median + locator + Jev) from the epoch's lens and, when it
    # validates within the match (<= corner_resolve_max_m), saved as a new
    # version anchored at the match and used immediately. Otherwise the corner
    # view is dropped from the job (corner_drop_on_fail). DEFAULT ON (user
    # decision Nov 11; operator spot-checks on /calibrate). Off = old behaviour.
    Knob("corner_per_match_calib", "CORNER_PER_MATCH_CALIB", bool, True,
         section="Fusion & field position"),
    Knob("corner_drop_on_fail", "CORNER_DROP_ON_FAIL", bool, True,
         section="Fusion & field position"),
    Knob("corner_check_max_m", "CORNER_CHECK_MAX_M", float, 0.25, min=0.01, max=5.0,
         section="Fusion & field position"),
    Knob("corner_resolve_max_m", "CORNER_RESOLVE_MAX_M", float, 0.15, min=0.01, max=5.0,
         section="Fusion & field position"),
    # Same-view fallback for the per-match corner check. 2025 corners (2025hop)
    # see only 1-2 AprilTags, so the tag check above is inconclusive on every
    # match and the re-solve cannot run either (needs >= 3 tags) -> both corners
    # were dropped from every job although the cameras never moved. When NO check
    # frame shows enough tags (never when tags measure the epoch off), this
    # match's median corner view is registered onto the epoch's calibration
    # frame (ORB + RANSAC homography; reference median cached as
    # calibration_refview_*.jpg in uploads); if no crop corner/centre moves more
    # than corner_same_view_max_px the epoch is kept. 2025hop qm1 vs qm2-4: 1-7 px
    # on ~965x347 crops. DEFAULT ON. Off = drop as before.
    Knob("corner_same_view_fallback", "CORNER_SAME_VIEW_FALLBACK", bool, True,
         section="Fusion & field position"),
    Knob("corner_same_view_max_px", "CORNER_SAME_VIEW_MAX_PX", float, 8.0, min=0.5,
         max=50.0, section="Fusion & field position"),
    # Auto-calibrate a camera that MOVED between matches, at job start
    # (auto_calibrate.resolve_moved_camera). Main: when the start-of-job drift
    # check fails, this match's footage is solved on its own — pose re-solve
    # against the epoch's lens, then a fresh full solve (lens + pose), then both
    # again without low-support tag decodes (tags seen in < max(5, 5%) of the
    # best tag's frames). A solve passing the usual gates (validation <=
    # CALIB_DRIFT_MAX_M, plausibility, tag spread) is saved as a new version
    # anchored at this match, logged per match (reviewable / rejectable on
    # /calibrate) and used immediately; otherwise the job aborts with the manual
    # calibration prompt as before. Corners: after the per-lens re-solves fail,
    # the corner step tries the same fresh full solve (+ low-support retry)
    # under corner_resolve_max_m. 2026hop qm45 main (re-placed, lower + zoomed):
    # full solve without 3 bogus decodes -> 0.02 m. DEFAULT ON (user request
    # Nov 12). Off = old behaviour (staged pose autofix for main).
    Knob("calib_auto_on_move", "CALIB_AUTO_ON_MOVE", bool, True,
         section="Fusion & field position"),
    # Event lineup store (docs/superpowers/specs/2026-11-10-event-lineup-design.md):
    # window_labels' lineup card borrows rows, for teams this match did not
    # read at p >= 0.85 twice, from the anchors earlier matches of the event
    # stored (frc_scorer.db event_anchors, namespace "live", tiles under
    # data/event_anchors/), and this match's anchors are written back after
    # its labels are applied. Needs window_labels (alone it does nothing and
    # logs once). Adds the colour bar and the colour sentence to the card.
    # DEFAULT OFF pending the offline evaluation (spec section 8).
    Knob("event_lineup", "EVENT_LINEUP", bool, False, section="Identity & votes"),
    # Robot appearance embedder (re-ID relink, identity veto, gallery, the
    # selective TBD appearance term). "dinov3" = web/embedder.py RobotEmbedder
    # (frozen DINOv3-S CLS 384-d, 8% inset crop, ROBOT_EMBEDDER_MODEL env
    # route). "s4" = the teacher-distilled same-robot embedder
    # (web/students/s4_embedder.py: DINOv3-S fine-tuned with SupCon, 128-d,
    # harvester crop = box + 10% pad letterboxed, unmasked, full-res frame);
    # every cosine threshold switches to its S4 value
    # (web/embedder.py S4_THRESHOLDS, spec §16). Falls back to dinov3 with a
    # warning when the checkpoint or CUDA is missing. Gallery / crop rows stay
    # separated by embed_model. DEFAULT s4 since Nov 11 (Nick): live qm8 x2
    # misID 10-13 -> 7 s, wrong 1.3-1.7 -> 0.9-1.0%; unseen 2026casnd x3
    # (spec section 17) no audited regression, appearance-veto churn -12..-53%.
    # ROBOT_EMBEDDER=dinov3 to roll back.
    Knob("robot_embedder", "ROBOT_EMBEDDER", str, "s4",
         choices=("dinov3", "s4"), section="Identity & votes"),
    # Exact slot discontinuity logging (web/field/slot_boundaries.py): one
    # diagnostic JSONL "slot_boundary" row per (frame, slot) with a seed /
    # expiry / coasting re-acquisition / feed change / identity birth /
    # corner-only transition, with its geometry. Observation-only; lets
    # scripts/identity_offline_check.py --segments cut on exact boundaries
    # instead of reconstructing them. Implied by identity_segment_resolve.
    Knob("slot_boundary_log", "SLOT_BOUNDARY_LOG", bool, False,
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
    # Per-camera appearance (identity layer, web/field/identity.py): fusion
    # averages every camera's crop embedding into one, but the same robot
    # looks different from main and from a corner. 2025hop qm2 S4 cosines,
    # same slot: main-vs-main (<=2 s apart) median 0.93 (6% < 0.59 floor),
    # corner-vs-corner 0.92 (5%), main-vs-corner SAME frame 0.62 (42%) — so
    # a corner-only observation (robot hidden from main, e.g. behind the
    # reef) was vetoed against a main-built identity; with corner cameras on
    # 2025hop births rose 2-3x (qm3 49 -> 134), mostly appearance_veto. ON:
    # identities keep per-camera EMA/anchor/exemplars and only same-camera
    # embeddings are compared (max over shared cameras); no shared camera =
    # neutral appearance (no veto, no bonus). Also applies to the appearance
    # divergence read trigger. DEFAULT ON (Nov 12, Nick: "if it helps, flip
    # it"). Live A/B, corners in: 2025hop births qm2 121->83, qm3 132->107,
    # qm4 155->102 (appearance vetoes halved), wrong-alliance labels 12->7,
    # 40->25, 124->128; 2026cur qm8 GT right 60->58%, wrong 1->1%, misID
    # 8.5->6.2 s. Cost: named time / right% 0-2 pts lower on every match.
    Knob("identity_per_cam_appearance", "IDENTITY_PER_CAM_APPEARANCE", bool, True,
         section="Identity & votes"),
    # Per-camera appearance for the FUSED-ID relink (FusedTrackRegistry, the
    # mask-tracker tracklet path): a cluster whose member keys are all new relinks onto
    # an inactive fused id on appearance; the averaged multi-camera embedding
    # rarely matches across views (main-vs-corner same-robot cosine ~0.62), so
    # the miss becomes a new fused id -> slot track change + crop-buffer flush.
    # ON: relink similarity = max over cameras the cluster and the id share
    # (per-camera EMA/anchor/exemplars); no shared camera = not a candidate.
    # Missing per-camera data on either side -> averaged comparison. OFF =
    # today's averaged relink. The always-on fused_ids gauge counts outcomes.
    # VERDICT Nov 12: keep OFF. Live A/B (corners in): ~99% of assignments are
    # member-key continuations; of 26-36 new fused ids/match only 0-1 had a
    # same-camera match above the floor (created_samecam_above_floor), and ON
    # blocks cross-view relinks that were passing (relinks -1..-6/match).
    # 2026cur qm8 GT right 60->62%, wrong 2->2%, misID 12.5->14.0 s; 2025hop
    # qm2 flat, qm4 robots/frame 4.58->4.32, named time 68->62%.
    Knob("fused_per_cam_relink", "FUSED_PER_CAM_RELINK", bool, False,
         section="Identity & votes"),
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
    # Fused-id far split: in full-SAM3 tracking a corner SAM object lives all
    # match; when it ends up on a DIFFERENT robot than the one it was first
    # cross-view-paired with, its (camera, sam_id) key still maps to that
    # robot's fused id, and the same-id merge above folded the second robot
    # into the first (2025hop qm2: 103 of 546 lost GT robot-moments, the corner
    # det absorbed into a fused obs a median 4.8 m away, range 1.5-5.8 m).
    # ON: (1) FusedTrackRegistry unbinds a key whose id is also claimed this
    # frame by a cluster farther than fused_split_far_m (the claimant nearest
    # the id's last position keeps it), so the moved cluster continues /
    # relinks / gets its own id; (2) merge_same_id_observations no longer
    # merges a same-id member farther than fused_split_far_m from the retained
    # member — it gets a fresh fused id. One switch: (1) fixes the source,
    # (2) is the safety net for residual same-id groups.
    # fused_split_far_m 2.5: same-robot same-id members differ by corner range
    # error (~0.3-1 m typical, ~2 m bad) and fusion only pairs views within
    # 0.6 m (isotropic) / 1.0 m cross-ray; 2.5 m clears a bad corner read with
    # margin, matches sticky_max_dist_m (the tracker's own "too far to be the
    # same robot" gate) and sits below the observed absorptions (median 4.8 m).
    # DEFAULT ON (Nov 14 2026 live A/B, 4 hand-GT matches, R1: cov 85.8->86.6,
    # misID 8.6->7.6 s, wrong flat; full-SAM3 2025hop_qm2 cov 79->83, labels
    # right 71->76). See fusion.py / fused_track_id.py.
    Knob("fused_split_far", "FUSED_SPLIT_FAR", bool, True,
         section="Fusion & field position"),
    Knob("fused_split_far_m", "FUSED_SPLIT_FAR_M", float, 2.5, min=0.5,
         max=20.0, section="Fusion & field position"),
    # FusedTrackRegistry.assign returns per-observation assignment evidence in
    # PASS order (direct key matches, then appearance relinks / new-with-
    # embedding, then new-without-embedding), but merge_same_id_observations
    # and the identity-registry sync pair it with fused observations BY INDEX —
    # on any frame where a new / relinked cluster precedes a continued one, a
    # slot's identity evidence (continuity type, relink margin) comes from a
    # different robot. ON returns it in input order. Ids are unchanged; only
    # which evidence each fused observation carries downstream. DEFAULT ON
    # since Nov 15 2026: live A/B on the 4 GT matches x2 was neutral to
    # slightly positive everywhere (misID qm8 10.1->8.5, hop_qm45 23.4->21.9,
    # 2025hop 5.25->4.25 s, casnd flat; coverage identical).
    Knob("fused_evidence_input_order", "FUSED_EVIDENCE_INPUT_ORDER", bool,
         True, section="Fusion & field position"),
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
    # --- field-tracker re-acquisition (slot robot changes) ---
    # Replay-A/B'd with scripts/replay_field_tracker.py on TRACKER_TRACE dumps.
    # assign_partial_fix: the slot<->detection search summed raw costs, so when
    # every full pairing held one gated-out (inf) pair (an extra out-of-gate
    # det or slot — ~7% of qm8 calls) all sums tied at inf and the first
    # permutation's in-gate pairs were applied, however poor. Fixed: gated-out
    # pairs cost a large constant, so in-gate matches are maximised then
    # cost-minimised. Replays: slot robot steals qm8 55->25, qm11 113->48.
    # assign_partial_fix + coast_gate_phys + corner_dup_suppress ("combo") are
    # DEFAULT ON since the Oct 31 2026 live full-match A/B (stride 2, Clef +
    # census, identity_offline_check): correct auto slots qm8 2->5/6, qm11
    # 1->4/6; held-out read agreement qm8 38->61%, qm11 66->87%; coverage
    # flat. reseed_identity stays OFF (neutral/negative on replays).
    Knob("assign_partial_fix", "ASSIGN_PARTIAL_FIX", bool, True,
         section="Fusion & field position"),
    # coast_gate_phys: soften the coasting gate's widening (up to 4x gate_m)
    # with physics — the distance a detection lies beyond the robot's reach
    # (phys_reach_mps x time since the slot's last measurement + slack) is
    # added to its cost, and the pair is rejected once that leaves the
    # widened gate. Never tighter than the base gate; visible slots unchanged.
    Knob("coast_gate_phys", "COAST_GATE_PHYS", bool, True,
         section="Fusion & field position"),
    # reseed_identity: leftover detections seed the free slot whose last
    # measurement explains them (min-cost) instead of the lowest-numbered one,
    # and a slot that expired < reseed_hold_s ago is not reseeded by a
    # detection beyond its physical reach (the 12 m expire->reseed teleports).
    Knob("reseed_identity", "RESEED_IDENTITY", bool, False,
         section="Fusion & field position"),
    # corner_dup_suppress: a corner-only detection within corner_dup_radius_m
    # of a main-backed same-alliance detection whose fused cluster lacks that
    # corner camera is that robot's unfused corner view (e.g. corner_right's
    # ~1.2 m lateral offset vs main and corner_left defeats the 1.0 m ray
    # gate); it is dropped before tracking, so it can't seed a duplicate slot.
    Knob("corner_dup_suppress", "CORNER_DUP_SUPPRESS", bool, True,
         section="Fusion & field position"),
    Knob("phys_reach_mps", "PHYS_REACH_MPS", float, 5.0, min=1.0, max=12.0,
         section="Fusion & field position"),
    Knob("phys_reach_slack_m", "PHYS_REACH_SLACK_M", float, 1.0, min=0.0, max=5.0,
         section="Fusion & field position"),
    Knob("phys_corner_slack_m", "PHYS_CORNER_SLACK_M", float, 2.0, min=0.0,
         max=6.0, section="Fusion & field position"),
    Knob("reseed_hold_s", "RESEED_HOLD_S", float, 1.5, min=0.0, max=10.0,
         section="Fusion & field position"),
    Knob("corner_dup_radius_m", "CORNER_DUP_RADIUS_M", float, 2.0, min=0.5,
         max=4.0, section="Fusion & field position"),
    # alliance_by_territory: decide an unknown-colour robot's alliance by the
    # field half it stands in (x < L/2 -> blue). An older-game assumption: in
    # 2026 robots cross midfield constantly. qm8 hand GT (Nov 7 2026): in the
    # OPPONENT's half a robot is followed by a WRONG-alliance slot 63% of its
    # tracked time (own half: 4%); ~1/3 of all mis-identifications are
    # cross-alliance. False = territory-free: in the field tracker an
    # unknown-colour detection may only EXTEND a live/coasting slot of either
    # alliance (known-colour assignment first, then one joint pass for the
    # leftover unknowns) and never seeds one; an unknown-alliance identity
    # birth is deferred; the SAM prompt/re-prompt gap-fill cap counts the whole
    # field (<= 6) instead of 3 per half. DEFAULT True (legacy, byte-identical)
    # pending A/B (ab_qm8 territory-free).
    # DEFAULT FLIPPED to False Nov 7 2026 (identity combo; see identity_segment_resolve).
    Knob("alliance_by_territory", "ALLIANCE_BY_TERRITORY", bool, False,
         section="Fusion & field position"),
    # Duplicate-slot guards (FieldTracker). Hand GT on qm8: most slot "swaps"
    # are duplicates — two slots on one robot (a second main-view SAM object
    # seeds a free slot next to the robot's existing slot, then the two take
    # turns). dup_seed_guard_m: no seed within this distance of any active
    # slot. dup_reap_m / dup_reap_updates: two active slots within dup_reap_m
    # for that many consecutive updates -> expire the younger. Offline replay
    # vs GT (qm8, with bumper_color_read=strict + alliance_by_territory=False):
    # slot purity 47.8 -> 61.4%, robot-on-main-slot 43.9 -> 54.4%, switches
    # 110 -> 88, coverage -0.3 pt. DEFAULT 0 = off pending live A/B.
    # DEFAULTS FLIPPED Nov 7 2026 (identity combo): guard 1.0 m, reap 0.7 m x 8.
    Knob("dup_seed_guard_m", "DUP_SEED_GUARD_M", float, 1.0, min=0.0, max=3.0,
         section="Fusion & field position"),
    Knob("dup_reap_m", "DUP_REAP_M", float, 0.7, min=0.0, max=3.0,
         section="Fusion & field position"),
    Knob("dup_reap_updates", "DUP_REAP_UPDATES", int, 8, min=1, max=60,
         section="Fusion & field position"),
    # colourless_stationary_gate: a colourless (bumper "unknown") fused
    # detection whose fused track_id has stayed within
    # colourless_stationary_span_m for the last colourless_stationary_s
    # seconds (tracker time, fps-independent) is a ghost parked on field
    # structure; it is excluded from the tracker update (claims no slot, seeds
    # none). Coloured and track_id-less detections are never gated. Why:
    # 2025hop_qm2 full SAM3 — main-only colourless objects on the processor /
    # reef held roster slots of EITHER alliance for most of the match; a static
    # no-seed zone can't fix it (it takes over existing slots). Offline trace
    # prefilter (1.0 m / 50 updates at 5 fps): GT coverage 75.6 -> 80.5%, no
    # robot worse, 96% of dropped obs on no robot. DEFAULT OFF pending live A/B.
    Knob("colourless_stationary_gate", "COLOURLESS_STATIONARY_GATE", bool, False,
         section="Fusion & field position"),
    Knob("colourless_stationary_s", "COLOURLESS_STATIONARY_S", float, 10.0,
         min=1.0, max=60.0, section="Fusion & field position"),
    Knob("colourless_stationary_span_m", "COLOURLESS_STATIONARY_SPAN_M", float,
         1.0, min=0.1, max=5.0, section="Fusion & field position"),
    # blind_zone_coast: at tracking start, compute the field cells NO camera
    # can usefully see from the calibrations (web/field/blind_zone.py: robot
    # ground point inside the view inset by blind_zone_margin_px (and, with
    # blind_zone_top_probe_m > 0, a point that far up too), depth/lens-fold
    # checks, overlay masks, SAM top crop; non-main views also need >=
    # blind_corner_min_range_m ground range). Why: some events' main views cut off the near corners and the
    # 2026 corner cams sit ~0.6-0.9 m outside those corners, so a robot that
    # drives into the gap is lost by every camera and its slot expired after
    # max_coast_open updates. With the knob a slot within blind_zone_margin_m
    # of a camera's dead area is occluded FOR THAT CAMERA (the ALL-cameras
    # merge then occludes a slot no camera sees); the tracker holds such a
    # slot in place (no velocity drift) for blind_zone_max_coast updates
    # (0 = max_coast_occluded) and lets it re-acquire only detections within
    # gate_m of the zone. Zone logged at job start + replay summary
    # "blind_zone". blind_corner_min_range_m 3.0: under det_frac_backstop
    # "always" a 0.6 m-tall robot spans > 50% of the 337-377 px 2026 corner
    # tile height until ~3.0 m (measured on 2026cur/casnd/hop calibrations).
    # margin_px 0 / top probe 0: hand GT marks robots "visible" with their
    # ground point 1-30 px inside main's side edges (2026cur_qm8: 10 px + a
    # 0.4 m probe put 779 GT-visible keyframes in the zone; 0 px, no probe 61).
    # DEFAULT ON (Nov 14 2026 live A/B, 4 hand-GT matches, R1: cov 85.8->86.5,
    # GT-"estimated" (robot unseen) coverage 48.9->52.0, casnd 50->63; wrong /
    # misID flat).
    Knob("blind_zone_coast", "BLIND_ZONE_COAST", bool, True,
         section="Fusion & field position"),
    Knob("blind_corner_min_range_m", "BLIND_CORNER_MIN_RANGE_M", float, 3.0,
         min=0.0, max=10.0, section="Fusion & field position"),
    Knob("blind_zone_margin_px", "BLIND_ZONE_MARGIN_PX", float, 0.0,
         min=-100.0, max=200.0, section="Fusion & field position"),
    Knob("blind_zone_top_probe_m", "BLIND_ZONE_TOP_PROBE_M", float, 0.0,
         min=0.0, max=1.5, section="Fusion & field position"),
    Knob("blind_zone_margin_m", "BLIND_ZONE_MARGIN_M", float, 0.3,
         min=0.0, max=2.0, section="Fusion & field position"),
    Knob("blind_zone_max_coast", "BLIND_ZONE_MAX_COAST", int, 0,
         min=0, max=3000, section="Fusion & field position"),
    # bumper_color_read: re-read each main-backed tracker input's bumper colour
    # from the main-view pixels every frame (web/field/bumper_colour.py: push
    # pos 0.475 m from the camera to the robot centre, project a 0.95 m square
    # bumper band z 0.03-0.16 m, HSV red-vs-blue fraction excluding the
    # static red/blue of a ~40-frame median background). Why: qm8 hand GT
    # (Nov 7 2026) — ~40% of tracker inputs carry colour "unknown" (selective
    # SAM tracks are TBD) and take their alliance from the field half (63%
    # wrong-colour-slot time in the opponent's half); the 6% known-but-wrong
    # inputs are 99% STALE (SAM colour fixed at seed, mask slid onto the other
    # alliance's robot). Offline read vs GT: own half 74% correct / 0.1% wrong
    # / 26% undecided; opponent half 32% / 2.3% / 66%. No per-track memory
    # (harmful: tracks swap robots). Corner-only inputs untouched.
    # "off" (default, byte-identical); "override" = a decided read replaces
    # the pipeline colour, undecided keeps it; "strict" = colour := the read
    # (undecided -> unknown, discarding stale pipeline colours). Pending A/B
    # (ab_qm8 bumper-colour).
    # DEFAULT FLIPPED to "strict" Nov 7 2026 (identity combo; pairs with
    # alliance_by_territory=False — strict leaves more inputs colourless).
    Knob("bumper_color_read", "BUMPER_COLOR_READ", str, "strict",
         choices=("off", "override", "strict"), section="Fusion & field position"),
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
    # HUD video rendered after finalize (docs/superpowers/specs/
    # 2026-11-10-hud-post-render-design.md): the job records a draw list instead
    # of encoding hud.mp4 live, and a render subprocess burns the FINAL labels
    # into the video once the replay is saved. The live preview is unchanged.
    # DEFAULT ON since Nov 11 with window_labels, so replay videos show the
    # final (window) labels, not the live ones. HUD_POST_RENDER=0 to disable.
    Knob("hud_post_render", "HUD_POST_RENDER", bool, True,
         section="Streaming & output"),
]

_BY_NAME = {k.name: k for k in KNOBS}
# Legacy override key -> canonical knob name.
_ALIASES = {a: k.name for k in KNOBS for a in k.aliases}


def canonical_knob_name(name: str) -> str:
    """The canonical knob name for ``name`` (itself unless it is a legacy
    alias such as ``sam2_track_fps``)."""
    return _ALIASES.get(name, name)


def normalize_overrides(overrides: Optional[dict]) -> dict:
    """Rewrite legacy alias keys in an override/config dict to canonical knob
    names. A canonical key wins over its alias when both are present; keys
    that are neither (unknown knobs, unrelated fields) pass through."""
    overrides = dict(overrides or {})
    out = {k: v for k, v in overrides.items() if k not in _ALIASES}
    for key, value in overrides.items():
        canon = _ALIASES.get(key)
        if canon is not None and canon not in out:
            out[canon] = value
    return out


def knob_default(name: str) -> Any:
    """Code default of the knob named ``name`` (canonical or legacy alias)."""
    return _BY_NAME[canonical_knob_name(name)].default


def _env_value(knob: Knob) -> Optional[str]:
    """The knob's env value: its canonical variable, else the first set legacy
    alias (empty strings count as unset, as before)."""
    for var in (knob.env, *knob.env_aliases):
        val = os.environ.get(var)
        if val not in (None, ""):
            return val
    return None

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
    name = canonical_knob_name(name)
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
    env_val = _env_value(knob)
    if env_val is not None:
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
        overrides = normalize_overrides(overrides)
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
                env_val = _env_value(knob)
                raw = env_val if env_val is not None else knob.default
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
