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


KNOBS = [
    # --- tracking backend ---
    Knob("tracking_backend", "TRACKING_BACKEND", str, "sam3",
         choices=("sam2", "sam3")),
    Knob("sam2_model_size", "SAM2_MODEL_SIZE", str, "small",
         choices=("tiny", "small", "base_plus", "large")),
    Knob("sam2_track_fps", "SAM2_TRACK_FPS", float, 8.0, min=0.5, max=60.0),
    # Floor 0.25: below that SAM2 is starved of pixels; above 1.0 only
    # upsamples (no information) and inflates mask RAM.
    Knob("sam2_export_scale", "SAM2_EXPORT_SCALE", float, 0.5, min=0.25, max=1.0),
    Knob("sam2_max_gpu_frames", "SAM2_MAX_GPU_FRAMES", int, 0, min=0),
    # Seed at the active-period open (0.0) rather than +3s: the +3s single seed
    # left clock 0:20->0:17 untracked and missed robots still en route. Validated
    # 2026-07-06 as lever 2 across qm8/10/11/12 (agg meanNN improved on every
    # match; 6800/870 late-start robots now seed at ~0 lag).
    Knob("sam2_prompt_offset_seconds", "SAM2_PROMPT_OFFSET_SECONDS", float, 0.0,
         min=0.0),
    Knob("reprompt_seconds", "REPROMPT_SECONDS", float, 2.0, min=0.1),
    # qm12 0:20-0:16 — during the first `opening_window_seconds` of active
    # tracking, re-prompt every `opening_reprompt_seconds` (denser than
    # reprompt_seconds) so robots streaming into center seed within a fraction
    # of a second instead of waiting a full ~2s boundary; 0 on either disables.
    # DEFAULT ON (0.5s / 6s) — validated 2026-07-06, no cross-match regression.
    Knob("opening_reprompt_seconds", "OPENING_REPROMPT_SECONDS", float, 0.5, min=0.0),
    Knob("opening_window_seconds", "OPENING_WINDOW_SECONDS", float, 6.0, min=0.0),
    # Motion override: frame-diff pixel motion as positive robot evidence at
    # re-prompt boundaries — a moving detection claimed by a stationary mask
    # seeds anyway, and may evict a stationary once-absent object from a
    # capped roster (qm11: sprinting robot starved by already_tracked/
    # roster_cap against equipment-parked masks). DEFAULT OFF pending A/B.
    Knob("motion_override", "SAM2_MOTION_OVERRIDE_ENABLED", bool, False),
    # Min mean-abs-gray-diff (normalized /255) inside the candidate box, one
    # track_fps step apart, to count as "moving". Static-scene JPEG noise is
    # <0.01; a robot sweeping its own box scores >0.06.
    Knob("motion_energy_thresh", "SAM2_MOTION_ENERGY_THRESH", float, 0.04,
         min=0.0, max=1.0),
    # Motion-gated low-confidence admission: run the bumper detector down to
    # det_conf_floor at seed/re-prompt frames and admit the sub-threshold band
    # only when the box shows pixel motion (qm11 6800: dark blurred bumper
    # read 0.26-0.34 vs the 0.4 seed conf — blur lowers confidence, and the
    # blur IS the motion evidence). DEFAULT OFF pending A/B.
    Knob("motion_gated_conf", "MOTION_GATED_CONF_ENABLED", bool, False),
    Knob("det_conf_floor", "DET_CONF_FLOOR", float, 0.2, min=0.01, max=1.0),
    # Detector-independent seed candidates from frame-diff blobs, fed into
    # the colorless gap-fill path (color backfilled later). Rescues a robot
    # YOLO whiffs on entirely (heavy blur, dark bumpers). DEFAULT OFF
    # pending A/B.
    Knob("motion_seed_blobs", "MOTION_SEED_BLOBS_ENABLED", bool, False),
    # Blob seeding is scoped to the first N seconds of the active period
    # (the auto-start sprint problem); match-long admission tripled roster
    # churn on qm11. 0 disables the scope (blobs whole match).
    Knob("motion_blob_window_s", "MOTION_BLOB_WINDOW_S", float, 6.0, min=0.0),
    # Seed candidates adopt the union of the bumper-expanded box and the
    # overlapping whole-robot detection (bumper color kept). A blurred
    # bumper expands to a narrow sliver — a poor SAM prompt; the whole-robot
    # detector knows the true extent. DEFAULT OFF pending A/B.
    Knob("seed_union_extent", "SEED_UNION_EXTENT_ENABLED", bool, False),
    Knob("sam2_vos_optimized", "SAM2_VOS_OPTIMIZED", bool, False),
    # SAM3 chunked sessions: frames per GPU-resident window (~14 MB/frame).
    # Bigger = fewer identity-risky chunk boundaries; cap is VRAM, not speed.
    # Default lowered 400->200 for 24GB cards: 400 peaked the whole card and OOM'd
    # when other state was co-resident (see web/field/sam3_tracker.py header).
    Knob("sam3_chunk_frames", "SAM3_CHUNK_FRAMES", int, 200, min=16, max=2000),
    # Corner-cam decimation: track corner views at track_fps/stride. SAM
    # propagation cost scales with frames/view (export resolution is NOT a
    # lever — the model resizes everything to a fixed square). Stride 2 cut
    # corner propagation ~39% on the qm24 smoke, BUT cost confirmed locks
    # (5/6 at stride 1 vs 3/6 and 1/6 across two stride-2 runs) — corner
    # reads carry 2.5x vote weight and halved crop opportunities starve
    # confirmation. Default OFF; opt in per job when speed outweighs
    # identity. Wall-clock-denominated tracker knobs (reprompt_seconds,
    # evict_dark_frames) are rescaled per view automatically.
    Knob("corner_track_stride", "CORNER_TRACK_STRIDE", int, 1, min=1, max=8),
    # Dense head for corner decimation: corners run full-rate for the first
    # N seconds of tracked footage (the lock-formation window), then pay the
    # stride. Only meaningful when corner_track_stride > 1. Smoke A/B
    # (qm24 45s, June 12 2026): stride 2 + 20s head got 2/6 locks vs 3/6
    # same-day baseline — softer than pure stride 2 (1/6) but still never
    # matched baseline, so decimation stays opt-in.
    Knob("corner_dense_seconds", "CORNER_DENSE_SECONDS", float, 25.0, min=0.0),
    Knob("cull_hijack_enabled", "CULL_HIJACK_ENABLED", bool, False),
    # Scope the absence-based cull to the main view: mid-field equipment is
    # fully visible to main-view YOLO (absence there is strong evidence),
    # while corner views are where qm24's false culls killed real robots
    # parked in detector blind spots. Only meaningful with cull_hijack_enabled.
    Knob("cull_main_view_only", "CULL_MAIN_VIEW_ONLY", bool, True),
    # Person-overlap veto: cull a stationary mask that spends consecutive
    # boundaries mostly inside a person detection — a positive hijack signal,
    # independent of the absence cull (qm11: R3 spent all of auto on a ref).
    Knob("person_veto_enabled", "SAM2_PERSON_VETO_ENABLED", bool, False),
    # Zone-blind corroboration: detections whose ground contact lies inside a
    # static-structure zone (UNinflated rects) never count as mask
    # corroboration, so a bumper-lookalike FP on equipment (2026cur OVERFLOW
    # bin base reads "1477") can't vouch for its own hijacked mask and the
    # absence cull can fire. Robots can't park inside an equipment footprint,
    # so blinding corroboration there is safe. Pair with cull_hijack_enabled.
    Knob("zone_blind_corroboration", "ZONE_BLIND_CORROB", bool, False),
    # Zone-blind VOTING (identity layer): a VLM read taken while the track's
    # field position is inside a raw static zone is printed equipment text
    # (2026cur OVERFLOW crate reads "1477" at digit_prob 1.0), never a
    # bumper — discard the vote. Default ON: it only acts inside explicit
    # equipment footprints, where a robot cannot be.
    Knob("zone_blind_votes", "VLM_ZONE_BLIND_VOTES", bool, True),
    Knob("reanchor_enabled", "SAM2_REANCHOR_ENABLED", bool, False),
    # Evidence-ranked roster: re-prompt a dark object claimed by a fresh
    # detection in place, and let weak-evidence roster members (dark or
    # repeatedly detector-orphaned) lose their slot to detector-backed
    # newcomers. The roster holds the 6 objects most likely to be robots.
    Knob("dark_reanchor_enabled", "SAM2_DARK_REANCHOR_ENABLED", bool, True),
    Knob("evict_dark_frames", "SAM2_EVICT_DARK_FRAMES", int, 8, min=1),
    Knob("evict_min_absences", "SAM2_EVICT_MIN_ABSENCES", int, 2, min=1),
    # Seed-box plausibility gate: a detection box taller than this aspect
    # (h/w) may not seed, re-anchor, or evict at SAM reprompt boundaries; 0
    # disables. qm12 hub-face sliver (aspect 4.47) evicted a real track.
    # DEFAULT ON (3.0) — validated 2026-07-06; blocked the sliver hijack with
    # no cross-match regression (rejects only geometrically-impossible boxes).
    Knob("seed_box_max_aspect", "SEED_BOX_MAX_ASPECT", float, 3.0, min=0.0, max=20.0),
    # Capped-roster eviction requires the candidate detection to prove pixel
    # motion when a motion probe is available (motion_override); static
    # candidates fall through to roster_cap rejection. No-op without
    # motion_fn. qm12: a static hub-face sliver evicted a real robot.
    # DEFAULT ON — validated 2026-07-06 (no-op unless motion_override is also on).
    Knob("evict_requires_motion", "EVICT_REQUIRES_MOTION", bool, True),
    # --- identity / VLM ---
    Knob("team_reader", "TEAM_READER", str, "local",
         choices=("local", "deepseek", "vllm")),
    # Phase-exclusive vLLM: stop the server during tracking (its ~11GB pool
    # plus SAM3's ~11GB peak OOMs the 24GB card; it is idle in that phase
    # anyway), start it for the analysis loop's reads, stop it after the job.
    Knob("vllm_phase_exclusive", "VLLM_PHASE_EXCLUSIVE", bool, True),
    Knob("gallery_enabled", "GALLERY_ENABLED", bool, True),
    # Trust model: deposits are saved as unverified CANDIDATES; only
    # human-verified exemplars (gallery UI checkbox, or minted from identity
    # corrections) vote at match time. Unverified deposits were poisoning
    # identity — one bad row votes wrong in every later match, and DINOv2
    # can't tell same-alliance robots apart to catch it.
    Knob("gallery_verified_only", "GALLERY_VERIFIED_ONLY", bool, True),
    # --- cross-camera fusion ---
    # Ray-aware cross-camera merge gate: corner cameras sit near carpet level,
    # so their range estimate is nearly meaningless (measured 1.1-3.8 m off on
    # 2026hop qm24 vs the 0.6 m isotropic radius) while bearing stays useful —
    # same-robot views were splitting into separate fused identities. The gate
    # merges on loose bearing agreement (fuse_ray_cross_m) with generous
    # range-proportional along-ray slack; contradicting per-frame bumper
    # colors veto a merge. Corners are identity evidence first, geometry
    # second. See web/field/fusion.py.
    Knob("fuse_ray_gate", "FUSE_RAY_GATE", bool, True),
    Knob("fuse_ray_cross_m", "FUSE_RAY_CROSS_M", float, 1.0, min=0.1, max=2.0),
    # Appearance-relink spatial gate: DINOv2 can't tell same-alliance robots
    # apart, so the fused-track registry was relinking one red robot onto
    # another red robot's inactive id (distinct robots collapse to one track).
    # When > 0, an unmatched observation may only relink to an inactive fused id
    # whose last-known position is within this many meters — a robot can't
    # teleport across the field. 0 = off (no spatial gate). See fused_track_id.py.
    Knob("relink_max_dist_m", "RELINK_MAX_DIST_M", float, 0.0, min=0.0, max=20.0),
    # Fusion same-id merge guard: don't collapse a same-track_id group that has
    # >=2 independent main-camera detections (distinct robots the registry
    # mis-identified) — only collapse genuine corner-camera duplicates. Lets the
    # tracker's sticky gate give each robot its own slot. See fusion.py.
    # DEFAULT ON: A/B on 2026cur qm8 (red coverage 58->86%) and qm2 (aggregate
    # mean-NN 1.52->0.82) both improved with this + sticky_max_dist_m.
    Knob("merge_main_guard", "MERGE_MAIN_GUARD", bool, True),
    # --- field tracker position stability (field-map-v2 spec) ---
    # Corner-only measurements (no main-view box) get inflated KF noise —
    # low cameras have poor range; 1.0 restores full-trust updates.
    Knob("corner_only_r_scale", "CORNER_ONLY_R_SCALE", float, 8.0,
         min=1.0, max=100.0),
    # Per-missed-frame velocity decay while coasting: occluded robots ease
    # to a stop instead of sailing away; 1.0 restores constant-velocity.
    Knob("coast_vel_decay", "COAST_VEL_DECAY", float, 0.85,
         min=0.0, max=1.0),
    # Sticky-rebind distance gate: a detection carrying a known SAM3 track_id
    # re-binds to that id's slot even when the registry mis-assigned the same id
    # to a different robot far away (distinct robots collapse into one slot).
    # When > 0, a sticky re-bind is only honored if the detection is within this
    # many meters of the slot's predicted position; otherwise it falls through to
    # distance matching and can claim its own slot. 0 = off. See tracker.py.
    # DEFAULT 2.5 m: A/B-validated on qm8 + qm2 with merge_main_guard (above).
    Knob("sticky_max_dist_m", "STICKY_MAX_DIST_M", float, 2.5, min=0.0, max=20.0),
    # Occlusion re-acquisition guard: while a slot coasts hub-occluded, a
    # detection sitting on a static-structure footprint (bump/hub/trench) may
    # not re-bind that slot or seed a new one until its track_id proves motion
    # over recent frames. Targets the bump/depot mask-hijack after a robot
    # drives behind the hub (qm10 blue side). DEFAULT OFF pending A/B.
    Knob("occlusion_rebind_guard", "OCCLUSION_REBIND_GUARD", bool, False),
    # Scope of the too-large image-fraction backstop in detect_bridge.
    # "always" (legacy): drop any box spanning >50% of the frame in either
    # dimension. "uncalibrated": apply only when the pose-based width checks
    # could not run — a close robot legitimately fills most of a short corner
    # panel's height (qm10: 4027 invisible to corner_left at close range).
    Knob("det_frac_backstop", "DET_FRAC_BACKSTOP", str, "always",
         choices=("always", "uncalibrated")),
    # --- match-period gating ---
    # Process ONLY the autonomous window (clock-locked) instead of the whole
    # match. Used to A/B against Manta human-scouted auto paths, which only
    # cover auto. Needs a clean clock lock on the auto countdown; if the scan
    # can't find auto, the job logs a warning and runs the legacy full range.
    Knob("auto_period_only", "AUTO_PERIOD_ONLY", bool, False),
    # --- live-view streaming ---
    Knob("frame_stream_interval_sec", "FRAME_STREAM_INTERVAL_SEC", float, 0.1,
         min=0.0, max=5.0),
    Knob("frame_stream_jpeg_quality", "FRAME_STREAM_JPEG_QUALITY", int, 70,
         min=1, max=100),
    # --- verification overlays ---
    # Draw the reprojected field wireframe (perimeter, grid, hubs, bumps, towers,
    # trenches) onto the main view. Baked into both the live stream and the recorded
    # replay so the operator can confirm the calibration/results line up with the
    # real field. Off by default (it's a visual-check pass, not a scoring change).
    Knob("field_wireframe_overlay", "FIELD_WIREFRAME_OVERLAY", bool, False),
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
