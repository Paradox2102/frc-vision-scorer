"""Typed registry of per-job inference knobs.

One place defining every job-scoped A/B knob: its override key, env var,
type, default, and valid range. JobConfig resolves override > env > default
at job creation and validates everything up front, so a bad knob fails the
submit with a clear message instead of failing minutes into inference.

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
    Knob("sam2_prompt_offset_seconds", "SAM2_PROMPT_OFFSET_SECONDS", float, 3.0,
         min=0.0),
    Knob("reprompt_seconds", "REPROMPT_SECONDS", float, 2.0, min=0.1),
    Knob("sam2_vos_optimized", "SAM2_VOS_OPTIMIZED", bool, False),
    # SAM3 chunked sessions: frames per GPU-resident window (~14 MB/frame).
    # Bigger = fewer identity-risky chunk boundaries; cap is VRAM, not speed.
    Knob("sam3_chunk_frames", "SAM3_CHUNK_FRAMES", int, 400, min=16, max=2000),
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
    Knob("reanchor_enabled", "SAM2_REANCHOR_ENABLED", bool, False),
    # Evidence-ranked roster: re-prompt a dark object claimed by a fresh
    # detection in place, and let weak-evidence roster members (dark or
    # repeatedly detector-orphaned) lose their slot to detector-backed
    # newcomers. The roster holds the 6 objects most likely to be robots.
    Knob("dark_reanchor_enabled", "SAM2_DARK_REANCHOR_ENABLED", bool, True),
    Knob("evict_dark_frames", "SAM2_EVICT_DARK_FRAMES", int, 8, min=1),
    Knob("evict_min_absences", "SAM2_EVICT_MIN_ABSENCES", int, 2, min=1),
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
    # --- live-view streaming ---
    Knob("frame_stream_interval_sec", "FRAME_STREAM_INTERVAL_SEC", float, 0.1,
         min=0.0, max=5.0),
    Knob("frame_stream_jpeg_quality", "FRAME_STREAM_JPEG_QUALITY", int, 70,
         min=1, max=100),
]

_BY_NAME = {k.name: k for k in KNOBS}


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
