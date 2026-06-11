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
    Knob("tracking_backend", "TRACKING_BACKEND", str, "sam2",
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
    Knob("cull_hijack_enabled", "CULL_HIJACK_ENABLED", bool, False),
    Knob("reanchor_enabled", "SAM2_REANCHOR_ENABLED", bool, False),
    # --- identity / VLM ---
    Knob("team_reader", "TEAM_READER", str, "local",
         choices=("local", "deepseek", "vllm")),
    Knob("gallery_enabled", "GALLERY_ENABLED", bool, True),
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
