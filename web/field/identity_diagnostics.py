"""Structured performance and decision logs for track identity and VLM scheduling.

Ensures explainable inference decisions and performance tracing.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set
import numpy as np


class DiagnosticEvents:
    # SAM2
    SAM2_OBJECT_CREATED = "sam2.object_created"
    SAM2_OBJECT_RETIRED = "sam2.object_retired"
    SAM2_REPROMPT_CANDIDATE = "sam2.reprompt_candidate"
    SAM2_REPROMPT_REJECTED = "sam2.reprompt_rejected"
    SAM2_MASK_GAP = "sam2.mask_gap"
    SAM2_MASK_REJECTED = "sam2.mask_rejected"
    SAM2_CHUNK_COMPLETE = "sam2.chunk_complete"
    SAM2_OBJECT_CULLED = "sam2.object_culled"
    SAM2_OBJECT_REANCHORED = "sam2.object_reanchored"
    SAM2_DETECTOR_ERROR = "sam2.detector_error"
    SAM2_COLOR_BACKFILLED = "sam2.color_backfilled"

    # Fusion
    FUSION_ASSIGNMENT = "fusion.assignment"
    FUSION_RELINK = "fusion.relink"
    FUSION_LOW_MARGIN = "fusion.low_margin"
    FUSION_EMBEDDING_UPDATE_SKIPPED = "fusion.embedding_update_skipped"

    # Tracker
    TRACKER_ASSIGNMENT = "tracker.assignment"
    TRACKER_SLOT_EXPIRED = "tracker.slot_expired"
    TRACKER_GATE_REJECTED = "tracker.gate_rejected"

    # Identity
    IDENTITY_CREATED = "identity.created"
    IDENTITY_PROMOTED = "identity.promoted"
    IDENTITY_REATTACHED = "identity.reattached"
    IDENTITY_DETACHED = "identity.detached"
    IDENTITY_BLOCKED = "identity.blocked"
    IDENTITY_TOMBSTONED = "identity.tombstoned"
    IDENTITY_ASSIGNMENT_AMBIGUOUS = "identity.assignment_ambiguous"
    IDENTITY_BINDING_COMMITTED = "identity.binding_committed"
    IDENTITY_CONFIDENCE_CHANGED = "identity.confidence_changed"

    # Occlusion
    OCCLUSION_GROUP_OPENED = "occlusion.group_opened"
    OCCLUSION_GROUP_REEMERGED = "occlusion.group_reemerged"
    OCCLUSION_GROUP_RESOLVED = "occlusion.group_resolved"
    OCCLUSION_GROUP_UNRESOLVED = "occlusion.group_unresolved"

    # VLM
    VLM_SCHEDULED = "vlm.scheduled"
    VLM_DEFERRED = "vlm.deferred"
    VLM_CROP_UNAVAILABLE = "vlm.crop_unavailable"
    VLM_RESULT = "vlm.result"
    VLM_AUDIT_OVERDUE = "vlm.audit_overdue"
    VLM_CONTRADICTION = "vlm.contradiction"

    # Scoring
    SCORING_ATTRIBUTED = "scoring.attributed"
    SCORING_DEFERRED = "scoring.deferred"
    SCORING_BACKFILLED = "scoring.backfilled"

    # Performance
    PERFORMANCE_SLOW_STAGE = "performance.slow_stage"

    # Invariants
    INVARIANT_VIOLATION = "invariant.violation"


class DiagnosticReasons:
    BELOW_SCORE_FLOOR = "below_score_floor"
    BELOW_MARGIN = "below_margin"
    ALLIANCE_CONFLICT = "alliance_conflict"
    ROSTER_CAP = "roster_cap"
    LIFETIME_CAP = "lifetime_cap"
    IDENTITY_ALREADY_ACTIVE = "identity_already_active"
    NO_USABLE_CROP = "no_usable_crop"
    VLM_TOP_K_STARVATION = "vlm_top_k_starvation"
    UNREADABLE_RESULT = "unreadable_result"
    REATTACH_CONFIDENT = "reattach_confident"
    MASK_IMPLAUSIBLE = "mask_implausible"
    DETECTOR_ABSENT_STATIONARY = "detector_absent_stationary"
    ALREADY_TRACKED = "already_tracked"
    EVICTED_FOR_NEW_DETECTION = "evicted_for_new_detection"
    STATIC_STRUCTURE = "static_structure"


ALL_EVENTS = {getattr(DiagnosticEvents, k) for k in dir(DiagnosticEvents) if not k.startswith("_")}
# The SAM tracker code is shared between backends and emits events under its
# DIAG_PREFIX ("sam2" or "sam3"). Mirror every sam2.* event for sam3 so SAM3
# runs aren't schema-rejected (they were silently dropped: qm10 dropped_logs=201).
ALL_EVENTS |= {f"sam3.{e.split('.', 1)[1]}" for e in ALL_EVENTS if e.startswith("sam2.")}
ALL_REASONS = {getattr(DiagnosticReasons, k) for k in dir(DiagnosticReasons) if not k.startswith("_")}


class IdentityDiagnosticRecorder:
    """Records diagnostic decision events and performance timings."""

    def __init__(self, clock=None, max_events: int = 1000) -> None:
        self.clock = clock or time.perf_counter
        self._max_events = max_events
        self._events: List[Dict[str, Any]] = []
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, Any] = {}
        self._timings: Dict[str, List[float]] = {}
        self._dropped_logs = 0
        self._context: Dict[str, Any] = {}

    def set_context(self, **kwargs) -> None:
        """Set shared frame/match metadata for subsequent events."""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear the shared metadata context."""
        self._context.clear()

    def increment(self, counter: str, amount: int = 1) -> None:
        """Increment a diagnostic counter."""
        self._counters[counter] = self._counters.get(counter, 0) + amount

    def gauge(self, name: str, value: Any) -> None:
        """Update a gauge value (scalar or serializable)."""
        self._gauges[name] = self._normalize(value)

    def record(self, event: str, **fields) -> None:
        """Record a structured decision or system state transition."""
        try:
            # Perform schema validation checks
            if event not in ALL_EVENTS:
                raise ValueError(f"Unknown diagnostic event type: {event}")

            reason = fields.pop("reason", None)
            if reason is not None and reason not in ALL_REASONS:
                raise ValueError(f"Unknown diagnostic reason: {reason}")

            details = fields.pop("details", {})
            if not isinstance(details, dict):
                details = {"value": details}

            # Merge other kwargs into details
            if fields:
                details.update(fields)

            payload = {
                "schema_version": 1,
                "event": event,
                "job_id": self._context.get("job_id"),
                "source_frame": self._context.get("source_frame"),
                "processed_frame": self._context.get("processed_frame"),
                "match_sec": self._context.get("match_sec"),
                "camera": self._context.get("camera"),
                "slot_id": self._context.get("slot_id"),
                "identity_id": self._context.get("identity_id"),
                "fused_track_id": self._context.get("fused_track_id"),
                "reason": reason,
                "details": self._normalize(details),
            }

            # Direct overrides from details if present (standard schema fields)
            for key in ["camera", "slot_id", "identity_id", "fused_track_id", "job_id", "source_frame", "processed_frame", "match_sec"]:
                if key in payload["details"]:
                    payload[key] = payload["details"].pop(key)

            if len(self._events) < self._max_events:
                self._events.append(payload)
            else:
                self._dropped_logs += 1
        except Exception:
            self._dropped_logs += 1

    @contextmanager
    def stage_timer(self, stage: str, **context):
        """Context manager to track execution time of a stage and alert if slow."""
        t0 = self.clock()
        try:
            yield
        finally:
            elapsed = self.clock() - t0
            if stage not in self._timings:
                self._timings[stage] = []
            self._timings[stage].append(elapsed)

            slow_threshold = context.get("slow_threshold")
            if slow_threshold is not None and elapsed > slow_threshold:
                self.record(
                    DiagnosticEvents.PERFORMANCE_SLOW_STAGE,
                    reason=None,
                    details={
                        "stage": stage,
                        "duration": elapsed,
                        "threshold": slow_threshold,
                        **context,
                    },
                )

    def summary(self) -> Dict[str, Any]:
        """Generate timing percentile aggregates, counters, and gauges."""
        timing_summary = {}
        for stage, durations in self._timings.items():
            if not durations:
                continue
            arr = np.array(durations)
            timing_summary[stage] = {
                "count": len(durations),
                "total": float(np.sum(arr)),
                "mean": float(np.mean(arr)),
                "max": float(np.max(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            }
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "dropped_logs": self._dropped_logs,
            "timings": timing_summary,
        }

    def flush_jsonl(self, path: str) -> None:
        """Write all cached events out as a JSONL log file."""
        try:
            with open(path, "w") as f:
                for ev in self._events:
                    f.write(json.dumps(ev) + "\n")
        except Exception:
            self._dropped_logs += len(self._events)

    def _normalize(self, val: Any) -> Any:
        """Ensure types are JSON serializable (handling numpy scalars, arrays, collections)."""
        if isinstance(val, (np.integer, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(val)
        if isinstance(val, (np.floating, np.float16, np.float32, np.float64)):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, dict):
            return {k: self._normalize(v) for k, v in val.items()}
        if isinstance(val, (list, tuple, set)):
            return [self._normalize(x) for x in val]
        return val
