"""Pure helpers for VLM-based field quality checks (no app state, no I/O).

Two consumers in web/app.py:
- equipment adjudication: is a long-static track actually a robot? Equipment
  hijacks are YOLO-corroborated (the detector false-fires on the equipment
  itself), so detector-based culls can't catch them — the VLM is the only
  independent modality short of a hard-negative retrain.
- full-field census (Task 4 of the same plan): reconcile what the VLM sees
  in a whole frame against the live track roster.
"""
from __future__ import annotations

import json


def track_is_static(history, now: float, window_sec: float, max_move_m: float) -> bool:
    """True when `history` proves the track sat still for the last `window_sec`.

    `history` is an iterable of (t, x, y) field-coordinate samples (meters).
    Requires samples covering (nearly) the whole window — a track we only
    just started observing cannot be declared static. Every sample in the
    window must lie within `max_move_m` of the window's oldest sample.
    """
    window = [(t, x, y) for (t, x, y) in history if now - t <= window_sec]
    if not window:
        return False
    t0, x0, y0 = window[0]
    # 0.9 slack: sample cadence rarely lands exactly on the window edge.
    if (now - t0) < window_sec * 0.9:
        return False
    return all(
        ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 <= max_move_m
        for (_t, x, y) in window
    )


_OBJECT_CLASSES = ("robot", "equipment", "person", "unclear")


def _extract_json(text):
    """Best-effort dict from VLM output (handles ``` fences and prose)."""
    if not isinstance(text, str) or not text:
        return None
    clean = text
    if "```" in clean:
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(clean[start:end])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_object_class(text) -> str:
    """Parse the adjudication reply -> robot|equipment|person|unclear.

    Anything malformed or unexpected is "unclear" — the caller treats
    unclear as no-evidence, never as an equipment verdict."""
    parsed = _extract_json(text)
    if parsed is None:
        return "unclear"
    value = parsed.get("object")
    if isinstance(value, str) and value.strip().lower() in _OBJECT_CLASSES:
        return value.strip().lower()
    return "unclear"
