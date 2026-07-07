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


def _clean_box(raw):
    """[x1,y1,x2,y2] floats, or None if malformed."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        return [float(v) for v in raw]
    except (TypeError, ValueError):
        return None


def parse_census(text):
    """Parse the full-field census reply.

    Returns {"robots": [{"box": [...], "color": red|blue|unknown}],
             "lookalikes": [{"box": [...], "kind": str}]} with malformed
    entries dropped, or None when no JSON object could be extracted."""
    parsed = _extract_json(text)
    if parsed is None:
        return None
    robots = []
    for entry in parsed.get("robots") or []:
        if not isinstance(entry, dict):
            continue
        box = _clean_box(entry.get("box"))
        if box is None:
            continue
        color = entry.get("color")
        robots.append({"box": box,
                       "color": color if color in ("red", "blue") else "unknown"})
    lookalikes = []
    for entry in parsed.get("lookalikes") or []:
        if not isinstance(entry, dict):
            continue
        box = _clean_box(entry.get("box"))
        if box is None:
            continue
        lookalikes.append({"box": box, "kind": str(entry.get("kind") or "unknown")})
    return {"robots": robots, "lookalikes": lookalikes}


def boxes_to_field_points(boxes, scale, image_to_field):
    """Project census boxes to field coordinates via ground contact.

    The VLM saw a frame downscaled by `scale`; divide back to full-view
    pixels, take the bottom-center (the ground-contact point — same
    convention as detection projection), and run it through the view's
    `image_to_field`. Boxes whose projection raises are skipped: the VLM
    sometimes boxes objects outside the calibrated carpet."""
    pts = []
    for box in boxes:
        u = (box[0] + box[2]) / 2.0 / scale
        v = box[3] / scale
        try:
            xy = image_to_field((u, v))
            pts.append((float(xy[0]), float(xy[1])))
        except Exception:
            continue
    return pts


def reconcile_census(census_pts, track_positions, match_radius_m=1.5):
    """Greedy nearest one-to-one matching of census points to live tracks.

    census_pts: [(x, y)] field meters; track_positions: {slot_id: (x, y)}.
    Returns {"matched": [[slot_id, census_idx, dist_m]],
             "unmatched_census": [census_idx],   # robot the tracker misses
             "unmatched_tracks": [slot_id]}       # track the VLM can't see
    """
    pairs = []
    for idx, (cx, cy) in enumerate(census_pts):
        for slot_id, (tx, ty) in track_positions.items():
            d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
            if d <= match_radius_m:
                pairs.append((d, idx, slot_id))
    pairs.sort()
    matched, used_idx, used_slots = [], set(), set()
    for d, idx, slot_id in pairs:
        if idx in used_idx or slot_id in used_slots:
            continue
        used_idx.add(idx)
        used_slots.add(slot_id)
        matched.append([slot_id, idx, round(d, 2)])
    return {
        "matched": matched,
        "unmatched_census": [i for i in range(len(census_pts)) if i not in used_idx],
        "unmatched_tracks": sorted(s for s in track_positions if s not in used_slots),
    }
