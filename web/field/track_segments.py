"""Per-segment track resolution: split a slot's track at suspicious
reacquisitions (a coast gap AND a spatial jump), classify each segment, and
drop phantom / wrong-robot segments that sit inside an otherwise-robot slot.

A slot's stored track is often a concatenation of different physical objects
stitched together by wrong reacquisitions after occlusion (e.g. a 9 s static
phantom, then a real moving robot). A single whole-slot verdict keeps the whole
thing; resolving *per segment* lets us drop the alien pieces while keeping the
robot. Do-no-harm: we only drop a segment that is clearly non-robot while the
slot also has a robot segment; anything ambiguous is left intact.

Pure — no torch, no app imports — like track_evidence.py, so it is unit-testable
in isolation. Decisions use motion / person-overlap / team-read only (never the
appearance gallery, which is unreliable across viewing angles).
"""
from __future__ import annotations

import math
from collections import defaultdict

from web.field.track_evidence import (
    DISPLACEMENT_WINDOW, TrackEvidence, classify_track, slot_label_to_id)


def split_at_jumps(samples, max_gap_s: float = 1.0, jump_m: float = 2.5,
                   teleport_speed_m_s: float = 8.0):
    """Split time-ordered ``(t_s, x, y)`` samples into ``[start, end)`` index
    ranges at suspicious reacquisitions. A new segment begins on a spatial step
    over ``jump_m`` that is EITHER across a coast gap (dt > ``max_gap_s``) OR at
    an impossible speed (> ``teleport_speed_m_s``, well above a robot's ~5.5 m/s).

    The two conditions cover the two data densities we run on: the sparse
    autonomous-path stream shows wrong reacquisitions as a big step across a
    time gap; the DENSE identity log (a row every frame, even while coasting)
    shows the same event as a teleport — a big step at tiny dt with no gap. The
    ``jump_m`` floor keeps genuine continuous fast moves (small per-frame steps)
    from splitting; ambiguous same-robot segments are kept by the resolve step."""
    n = len(samples)
    if n == 0:
        return []
    segments = []
    start = 0
    for i in range(1, n):
        dt = samples[i][0] - samples[i - 1][0]
        step = math.hypot(samples[i][1] - samples[i - 1][1],
                          samples[i][2] - samples[i - 1][2])
        speed = step / dt if dt > 1e-9 else float("inf")
        if step > jump_m and (dt > max_gap_s or speed > teleport_speed_m_s):
            segments.append((start, i))
            start = i
    segments.append((start, n))
    return segments


def segment_evidence(entries) -> TrackEvidence:
    """Build a TrackEvidence from one segment's identity-log rows (the same
    fields ``track_evidence._evidence_keyed`` consumes)."""
    te = TrackEvidence(track_id=0)
    for s in entries:
        te.observe_frame(s["frame_num"], bool(s.get("person_overlap")))
        pos = s.get("pos")
        if pos is not None:
            te.add_position(s["frame_num"], pos[0], pos[1])
        if s.get("confirmed"):
            te.confirm(s["frame_num"], team_read=True)
    return te


def classify_segment(entries, displacement_window: int = DISPLACEMENT_WINDOW) -> str:
    """'robot' | 'human' | 'static' for one segment, via the shared classifier."""
    return classify_track(segment_evidence(entries),
                          displacement_window=displacement_window)


def resolve_segments(verdicts) -> list[bool]:
    """Do-no-harm keep-mask. Drop a non-robot (static/human) segment ONLY when
    the slot also has a robot segment (the slot IS a robot but carries a phantom
    or alien piece). Single-segment slots, all-robot slots, and wholly-non-robot
    slots are kept intact — the last is left to the existing per-slot verdict."""
    if len(verdicts) <= 1:
        return [True] * len(verdicts)
    has_robot = any(v == "robot" for v in verdicts)
    if not has_robot:
        return [True] * len(verdicts)
    return [v == "robot" for v in verdicts]


def slot_dropped_ranges(identity_log, video_fps: float,
                        max_gap_s: float = 1.0, jump_m: float = 2.5,
                        displacement_window: int = DISPLACEMENT_WINDOW):
    """Per slot_id: split its time-ordered log rows at gap-jumps, classify each
    segment, and return the INCLUSIVE ``(lo_frame, hi_frame)`` ranges of the
    segments the do-no-harm resolver drops. Rows with no position are skipped."""
    by_slot = defaultdict(list)
    for s in identity_log:
        if s.get("pos") is not None:
            by_slot[s["slot_id"]].append(s)
    fps = video_fps or 1.0
    out = {}
    for sid, rows in by_slot.items():
        rows.sort(key=lambda r: r["frame_num"])
        samples = [(r["frame_num"] / fps, r["pos"][0], r["pos"][1]) for r in rows]
        segs = split_at_jumps(samples, max_gap_s, jump_m)
        if len(segs) <= 1:
            continue
        verdicts = [classify_segment(rows[a:b], displacement_window)
                    for (a, b) in segs]
        keep = resolve_segments(verdicts)
        dropped = [(rows[a]["frame_num"], rows[b - 1]["frame_num"])
                   for (a, b), k in zip(segs, keep) if not k]
        if dropped:
            out[sid] = dropped
    return out


def filter_paths_by_dropped(autonomous_paths, dropped):
    """Return a new auto-paths dict with samples removed whose ``frame_num``
    falls in a dropped ``(lo, hi)`` inclusive range for that slot. Auto paths are
    keyed by slot label (B1..R3); ``dropped`` is keyed by slot_id."""
    out = {}
    for label, samples in autonomous_paths.items():
        sid = slot_label_to_id(label)
        ranges = dropped.get(sid, []) if sid is not None else []
        if not ranges:
            out[label] = samples
            continue
        out[label] = [s for s in samples
                      if not any(lo <= s["frame_num"] <= hi for (lo, hi) in ranges)]
    return out
