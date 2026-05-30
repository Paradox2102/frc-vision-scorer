"""Pure metric functions for the SAM2 tracking spike. No torch/SAM2 imports —
unit-testable in isolation. Identity swaps are counted against hand-labeled
anchors that pin a known physical robot to a screen point at sampled frames."""
from __future__ import annotations
import math


def _nearest_track(tracks: dict, point) -> int | None:
    best, bd = None, math.inf
    for tid, (x, y) in tracks.items():
        d = (x - point[0]) ** 2 + (y - point[1]) ** 2
        if d < bd:
            best, bd = tid, d
    return best


def count_id_switches(tracker_frames: dict, anchors: list) -> int:
    """anchors: list of (frame, point, label). For each label, walk its anchors in
    frame order; the track assigned to the label is the track nearest that anchor's
    point on that frame. Each time that track id changes, count one switch.

    Counting is per-label: one physical crossing that swaps two robots produces two
    switches (one per robot). That is fine — both trackers are scored with the same
    function, so the relative comparison (the gate's >=50% reduction) is valid."""
    by_label: dict[str, list] = {}
    for frame, point, label in sorted(anchors, key=lambda a: a[0]):
        by_label.setdefault(label, []).append((frame, point))
    switches = 0
    for label, seq in by_label.items():
        prev_tid = None
        for frame, point in seq:
            tracks = tracker_frames.get(frame, {})
            tid = _nearest_track(tracks, point)
            if prev_tid is not None and tid is not None and tid != prev_tid:
                switches += 1
            if tid is not None:
                prev_tid = tid
    return switches


def continuity_stats(tracker_frames: dict) -> dict:
    """Track births/deaths and lifetime stats from per-frame {track_id: pos}."""
    frames = sorted(tracker_frames)
    seen, births, deaths, lifetimes = set(), 0, 0, {}
    prev_ids: set = set()
    for f in frames:
        ids = set(tracker_frames[f])
        births += len(ids - prev_ids)
        deaths += len(prev_ids - ids)
        for tid in ids:
            lifetimes[tid] = lifetimes.get(tid, 0) + 1
        seen |= ids
        prev_ids = ids
    deaths += len(prev_ids)  # tracks alive at the end "die" at clip end
    return {
        "n_tracks": len(seen),
        "births": births,
        "deaths": deaths,
        "mean_frames_per_track": (sum(lifetimes.values()) / len(lifetimes)) if lifetimes else 0.0,
    }
