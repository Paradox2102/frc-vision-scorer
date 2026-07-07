"""Gap-aware quality helpers for time-stamped robot paths.

Our auto paths (``summary.autonomous_paths[slot]``) record a sample only on
frames where the slot is *visible*. A tracking dropout (occlusion or lost
track) therefore appears as a large timestamp gap between two consecutive
samples, and any consumer that connects the samples as a single polyline
fabricates straight-line motion across the gap — inflating path length and
drawing routes the robot never took.

These pure helpers let consumers (the Manta comparison, overlays, and — later —
the tracker's own reporting) break a path at its dropouts and quantify how much
of the window was actually tracked. A "gap" is any step whose timestamp delta
exceeds ``max_gap_s``; the normal per-frame cadence is well under that.
"""
from __future__ import annotations

import numpy as np


def gap_segments(timestamps, max_gap_s: float = 1.0) -> list[tuple[int, int]]:
    """Split ``timestamps`` into contiguous ``[start, end)`` index ranges.

    A new segment begins whenever the gap to the previous timestamp exceeds
    ``max_gap_s``. Timestamps are assumed non-decreasing (path sample order).
    """
    ts = [float(t) for t in timestamps]
    if not ts:
        return []
    segments: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(ts)):
        if ts[i] - ts[i - 1] > max_gap_s:
            segments.append((start, i))
            start = i
    segments.append((start, len(ts)))
    return segments


def tracked_length_m(points, timestamps, max_gap_s: float = 1.0) -> float:
    """Polyline length in meters, summing only *within-segment* steps.

    The straight-line "bridge" a naive polyline draws across a dropout is
    excluded, so the result reflects distance the robot was actually observed
    to travel rather than interpolation across missing tracking.
    """
    pts = np.asarray(points, float).reshape(-1, 2)
    total = 0.0
    for a, b in gap_segments(timestamps, max_gap_s):
        if b - a >= 2:
            seg = pts[a:b]
            total += float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
    return total


def coverage(timestamps, max_gap_s: float = 1.0, window_s: float | None = None) -> dict:
    """Summarize how completely a path covers its time window.

    Returns ``n_samples``, ``span_s`` (last-first timestamp), ``tracked_s``
    (summed duration of continuous-tracking runs), ``coverage_frac``
    (``tracked_s`` over ``window_s`` if given, else over ``span_s``),
    ``n_gaps`` and ``max_gap_s`` (largest observed dropout, 0 if none).
    """
    ts = [float(t) for t in timestamps]
    n = len(ts)
    span = (ts[-1] - ts[0]) if n >= 2 else 0.0
    segs = gap_segments(ts, max_gap_s)
    tracked = sum(ts[b - 1] - ts[a] for a, b in segs)
    deltas = [ts[i] - ts[i - 1] for i in range(1, n)]
    gaps = [d for d in deltas if d > max_gap_s]
    denom = window_s if window_s else span
    frac = (tracked / denom) if denom and denom > 0 else (1.0 if n else 0.0)
    return {
        "n_samples": n,
        "span_s": round(span, 4),
        "tracked_s": round(tracked, 4),
        "coverage_frac": round(frac, 4),
        "n_gaps": len(gaps),
        "max_gap_s": round(max(gaps), 4) if gaps else 0.0,
    }


def quality_summary(samples) -> dict:
    """Aggregate the per-sample quality signals attached to auto-path samples.

    Newer samples carry ``tracker_prob`` (SAM per-object confidence, float or
    None), ``corner_only`` (bool) and ``occluded`` (cause string or None). Legacy
    replays predate these keys; such samples are EXCLUDED from the tracker_prob
    aggregates (which are reported None with ``n_with_prob=0`` when none carry a
    prob) and treated as not-corner-only / not-occluded for the flag fractions.
    A sample with ``tracker_prob`` explicitly None counts as prob-less too.

    Returns ``mean_tracker_prob``, ``min_tracker_prob``, ``frac_low_conf``
    (prob < 0.5, over the prob-carrying samples), ``frac_corner_only``,
    ``frac_occluded`` (over all samples), ``n_samples`` and ``n_with_prob``.
    """
    n = len(samples)
    probs = [float(s["tracker_prob"]) for s in samples
             if s.get("tracker_prob") is not None]
    n_with_prob = len(probs)
    corner = sum(1 for s in samples if s.get("corner_only"))
    occ = sum(1 for s in samples if s.get("occluded"))
    return {
        "n_samples": n,
        "n_with_prob": n_with_prob,
        "mean_tracker_prob": round(sum(probs) / n_with_prob, 4) if probs else None,
        "min_tracker_prob": round(min(probs), 4) if probs else None,
        "frac_low_conf": (round(sum(1 for p in probs if p < 0.5) / n_with_prob, 4)
                          if probs else None),
        "frac_corner_only": round(corner / n, 4) if n else 0.0,
        "frac_occluded": round(occ / n, 4) if n else 0.0,
    }
