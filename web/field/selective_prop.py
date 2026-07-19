"""Image-space tracking-by-detection (TBD) tracker with assignment margins.

Re-implementation of the cheap per-frame tracker from arXiv 2606.13033. Every
frame we run a Hungarian match between constant-velocity track predictions and
raw detections using an expanded-IoU cost (Deep-EIoU style). The *margin* of
each assignment -- how much cheaper the winning track is than the next-best
competitor for the same detection (paper Eq. 1) -- is the ambiguity signal that
Task 3's window finder consumes to decide where expensive SAM3 mask propagation
is actually needed.

Pure logic: numpy + scipy.optimize.linear_sum_assignment only. No torch, no GPU.
Determinism: tracks are iterated in oid order (cost-matrix rows), detections in
input order (columns); nothing depends on RNG or wall-clock.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

# Cost assigned to a track/detection pair whose expanded boxes do not overlap.
# Large but finite so linear_sum_assignment stays well-conditioned; any such
# match is discarded afterwards and never counts as a margin competitor.
FORBIDDEN = 1e6
# Additive penalty when a detection's known color contradicts the track's.
COLOR_PENALTY = 0.3


@dataclass
class SelectiveConfig:
    """Knobs for the selective-mask-propagation pipeline.

    Mirrors the ``selective_*`` knobs registered in web/job_config.py (Task 1).
    Only the TBD-tracker-relevant fields are consumed by this module; the rest
    are carried for Tasks 3-4 (window finder / propagation gate).
    """
    margin_entry: float = 0.05
    margin_exit: float = 0.10      # also SEED_MARGIN
    min_streak: int = 5            # SEED_CONSECUTIVE and EXIT_CONSECUTIVE
    gap_trigger: int = 7
    window_max_frames: int = 120
    seed_clean_iou: float = 0.10   # not knob-exposed (paper constant)
    ioma_exit: float = 0.80        # not knob-exposed
    mask_overlap_exit: float = 0.90
    area_degradation: float = 0.25
    border_margin_px: int = 20
    max_missed: int = 30           # TBD track death (die when missed > this)
    iou_expand: float = 0.25       # expanded-IoU box inflation factor


@dataclass
class FrameObs:
    """One (frame, oid) TBD observation.

    ``box`` is the matched (or spawning) detection's box in export coords;
    ``margin`` is this assignment's margin (inf when unambiguous / new / solo);
    ``color`` is the detection's raw color for this frame (None if unknown).
    """
    box: tuple[int, int, int, int]
    margin: float
    color: str | None


def _expand_box(box, expand: float):
    """Inflate a box by ``expand`` fraction of its own width/height per side."""
    x1, y1, x2, y2 = box
    dx = (x2 - x1) * expand
    dy = (y2 - y1) * expand
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)


def expanded_iou(a, b, expand: float) -> float:
    """IoU of two boxes after each is inflated by ``expand`` per side.

    Deep-EIoU-style expansion helps fast movers whose raw boxes barely touch.
    Returns 0.0 when the expanded boxes do not overlap.
    """
    ax1, ay1, ax2, ay2 = _expand_box(a, expand)
    bx1, by1, bx2, by2 = _expand_box(b, expand)
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


class _Track:
    """Internal per-object state (never leaked across the public interface)."""

    __slots__ = ("oid", "box", "color", "_centers", "missed", "last_seen")

    def __init__(self, oid: int, box, color, frame_idx: int):
        self.oid = oid
        self.box = tuple(int(v) for v in box)
        self.color = color            # sticky: first *known* color wins
        self._centers = [_center(box)]
        self.missed = 0
        self.last_seen = frame_idx

    def velocity(self):
        """Average of the last <=3 center deltas; (0, 0) for new tracks."""
        c = self._centers[-4:]
        if len(c) < 2:
            return (0.0, 0.0)
        dxs = [c[i][0] - c[i - 1][0] for i in range(1, len(c))]
        dys = [c[i][1] - c[i - 1][1] for i in range(1, len(c))]
        return (sum(dxs) / len(dxs), sum(dys) / len(dys))

    def predicted_box(self):
        """Last box translated by the constant-velocity estimate."""
        vx, vy = self.velocity()
        x1, y1, x2, y2 = self.box
        return (x1 + vx, y1 + vy, x2 + vx, y2 + vy)

    def observe(self, box, color, frame_idx: int):
        self.box = tuple(int(v) for v in box)
        self._centers.append(_center(box))
        if self.color is None and color is not None:
            self.color = color        # first known color sticks
        self.missed = 0
        self.last_seen = frame_idx


class TbdTracker:
    """Deterministic tracking-by-detection with per-assignment margins."""

    def __init__(self, cfg: SelectiveConfig, first_oid: int = 1):
        self.cfg = cfg
        self._tracks: dict[int, _Track] = {}
        self._next_oid = first_oid
        self._history: dict[int, dict[int, FrameObs]] = {}
        self._gap_reappearances: list[tuple[int, int, int]] = []

    # -- public read-only interface (consumed by Tasks 3-4) ----------------
    @property
    def history(self) -> dict[int, dict[int, FrameObs]]:
        return self._history

    @property
    def gap_reappearances(self) -> list[tuple[int, int, int]]:
        return self._gap_reappearances

    @property
    def next_oid(self) -> int:
        return self._next_oid

    # -- core step ---------------------------------------------------------
    def _pair_cost(self, track: _Track, det_box, det_color) -> float:
        eiou = expanded_iou(track.predicted_box(), det_box, self.cfg.iou_expand)
        if eiou <= 0.0:
            return FORBIDDEN
        cost = 1.0 - eiou
        if (det_color is not None and track.color is not None
                and det_color != track.color):
            cost += COLOR_PENALTY
        return cost

    def step(self, frame_idx: int,
             dets: list[tuple[tuple, str | None]]) -> dict[int, FrameObs]:
        """Advance one frame.

        ``dets = [(box_xyxy, color), ...]``. Returns this frame's matched and
        newly-spawned observations keyed by oid. Also appends to ``history``.
        """
        live = sorted(self._tracks.values(), key=lambda t: t.oid)  # rows by oid
        obs: dict[int, FrameObs] = {}
        matched_oids: set[int] = set()
        matched_dets: set[int] = set()

        if live and dets:
            n, m = len(live), len(dets)
            cost = np.empty((n, m), dtype=float)
            for i, trk in enumerate(live):
                for j, (dbox, dcolor) in enumerate(dets):
                    cost[i, j] = self._pair_cost(trk, dbox, dcolor)

            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                mcost = cost[r, c]
                if mcost >= FORBIDDEN:
                    continue                       # discard forbidden matches
                trk = live[r]
                dbox, dcolor = dets[c]
                # Margin (paper Eq. 1): cheapest finite competitor in this
                # detection's column minus the winning cost. No finite
                # competitor -> unambiguous -> inf.
                competitors = [cost[i, c] for i in range(n)
                               if i != r and cost[i, c] < FORBIDDEN]
                margin = (min(competitors) - mcost) if competitors else float("inf")
                # Gap reappearance: a still-alive track that went dark for
                # >= gap_trigger frames and now matches again.
                if trk.missed >= self.cfg.gap_trigger:
                    self._gap_reappearances.append(
                        (trk.oid, trk.last_seen, frame_idx))
                trk.observe(dbox, dcolor, frame_idx)
                matched_oids.add(trk.oid)
                matched_dets.add(c)
                obs[trk.oid] = FrameObs(
                    box=tuple(int(v) for v in dbox),
                    margin=float(margin),
                    color=dcolor,
                )

        # Unmatched live tracks coast (miss counter climbs).
        for trk in live:
            if trk.oid not in matched_oids:
                trk.missed += 1

        # Unmatched detections spawn fresh tracks (input order); margin inf.
        for j, (dbox, dcolor) in enumerate(dets):
            if j in matched_dets:
                continue
            oid = self._next_oid
            self._next_oid += 1
            self._tracks[oid] = _Track(oid, dbox, dcolor, frame_idx)
            obs[oid] = FrameObs(
                box=tuple(int(v) for v in dbox),
                margin=float("inf"),
                color=dcolor,
            )

        # Reap dead tracks: die when missed exceeds max_missed (never revive).
        dead = [oid for oid, trk in self._tracks.items()
                if trk.missed > self.cfg.max_missed]
        for oid in dead:
            del self._tracks[oid]

        # Emit in oid order for stable, deterministic output.
        obs = {oid: obs[oid] for oid in sorted(obs)}
        self._history[frame_idx] = obs
        return obs
