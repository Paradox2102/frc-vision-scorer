"""Fixed-3v3 field-space multi-object tracker (pure NumPy).

6 slots (3 blue + 3 red), each a constant-velocity Kalman filter in field meters.
Per frame, per alliance, assign that color's <=3 detections to its 3 slots by
minimum-cost matching, brute-forced over small permutation sets. Update matched, coast
unmatched. Occluded slots coast far longer and cannot be re-seeded. Alliance sticky."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations
from typing import Optional
import numpy as np

ALLIANCES = ("blue", "red")


@dataclass
class FieldDetection:
    pos: tuple
    color: str
    box: tuple
    score: float = 1.0


@dataclass
class TrackOut:
    slot_id: int
    alliance: str
    pos: tuple
    vel: tuple
    box: Optional[tuple]
    state: str   # "visible" | "coasting" | "empty"


class _KF:
    """Constant-velocity Kalman filter. State vector: [x, y, vx, vy]."""

    def __init__(self, xy, q=2.0, r=0.05):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], float)
        self.P = np.diag([r, r, 4.0, 4.0])
        self.q, self.r = q, r

    def predict(self, dt):
        F = np.eye(4); F[0, 2] = dt; F[1, 3] = dt
        self.x = F @ self.x
        G = np.array([[0.5*dt*dt, 0], [0, 0.5*dt*dt], [dt, 0], [0, dt]])
        self.P = F @ self.P @ F.T + G @ (self.q * np.eye(2)) @ G.T

    def update(self, z):
        H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1.0
        S = H @ self.P @ H.T + self.r * np.eye(2)
        Kg = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + Kg @ (np.asarray(z, float) - H @ self.x)
        self.P = (np.eye(4) - Kg @ H) @ self.P

    @property
    def pos(self): return (float(self.x[0]), float(self.x[1]))

    @property
    def vel(self): return (float(self.x[2]), float(self.x[3]))


@dataclass
class _Track:
    slot_id: int
    alliance: str
    kf: Optional[_KF] = None
    box: Optional[tuple] = None
    missed: int = 0
    hits: int = 0
    coasting: bool = False
    active: bool = False


class FieldTracker:
    def __init__(self, field_length, gate_m=2.5, max_coast_open=8,
                 max_coast_occluded=90, color_penalty=3.0):
        self.gate = gate_m
        self.field_length = field_length
        self.max_open = max_coast_open
        self.max_occ = max_coast_occluded
        self.color_penalty = color_penalty
        self.tracks = []
        sid = 0
        for a in ALLIANCES:
            for _ in range(3):
                self.tracks.append(_Track(slot_id=sid, alliance=a))
                sid += 1

    def _slots(self, alliance):
        return [t for t in self.tracks if t.alliance == alliance]

    def update(self, detections, occlusion, dt):
        # Predict step for all active tracks
        for t in self.tracks:
            if t.active:
                t.kf.predict(dt)

        # Initialise assignment map: detection index -> slot_id
        self.last_det_to_slot = {}

        # Assign detections per alliance.
        # Unknown-color detections go only to the alliance whose half they're in
        # (x < field_length/2 → "blue", else "red"), preventing a single unknown
        # detection from seeding slots in both alliances simultaneously.
        half = self.field_length / 2.0
        indexed = list(enumerate(detections))
        for alliance in ALLIANCES:
            def _territory(pos):
                return "blue" if pos[0] < half else "red"
            pairs = [(i, d) for (i, d) in indexed
                     if d.color == alliance
                     or (d.color == "unknown" and _territory(d.pos) == alliance)]
            self._assign(
                self._slots(alliance),
                pairs,
                alliance,
            )

        # Coast/expire logic: runs after _assign has updated missed counts
        for t in self.tracks:
            if not t.active:
                continue
            occ = occlusion.get(t.slot_id, {}).get("view_occluded", False)
            budget = self.max_occ if occ else self.max_open
            if t.missed > 0:
                t.coasting = True
            if t.missed > budget:
                # Expire the track
                t.active = False
                t.kf = None
                t.box = None
                t.coasting = False

        return self._emit()

    def _assign(self, slots, dets_with_idx, alliance):
        """Match active tracks to dets, update matched, coast unmatched, seed free slots."""
        active_slots = [t for t in slots if t.active]
        free_slots = [t for t in slots if not t.active]

        # Unpack original indices and detection objects from paired list
        idxs = [i for (i, _) in dets_with_idx]
        dets = [d for (_, d) in dets_with_idx]

        if active_slots and dets:
            # Build cost matrix: rows=active_slots, cols=dets
            n = len(active_slots)
            m = len(dets)
            cost = np.full((n, m), np.inf)
            for i, t in enumerate(active_slots):
                tp = np.array(t.kf.pos)
                for j, d in enumerate(dets):
                    dp = np.array(d.pos)
                    dist = float(np.linalg.norm(tp - dp))
                    # Widen the gate for coasting tracks (more missed frames →
                    # wider gate), but cap the multiplier to prevent teleportation.
                    effective_gate = self.gate * min(4.0, 1.0 + 0.5 * t.missed)
                    if dist <= effective_gate:
                        c = dist
                        if d.color == "unknown":
                            c += self.color_penalty
                        cost[i, j] = c

            # Find best assignment (brute force over <=3! permutations)
            rows, cols = self._best_assignment(cost)

            # Determine which (slot_idx, det_idx) pairs are valid (gated)
            matched_slot_indices = set()
            matched_det_indices = set()
            for r, c in zip(rows, cols):
                if np.isfinite(cost[r, c]):
                    # Update matched slot
                    t = active_slots[r]
                    t.kf.update(dets[c].pos)
                    t.box = dets[c].box
                    t.missed = 0
                    t.hits += 1
                    t.coasting = False
                    matched_slot_indices.add(r)
                    matched_det_indices.add(c)
                    self.last_det_to_slot[idxs[c]] = t.slot_id

            # Increment missed for unmatched active slots
            for i, t in enumerate(active_slots):
                if i not in matched_slot_indices:
                    t.missed += 1

            # Seed leftover detections into free slots
            leftover_pairs = [(j, dets[j]) for j in range(len(dets)) if j not in matched_det_indices]
            for j, d in leftover_pairs:
                if not free_slots:
                    break
                slot = free_slots.pop(0)
                slot.kf = _KF(d.pos)
                slot.box = d.box
                slot.missed = 0
                slot.hits = 1
                slot.coasting = False
                slot.active = True
                self.last_det_to_slot[idxs[j]] = slot.slot_id

        elif active_slots and not dets:
            # No detections: all active slots missed
            for t in active_slots:
                t.missed += 1

        else:
            # No active slots: seed as many dets as we have free slots
            for j, d in enumerate(dets):
                if not free_slots:
                    break
                slot = free_slots.pop(0)
                slot.kf = _KF(d.pos)
                slot.box = d.box
                slot.missed = 0
                slot.hits = 1
                slot.coasting = False
                slot.active = True
                self.last_det_to_slot[idxs[j]] = slot.slot_id

    def _best_assignment(self, cost):
        """Brute-force minimum-cost assignment for ≤3×3 cost matrices.

        When n > m (more tracks than detections), we iterate permutations of
        track indices choosing m of them to match, paired with all m! det orderings.
        For a 3×3 matrix this evaluates P(3,3)×P(3,3) = 36 candidate pairings.
        Returns (track_rows, det_cols) for the best finite-cost pairs found.
        """
        n, m = cost.shape
        best, best_cost = None, None
        for track_rows in permutations(range(n), min(n, m)):
            for det_cols in permutations(range(m), min(n, m)):
                rows = list(track_rows)
                cols = list(det_cols)
                c = sum(cost[rows[k], cols[k]] for k in range(len(rows)))
                if best_cost is None or c < best_cost:
                    best_cost = c
                    best = (rows, cols)
        return best if best else ([], [])

    def _emit(self):
        # Only active (visible or coasting) tracks are emitted; empty slots are omitted.
        out = []
        for t in self.tracks:
            if t.active:
                out.append(TrackOut(
                    t.slot_id, t.alliance, t.kf.pos, t.kf.vel,
                    t.box, "coasting" if t.coasting else "visible",
                ))
        return out
