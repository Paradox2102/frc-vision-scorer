"""Fixed-3v3 field-space multi-object tracker (pure NumPy).

6 slots (3 blue + 3 red), each a constant-velocity Kalman filter in field meters.
Per frame, per alliance, assign that color's <=3 detections to its 3 slots by
minimum-cost matching, brute-forced over small permutation sets. Update matched, coast
unmatched. Occluded slots coast far longer and cannot be re-seeded. Alliance sticky."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from itertools import permutations
from typing import Optional
import numpy as np

from web.field.zones import point_in_zones

ALLIANCES = ("blue", "red")


@dataclass
class FieldDetection:
    pos: tuple
    color: str
    box: Optional[tuple]
    score: float = 1.0
    track_id: int | None = None
    embedding: Optional[np.ndarray] = None
    # Masked whole-robot crop (base64 JPEG) behind `embedding`, for the gallery UI.
    robot_crop_b64: Optional[str] = None


@dataclass
class TrackOut:
    slot_id: int
    alliance: str
    pos: tuple
    vel: tuple
    box: Optional[tuple]
    state: str   # "visible" | "coasting" | "empty"
    # Position currently sourced from corner cameras only (no main-view box).
    # Corner ground projections have poor range; the map flags these dots.
    corner_only: bool = False


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

    def update(self, z, r_scale: float = 1.0):
        H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1.0
        S = H @ self.P @ H.T + (self.r * r_scale) * np.eye(2)
        Kg = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + Kg @ (np.asarray(z, float) - H @ self.x)
        self.P = (np.eye(4) - Kg @ H) @ self.P

    def decay_velocity(self, factor: float):
        self.x[2:4] *= factor

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
    corner_only: bool = False
    # Went into coast while hub-occluded; stays set until re-acquired. Gates the
    # occlusion re-acquisition guard (suspect-zone detections need motion proof).
    occ_coasting: bool = False


class FieldTracker:
    def __init__(self, field_length, gate_m=2.5, max_coast_open=8,
                 max_coast_occluded=90, color_penalty=3.0, zones=None,
                 corner_only_r_scale=8.0, coast_vel_decay=0.85,
                 max_speed_m_s=5.5, sticky_max_dist_m=0.0,
                 suspect_zones=None, occlusion_rebind_guard=False,
                 guard_motion_m=0.3, guard_motion_window=5):
        self.gate = gate_m
        # Sticky-rebind distance gate (meters); 0 = off. A detection carrying a
        # known track_id normally re-binds to that id's slot unconditionally; when
        # this is > 0, a re-bind farther than this from the slot is rejected
        # (the registry mis-assigned one id to two robots) and the detection is
        # routed to distance matching so it can claim its own slot.
        self.sticky_max_dist_m = float(sticky_max_dist_m)
        self.field_length = field_length
        self.max_open = max_coast_open
        self.max_occ = max_coast_occluded
        self.color_penalty = color_penalty
        # Position-stability knobs (see field-map-v2 spec): corner-only
        # measurements get inflated noise (poor range from low cameras),
        # coasting velocity decays toward rest, and measurements implying
        # impossible robot speed are downweighted instead of followed.
        self.corner_only_r_scale = float(corner_only_r_scale)
        self.coast_vel_decay = float(coast_vel_decay)
        self.max_speed = float(max_speed_m_s)
        # Static-structure zones (field meters). A new slot is never seeded from
        # a detection inside a zone; existing tracks update normally.
        self.zones = list(zones) if zones else []
        # Occlusion re-acquisition guard: while a slot coasts hub-occluded, a
        # detection inside a suspect zone (bump/hub/trench footprint — places a
        # false positive parks) may not claim that slot or seed a new one until
        # its track_id displaces >= guard_motion_m within the last
        # guard_motion_window update calls. Robots legitimately drive over these
        # structures, so unlike `zones` this is motion-conditioned, not a hard block.
        self.suspect_zones = list(suspect_zones) if suspect_zones else []
        self.occlusion_rebind_guard = bool(occlusion_rebind_guard)
        self.guard_motion_m = float(guard_motion_m)
        self.guard_motion_window = int(guard_motion_window)
        self._det_history: dict[int, deque] = {}  # track_id -> (frame_idx, pos)
        self._frame_idx = 0
        self.tracks = []
        sid = 0
        for a in ALLIANCES:
            for _ in range(3):
                self.tracks.append(_Track(slot_id=sid, alliance=a))
                sid += 1
        # Maps stable track_id -> slot_id for sticky binding.
        self._id_to_slot: dict[int, int] = {}
        self.binding_events: list[dict] = []
        self.assignment_evidence: dict[int, dict] = {}

    def _slots(self, alliance):
        return [t for t in self.tracks if t.alliance == alliance]

    def _record_det_history(self, detections):
        """Track recent positions per stable track_id for the motion proof."""
        for d in detections:
            if d.track_id is None:
                continue
            h = self._det_history.get(d.track_id)
            if h is None:
                h = self._det_history[d.track_id] = deque(
                    maxlen=self.guard_motion_window + 3)
            h.append((self._frame_idx, (float(d.pos[0]), float(d.pos[1]))))
        if len(self._det_history) > 256:
            stale = self._frame_idx - 3 * self.guard_motion_window
            for tid in [tid for tid, h in self._det_history.items()
                        if h[-1][0] < stale]:
                del self._det_history[tid]

    def _det_moving(self, d):
        """Has this detection's track_id displaced >= guard_motion_m within the
        last guard_motion_window update calls? track_id-less detections can
        never prove motion (conservative: guard treats them as stationary)."""
        if d.track_id is None:
            return False
        h = self._det_history.get(d.track_id)
        if not h:
            return False
        recent = [p for f, p in h if f > self._frame_idx - self.guard_motion_window]
        if len(recent) < 2:
            return False
        span = 0.0
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                dx = recent[i][0] - recent[j][0]
                dy = recent[i][1] - recent[j][1]
                span = max(span, (dx * dx + dy * dy) ** 0.5)
        return span >= self.guard_motion_m

    def _guard_blocks(self, slot, d):
        """Suspect-zone detection may not claim an occlusion-coasting slot
        until it proves motion."""
        return (self.occlusion_rebind_guard and slot.occ_coasting
                and point_in_zones(d.pos, self.suspect_zones)
                and not self._det_moving(d))

    def update(self, detections, occlusion, dt):
        self.binding_events = []
        self.assignment_evidence = {}
        self._frame_idx += 1
        if self.occlusion_rebind_guard:
            self._record_det_history(detections)
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
                dt,
            )

        # Coast/expire logic: runs after _assign has updated missed counts
        for t in self.tracks:
            if not t.active:
                continue
            occ = occlusion.get(t.slot_id, {}).get("view_occluded", False)
            budget = self.max_occ if occ else self.max_open
            if t.missed > 0:
                t.coasting = True
                if occ:
                    t.occ_coasting = True
                if t.kf is not None:
                    t.kf.decay_velocity(self.coast_vel_decay)
            if t.missed > budget:
                # Expire the track
                t.active = False
                t.kf = None
                t.box = None
                t.coasting = False
                t.occ_coasting = False
                tids_to_del = [tid for tid, sid in self._id_to_slot.items() if sid == t.slot_id]
                for tid in tids_to_del:
                    del self._id_to_slot[tid]
                    self.binding_events.append({
                        "event": "slot_expired",
                        "slot_id": t.slot_id,
                        "track_id": tid,
                        "alliance": t.alliance,
                    })

        return self._emit()

    def _assign(self, slots, dets_with_idx, alliance, dt):
        """Match active tracks to dets, update matched, coast unmatched, seed free slots."""
        active_slots = [t for t in slots if t.active]
        free_slots = [t for t in slots if not t.active]

        # A track_id carried by 2+ detections in this frame is AMBIGUOUS: the
        # registry mis-assigned one id to distinct robots (merge_main_guard
        # passes them through so each keeps its own slot). Sticky binding and
        # mapping updates for such an id would flip-flop _id_to_slot between
        # the robots' slots every frame (qm11 R1/R2/R3 churn); route these to
        # distance matching, which preserves per-slot continuity, and learn
        # nothing from them.
        tid_counts: dict = {}
        for _i, d in dets_with_idx:
            if d.track_id is not None:
                tid_counts[d.track_id] = tid_counts.get(d.track_id, 0) + 1
        ambiguous_tids = {t for t, n in tid_counts.items() if n > 1}

        # ---- Sticky binding pass (track_id is not None) ----
        # Process before building the cost matrix so these detections are
        # removed from the pool and never cause an index off-by-one.
        remaining_pairs = []
        for orig_idx, d in dets_with_idx:
            if d.track_id is None or d.track_id in ambiguous_tids:
                remaining_pairs.append((orig_idx, d))
                continue

            tid = d.track_id
            if tid in self._id_to_slot:
                # Re-bind to the known slot.
                slot_id = self._id_to_slot[tid]
                slot = next((t for t in slots if t.slot_id == slot_id), None)
                if slot is not None and slot.active:
                    if (self.sticky_max_dist_m > 0 and slot.kf is not None
                            and float(np.linalg.norm(
                                np.array(slot.kf.pos) - np.array(d.pos)))
                            > self.sticky_max_dist_m):
                        # Same track_id but implausibly far from its slot: the
                        # registry mis-assigned one id to two robots. Don't
                        # collapse them — route to distance matching so this
                        # detection can claim its own slot.
                        remaining_pairs.append((orig_idx, d))
                        continue
                    if self._guard_blocks(slot, d):
                        # Known id gone motionless on a static structure while
                        # its slot is hub-occluded: hold the binding. The det is
                        # consumed (it must not seed elsewhere); the slot stays
                        # coasting and re-binds when this id proves motion.
                        self.binding_events.append({
                            "event": "occlusion_rebind_held",
                            "slot_id": slot.slot_id,
                            "track_id": tid,
                            "alliance": slot.alliance,
                        })
                        continue
                    self._measured_update(slot, d, dt)
                    slot.box = d.box
                    slot.missed = 0
                    slot.hits += 1
                    slot.coasting = False
                    slot.occ_coasting = False
                    self.last_det_to_slot[orig_idx] = slot.slot_id
                    self.binding_events.append({
                        "event": "slot_rebound",
                        "slot_id": slot.slot_id,
                        "track_id": tid,
                        "alliance": slot.alliance,
                    })
                    # Remove this slot from active_slots so it won't be
                    # processed again in the distance-matching pass.
                    if slot in active_slots:
                        active_slots.remove(slot)
                    continue
                # Slot expired or doesn't exist — fall through to seed path
                # (remove stale mapping so it can be re-seeded below)
                del self._id_to_slot[tid]

            # New/unknown track_id: do NOT seed a fresh slot in this sticky pass.
            # Route it into the distance-matching pass below, which binds a
            # detection to the nearest active slot — whether that slot is still
            # VISIBLE (SAM3 re-ided a continuously-tracked robot mid-track: the qm8
            # B1<->B3 flip, which happened with no occlusion) or COASTING (a robot
            # reappearing after occlusion with a new mask id). Only a detection that
            # matches no active slot within gate seeds a new slot there. Seeding
            # before matching is exactly what spawned the duplicate slot.
            remaining_pairs.append((orig_idx, d))

        # Replace dets_with_idx with the non-sticky remainder for the
        # existing distance-matching / seeding path below.
        dets_with_idx = remaining_pairs

        # Unpack original indices and detection objects from paired list
        idxs = [i for (i, _) in dets_with_idx]
        dets = [d for (_, d) in dets_with_idx]

        guard_blocked = set()  # det indices barred from seeding this call
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
                        if self._guard_blocks(t, d):
                            # In-gate suspect-zone detection vs an occlusion-
                            # coasting slot: leave the pair unmatchable and bar
                            # the det from seeding (a bump false positive must
                            # not become a robot while the real one is hidden).
                            guard_blocked.add(j)
                            self.binding_events.append({
                                "event": "occlusion_rebind_held",
                                "slot_id": t.slot_id,
                                "track_id": d.track_id,
                                "alliance": t.alliance,
                            })
                            continue
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
                    self._measured_update(t, dets[c], dt)
                    t.box = dets[c].box
                    t.missed = 0
                    t.hits += 1
                    t.coasting = False
                    t.occ_coasting = False
                    matched_slot_indices.add(r)
                    matched_det_indices.add(c)
                    self.last_det_to_slot[idxs[c]] = t.slot_id
                    if (dets[c].track_id is not None
                            and dets[c].track_id not in ambiguous_tids):
                        self._id_to_slot[dets[c].track_id] = t.slot_id
                    
                    # Store assignment evidence with margin
                    margin = np.inf
                    other_costs = [cost[r, j] for j in range(m) if j != c]
                    valid_others = [v for v in other_costs if np.isfinite(v)]
                    if valid_others:
                        margin = float(min(valid_others) - cost[r, c])
                    
                    self.assignment_evidence[t.slot_id] = {
                        "distance": float(cost[r, c]),
                        "margin": margin,
                        "detection_idx": idxs[c],
                    }

            # Increment missed for unmatched active slots
            for i, t in enumerate(active_slots):
                if i not in matched_slot_indices:
                    t.missed += 1

            # Seed leftover detections into free slots — but never seed a new
            # slot from a detection inside a static-structure zone (a real robot
            # is never born there; it drives in as an already-active track).
            leftover_pairs = [(j, dets[j]) for j in range(len(dets)) if j not in matched_det_indices]
            for j, d in leftover_pairs:
                if not free_slots:
                    break
                if j in guard_blocked or point_in_zones(d.pos, self.zones):
                    continue
                slot = free_slots.pop(0)
                slot.kf = _KF(d.pos)
                slot.box = d.box
                slot.corner_only = d.box is None
                slot.missed = 0
                slot.hits = 1
                slot.coasting = False
                slot.active = True
                self.last_det_to_slot[idxs[j]] = slot.slot_id
                if d.track_id is not None and d.track_id not in ambiguous_tids:
                    self._id_to_slot[d.track_id] = slot.slot_id
                    self.binding_events.append({
                        "event": "slot_seeded",
                        "slot_id": slot.slot_id,
                        "track_id": d.track_id,
                        "alliance": slot.alliance,
                    })

        elif active_slots and not dets:
            # No detections: all active slots missed
            for t in active_slots:
                t.missed += 1

        else:
            # No active slots: seed as many dets as we have free slots, skipping
            # any detection sitting inside a static-structure zone.
            for j, d in enumerate(dets):
                if not free_slots:
                    break
                if point_in_zones(d.pos, self.zones):
                    continue
                slot = free_slots.pop(0)
                slot.kf = _KF(d.pos)
                slot.box = d.box
                slot.corner_only = d.box is None
                slot.missed = 0
                slot.hits = 1
                slot.coasting = False
                slot.active = True
                self.last_det_to_slot[idxs[j]] = slot.slot_id
                if d.track_id is not None and d.track_id not in ambiguous_tids:
                    self._id_to_slot[d.track_id] = slot.slot_id
                    self.binding_events.append({
                        "event": "slot_seeded",
                        "slot_id": slot.slot_id,
                        "track_id": d.track_id,
                        "alliance": slot.alliance,
                    })

    def _measured_update(self, t, d, dt):
        """KF update with measurement noise scaled by source quality.

        Corner-only detections (no main-view box) carry poor range from the
        low cameras; measurements implying impossible robot speed are noise
        or identity errors. Both lean the state on the motion model instead
        of being followed."""
        scale = 1.0
        if d.box is None:
            scale *= self.corner_only_r_scale
        if dt and dt > 0:
            speed = float(np.linalg.norm(
                np.asarray(d.pos, float) - np.asarray(t.kf.pos, float))) / dt
            if speed > self.max_speed:
                scale *= (speed / self.max_speed) ** 2
        t.kf.update(d.pos, r_scale=scale)
        t.corner_only = d.box is None

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
                    corner_only=t.corner_only,
                ))
        return out
