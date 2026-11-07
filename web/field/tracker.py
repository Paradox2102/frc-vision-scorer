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
# Stand-in cost of a gated-out pair under assign_partial_fix: larger than any
# sum of in-gate costs, so fewer matches can never beat more.
_GATED_OUT_COST = 1e6


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
    # SAM tracker per-object confidence for this frame (SAM3 out_tracker_probs),
    # or None. PAYLOAD ONLY — recorded onto the matched slot for path-sample
    # quality metadata; it never enters cost, gating, fusion weights, or any
    # tracking decision.
    tracker_prob: Optional[float] = None
    # Cameras whose detections fused into this one (fusion payload; empty when
    # unknown). Only corner_dup_suppress reads it: a corner-only detection is
    # a duplicate of a nearby main-backed one only if that one's cluster lacks
    # this corner camera (else the corner saw two robots there).
    cams: tuple = ()


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
    # SAM tracker confidence of the detection that last measured this slot, or
    # None (legacy / no prob / coasting-from-a-prob-less measurement). Payload
    # only — see FieldDetection.tracker_prob.
    tracker_prob: Optional[float] = None


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
    # SAM tracker confidence of the detection that last measured this slot (payload
    # only, for path-sample quality metadata). None until a prob-carrying
    # detection measures the slot; left untouched while the slot coasts.
    tracker_prob: Optional[float] = None
    # Went into coast while hub-occluded; stays set until re-acquired. Gates the
    # occlusion re-acquisition guard (suspect-zone detections need motion proof).
    occ_coasting: bool = False
    # The position of the last detection that measured this slot, and the
    # slot's position when it last expired. Observation-only (slot_events)
    # unless coast_gate_phys / reseed_identity reason from last_meas_pos.
    last_meas_pos: Optional[tuple] = None
    expired_pos: Optional[tuple] = None
    # Physical-reach bookkeeping (coast_gate_phys / reseed_identity): update
    # index of the last measurement, whether it was corner-only, and the update
    # index at which the slot last expired.
    last_meas_k: Optional[int] = None
    last_meas_corner: bool = False
    expired_k: Optional[int] = None


class FieldTracker:
    def __init__(self, field_length, gate_m=2.5, max_coast_open=8,
                 max_coast_occluded=90, color_penalty=3.0, zones=None,
                 corner_only_r_scale=8.0, coast_vel_decay=0.85,
                 max_speed_m_s=5.5, sticky_max_dist_m=0.0,
                 suspect_zones=None, occlusion_rebind_guard=False,
                 guard_motion_m=0.3, guard_motion_window=5,
                 kin_gate_enabled=False, kin_max_speed_mps=6.5,
                 record_history=False,
                 coast_gate_phys=False, reseed_identity=False,
                 corner_dup_suppress=False, phys_reach_mps=5.0,
                 phys_reach_slack_m=1.0, phys_corner_slack_m=2.0,
                 reseed_hold_s=1.5, reseed_fresh_cost_m=3.0,
                 corner_dup_radius_m=2.0, assign_partial_fix=False,
                 phys_excess_weight=2.0, alliance_by_territory=True):
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
        # Kinematic plausibility gate: hard-reject an association implying a
        # robot speed beyond what's physically possible (FRC robots sustain
        # <= ~5.5-6 m/s). The KF's _measured_update already downweights fast
        # measurements via r_scale; this is an additional hard rejection so
        # the detection is routed elsewhere (distance match / seed a free
        # slot) instead of dragging the existing slot along. Default off.
        self.kin_gate_enabled = bool(kin_gate_enabled)
        self.kin_max_speed = float(kin_max_speed_mps)
        # Physical reach (all default OFF; see _reach_excess). A robot last
        # measured at P, gap seconds ago, is within phys_reach_mps * gap of P,
        # plus measurement slack (phys_reach_slack_m; + phys_corner_slack_m
        # when either end is a corner-only position, whose range is poor).
        # - coast_gate_phys: a COASTING slot's widened gate (up to 4x gate_m)
        #   is softened by the physics: the excess beyond reach is added to the
        #   association cost, and a pair whose penalised cost leaves the
        #   widened gate is rejected. Visible slots and the base gate_m are
        #   untouched, so steady tracking is unchanged.
        # - reseed_identity: leftover detections seed the FREE slot whose last
        #   measurement best explains them (min-cost assignment) instead of the
        #   lowest-numbered one; a slot that expired < reseed_hold_s ago is
        #   held (not reseeded) by a detection beyond its physical reach.
        #   A never-seeded slot costs reseed_fresh_cost_m.
        # - corner_dup_suppress: a corner-only detection within
        #   corner_dup_radius_m of a main-backed detection of the same alliance
        #   whose fused cluster lacks that corner camera is that robot's
        #   unfused corner view (fusion missed the merge). It is dropped
        #   before matching: it can't seed a second slot onto the robot, and a
        #   slot riding it falls back to the main view (see _corner_dups).
        # - phys_excess_weight: cost per meter of reach excess (coast gate).
        # assign_partial_fix: _best_assignment sums raw costs, so when EVERY
        # full pairing contains a gated-out (inf) pair — e.g. one extra
        # out-of-gate detection or slot — all sums tie at inf and the FIRST
        # permutation's finite pairs are applied, however poor (a coasting slot
        # takes the det a visible slot sits on). With the fix a gated-out pair
        # costs a large finite constant instead, so the search maximises the
        # number of in-gate matches, then minimises their cost. Default OFF.
        self.assign_partial_fix = bool(assign_partial_fix)
        self.coast_gate_phys = bool(coast_gate_phys)
        self.reseed_identity = bool(reseed_identity)
        self.corner_dup_suppress = bool(corner_dup_suppress)
        self.phys_reach_mps = float(phys_reach_mps)
        self.phys_excess_weight = float(phys_excess_weight)
        self.phys_reach_slack_m = float(phys_reach_slack_m)
        self.phys_corner_slack_m = float(phys_corner_slack_m)
        self.reseed_hold_s = float(reseed_hold_s)
        self.reseed_fresh_cost_m = float(reseed_fresh_cost_m)
        self.corner_dup_radius_m = float(corner_dup_radius_m)
        # alliance_by_territory (default True = legacy): an unknown-colour
        # detection is offered only to the alliance whose field half it is in,
        # and may seed a slot there. False (territory-free; robots cross
        # midfield constantly in 2026): known-colour detections are assigned
        # per alliance first, then ONE joint pass offers the leftover unknown
        # detections to the still-unmatched pre-existing slots of BOTH
        # alliances (each det claimed at most once, min total cost); unknown
        # detections never seed. See _assign_territory_free.
        self.alliance_by_territory = bool(alliance_by_territory)
        self._det_history: dict[int, deque] = {}  # track_id -> (frame_idx, pos)
        self._frame_idx = 0
        # Opt-in measurement history for the offline RTS smoothing pass
        # (web/field/smoother.py). Observation-only: never read by any
        # tracking decision. slot_id -> list of segments; a segment is the
        # record list from one seed until the frame before expiry.
        self.record_history = bool(record_history)
        self.history: dict[int, list[list[dict]]] = {}
        self.tracks = []
        sid = 0
        for a in ALLIANCES:
            for _ in range(3):
                self.tracks.append(_Track(slot_id=sid, alliance=a))
                sid += 1
        # Maps stable track_id -> slot_id for sticky binding.
        self._id_to_slot: dict[int, int] = {}
        self.binding_events: list[dict] = []
        # Observation-only per-update log of the events that can change a slot's
        # physical robot (seed / expire / coasting re-acquisition), with the
        # geometry behind each (web/field/slot_boundaries.py). Unlike
        # binding_events it also covers track_id-less detections. Never read by
        # any tracking decision.
        self.slot_events: list[dict] = []
        self.assignment_evidence: dict[int, dict] = {}

    def config_dict(self) -> dict:
        """Constructor kwargs reproducing this tracker (TRACKER_TRACE header,
        consumed by scripts/replay_field_tracker.py). JSON-serialisable."""
        def _z(zs):
            return [[float(v) for v in z] for z in zs]
        return {
            "field_length": float(self.field_length), "gate_m": float(self.gate),
            "max_coast_open": int(self.max_open), "max_coast_occluded": int(self.max_occ),
            "color_penalty": float(self.color_penalty), "zones": _z(self.zones),
            "corner_only_r_scale": self.corner_only_r_scale,
            "coast_vel_decay": self.coast_vel_decay, "max_speed_m_s": self.max_speed,
            "sticky_max_dist_m": self.sticky_max_dist_m,
            "suspect_zones": _z(self.suspect_zones),
            "occlusion_rebind_guard": self.occlusion_rebind_guard,
            "guard_motion_m": self.guard_motion_m,
            "guard_motion_window": self.guard_motion_window,
            "kin_gate_enabled": self.kin_gate_enabled,
            "kin_max_speed_mps": self.kin_max_speed,
            "coast_gate_phys": self.coast_gate_phys,
            "reseed_identity": self.reseed_identity,
            "corner_dup_suppress": self.corner_dup_suppress,
            "phys_reach_mps": self.phys_reach_mps,
            "phys_excess_weight": self.phys_excess_weight,
            "phys_reach_slack_m": self.phys_reach_slack_m,
            "phys_corner_slack_m": self.phys_corner_slack_m,
            "reseed_hold_s": self.reseed_hold_s,
            "reseed_fresh_cost_m": self.reseed_fresh_cost_m,
            "corner_dup_radius_m": self.corner_dup_radius_m,
            "assign_partial_fix": self.assign_partial_fix,
            "alliance_by_territory": self.alliance_by_territory,
        }

    def _reach_excess(self, t, pos, det_corner, dt) -> float:
        """Meters by which ``pos`` lies beyond where slot ``t``'s robot can
        physically be: phys_reach_mps x the time since its last measurement,
        plus measurement slack (more when the detection or that measurement is
        corner-only — corner ground projections have poor range). 0 = within
        reach; also 0 when the slot has no measurement to reason from."""
        if t.last_meas_pos is None or t.last_meas_k is None or not dt or dt <= 0:
            return 0.0
        gap_s = (self._frame_idx - t.last_meas_k) * dt
        reach = self.phys_reach_mps * gap_s + self.phys_reach_slack_m
        if det_corner or t.last_meas_corner:
            reach += self.phys_corner_slack_m
        dist = float(np.hypot(pos[0] - t.last_meas_pos[0], pos[1] - t.last_meas_pos[1]))
        return max(0.0, dist - reach)

    def _corner_dups(self, dets_with_idx, context=()) -> set:
        """Original indices of corner-only detections that are an unfused
        corner view of a main-backed detection of the same alliance pool this
        frame: within corner_dup_radius_m of it, and (when camera membership
        is known) from a corner camera that main-backed cluster lacks — if
        the cluster already holds that camera's view, the corner camera saw a
        second robot there (e.g. one hidden from main), not a duplicate.
        ``context``: extra ``(idx, det)`` pairs that count as main-backed
        partners but are not themselves candidates (territory-free passes,
        where the legacy pool's unknown/known mix is split across passes)."""
        if not self.corner_dup_suppress:
            return set()
        mains = [d for _i, d in list(dets_with_idx) + list(context)
                 if d.box is not None]
        r = self.corner_dup_radius_m
        out = set()
        for i, d in dets_with_idx:
            if d.box is not None:
                continue
            own = set(d.cams or ())
            for m in mains:
                if float(np.hypot(d.pos[0] - m.pos[0], d.pos[1] - m.pos[1])) > r:
                    continue
                if own and m.cams and own & set(m.cams):
                    continue
                out.add(i)
                break
        return out

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

    def _kin_violates(self, t, d, dt):
        """Implied speed of detection d claiming slot t, vs the physical cap.
        Coasting slots were last measured (missed+1) intervals ago."""
        if not self.kin_gate_enabled or t.kf is None or not dt or dt <= 0:
            return False
        gap_s = dt * (t.missed + 1)
        dist = float(np.linalg.norm(
            np.asarray(d.pos, float) - np.asarray(t.kf.pos, float)))
        return (dist / gap_s) > self.kin_max_speed

    def update(self, detections, occlusion, dt, portals=None, frame_num=None):
        self.binding_events = []
        self.slot_events = []
        self.assignment_evidence = {}
        self._pending_meas: dict[int, tuple] = {}   # slot_id -> (z, r_scale)
        self._pending_seed: set[int] = set()
        # Portal gate: {slot_id: (x, y, radius_m)}. Only consulted for slots
        # that are occ_coasting; lets an occluded slot re-acquire a detection
        # near its predicted hub-exit point even outside the plain distance
        # gate. Empty/None when the occlusion_portal knob is off.
        self._portals = dict(portals or {})
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
        if not self.alliance_by_territory:
            self._assign_territory_free(indexed, dt)
        else:
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
                t.expired_pos = t.kf.pos if t.kf is not None else None
                t.expired_k = self._frame_idx
                self.slot_events.append({
                    "event": "expire", "slot_id": t.slot_id,
                    "pos": t.expired_pos, "missed": int(t.missed)})
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

        if self.record_history:
            for t in self.tracks:
                if not t.active:
                    continue
                meas = self._pending_meas.get(t.slot_id)
                rec = {
                    "frame_num": frame_num,
                    "dt": float(dt),
                    "z": meas[0] if meas else None,
                    "r_scale": meas[1] if meas else None,
                    "seed": t.slot_id in self._pending_seed,
                    "corner_only": bool(t.corner_only),
                }
                segs = self.history.setdefault(t.slot_id, [])
                if rec["seed"] or not segs:
                    segs.append([rec])
                else:
                    segs[-1].append(rec)

        return self._emit()

    def _assign_territory_free(self, indexed, dt):
        """alliance_by_territory=False: assignment without field-half alliance.

        1. Per alliance, that colour's KNOWN detections are matched/seeded
           exactly as in the legacy pass (unknown dets are context only: they
           still count for corner-dup partners and track_id ambiguity, as they
           would have in the legacy same-half pool).
        2. ONE joint pass offers the leftover UNKNOWN-colour detections to the
           slots of BOTH alliances that were active before this frame and are
           still unmatched (same gates/penalties, incl. color_penalty). A
           single min-cost assignment over that joint matrix means one
           detection can claim at most one slot. Never seeds.
        3. Pre-existing active slots still unmatched miss this frame (the
           missed counter is deferred to here so step 2 sees the same coast
           gate the legacy pass would).
        Deterministic: no ordering between alliances matters (step 1's
        passes touch disjoint slots and detections)."""
        pre_active = {t.slot_id for t in self.tracks if t.active}
        known = [(i, d) for i, d in indexed if d.color in ALLIANCES]
        unknown = [(i, d) for i, d in indexed if d.color == "unknown"]
        for alliance in ALLIANCES:
            self._assign(self._slots(alliance),
                         [(i, d) for i, d in known if d.color == alliance],
                         alliance, dt, count_miss=False, context=unknown)
        claimed = set(self.last_det_to_slot.values())
        eligible = {sid for sid in pre_active if sid not in claimed}
        if unknown and eligible:
            self._assign(list(self.tracks), unknown, "unknown", dt,
                         seed=False, count_miss=False, eligible=eligible,
                         context=known)
        claimed = set(self.last_det_to_slot.values())
        for t in self.tracks:
            if t.slot_id in pre_active and t.active and t.slot_id not in claimed:
                t.missed += 1

    def _joint_assignment(self, cost):
        """Min-cost assignment for the territory-free joint pass (up to 6
        slots x N unknown dets — too big for the brute force). Gated-out
        pairs cost a large constant, so in-gate matches are maximised then
        cost-minimised (the assign_partial_fix semantics); callers drop
        non-finite pairs. Deterministic."""
        from scipy.optimize import linear_sum_assignment
        c = np.where(np.isfinite(cost), cost, _GATED_OUT_COST)
        rows, cols = linear_sum_assignment(c)
        return [int(r) for r in rows], [int(k) for k in cols]

    def _assign(self, slots, dets_with_idx, alliance, dt, *, seed=True,
                count_miss=True, eligible=None, context=()):
        """Match active tracks to dets, update matched, coast unmatched, seed free slots.

        Legacy calls use the defaults. Territory-free passes
        (_assign_territory_free) may disable seeding (``seed``) and the
        missed-counter increment (``count_miss``), restrict matching to the
        slot ids in ``eligible`` (others are neither matched nor sticky-
        rebound; a sticky id pointing at one routes to distance matching),
        and pass ``context`` detections that count for corner-dup partners
        and track_id ambiguity without being matchable here."""
        active_slots = [t for t in slots if t.active
                        and (eligible is None or t.slot_id in eligible)]
        free_slots = [t for t in slots if not t.active] if seed else []

        # corner_dup_suppress: an unfused corner view of a main-backed robot is
        # that robot measured twice (worse geometry); it is dropped from the
        # pool before any matching or seeding, so it can neither seed a second
        # slot nor keep a slot riding it (that slot re-acquires the main view).
        dup_idx = self._corner_dups(dets_with_idx, context)
        if dup_idx:
            self.slot_events.append({"event": "corner_dup", "slot_id": None,
                                     "det_idx": sorted(int(i) for i in dup_idx),
                                     "alliance": alliance})
            dets_with_idx = [(i, d) for i, d in dets_with_idx if i not in dup_idx]

        # A track_id carried by 2+ detections in this frame is AMBIGUOUS: the
        # registry mis-assigned one id to distinct robots (merge_main_guard
        # passes them through so each keeps its own slot). Sticky binding and
        # mapping updates for such an id would flip-flop _id_to_slot between
        # the robots' slots every frame (qm11 R1/R2/R3 churn); route these to
        # distance matching, which preserves per-slot continuity, and learn
        # nothing from them.
        tid_counts: dict = {}
        for _i, d in list(dets_with_idx) + list(context):
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
                if (eligible is not None and slot is not None and slot.active
                        and slot.slot_id not in eligible):
                    # Territory-free joint pass: the id's slot already took a
                    # measurement this frame (or was just seeded). Keep the
                    # mapping; let distance matching place this detection.
                    remaining_pairs.append((orig_idx, d))
                    continue
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
                    if self._kin_violates(slot, d, dt):
                        # Implied speed exceeds the physical cap: the
                        # registry handed this id to a robot it isn't. Refuse
                        # the rebind and let distance matching decide.
                        gap_s = dt * (slot.missed + 1)
                        dist = float(np.linalg.norm(
                            np.array(slot.kf.pos) - np.array(d.pos)))
                        self.binding_events.append({
                            "event": "kinematic_reject",
                            "slot_id": slot.slot_id,
                            "track_id": tid,
                            "alliance": slot.alliance,
                            "speed_mps": round(dist / gap_s, 1),
                        })
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
                    self._note_reacquire(slot, d, dt, "sticky", None)
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
        portal_pairs = set()   # (slot row, det col) admitted via a portal (slot_events only)
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
                    in_gate = dist <= effective_gate
                    # Portal second chance: an occlusion-coasting slot may
                    # re-acquire a detection near its predicted hub-exit point
                    # even beyond the plain distance gate.
                    portal = self._portals.get(t.slot_id) if t.occ_coasting else None
                    if not in_gate and portal is not None:
                        px, py, pr = portal
                        d_portal = float(np.hypot(d.pos[0] - px, d.pos[1] - py))
                        if d_portal <= pr:
                            in_gate = True
                            dist = d_portal + 0.5   # small penalty vs direct matches
                            portal_pairs.add((i, j))
                    if in_gate:
                        if self._kin_violates(t, d, dt):
                            # Implied speed exceeds the physical cap: leave
                            # this pair at cost=inf so it can't win the
                            # assignment (the detection may seed/claim
                            # elsewhere — it might be a different robot).
                            gap_s = dt * (t.missed + 1)
                            self.binding_events.append({
                                "event": "kinematic_reject",
                                "slot_id": t.slot_id,
                                "track_id": d.track_id,
                                "alliance": t.alliance,
                                "speed_mps": round(dist / gap_s, 1),
                            })
                            continue
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
                        if (self.coast_gate_phys and t.missed > 0
                                and (i, j) not in portal_pairs):
                            # Soft physical gate: penalise the distance beyond
                            # the robot's reach since its last measurement;
                            # reject once the penalised cost leaves the
                            # (widened) gate.
                            excess = self._reach_excess(t, d.pos, d.box is None, dt)
                            if excess > 0:
                                c += self.phys_excess_weight * excess
                                if c > effective_gate:
                                    self.slot_events.append({
                                        "event": "coast_gate_reject",
                                        "slot_id": t.slot_id, "missed": int(t.missed),
                                        "det_idx": int(idxs[j]),
                                        "excess_m": round(excess, 2)})
                                    continue
                        if d.color == "unknown":
                            c += self.color_penalty
                        cost[i, j] = c

            # Find best assignment (brute force over <=3! permutations;
            # the territory-free joint pass spans both alliances -> LSA).
            if eligible is not None:
                rows, cols = self._joint_assignment(cost)
            else:
                rows, cols = self._best_assignment(cost)

            # Determine which (slot_idx, det_idx) pairs are valid (gated)
            matched_slot_indices = set()
            matched_det_indices = set()
            for r, c in zip(rows, cols):
                if np.isfinite(cost[r, c]):
                    # Update matched slot
                    t = active_slots[r]
                    self._note_reacquire(
                        t, dets[c], dt, "portal" if (r, c) in portal_pairs else "distance",
                        self.gate * min(4.0, 1.0 + 0.5 * t.missed))
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
            if count_miss:
                for i, t in enumerate(active_slots):
                    if i not in matched_slot_indices:
                        t.missed += 1

            # Seed leftover detections into free slots — but never seed a new
            # slot from a detection inside a static-structure zone (a real robot
            # is never born there; it drives in as an already-active track).
            cands = [(idxs[j], dets[j]) for j in range(len(dets))
                     if j not in matched_det_indices and j not in guard_blocked
                     and not point_in_zones(dets[j].pos, self.zones)]
            for slot, (orig_idx, d) in self._seed_pairs(free_slots, cands, dt):
                self._seed_slot(slot, d, orig_idx, ambiguous_tids)

        elif active_slots and not dets:
            # No detections: all active slots missed
            if count_miss:
                for t in active_slots:
                    t.missed += 1

        else:
            # No active slots: seed as many dets as we have free slots, skipping
            # any detection sitting inside a static-structure zone.
            cands = [(idxs[j], d) for j, d in enumerate(dets)
                     if not point_in_zones(d.pos, self.zones)]
            for slot, (orig_idx, d) in self._seed_pairs(free_slots, cands, dt):
                self._seed_slot(slot, d, orig_idx, ambiguous_tids)

    def _seed_pairs(self, free_slots, cands, dt):
        """Which free slot each seedable detection ``(orig_idx, det)`` seeds.

        Default: the lowest-numbered free slots, in detection order.
        reseed_identity: a min-cost assignment where a never-seeded slot costs
        reseed_fresh_cost_m and an expired slot costs its physical-reach excess
        (plus a 0.1 x distance tie-break) from its last measurement; a slot
        that expired less than reseed_hold_s ago is HELD — never reseeded by a
        detection beyond its reach (a recently lost robot is likelier to
        reappear than to teleport across the field)."""
        if not free_slots or not cands:
            return []
        if not self.reseed_identity:
            return list(zip(free_slots, cands))
        big = 1e6
        cost = np.zeros((len(free_slots), len(cands)))
        for a, t in enumerate(free_slots):
            for b, (_i, d) in enumerate(cands):
                if t.last_meas_pos is None:
                    cost[a, b] = self.reseed_fresh_cost_m
                    continue
                excess = self._reach_excess(t, d.pos, d.box is None, dt)
                since = ((self._frame_idx - t.expired_k) * dt
                         if t.expired_k is not None and dt else float("inf"))
                if excess > 0 and since < self.reseed_hold_s:
                    cost[a, b] = big
                    continue
                dist = float(np.hypot(d.pos[0] - t.last_meas_pos[0],
                                      d.pos[1] - t.last_meas_pos[1]))
                cost[a, b] = excess + 0.1 * dist
        rows, cols = self._best_assignment(cost)
        pairs = []
        for r, c in zip(rows, cols):
            if cost[r, c] >= big:
                self.slot_events.append({
                    "event": "reseed_hold", "slot_id": free_slots[r].slot_id,
                    "det_idx": int(cands[c][0])})
                continue
            pairs.append((free_slots[r], cands[c]))
        return pairs

    def _seed_slot(self, slot, d, orig_idx, ambiguous_tids):
        """Seed free ``slot`` from detection ``d`` (fused index ``orig_idx``)."""
        self._note_seed(slot, d)
        slot.kf = _KF(d.pos)
        slot.box = d.box
        slot.corner_only = d.box is None
        slot.tracker_prob = d.tracker_prob  # payload only
        slot.missed = 0
        slot.hits = 1
        slot.coasting = False
        slot.active = True
        if self.record_history:
            self._pending_seed.add(slot.slot_id)
            self._pending_meas[slot.slot_id] = (
                (float(d.pos[0]), float(d.pos[1])), 1.0)
        self.last_det_to_slot[orig_idx] = slot.slot_id
        if d.track_id is not None and d.track_id not in ambiguous_tids:
            self._id_to_slot[d.track_id] = slot.slot_id
            self.binding_events.append({
                "event": "slot_seeded",
                "slot_id": slot.slot_id,
                "track_id": d.track_id,
                "alliance": slot.alliance,
            })

    def _note_seed(self, slot, d):
        """slot_events: a free slot (re)seeded from detection ``d``. Records the
        distance from where the slot last expired (None on its first seed)."""
        pos = (float(d.pos[0]), float(d.pos[1]))
        prev = slot.expired_pos
        self.slot_events.append({
            "event": "seed", "slot_id": slot.slot_id, "pos": pos,
            "track_id": d.track_id, "prev_pos": prev,
            "seed_dist_m": (float(np.hypot(pos[0] - prev[0], pos[1] - prev[1]))
                            if prev is not None else None),
            "corner_only": d.box is None})
        slot.last_meas_pos = pos
        slot.last_meas_k = self._frame_idx
        slot.last_meas_corner = d.box is None

    def _note_reacquire(self, t, d, dt, via, gate_m):
        """slot_events: a COASTING slot (missed > 0) matched detection ``d``.
        ``jump_m`` is measured from the slot's last measurement, ``pred_jump_m``
        from its coasted prediction; ``speed_mps`` is jump_m over the coast."""
        if t.missed <= 0 or t.kf is None:
            return
        dp = np.asarray(d.pos, float)
        last = t.last_meas_pos
        jump = float(np.linalg.norm(dp - np.asarray(last, float))) if last is not None else None
        span = (dt or 0.0) * (t.missed + 1)
        self.slot_events.append({
            "event": "reacquire", "slot_id": t.slot_id, "missed": int(t.missed),
            "jump_m": jump,
            "pred_jump_m": float(np.linalg.norm(dp - np.asarray(t.kf.pos, float))),
            "speed_mps": (jump / span) if (jump is not None and span > 0) else None,
            "gate_m": gate_m, "via": via, "track_id": d.track_id,
            "corner_only": d.box is None})

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
        t.last_meas_pos = (float(d.pos[0]), float(d.pos[1]))
        t.last_meas_k = self._frame_idx
        t.last_meas_corner = d.box is None
        # Payload: record the measuring detection's SAM confidence on the slot.
        # After the KF update so it can never feed back into the state estimate.
        t.tracker_prob = d.tracker_prob
        if self.record_history:
            self._pending_meas[t.slot_id] = (
                (float(d.pos[0]), float(d.pos[1])), float(scale))

    def _best_assignment(self, cost):
        """Brute-force minimum-cost assignment for ≤3×3 cost matrices.

        When n > m (more tracks than detections), we iterate permutations of
        track indices choosing m of them to match, paired with all m! det orderings.
        For a 3×3 matrix this evaluates P(3,3)×P(3,3) = 36 candidate pairings.
        Returns (track_rows, det_cols) for the best finite-cost pairs found.
        """
        n, m = cost.shape
        if self.assign_partial_fix:
            # Gated-out pairs (inf) become a large finite cost so partial
            # matchings stay comparable; callers drop non-finite pairs.
            cost = np.where(np.isfinite(cost), cost, _GATED_OUT_COST)
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
                    tracker_prob=t.tracker_prob,
                ))
        return out
