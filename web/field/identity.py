from __future__ import annotations
import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
from web.embedder import cosine_similarity

@dataclass
class RobotIdentity:
    identity_id: int
    alliance: str
    team_number: str | None = None
    is_confirmed: bool = False
    appearance_embedding: np.ndarray | None = None
    anchor_embedding: np.ndarray | None = None
    recent_exemplars: list[np.ndarray] = field(default_factory=list)
    active_slot: int | None = None
    last_seen_frame: int = 0
    last_seen_pos: tuple[float, float] | None = None
    confidence: float = 0.5
    last_vlm_time: float = -999.0
    last_success_time: float = -999.0
    # Field position at the last VLM attempt — the novelty gate skips a failed
    # retry until the robot has moved enough to present a new view.
    last_vlm_pos: tuple[float, float] | None = None
    backoff_until: float = -999.0
    vlm_fail_count: int = 0
    needs_immediate_read: bool = False
    # Last time an immediate read was queued — the escalation cooldown
    # (should_queue_immediate_read) keys off this and last_vlm_time.
    last_immediate_queue_time: float = -999.0
    has_vlm_read: bool = False
    # VLM equipment adjudication (observe-only): consecutive non-robot
    # verdicts on this identity's crop, and the last time we asked. The
    # streak resets on a "robot" verdict; nothing acts on it yet — it is
    # diagnostic evidence for grading the signal before wiring corrections.
    equipment_streak: int = 0
    last_adjudicate_time: float = -999.0
    vlm_agrees: bool = False
    vlm_contradicts: bool = False
    is_ambiguous: bool = False
    # Churn telemetry + young-identity warm-up: the update_frame frame this
    # identity was created on, and how many observations it has matched
    # (the birth observation counts as 1).
    born_frame: int = 0
    obs_count: int = 0

@dataclass
class IdentityObservation:
    slot_id: int
    pos: tuple[float, float]
    alliance: str
    embedding: np.ndarray | None = None
    team_number: str | None = None
    team_conf: float = 0.0
    relink_margin: float = 1.0
    continuity_evidence: str = "initial"
    # True when the field tracker reported a binding disruption for this slot
    # this frame (seeded / expired / kinematic_reject / occlusion_rebind_held,
    # or the slot is now fed by a different track id than last frame). A
    # disrupted slot never gets the identity_continuity pass.
    slot_disrupted: bool = False

# Field-tracker binding events (FieldTracker.binding_events) that break slot
# continuity for the identity_continuity gate. slot_rebound is deliberately
# absent: it fires every frame a known track id re-binds to its own slot (the
# steady state); a slot fed by a different track id than last frame is
# detected by the caller (InferenceJob._sync_identity_registry_frame).
DISRUPTING_BINDING_EVENTS = frozenset(
    {"slot_seeded", "slot_expired", "kinematic_reject", "occlusion_rebind_held"})


def live_robot_tracks(robot_tracks: dict, registry) -> dict:
    """``robot_tracks`` restricted to identities still alive in ``registry``.

    robot_tracks is keyed by identity id and never pruned (qm11 full match:
    575 entries, ~6 alive); cap enforcement pops evicted identities from
    ``registry.identities`` and leaves their tracks behind as history. Team
    arbitration (Hungarian, elimination, HUD claims) must only consider the
    live ones, or a dead identity's stale votes steal a team from the robot
    actually on the field. ``registry`` None (legacy stand-ins) -> all live.
    """
    if registry is None:
        return dict(robot_tracks)
    alive = registry.identities
    return {rid: r for rid, r in robot_tracks.items() if rid in alive}


# Association cost terms (see IdentityRegistry.update_frame). Named so the
# churn telemetry can say which term pushed a rejected match over threshold.
BASE_COST = 0.5
VETO_PENALTY = 0.5            # max cosine < app_floor
APP_BONUS = 0.3               # scaled by how far max cosine clears app_floor
FAR_DIST_M = 3.5
FAR_PENALTY = 1.0             # moved > FAR_DIST_M since last seen
DIST_WEIGHT = 0.1             # per metre, within FAR_DIST_M
TEAM_MATCH_BONUS = 0.4
TEAM_CONFLICT_PENALTY = 0.8

# Why an observation did NOT join an existing identity (=> identity birth).
BIRTH_REASONS = ("appearance_veto", "distance", "team_conflict", "no_candidate", "other")


def _reject_reason(cost: float, veto: float, far: float, team: float,
                   threshold: float) -> str:
    """The single cost term whose removal would have let the match through;
    "other" when no one term is decisive on its own."""
    if veto > 0 and cost - veto < threshold:
        return "appearance_veto"
    if far > 0 and cost - far < threshold:
        return "distance"
    if team > 0 and cost - team < threshold:
        return "team_conflict"
    return "other"


def churn_summary(stats, final_ages_s: dict, span_s: float | None,
                  continuity: bool) -> tuple[str, dict]:
    """End-of-job identity churn line + JSON-able payload from
    ``IdentityRegistry.stats``. ``final_ages_s`` maps slot label -> age in
    seconds (born -> last seen) of the identity holding that slot at the end;
    ``span_s`` is the processed match span used for the births/min rate."""
    births = int(stats.get("births", 0))
    by_reason = {r: int(stats.get(f"births_{r}", 0)) for r in BIRTH_REASONS}
    eligible = int(stats.get("births_veto_continuity_eligible", 0))
    per_min = round(births / span_s * 60.0, 1) if span_s and span_s > 0 else None
    parts = []
    for r in BIRTH_REASONS:
        txt = f"{r} {by_reason[r]}"
        if r == "appearance_veto":
            txt += f" [continuity-eligible {eligible}]"
        parts.append(txt)
    ages = " ".join(f"{k} {v:.1f}s" for k, v in final_ages_s.items())
    line = (f"[identity] births {births} ({', '.join(parts)})"
            + (f" {per_min}/min" if per_min is not None else "")
            + f", evictions {int(stats.get('evictions', 0))}"
            + f", cap_rejected {int(stats.get('cap_rejected', 0))}"
            + f", matches {int(stats.get('matches', 0))}"
            + f", continuity={'on' if continuity else 'off'}"
            + f" holds {int(stats.get('continuity_holds', 0))}"
            + f" warmup_updates {int(stats.get('warmup_updates', 0))}"
            + f", final ages {ages or '-'}")
    payload = {
        "births": births,
        "births_per_min": per_min,
        "births_by_reason": by_reason,
        "births_veto_continuity_eligible": eligible,
        "evictions": int(stats.get("evictions", 0)),
        "cap_rejected": int(stats.get("cap_rejected", 0)),
        "matches": int(stats.get("matches", 0)),
        "continuity": bool(continuity),
        "continuity_holds": int(stats.get("continuity_holds", 0)),
        "warmup_updates": int(stats.get("warmup_updates", 0)),
        "final_ages_s": dict(final_ages_s),
    }
    return line, payload


class IdentityRegistry:
    """Durable robot identities on top of the field tracker's 6 slots.

    Continuity knob (``continuity``, job knob ``identity_continuity``, default
    OFF): when an observation lands on the SAME slot the identity held on the
    previous update, the tracker reported no binding disruption for that slot
    (``IdentityObservation.slot_disrupted``), the slot moved less than
    ``continuity_max_m`` and there is no conflicting team evidence, a failed
    appearance check costs ``continuity_soft_penalty`` instead of the full
    veto. DINOv2 cannot reliably tell same-alliance robots apart and a single
    robot's crop changes with viewpoint, so on an undisturbed slot the
    tracker's continuity is the stronger evidence; genuine swaps still split
    (disruption events, position jumps, team conflicts keep the full veto).

    Warm-up (``warmup_frames``, only with ``continuity``): the EMA/exemplar
    update normally needs confidence >= ``app_update_floor`` (~5 frames for a
    new identity), so a young identity only ever compares against its birth
    embedding and the next viewpoint change splits it again. For its first
    ``warmup_frames`` observations (birth included) an identity may update at
    any confidence — still gated on ``relink_margin`` — filling the 5-sample
    exemplar FIFO once. The birth ``anchor_embedding`` never changes, so an
    early bad sample cannot erase the birth appearance; mature identities keep
    the confidence gate unchanged.

    Telemetry (always on, cheap): ``stats`` counts births by reason
    (``births_<reason>``, reasons in BIRTH_REASONS), evictions, cap-rejected
    creations, matches, continuity holds and warm-up updates.
    ``frame_rejections`` / ``frame_evictions`` describe the latest
    update_frame only (reset each call) for per-frame diagnostic rows.
    """

    def __init__(
        self,
        association_threshold: float = 0.75,
        app_floor: float = 0.55,
        ema_alpha: float = 0.3,
        app_update_floor: float = 0.6,
        continuity: bool = False,
        continuity_max_m: float = 1.0,
        continuity_soft_penalty: float = 0.1,
        warmup_frames: int = 0,
        alliance_by_territory: bool = True,
        update_margin: float = 0.05,
    ) -> None:
        self.identities: dict[int, RobotIdentity] = {}
        # Relink-margin gate on appearance EMA updates (cosine units of the
        # job's embedder: web.embedder.EmbedThresholds.update_margin).
        self.update_margin = float(update_margin)
        # alliance_by_territory (job knob, default True = legacy): a birth
        # from an observation with no known alliance takes its alliance from
        # the field half (alliance_side_fn). False: that birth is DEFERRED
        # (no identity) until an observation with a known alliance arrives —
        # an unknown-alliance observation can still MATCH an existing
        # identity of either alliance (the cost's alliance check skips it).
        self.alliance_by_territory = bool(alliance_by_territory)
        self.association_threshold = association_threshold
        self.app_floor = app_floor
        self.ema_alpha = ema_alpha
        self.app_update_floor = app_update_floor
        self.continuity = bool(continuity)
        self.continuity_max_m = float(continuity_max_m)
        self.continuity_soft_penalty = float(continuity_soft_penalty)
        self.warmup_frames = int(warmup_frames)
        self._next_id = 0
        self.stats: Counter = Counter()
        self.frame_rejections: list[dict] = []
        self.frame_evictions: list[int] = []
        self._last_frame_idx: int | None = None

    def get_active_identities(self) -> list[RobotIdentity]:
        return [ident for ident in self.identities.values() if ident.active_slot is not None]

    def get_inactive_identities(self) -> list[RobotIdentity]:
        return [ident for ident in self.identities.values() if ident.active_slot is None]

    def _enforce_caps_to_limit(
        self,
        alliance: str,
        limit: int,
        protected_ids: set[int] | None = None,
    ) -> bool:
        protected_ids = protected_ids or set()
        idents = [ident for ident in self.identities.values() if ident.alliance == alliance]
        while len(idents) > limit:
            provisional_inactive = [
                ident for ident in idents 
                if not ident.is_confirmed
                and ident.active_slot is None
                and ident.identity_id not in protected_ids
            ]
            if not provisional_inactive:
                return False
            # Sort by last_seen_frame (ascending), then confidence (ascending)
            provisional_inactive.sort(key=lambda x: (x.last_seen_frame, x.confidence))
            victim = provisional_inactive[0]
            self.identities.pop(victim.identity_id)
            idents.remove(victim)
            self.frame_evictions.append(victim.identity_id)
            self.stats["evictions"] += 1
        return True

    def update_frame(self, frame_idx: int, observations: list[IdentityObservation], alliance_side_fn=None) -> None:
        self.frame_rejections = []
        self.frame_evictions = []
        prev_frame = self._last_frame_idx
        self._last_frame_idx = frame_idx

        # Who held each slot on the previous update (continuity gate +
        # telemetry), then reset active slot bindings.
        prev_holder: dict[int, int] = {}
        for ident in self.identities.values():
            if (ident.active_slot is not None and prev_frame is not None
                    and ident.last_seen_frame == prev_frame):
                prev_holder[ident.active_slot] = ident.identity_id
            ident.active_slot = None

        if not observations:
            return

        # Build global cost matrix
        # Rows = observations, Cols = existing identities
        num_rows = len(observations)
        existing_ids = sorted(list(self.identities.keys()))
        num_cols = len(existing_ids)

        assigned_id_map = {} # row_idx -> identity_id
        rejected: dict[int, dict] = {}  # row_idx -> rejection telemetry row
        
        if num_cols > 0:
            cost_matrix = np.zeros((num_rows, num_cols))
            all_sims = np.zeros((num_rows, num_cols))
            # (r, c) -> (max_sim, dist, veto, far, team, eligible)
            detail: dict[tuple[int, int], tuple] = {}
            
            for r in range(num_rows):
                obs = observations[r]
                for c in range(num_cols):
                    ident = self.identities[existing_ids[c]]
                    
                    # 1. Alliance check
                    if obs.alliance in {"red", "blue"} and ident.alliance in {"red", "blue"} and obs.alliance != ident.alliance:
                        cost_matrix[r, c] = 999.0
                        continue
                        
                    cost = BASE_COST
                    dist = None
                    if ident.last_seen_pos is not None and obs.pos is not None:
                        dist = float(np.linalg.norm(np.array(obs.pos) - np.array(ident.last_seen_pos)))
                    team_conflict = (obs.team_number is not None
                                     and ident.team_number is not None
                                     and obs.team_number != ident.team_number)
                    # Continuity: same slot as last frame's holder, no tracker
                    # disruption, small move, no team conflict. Computed even
                    # with the knob off so telemetry can say how many veto
                    # births it would have prevented.
                    eligible = (prev_holder.get(obs.slot_id) == ident.identity_id
                                and not obs.slot_disrupted
                                and dist is not None and dist < self.continuity_max_m
                                and not team_conflict)
                    max_sim = None
                    veto = 0.0
                    
                    # 2. Appearance similarity
                    if obs.embedding is not None and ident.appearance_embedding is not None:
                        sims = [cosine_similarity(obs.embedding, ident.appearance_embedding)]
                        if ident.anchor_embedding is not None:
                            sims.append(cosine_similarity(obs.embedding, ident.anchor_embedding))
                        for ex in ident.recent_exemplars:
                            sims.append(cosine_similarity(obs.embedding, ex))
                        max_sim = max(sims)
                        all_sims[r, c] = max_sim
                        
                        if max_sim < self.app_floor:
                            veto = (self.continuity_soft_penalty
                                    if (self.continuity and eligible) else VETO_PENALTY)
                            cost += veto
                        else:
                            cost -= APP_BONUS * (max_sim - self.app_floor) / (1.0 - self.app_floor)
                            
                    # 3. Spatial distance
                    far = 0.0
                    if dist is not None:
                        if dist > FAR_DIST_M:
                            far = FAR_PENALTY
                            cost += FAR_PENALTY
                        else:
                            cost += DIST_WEIGHT * dist
                            
                    # 4. Roster consistency
                    team = 0.0
                    if obs.team_number is not None and ident.team_number is not None:
                        if obs.team_number == ident.team_number:
                            cost -= TEAM_MATCH_BONUS
                        else:
                            team = TEAM_CONFLICT_PENALTY
                            cost += TEAM_CONFLICT_PENALTY
                            
                    cost_matrix[r, c] = cost
                    detail[(r, c)] = (max_sim, dist, veto, far, team, eligible)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                cost = cost_matrix[r, c]
                d = detail.get((r, c))
                if cost < self.association_threshold:
                    assigned_id_map[r] = existing_ids[c]
                    if (d is not None and self.continuity and d[2] > 0
                            and d[2] < VETO_PENALTY
                            and cost - d[2] + VETO_PENALTY >= self.association_threshold):
                        self.stats["continuity_holds"] += 1
                elif d is not None:   # a real same-alliance candidate, rejected
                    max_sim, dist, veto, far, team, eligible = d
                    obs = observations[r]
                    rejected[r] = {
                        "slot_id": obs.slot_id,
                        "candidate_id": existing_ids[c],
                        "max_sim": None if max_sim is None else float(max_sim),
                        "distance_m": dist,
                        "cost": float(cost),
                        "reason": _reject_reason(float(cost), veto, far, team,
                                                 self.association_threshold),
                        "continuity_eligible": bool(eligible),
                        "slot_disrupted": bool(obs.slot_disrupted),
                        "outcome": "born",
                    }

        # Apply assignments or create new identities
        protected_assignment_ids = set(assigned_id_map.values())
        for r, obs in enumerate(observations):
            if r in assigned_id_map:
                fused_id = assigned_id_map[r]
                ident = self.identities[fused_id]
                ident.active_slot = obs.slot_id
                ident.last_seen_frame = frame_idx
                ident.last_seen_pos = obs.pos
                ident.obs_count += 1
                self.stats["matches"] += 1
                
                # Expose relink margin if match had other choices
                # Margin = cost - next_best_cost
                # (For simplicity and test matching, we use observations' own margins or calculate it if needed)
                
                # Update appearance embedding
                if obs.embedding is not None:
                    if ident.appearance_embedding is None:
                        ident.appearance_embedding = obs.embedding
                        ident.anchor_embedding = obs.embedding
                        ident.recent_exemplars = [obs.embedding]
                    else:
                        mature_ok = ident.confidence >= self.app_update_floor
                        warm_ok = (self.continuity and self.warmup_frames > 0
                                   and ident.obs_count <= self.warmup_frames)
                        if obs.relink_margin >= self.update_margin and (mature_ok or warm_ok):
                            if not mature_ok:
                                self.stats["warmup_updates"] += 1
                            old_emb = ident.appearance_embedding
                            new_emb = obs.embedding
                            blended = (1.0 - self.ema_alpha) * old_emb + self.ema_alpha * new_emb
                            norm = np.linalg.norm(blended)
                            ident.appearance_embedding = blended / norm if norm > 0 else blended
                            
                            ident.recent_exemplars.append(obs.embedding)
                            if len(ident.recent_exemplars) > 5:
                                ident.recent_exemplars.pop(0)
            else:
                # Create a new identity
                row = rejected.get(r)
                reason = row["reason"] if row is not None else "no_candidate"
                if row is not None:
                    self.frame_rejections.append(row)
                alliance = obs.alliance
                if alliance not in {"red", "blue"} and not self.alliance_by_territory:
                    # Territory-free: never guess the alliance from field half.
                    self.stats["births_deferred_no_alliance"] += 1
                    if row is not None:
                        row["outcome"] = "deferred_no_alliance"
                    continue
                if alliance not in {"red", "blue"}:
                    if alliance_side_fn is not None and obs.pos is not None:
                        alliance = alliance_side_fn(obs.pos)
                    else:
                        alliance = "blue"
                
                # Enforce caps (stay under 3 for this alliance)
                current_count = sum(1 for ident in self.identities.values() if ident.alliance == alliance)
                if current_count >= 3:
                    success = self._enforce_caps_to_limit(
                        alliance, 2, protected_ids=protected_assignment_ids
                    )
                    if not success:
                        # Reject creation
                        self.stats["cap_rejected"] += 1
                        if row is not None:
                            row["outcome"] = "cap_rejected"
                        continue
                
                fused_id = self._next_id
                self._next_id += 1
                self.stats["births"] += 1
                self.stats[f"births_{reason}"] += 1
                if (reason == "appearance_veto" and row is not None
                        and row["continuity_eligible"]):
                    self.stats["births_veto_continuity_eligible"] += 1
                
                new_ident = RobotIdentity(
                    identity_id=fused_id,
                    alliance=alliance,
                    active_slot=obs.slot_id,
                    last_seen_frame=frame_idx,
                    last_seen_pos=obs.pos,
                    born_frame=frame_idx,
                    obs_count=1,
                )
                if obs.embedding is not None:
                    new_ident.appearance_embedding = obs.embedding
                    new_ident.anchor_embedding = obs.embedding
                    new_ident.recent_exemplars = [obs.embedding]
                
                self.identities[fused_id] = new_ident
