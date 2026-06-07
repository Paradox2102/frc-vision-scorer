from __future__ import annotations
import numpy as np
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
    backoff_until: float = -999.0
    vlm_fail_count: int = 0
    needs_immediate_read: bool = False
    has_vlm_read: bool = False
    vlm_agrees: bool = False
    vlm_contradicts: bool = False

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

class IdentityRegistry:
    def __init__(self, association_threshold: float = 0.75, app_floor: float = 0.55, ema_alpha: float = 0.3, app_update_floor: float = 0.6) -> None:
        self.identities: dict[int, RobotIdentity] = {}
        self.association_threshold = association_threshold
        self.app_floor = app_floor
        self.ema_alpha = ema_alpha
        self.app_update_floor = app_update_floor
        self._next_id = 0

    def get_active_identities(self) -> list[RobotIdentity]:
        return [ident for ident in self.identities.values() if ident.active_slot is not None]

    def get_inactive_identities(self) -> list[RobotIdentity]:
        return [ident for ident in self.identities.values() if ident.active_slot is None]

    def _enforce_caps_to_limit(self, alliance: str, limit: int) -> bool:
        idents = [ident for ident in self.identities.values() if ident.alliance == alliance]
        while len(idents) > limit:
            provisional_inactive = [
                ident for ident in idents 
                if not ident.is_confirmed and ident.active_slot is None
            ]
            if not provisional_inactive:
                return False
            # Sort by last_seen_frame (ascending), then confidence (ascending)
            provisional_inactive.sort(key=lambda x: (x.last_seen_frame, x.confidence))
            victim = provisional_inactive[0]
            self.identities.pop(victim.identity_id)
            idents.remove(victim)
        return True

    def update_frame(self, frame_idx: int, observations: list[IdentityObservation], alliance_side_fn=None) -> None:
        # Reset active slot bindings
        for ident in self.identities.values():
            ident.active_slot = None

        if not observations:
            return

        # Build global cost matrix
        # Rows = observations, Cols = existing identities
        num_rows = len(observations)
        existing_ids = sorted(list(self.identities.keys()))
        num_cols = len(existing_ids)

        assigned_id_map = {} # row_idx -> identity_id
        
        if num_cols > 0:
            cost_matrix = np.zeros((num_rows, num_cols))
            all_sims = np.zeros((num_rows, num_cols))
            
            for r in range(num_rows):
                obs = observations[r]
                for c in range(num_cols):
                    ident = self.identities[existing_ids[c]]
                    
                    # 1. Alliance check
                    if obs.alliance in {"red", "blue"} and ident.alliance in {"red", "blue"} and obs.alliance != ident.alliance:
                        cost_matrix[r, c] = 999.0
                        continue
                        
                    cost = 0.5
                    
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
                            cost += 0.5
                        else:
                            cost -= 0.3 * (max_sim - self.app_floor) / (1.0 - self.app_floor)
                            
                    # 3. Spatial distance
                    if ident.last_seen_pos is not None and obs.pos is not None:
                        dist = np.linalg.norm(np.array(obs.pos) - np.array(ident.last_seen_pos))
                        if dist > 3.5:
                            cost += 1.0
                        else:
                            cost += 0.1 * dist
                            
                    # 4. Roster consistency
                    if obs.team_number is not None and ident.team_number is not None:
                        if obs.team_number == ident.team_number:
                            cost -= 0.4
                        else:
                            cost += 0.8
                            
                    cost_matrix[r, c] = cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                cost = cost_matrix[r, c]
                if cost < self.association_threshold:
                    assigned_id_map[r] = existing_ids[c]

        # Apply assignments or create new identities
        for r, obs in enumerate(observations):
            if r in assigned_id_map:
                fused_id = assigned_id_map[r]
                ident = self.identities[fused_id]
                ident.active_slot = obs.slot_id
                ident.last_seen_frame = frame_idx
                ident.last_seen_pos = obs.pos
                
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
                        if obs.relink_margin >= 0.05 and ident.confidence >= self.app_update_floor:
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
                alliance = obs.alliance
                if alliance not in {"red", "blue"}:
                    if alliance_side_fn is not None and obs.pos is not None:
                        alliance = alliance_side_fn(obs.pos)
                    else:
                        alliance = "blue"
                
                # Enforce caps (stay under 3 for this alliance)
                current_count = sum(1 for ident in self.identities.values() if ident.alliance == alliance)
                if current_count >= 3:
                    success = self._enforce_caps_to_limit(alliance, 2)
                    if not success:
                        # Reject creation
                        continue
                
                fused_id = self._next_id
                self._next_id += 1
                
                new_ident = RobotIdentity(
                    identity_id=fused_id,
                    alliance=alliance,
                    active_slot=obs.slot_id,
                    last_seen_frame=frame_idx,
                    last_seen_pos=obs.pos
                )
                if obs.embedding is not None:
                    new_ident.appearance_embedding = obs.embedding
                    new_ident.anchor_embedding = obs.embedding
                    new_ident.recent_exemplars = [obs.embedding]
                
                self.identities[fused_id] = new_ident
