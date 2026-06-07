"""Stable cross-camera fused track ids for the SAM2 tracklet path.

SAM2 emits per-camera object ids that collide across cameras and restart per
prompt. SP3 fusion clusters per-camera detections of one physical robot; this
registry gives each such cluster ONE integer id that is stable across frames,
so the Phase-1 FieldTracker sticky binding turns it into a fixed slot.

A cluster is identified by its set of member keys ``(camera, per_camera_id)``.
Because each camera's id is stable within that camera, two clusters that share
ANY member key are the same physical robot — even across a member joining (a
corner-only robot later seen by main) or dropping out (main occluded). That is
exactly union-find over member keys: each member key remembers its fused id, and
a cluster that touches several known ids collapses them to one.

Pure: no torch / SAM2 / cv2 imports."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from web.embedder import cosine_similarity
from scipy.optimize import linear_sum_assignment


@dataclass
class FusedAssignmentEvidence:
    track_id: int
    type: str # "continued" | "appearance_relinked" | "merged" | "newly_created"
    margin: float


class FusedTrackRegistry:
    def __init__(self, app_floor: float = 0.55, ema_alpha: float = 0.3) -> None:
        # member key (camera, per_camera_id) -> fused id
        self._key_to_id: dict[tuple, int] = {}
        self._next_id = 0
        self.app_floor = app_floor
        self.ema_alpha = ema_alpha
        # running embedding per fused track
        self._id_to_embedding: dict[int, np.ndarray] = {}
        # initial embedding when track was created (anchor)
        self._id_to_anchor: dict[int, np.ndarray] = {}
        # last N frame embeddings to prevent EMA drift
        self._recent_exemplars: dict[int, list[np.ndarray]] = {}

    def get_embedding(self, track_id: int) -> "np.ndarray | None":
        """Running (EMA) appearance embedding for a fused track, or None if the
        track has no embedding yet. Used to mirror embeddings onto per-slot
        RobotTracks for the appearance gallery."""
        return self._id_to_embedding.get(track_id)

    def assign(self, observations: list) -> list[FusedAssignmentEvidence]:
        """Stamp a stable ``track_id`` onto each FusedObservation (and its
        ``detection``), mutating in place. Call once per frame, in order.
        Returns a list of FusedAssignmentEvidence corresponding to each observation."""
        active_ids_this_frame = set()
        evidence_list = []
        
        # 1. First pass: Handle observations that have known member keys.
        # We assign them their known fused ID, handle any collapses, and mark the ID as active.
        direct_matches = [] # list of (obs, fused_id, is_merge)
        unmatched_obs = []  # list of obs that don't match any known keys
        
        for obs in observations:
            keys = [tuple(m) for m in obs.members]
            known = {self._key_to_id[k] for k in keys if k in self._key_to_id}
            if known:
                fused_id = min(known)
                is_merge = len(known) > 1
                # Collapse any bridged ids: every key currently pointing at one of
                # the `known` ids is repointed to the winner.
                if is_merge:
                    losers = known - {fused_id}
                    for k, v in list(self._key_to_id.items()):
                        if v in losers:
                            self._key_to_id[k] = fused_id
                    # Clean up collapsed embeddings and merge their appearance state
                    for loser in losers:
                        if loser in self._id_to_embedding:
                            w_emb = self._id_to_embedding[fused_id]
                            l_emb = self._id_to_embedding[loser]
                            blended = 0.5 * w_emb + 0.5 * l_emb
                            norm = np.linalg.norm(blended)
                            self._id_to_embedding[fused_id] = blended / norm if norm > 0 else blended
                            self._id_to_embedding.pop(loser, None)
                        if loser in self._id_to_anchor:
                            self._id_to_anchor.pop(loser, None)
                        if loser in self._recent_exemplars:
                            self._recent_exemplars.setdefault(fused_id, []).extend(self._recent_exemplars[loser])
                            self._recent_exemplars.pop(loser, None)
                    
                    if fused_id in self._recent_exemplars and len(self._recent_exemplars[fused_id]) > 5:
                        self._recent_exemplars[fused_id] = self._recent_exemplars[fused_id][-5:]
                
                direct_matches.append((obs, fused_id, is_merge))
                active_ids_this_frame.add(fused_id)
            else:
                unmatched_obs.append(obs)

        # Apply assignments from direct matches
        for obs, fused_id, is_merge in direct_matches:
            obs.track_id = fused_id
            obs.detection.track_id = fused_id
            obs.relink_margin = 1.0
            obs.detection.relink_margin = 1.0
            
            ev_type = "merged" if is_merge else "continued"
            ev = FusedAssignmentEvidence(track_id=fused_id, type=ev_type, margin=1.0)
            obs.relink_evidence = ev
            obs.detection.relink_evidence = ev
            evidence_list.append(ev)
            
            # Update running embedding with EMA
            if obs.detection.embedding is not None:
                if self._id_to_embedding.get(fused_id) is None:
                    self._id_to_embedding[fused_id] = obs.detection.embedding
                    self._id_to_anchor[fused_id] = obs.detection.embedding
                    self._recent_exemplars[fused_id] = [obs.detection.embedding]
                else:
                    old_emb = self._id_to_embedding[fused_id]
                    new_emb = obs.detection.embedding
                    blended = (1.0 - self.ema_alpha) * old_emb + self.ema_alpha * new_emb
                    norm = np.linalg.norm(blended)
                    self._id_to_embedding[fused_id] = blended / norm if norm > 0 else blended
                    
                    exemplars = self._recent_exemplars.setdefault(fused_id, [])
                    exemplars.append(obs.detection.embedding)
                    if len(exemplars) > 5:
                        exemplars.pop(0)

        # 2. Second pass: Handle unmatched observations and match them globally against inactive tracks.
        # Only observations with embeddings can be matched using appearance.
        unmatched_with_emb = [obs for obs in unmatched_obs if obs.detection.embedding is not None]
        unmatched_no_emb = [obs for obs in unmatched_obs if obs.detection.embedding is None]
        
        inactive_ids = sorted(list(set(self._id_to_embedding.keys()) - active_ids_this_frame))
        
        relinked_matches = {} # index in unmatched_with_emb -> (inactive_id, similarity, margin)
        
        if unmatched_with_emb and inactive_ids:
            num_rows = len(unmatched_with_emb)
            num_cols = len(inactive_ids)
            cost_matrix = np.zeros((num_rows, num_cols))
            all_sims = np.zeros((num_rows, num_cols))
            
            for r in range(num_rows):
                obs = unmatched_with_emb[r]
                for c in range(num_cols):
                    inactive_id = inactive_ids[c]
                    # Compute max similarity over EMA, anchor, and recent exemplars
                    sim_candidates = []
                    sim_candidates.append(cosine_similarity(obs.detection.embedding, self._id_to_embedding[inactive_id]))
                    if inactive_id in self._id_to_anchor:
                        sim_candidates.append(cosine_similarity(obs.detection.embedding, self._id_to_anchor[inactive_id]))
                    for ex in self._recent_exemplars.get(inactive_id, []):
                        sim_candidates.append(cosine_similarity(obs.detection.embedding, ex))
                    
                    sim = max(sim_candidates)
                    all_sims[r, c] = sim
                    cost_matrix[r, c] = 1.0 - sim
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                sim = all_sims[r, c]
                inactive_id = inactive_ids[c]
                
                # Compute margin
                other_sims = [all_sims[r, j] for j in range(num_cols) if j != c]
                next_best = max(other_sims) if other_sims else -1.0
                margin = sim - next_best
                
                if sim >= self.app_floor:
                    relinked_matches[r] = (inactive_id, sim, margin)

        # Apply assignments from global relinking
        for r, obs in enumerate(unmatched_with_emb):
            if r in relinked_matches:
                fused_id, sim, margin = relinked_matches[r]
                obs.track_id = fused_id
                obs.detection.track_id = fused_id
                obs.relink_margin = margin
                obs.detection.relink_margin = margin
                active_ids_this_frame.add(fused_id)
                
                ev = FusedAssignmentEvidence(track_id=fused_id, type="appearance_relinked", margin=margin)
                obs.relink_evidence = ev
                obs.detection.relink_evidence = ev
                evidence_list.append(ev)
                
                # Quarantine uncertain updates
                if margin >= 0.05:
                    old_emb = self._id_to_embedding[fused_id]
                    new_emb = obs.detection.embedding
                    blended = (1.0 - self.ema_alpha) * old_emb + self.ema_alpha * new_emb
                    norm = np.linalg.norm(blended)
                    self._id_to_embedding[fused_id] = blended / norm if norm > 0 else blended
                    
                    exemplars = self._recent_exemplars.setdefault(fused_id, [])
                    exemplars.append(obs.detection.embedding)
                    if len(exemplars) > 5:
                        exemplars.pop(0)
            else:
                # Brand new robot
                fused_id = self._next_id
                self._next_id += 1
                obs.track_id = fused_id
                obs.detection.track_id = fused_id
                obs.relink_margin = 1.0
                obs.detection.relink_margin = 1.0
                active_ids_this_frame.add(fused_id)
                
                ev = FusedAssignmentEvidence(track_id=fused_id, type="newly_created", margin=1.0)
                obs.relink_evidence = ev
                obs.detection.relink_evidence = ev
                evidence_list.append(ev)
                
                self._id_to_embedding[fused_id] = obs.detection.embedding
                self._id_to_anchor[fused_id] = obs.detection.embedding
                self._recent_exemplars[fused_id] = [obs.detection.embedding]
                
        # For unmatched observations with no embeddings, they get fresh IDs
        for obs in unmatched_no_emb:
            fused_id = self._next_id
            self._next_id += 1
            obs.track_id = fused_id
            obs.detection.track_id = fused_id
            obs.relink_margin = 1.0
            obs.detection.relink_margin = 1.0
            active_ids_this_frame.add(fused_id)
            
            ev = FusedAssignmentEvidence(track_id=fused_id, type="newly_created", margin=1.0)
            obs.relink_evidence = ev
            obs.detection.relink_evidence = ev
            evidence_list.append(ev)

        # Register every member key of this cluster to the chosen id.
        for obs in observations:
            keys = [tuple(m) for m in obs.members]
            for k in keys:
                self._key_to_id[k] = obs.track_id
                
        return evidence_list
