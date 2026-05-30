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
from web.embedder import cosine_similarity


class FusedTrackRegistry:
    def __init__(self, app_floor: float = 0.55, ema_alpha: float = 0.3) -> None:
        # member key (camera, per_camera_id) -> fused id
        self._key_to_id: dict[tuple, int] = {}
        self._next_id = 0
        self.app_floor = app_floor
        self.ema_alpha = ema_alpha
        # running embedding per fused track
        self._id_to_embedding: dict[int, np.ndarray] = {}

    def assign(self, observations: list) -> None:
        """Stamp a stable ``track_id`` onto each FusedObservation (and its
        ``detection``), mutating in place. Call once per frame, in order."""
        active_ids_this_frame = set()
        for obs in observations:
            keys = [tuple(m) for m in obs.members]
            # Fused ids already known for any member key in this cluster.
            known = {self._key_to_id[k] for k in keys if k in self._key_to_id}
            if known:
                fused_id = min(known)
                # Collapse any bridged ids: every key currently pointing at one of
                # the `known` ids is repointed to the winner.
                if len(known) > 1:
                    losers = known - {fused_id}
                    for k, v in list(self._key_to_id.items()):
                        if v in losers:
                            self._key_to_id[k] = fused_id
                    # Clean up collapsed embeddings
                    for loser in losers:
                        self._id_to_embedding.pop(loser, None)
                
                # Update running embedding with EMA
                if obs.detection.embedding is not None:
                    if self._id_to_embedding.get(fused_id) is None:
                        self._id_to_embedding[fused_id] = obs.detection.embedding
                    else:
                        old_emb = self._id_to_embedding[fused_id]
                        new_emb = obs.detection.embedding
                        blended = (1.0 - self.ema_alpha) * old_emb + self.ema_alpha * new_emb
                        norm = np.linalg.norm(blended)
                        self._id_to_embedding[fused_id] = blended / norm if norm > 0 else blended
            else:
                # Try to re-link using embedding similarity against inactive tracks
                fused_id = None
                if obs.detection.embedding is not None and self._id_to_embedding:
                    # Inactive tracks: previously seen tracks that are NOT active in this frame yet
                    inactive_ids = set(self._id_to_embedding.keys()) - active_ids_this_frame
                    best_id, best_sim = None, -1.0
                    for inactive_id in inactive_ids:
                        sim = cosine_similarity(obs.detection.embedding, self._id_to_embedding[inactive_id])
                        if sim > best_sim:
                            best_id, best_sim = inactive_id, sim
                    if best_sim >= self.app_floor:
                        fused_id = best_id

                if fused_id is not None:
                    # Re-linked! Update running embedding with EMA
                    old_emb = self._id_to_embedding[fused_id]
                    new_emb = obs.detection.embedding
                    blended = (1.0 - self.ema_alpha) * old_emb + self.ema_alpha * new_emb
                    norm = np.linalg.norm(blended)
                    self._id_to_embedding[fused_id] = blended / norm if norm > 0 else blended
                else:
                    # Brand new robot
                    fused_id = self._next_id
                    self._next_id += 1
                    if obs.detection.embedding is not None:
                        self._id_to_embedding[fused_id] = obs.detection.embedding
            
            active_ids_this_frame.add(fused_id)
            # Register every member key of this cluster to the chosen id.
            for k in keys:
                self._key_to_id[k] = fused_id
            obs.track_id = fused_id
            obs.detection.track_id = fused_id
