"""Stable cross-camera fused track ids for the mask-tracker tracklet path.

The SAM mask tracker emits per-camera object ids that collide across cameras and restart per
prompt. SP3 fusion clusters per-camera detections of one physical robot; this
registry gives each such cluster ONE integer id that is stable across frames,
so the Phase-1 FieldTracker sticky binding turns it into a fixed slot.

A cluster is identified by its set of member keys ``(camera, per_camera_id)``.
Because each camera's id is stable within that camera, two clusters that share
ANY member key are the same physical robot — even across a member joining (a
corner-only robot later seen by main) or dropping out (main occluded). That is
exactly union-find over member keys: each member key remembers its fused id, and
a cluster that touches several known ids collapses them to one.

Appearance relink (``fused_per_cam_relink``, job knob, default OFF): fusion
averages every camera's crop embedding into ``FieldDetection.embedding``, but
one robot looks very different from main vs a broadcast corner (S4 main-vs-
corner same-robot median cosine 0.62, within one camera 0.92-0.93). The
registry therefore always keeps per-camera EMA / anchor / exemplar stores per
fused id (fed by ``FieldDetection.cam_embeddings``); with the knob ON the
relink similarity is the max over cameras the observation and the inactive id
share, and a pair sharing no camera is not a relink candidate. Either side
lacking per-camera data falls back to the averaged comparison. The legacy
averaged stores always update (``get_embedding`` consumers read them).

``stats`` (Counter, always on): per-observation outcome — continued, merged,
relinked, created_with_emb, created_no_emb — plus two diagnostics on
created_with_emb: created_crossview_only (some spatially plausible inactive id
shared no camera with it) and created_samecam_below_floor /
created_samecam_above_floor (best same-camera cosine vs a plausible inactive
id, below / at-or-above app_floor).

Far-key split (``split_far_m`` > 0, job knob ``fused_split_far``, default
OFF): union-find over member keys assumes a ``(camera, sam_id)`` key stays on
one robot. In full-SAM3 mode a corner SAM object lives all match and can end
up on a DIFFERENT robot (SAM object switch, or a wrong cross-view pairing);
its key then still maps to the first robot's id, and merge_same_id_observations
folded the second robot into the first (2025hop qm2: 103 lost GT
robot-moments, median 4.8 m apart). With the knob on, before ids are resolved,
every fused id claimed this frame by >= 2 clusters keeps one owner — the
claimant nearest the id's last-known position (fallback: main-backed, then most
keys bound to the id, then first) — and any other claimant farther than
``split_far_m`` from the owner has its keys for that id UNBOUND, so it
continues via its other keys, relinks, or gets a fresh id (and never collapses
the two ids together). ``split_off`` is the merge-time safety net for residual
same-id groups. Stats: key_unbound_far (keys), split_far (observations split
off at merge); ``last_frame_split`` holds this frame's counts.

Evidence order (``evidence_input_order``, job knob
``fused_evidence_input_order``, DEFAULT ON since Nov 15 2026; the class
default stays legacy False for direct construction): ``assign`` resolves observations
pass by pass (direct key matches, then appearance relinks / new-with-embedding,
then new-without-embedding) and legacy-returns the evidence list in that PASS
order, while its callers (merge_same_id_observations,
InferenceJob._sync_identity_registry_frame) index it by input position — so on
any frame where a new / relinked cluster precedes a continued one, evidence is
paired with the wrong fused observation. ON returns it in input order (each
entry is that observation's own ``relink_evidence``). Ids and registry state
are identical either way; only the returned list's order changes. Stats
evidence_misaligned / evidence_misaligned_type (always counted) give the
number of positions the legacy order mispairs / mispairs with a different
evidence type or margin.

Pure: no torch / SAM / cv2 imports."""
from __future__ import annotations
import numpy as np
from collections import Counter
from dataclasses import dataclass
from web.embedder import cosine_similarity
from scipy.optimize import linear_sum_assignment


@dataclass
class FusedAssignmentEvidence:
    track_id: int
    type: str # "continued" | "appearance_relinked" | "merged" | "newly_created"
    margin: float


class FusedTrackRegistry:
    def __init__(self, app_floor: float = 0.55, ema_alpha: float = 0.3,
                 relink_max_dist_m: float = 0.0, per_cam_relink: bool = False,
                 split_far_m: float = 0.0,
                 evidence_input_order: bool = False) -> None:
        # member key (camera, per_camera_id) -> fused id
        self._key_to_id: dict[tuple, int] = {}
        self._next_id = 0
        self.app_floor = app_floor
        self.ema_alpha = ema_alpha
        # Appearance-relink spatial gate (meters); 0 = off. Blocks relinking an
        # observation onto an inactive id whose last position is farther than
        # this — DINOv2 can't distinguish same-alliance robots, so without it one
        # red robot relinks onto another's id and distinct robots collapse.
        self.relink_max_dist_m = float(relink_max_dist_m)
        # last-known field position per fused id, for the spatial gate.
        self._id_to_pos: dict[int, tuple] = {}
        # running embedding per fused track
        self._id_to_embedding: dict[int, np.ndarray] = {}
        # initial embedding when track was created (anchor)
        self._id_to_anchor: dict[int, np.ndarray] = {}
        # last N frame embeddings to prevent EMA drift
        self._recent_exemplars: dict[int, list[np.ndarray]] = {}
        # Per-camera mirror of the three stores above (fused id -> camera ->
        # EMA / first-seen anchor / last-5 exemplars). Always kept (the
        # created_* diagnostics read them); relink reads them only when
        # per_cam_relink is on. Keys of _cam_appearance[id] = the cameras whose
        # embeddings the id has seen.
        self.per_cam_relink = bool(per_cam_relink)
        self._cam_appearance: dict[int, dict] = {}
        self._cam_anchor: dict[int, dict] = {}
        self._cam_exemplars: dict[int, dict] = {}
        self.stats: Counter = Counter()
        # Far-key split (module docstring); 0 = off.
        self.split_far_m = float(split_far_m)
        # This frame's far-split counts (reset by assign): keys unbound in the
        # pre-pass, observations split off by split_off (merge safety net).
        self.last_frame_split = {"unbound": 0, "split": 0}
        # Evidence order (module docstring): True = assign returns evidence
        # index-aligned to its input; False = legacy pass order.
        self.evidence_input_order = bool(evidence_input_order)

    def _claim_owner(self, fused_id: int, idxs: list, observations: list) -> int:
        """Owner among the observations claiming ``fused_id`` this frame: the
        one nearest the id's last-known position; without one, a main-backed
        claimant, then the one with the most keys bound to the id, then the
        first."""
        last = self._id_to_pos.get(fused_id)
        if last is not None:
            return min(idxs, key=lambda i: (
                float(np.hypot(observations[i].detection.pos[0] - last[0],
                               observations[i].detection.pos[1] - last[1])), i))

        def n_keys(i):
            return sum(1 for m in observations[i].members
                       if self._key_to_id.get(tuple(m)) == fused_id)
        return min(idxs, key=lambda i: (
            observations[i].detection.box is None, -n_keys(i), i))

    def _unbind_far_keys(self, observations: list) -> int:
        """Far-key split pre-pass (see module docstring). Returns the number of
        member keys unbound."""
        claims: dict[int, list] = {}
        for i, obs in enumerate(observations):
            ids = {self._key_to_id[k] for k in (tuple(m) for m in obs.members)
                   if k in self._key_to_id}
            for fid in sorted(ids):
                claims.setdefault(fid, []).append(i)
        unbound = 0
        for fid, idxs in claims.items():
            if len(idxs) < 2:
                continue
            owner = self._claim_owner(fid, idxs, observations)
            ox, oy = observations[owner].detection.pos
            for i in idxs:
                if i == owner:
                    continue
                px, py = observations[i].detection.pos
                if float(np.hypot(px - ox, py - oy)) <= self.split_far_m:
                    continue
                for m in observations[i].members:
                    k = tuple(m)
                    if self._key_to_id.get(k) == fid:
                        del self._key_to_id[k]
                        unbound += 1
        return unbound

    def split_off(self, observations: list, kept_obs=None) -> "FusedAssignmentEvidence":
        """Merge-time far split (merge_same_id_observations ``on_split``): the
        given same-id observations sit farther than ``split_far_m`` from the
        id's retained member, so they are another robot. Stamp ONE fresh fused
        id on all of them, repoint their member keys to it (so the key
        continues the new id next frame instead of re-claiming the old one),
        seed its appearance stores, and restore the old id's last position to
        the retained member's. Returns the new id's assignment evidence."""
        fused_id = self._next_id
        self._next_id += 1
        ev = FusedAssignmentEvidence(track_id=fused_id, type="newly_created",
                                     margin=1.0)
        for obs in observations:
            obs.track_id = fused_id
            obs.detection.track_id = fused_id
            obs.relink_margin = 1.0
            obs.detection.relink_margin = 1.0
            obs.relink_evidence = ev
            obs.detection.relink_evidence = ev
            for m in obs.members:
                self._key_to_id[tuple(m)] = fused_id
            emb = obs.detection.embedding
            if emb is not None and self._id_to_embedding.get(fused_id) is None:
                self._id_to_embedding[fused_id] = emb
                self._id_to_anchor[fused_id] = emb
                self._recent_exemplars[fused_id] = [emb]
            self._update_cam_stores(fused_id, obs, False)
            self.stats["split_far"] += 1
            self.last_frame_split["split"] += 1
        ref = next((o for o in observations if o.detection.box is not None),
                   observations[0])
        self._id_to_pos[fused_id] = (float(ref.detection.pos[0]),
                                     float(ref.detection.pos[1]))
        if kept_obs is not None and kept_obs.track_id is not None:
            self._id_to_pos[kept_obs.track_id] = (
                float(kept_obs.detection.pos[0]), float(kept_obs.detection.pos[1]))
        return ev

    def _update_cam_stores(self, fused_id: int, obs, update_ok: bool) -> None:
        """Per-camera mirror of the legacy appearance update (same shape as
        IdentityRegistry._update_cam_appearance): a camera seen for the first
        time seeds its EMA / anchor / exemplars ungated; a known camera takes
        the EMA + exemplar update only when ``update_ok``. Anchor never moves."""
        cam_embs = getattr(obs.detection, "cam_embeddings", None)
        if not cam_embs:
            return
        ema = self._cam_appearance.setdefault(fused_id, {})
        anchor = self._cam_anchor.setdefault(fused_id, {})
        exemplars = self._cam_exemplars.setdefault(fused_id, {})
        for cam, e in cam_embs.items():
            if e is None:
                continue
            if cam not in ema:
                ema[cam] = e
                anchor[cam] = e
                exemplars[cam] = [e]
                continue
            if not update_ok:
                continue
            blended = (1.0 - self.ema_alpha) * ema[cam] + self.ema_alpha * e
            norm = np.linalg.norm(blended)
            ema[cam] = blended / norm if norm > 0 else blended
            fifo = exemplars.setdefault(cam, [])
            fifo.append(e)
            if len(fifo) > 5:
                fifo.pop(0)

    def _merge_cam_stores(self, winner: int, loser: int) -> None:
        """Fold a collapsed id's per-camera stores into the winner, per camera
        exactly as the legacy merge does (adopt when the winner lacks the
        camera, else 50/50 EMA blend; winner's anchor kept; exemplars
        concatenated, last 5 kept)."""
        l_ema = self._cam_appearance.pop(loser, None) or {}
        l_anchor = self._cam_anchor.pop(loser, None) or {}
        l_ex = self._cam_exemplars.pop(loser, None) or {}
        if not (l_ema or l_anchor or l_ex):
            return
        w_ema = self._cam_appearance.setdefault(winner, {})
        w_anchor = self._cam_anchor.setdefault(winner, {})
        w_ex = self._cam_exemplars.setdefault(winner, {})
        for cam, e in l_ema.items():
            if cam not in w_ema:
                w_ema[cam] = e
            else:
                blended = 0.5 * w_ema[cam] + 0.5 * e
                norm = np.linalg.norm(blended)
                w_ema[cam] = blended / norm if norm > 0 else blended
        for cam, e in l_anchor.items():
            w_anchor.setdefault(cam, e)
        for cam, fifo in l_ex.items():
            merged = w_ex.setdefault(cam, [])
            merged.extend(fifo)
            if len(merged) > 5:
                w_ex[cam] = merged[-5:]

    def _samecam_similarity(self, obs, fused_id: int):
        """(shared, sim): ``shared`` is None when either side has no per-camera
        data (unclassifiable -> legacy); else whether any camera is shared, and
        ``sim`` the max cosine over shared cameras vs that camera's EMA, anchor
        and exemplars (None when nothing is shared)."""
        cam_embs = getattr(obs.detection, "cam_embeddings", None)
        ema = self._cam_appearance.get(fused_id)
        if not cam_embs or not ema:
            return None, None
        best = None
        for cam, e in cam_embs.items():
            if e is None or cam not in ema:
                continue
            cands = [ema[cam]]
            a = self._cam_anchor.get(fused_id, {}).get(cam)
            if a is not None:
                cands.append(a)
            cands.extend(self._cam_exemplars.get(fused_id, {}).get(cam, []))
            for ref in cands:
                c = cosine_similarity(e, ref)
                best = c if best is None else max(best, c)
        return best is not None, best

    def get_embedding(self, track_id: int) -> "np.ndarray | None":
        """Running (EMA) appearance embedding for a fused track, or None if the
        track has no embedding yet. Used to mirror embeddings onto per-slot
        RobotTracks for the appearance gallery."""
        return self._id_to_embedding.get(track_id)

    def assign(self, observations: list) -> list[FusedAssignmentEvidence]:
        """Stamp a stable ``track_id`` onto each FusedObservation (and its
        ``detection``), mutating in place. Call once per frame, in order.
        Returns a list of FusedAssignmentEvidence: index-aligned to
        ``observations`` when ``evidence_input_order`` is on, else (legacy) in
        pass order — direct matches, relinked / new-with-embedding,
        new-without-embedding (module docstring)."""
        active_ids_this_frame = set()
        evidence_list = []
        if self.split_far_m > 0:
            self.last_frame_split = {"unbound": 0, "split": 0}
            n = self._unbind_far_keys(observations)
            if n:
                self.stats["key_unbound_far"] += n
                self.last_frame_split["unbound"] = n
        
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
                            # Winner may have been created from an observation
                            # without an embedding; adopt the loser's instead
                            # of blending with nothing.
                            w_emb = self._id_to_embedding.get(fused_id)
                            l_emb = self._id_to_embedding.pop(loser)
                            if w_emb is None:
                                self._id_to_embedding[fused_id] = l_emb
                            else:
                                blended = 0.5 * w_emb + 0.5 * l_emb
                                norm = np.linalg.norm(blended)
                                self._id_to_embedding[fused_id] = blended / norm if norm > 0 else blended
                        if loser in self._id_to_anchor:
                            if fused_id not in self._id_to_anchor:
                                self._id_to_anchor[fused_id] = self._id_to_anchor[loser]
                            self._id_to_anchor.pop(loser, None)
                        if loser in self._recent_exemplars:
                            self._recent_exemplars.setdefault(fused_id, []).extend(self._recent_exemplars[loser])
                            self._recent_exemplars.pop(loser, None)
                        self._merge_cam_stores(fused_id, loser)
                    
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
            self.stats[ev_type] += 1
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
            # Continued/merged: per-camera stores update ungated, like the
            # legacy EMA above.
            self._update_cam_stores(fused_id, obs, True)

        # 2. Second pass: Handle unmatched observations and match them globally against inactive tracks.
        # Only observations with embeddings can be matched using appearance.
        unmatched_with_emb = [obs for obs in unmatched_obs if obs.detection.embedding is not None]
        unmatched_no_emb = [obs for obs in unmatched_obs if obs.detection.embedding is None]
        
        inactive_ids = sorted(list(set(self._id_to_embedding.keys()) - active_ids_this_frame))
        
        relinked_matches = {} # index in unmatched_with_emb -> (inactive_id, similarity, margin)
        # Per row: created_* diagnostics over spatially plausible inactive ids
        # ({"crossview": bool, "samecam": best same-camera cosine or None}).
        row_diag: dict[int, dict] = {}
        
        if unmatched_with_emb and inactive_ids:
            num_rows = len(unmatched_with_emb)
            num_cols = len(inactive_ids)
            cost_matrix = np.zeros((num_rows, num_cols))
            all_sims = np.zeros((num_rows, num_cols))
            
            for r in range(num_rows):
                obs = unmatched_with_emb[r]
                diag = row_diag.setdefault(r, {"crossview": False, "samecam": None})
                for c in range(num_cols):
                    inactive_id = inactive_ids[c]
                    shared, cam_sim = self._samecam_similarity(obs, inactive_id)
                    gate_ok = True
                    if self.relink_max_dist_m > 0 and inactive_id in self._id_to_pos:
                        px, py = self._id_to_pos[inactive_id]
                        d = float(np.hypot(obs.detection.pos[0] - px,
                                           obs.detection.pos[1] - py))
                        gate_ok = d <= self.relink_max_dist_m
                    if gate_ok and shared is False:
                        diag["crossview"] = True
                    if gate_ok and cam_sim is not None:
                        diag["samecam"] = (cam_sim if diag["samecam"] is None
                                           else max(diag["samecam"], cam_sim))
                    if self.per_cam_relink and shared is not None:
                        # Same-camera appearance only; no shared camera = no
                        # appearance evidence = not a relink candidate.
                        sim = cam_sim if shared else -1.0
                        if not gate_ok:
                            sim = -1.0
                        all_sims[r, c] = sim
                        cost_matrix[r, c] = 1.0 - sim
                        continue
                    # Compute max similarity over EMA, anchor, and recent exemplars
                    sim_candidates = []
                    sim_candidates.append(cosine_similarity(obs.detection.embedding, self._id_to_embedding[inactive_id]))
                    if inactive_id in self._id_to_anchor:
                        sim_candidates.append(cosine_similarity(obs.detection.embedding, self._id_to_anchor[inactive_id]))
                    for ex in self._recent_exemplars.get(inactive_id, []):
                        sim_candidates.append(cosine_similarity(obs.detection.embedding, ex))
                    
                    sim = max(sim_candidates)
                    # Spatial gate: a robot can't teleport across the field, so
                    # block relinking to an inactive id whose last position is
                    # implausibly far — stops same-appearance distinct robots
                    # (same-alliance) from collapsing onto one id.
                    if self.relink_max_dist_m > 0 and inactive_id in self._id_to_pos:
                        px, py = self._id_to_pos[inactive_id]
                        d = float(np.hypot(obs.detection.pos[0] - px,
                                           obs.detection.pos[1] - py))
                        if d > self.relink_max_dist_m:
                            sim = -1.0
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
                self.stats["relinked"] += 1
                # Same quarantine for the per-camera stores (new cameras seed).
                self._update_cam_stores(fused_id, obs, margin >= 0.05)
                
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
                self._update_cam_stores(fused_id, obs, False)

                self.stats["created_with_emb"] += 1
                diag = row_diag.get(r)
                if diag is not None:
                    if diag["crossview"]:
                        self.stats["created_crossview_only"] += 1
                    if diag["samecam"] is not None:
                        key = ("created_samecam_below_floor"
                               if diag["samecam"] < self.app_floor
                               else "created_samecam_above_floor")
                        self.stats[key] += 1
                
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
            self.stats["created_no_emb"] += 1

        # Register every member key of this cluster to the chosen id, and record
        # each id's latest field position for the appearance-relink spatial gate.
        for obs in observations:
            keys = [tuple(m) for m in obs.members]
            for k in keys:
                self._key_to_id[k] = obs.track_id
            if obs.track_id is not None and obs.detection.pos is not None:
                self._id_to_pos[obs.track_id] = (float(obs.detection.pos[0]),
                                                 float(obs.detection.pos[1]))

        # Diagnostic (always on): positions where the legacy pass-order list
        # pairs evidence with a different observation, and how many of those
        # hand over a different type / margin (what the identity sync reads).
        for obs, ev in zip(observations, evidence_list):
            own = obs.relink_evidence
            if ev is not own:
                self.stats["evidence_misaligned"] += 1
                if (ev.type, ev.margin) != (own.type, own.margin):
                    self.stats["evidence_misaligned_type"] += 1
        if self.evidence_input_order:
            # Every pass stamped obs.relink_evidence; read them back in input
            # order so callers can pair evidence[i] with observations[i].
            return [obs.relink_evidence for obs in observations]
        return evidence_list
