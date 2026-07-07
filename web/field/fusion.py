"""Camera fusion: raw field detections -> one observation per physical robot.

Greedy single-link clustering by field distance. Inputs are per-camera lists of
(FieldDetection, bumper_crop); a cluster of detections within radius_m is one
physical robot. The merged FieldDetection (weighted-average position, weighted
color vote) is what the tracker consumes; the FusedObservation also carries the
raw evidence so the identity step can read each camera's bumper crop with its
source weight and that camera's own projected field position.

SAM2 can occasionally propagate two object masks onto the same robot. Same-camera
detections are therefore also merged when their image boxes overlap strongly.
Requiring both field proximity and image overlap avoids collapsing merely-adjacent
robots.

Cross-camera positions disagree anisotropically: a low oblique camera keeps
accurate bearing but its ground projection slides AWAY from the camera whenever
the visible mask bottom sits above true ground contact (occlusion, crop clipping,
range). Measured on 2026hop qm24, same-robot main/corner pairs were 1.1-3.8 m
apart with only 0.1-0.3 m of cross-ray error — far outside any safe isotropic
radius. When ``camera_centers`` is provided, a second pass therefore merges
clusters whose detections agree in bearing (cross-ray gate) with a
range-proportional along-ray allowance. Two robots on the same ray cannot both
be visible to that camera, and clusters sharing any camera never merge, so the
along-ray slack cannot chain two real robots together.

Corner cameras sit near carpet level, so their range estimates are treated as
nearly meaningless: the along-ray allowance is deliberately generous (60% of
range, capped at 6 m). The cross-ray tolerance (default 1.0 m) also covers
more than pixel noise: a close robot clipped by the crop edge or showing only
its near face shifts the projected bottom-center laterally by up to half a
robot, on top of the main camera's own ~0.3 m error (all measured on 2026hop
qm24 with a verified-good calibration). Cross-pairing two real robots through
that slack is instead blocked by the per-frame color veto: detections with
known, contradicting bumper colors never merge. Corner cameras are
identity/bearing evidence first and geometry second — a merged cluster's
position always comes from main when main saw the robot."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
from web.field.tracker import FieldDetection


@dataclass
class FusedObservation:
    detection: FieldDetection                 # what the tracker sees
    evidence: list                            # [(source_camera, bumper_crop, weight, cam_pos), ...]
    members: list = field(default_factory=list)   # [(camera, per_camera_track_id), ...]
    track_id: int | None = None               # stable fused id (assigned by FusedTrackRegistry)


def _box_ios(box_a, box_b) -> float:
    """Intersection over smaller box area, or zero for missing/empty boxes."""
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    if smaller <= 0:
        return 0.0
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    return (iw * ih) / smaller


def _ray_gate_score(pos_ref, center_ref, pos_other, cross_m, along_frac,
                    along_max_m, near_slack_m):
    """Goodness (lower = better) of ``pos_other`` explaining the same robot as
    ``pos_ref`` seen from the camera at ``center_ref``, or None if rejected.

    Decomposes the residual into bearing (cross-ray) and range (along-ray)
    components. Projection error only pushes ``pos_ref`` away from its camera,
    so ``pos_other`` farther out than ``pos_ref`` (beyond ``near_slack_m``
    noise) is rejected outright."""
    c = np.array(center_ref, float)[:2]
    p = np.array(pos_ref, float)
    o = np.array(pos_other, float)
    ray = p - c
    rng = float(np.linalg.norm(ray))
    if rng < 1e-6:
        return None
    u = ray / rng
    rel = o - c
    along_err = rng - float(rel @ u)          # > 0: other is nearer the camera
    cross = abs(float(rel[0] * u[1] - rel[1] * u[0]))
    along_tol = min(along_max_m, along_frac * rng)
    if cross > cross_m or along_err > along_tol or along_err < -near_slack_m:
        return None
    return (cross / cross_m) ** 2 + (along_err / along_tol) ** 2


def _ray_gate_merge_clusters(clusters, camera_centers, cross_m, along_frac,
                             along_max_m, near_slack_m):
    """Merge clusters whose members agree in bearing from either camera.

    Greedy best-score-first with a hard invariant: merged clusters never share
    a camera (one camera sees one robot once), which also guarantees two
    main-camera robots can never collapse through a corner detection."""
    cams = [set(m[0] for m in cl) for cl in clusters]
    candidates = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if cams[i] & cams[j]:
                continue
            best = None
            for cam_a, det_a, _ca, _wa in clusters[i]:
                for cam_b, det_b, _cb, _wb in clusters[j]:
                    # Color veto: the wide geometric slack must never marry
                    # two robots the bumper evidence already distinguishes.
                    col_a, col_b = det_a.color, det_b.color
                    if (col_a and col_b and "unknown" not in (col_a, col_b)
                            and col_a != col_b):
                        continue
                    for (ref_cam, ref_det, other_det) in (
                            (cam_a, det_a, det_b), (cam_b, det_b, det_a)):
                        center = camera_centers.get(ref_cam)
                        if center is None:
                            continue
                        s = _ray_gate_score(
                            ref_det.pos, center, other_det.pos,
                            cross_m, along_frac, along_max_m, near_slack_m)
                        if s is not None and (best is None or s < best):
                            best = s
            if best is not None:
                candidates.append((best, i, j))
    candidates.sort()
    parent = list(range(len(clusters)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _score, i, j in candidates:
        ri, rj = find(i), find(j)
        if ri == rj or (cams[ri] & cams[rj]):
            continue
        parent[rj] = ri
        cams[ri] = cams[ri] | cams[rj]
    merged = {}
    for idx, cl in enumerate(clusters):
        merged.setdefault(find(idx), []).extend(cl)
    return list(merged.values())


def fuse_detections(
    per_camera: dict,
    weights: dict,
    radius_m: float = 0.6,
    same_camera_ios: float = 0.7,
    is_hub_occluded: Optional[Callable[[str, tuple], bool]] = None,
    camera_centers: Optional[dict] = None,
    ray_cross_m: float = 1.0,
    ray_along_frac: float = 0.6,
    ray_along_max_m: float = 6.0,
    ray_near_slack_m: float = 0.75,
) -> list:
    # flatten to (camera, det, crop, weight). The effective weight is the
    # camera's vote weight scaled by the detection's own score (1.0 unless a
    # producer attenuated it, e.g. the corner-camera range falloff) — so far
    # corner evidence supplements without overriding near/main evidence.
    items = []
    for cam, dets in per_camera.items():
        w = float(weights.get(cam, 1.0))
        for (det, crop) in dets:
            items.append((cam, det, crop, w * float(det.score or 1.0)))

    used = [False] * len(items)
    clusters = []
    for i in range(len(items)):
        if used[i]:
            continue
        cam_i, det_i, crop_i, w_i = items[i]
        cluster = [(cam_i, det_i, crop_i, w_i)]
        used[i] = True
        pi = np.array(det_i.pos, float)
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            cam_j, det_j, crop_j, w_j = items[j]
            close_in_field = (
                np.linalg.norm(pi - np.array(det_j.pos, float)) <= radius_m
            )
            if not close_in_field:
                continue
            if cam_j == cam_i and _box_ios(det_i.box, det_j.box) < same_camera_ios:
                continue
            cluster.append((cam_j, det_j, crop_j, w_j))
            used[j] = True
        clusters.append(cluster)
    if camera_centers:
        clusters = _ray_gate_merge_clusters(
            clusters, camera_centers, ray_cross_m, ray_along_frac,
            ray_along_max_m, ray_near_slack_m)
    return [_merge(c, is_hub_occluded=is_hub_occluded) for c in clusters]


def merge_same_id_observations(observations, evidence=None, main_guard=False):
    """Collapse observations that share a fused ``track_id`` into one per id.

    A fused track_id is one physical robot by definition (the registry unifies a
    robot's per-camera detections across frames), so two observations with the
    same id in a single frame are a split — e.g. a corner camera producing a
    second, duplicate mask whose ground projection has slid away from the real
    position. Left alone they reach the tracker as two detections and seed a
    phantom slot (the qm8 B1<->B3 churn).

    Merging preserves the multi-camera cooperation the design intends: the merged
    observation takes its geometry from a main-camera member when one exists (main
    is authoritative; corner range is unreliable) but unions every member and all
    evidence, so a corner camera's bumper read still contributes to the robot's
    identity, and a lone corner observation (main occluded) still carries the
    track. ``evidence`` (the per-observation assignment evidence, index-aligned to
    ``observations``) is collapsed in step, taking the retained member's entry.
    Returns ``(merged_observations, merged_evidence)``; ``merged_evidence`` is None
    when ``evidence`` is None. Observation order is preserved by first appearance."""
    groups: dict = {}
    order: list = []   # ("id", track_id) or ("obs", index) preserving first-seen order
    for i, o in enumerate(observations):
        tid = o.track_id
        if tid is None:                      # never assigned an id — pass through
            order.append(("obs", i))
            continue
        if tid not in groups:
            groups[tid] = []
            order.append(("id", tid))
        groups[tid].append(i)

    merged_obs = []
    merged_ev = [] if evidence is not None else None
    for kind, val in order:
        if kind == "obs":
            merged_obs.append(observations[val])
            if merged_ev is not None:
                merged_ev.append(evidence[val])
            continue
        idxs = groups[val]
        # Retain a main-backed member's geometry/evidence when one exists.
        ref_local = next((k for k, i in enumerate(idxs)
                          if observations[i].detection.box is not None), 0)
        n_main = sum(1 for i in idxs if observations[i].detection.box is not None)
        if len(idxs) == 1 or (main_guard and n_main >= 2):
            # Either nothing to merge, or the same-id group has >=2 independent
            # main-camera detections at different positions — those are DISTINCT
            # robots the registry mis-identified, not a corner duplicate. Pass
            # them through unmerged so the tracker can give each its own slot.
            for i in idxs:
                merged_obs.append(observations[i])
                if merged_ev is not None:
                    merged_ev.append(evidence[i])
        else:
            merged_obs.append(_merge_fused_group(
                [observations[i] for i in idxs], ref_local))
            if merged_ev is not None:
                merged_ev.append(evidence[idxs[ref_local]])
    return merged_obs, merged_ev


def _merge_fused_group(group, ref_local: int) -> FusedObservation:
    """Merge same-id FusedObservations: geometry/box/color from the ref member
    (a main-backed one when present), members + evidence unioned, embeddings
    averaged. Preserves the ref's track_id and relink annotations."""
    ref = group[ref_local]
    members, seen = [], set()
    for o in group:
        for m in o.members:
            key = tuple(m)
            if key not in seen:
                seen.add(key)
                members.append(m)
    evidence = [e for o in group for e in o.evidence]
    embs = [o.detection.embedding for o in group if o.detection.embedding is not None]
    if embs:
        avg = np.mean(embs, axis=0)
        norm = np.linalg.norm(avg)
        emb = avg / norm if norm > 0 else avg
    else:
        emb = None
    rd = ref.detection
    det = FieldDetection(pos=rd.pos, color=rd.color, box=rd.box,
                         score=rd.score, track_id=rd.track_id, embedding=emb,
                         robot_crop_b64=rd.robot_crop_b64,
                         tracker_prob=rd.tracker_prob)
    # Carry the registry's per-detection relink annotations forward if present.
    for attr in ("relink_margin", "relink_evidence"):
        if hasattr(rd, attr):
            setattr(det, attr, getattr(rd, attr))
    merged = FusedObservation(detection=det, evidence=evidence,
                              members=members, track_id=ref.track_id)
    for attr in ("relink_margin", "relink_evidence"):
        if hasattr(ref, attr):
            setattr(merged, attr, getattr(ref, attr))
    return merged


def _merge(cluster, is_hub_occluded=None) -> FusedObservation:
    position_members = cluster
    if is_hub_occluded is not None:
        visible_members = [
            c for c in cluster
            if not is_hub_occluded(c[0], c[1].pos)
        ]
        # If every view is blocked, retain the prior behavior rather than making
        # the observation disappear. The tracker can still coast/reassociate it.
        if visible_members:
            position_members = visible_members

    ws = np.array([c[3] for c in position_members], float)
    pos = np.array([c[1].pos for c in position_members], float)
    # Main is the reference geometry. Corner cameras are deliberately weighted
    # higher for identity evidence, but their low, oblique ground projection is
    # less stable and must not drag a valid main-camera position across the field.
    # Hub-occluded views are removed from position_members when another camera
    # has line of sight, so corner geometry becomes authoritative behind the hub.
    main_position_members = [c for c in position_members if c[0] == "main"]
    if main_position_members:
        main_ws = np.array([c[3] for c in main_position_members], float)
        main_pos = np.array([c[1].pos for c in main_position_members], float)
        avg = (main_pos * main_ws[:, None]).sum(axis=0) / main_ws.sum()
    else:
        avg = (pos * ws[:, None]).sum(axis=0) / ws.sum()
    main_members = [c for c in cluster if c[0] == "main"]
    # weighted color vote, ignoring "unknown"
    votes = {}
    for cam, det, crop, w in cluster:
        if det.color and det.color != "unknown":
            votes[det.color] = votes.get(det.color, 0.0) + w
    color = max(votes, key=votes.get) if votes else "unknown"
    # Boxes are camera-local pixels. Only a main-camera box may enter the field
    # tracker because its output is drawn/cropped in main-view coordinates.
    # Corner-only observations remain visible in their own HUD panels and radar.
    box = main_members[0][1].box if main_members else None
    # tracker_prob is PAYLOAD ONLY (path-sample quality). Carry the value from the
    # max-weight member — the detection that most influenced the fused position —
    # ignoring members with no prob. Never feeds fusion weights or the tracker.
    prob_members = [(w, det.tracker_prob) for (cam, det, _crop, w) in cluster
                    if det.tracker_prob is not None]
    tracker_prob = max(prob_members)[1] if prob_members else None
    evidence = [(cam, crop, w, (float(det.pos[0]), float(det.pos[1])))
                for (cam, det, crop, w) in cluster]
    # Keep one deterministic member key per camera. Same-camera duplicates are a
    # per-frame suppression signal, not proof that two SAM2 IDs should be aliased
    # forever: two real robots can briefly overlap during contact.
    members = []
    member_cameras = set()
    for cam, det, _crop, _w in cluster:
        if cam in member_cameras:
            continue
        members.append((cam, det.track_id))
        member_cameras.add(cam)

    # Weighted average and L2-normalize crop embeddings
    embs = [c[1].embedding for c in cluster if c[1].embedding is not None]
    emb_ws = [c[3] for c in cluster if c[1].embedding is not None]
    if embs:
        avg_emb = np.average(embs, axis=0, weights=emb_ws)
        norm = np.linalg.norm(avg_emb)
        merged_embedding = avg_emb / norm if norm > 0 else avg_emb
    else:
        merged_embedding = None

    return FusedObservation(
        detection=FieldDetection(pos=(float(avg[0]), float(avg[1])), color=color, box=box, embedding=merged_embedding, tracker_prob=tracker_prob),
        evidence=evidence,
        members=members,
    )
