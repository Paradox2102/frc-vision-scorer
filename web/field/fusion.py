"""Camera fusion: raw field detections -> one observation per physical robot.

Greedy single-link clustering by field distance. Inputs are per-camera lists of
(FieldDetection, bumper_crop); a cluster of detections within radius_m is one
physical robot. The merged FieldDetection (weighted-average position, weighted
color vote) is what the tracker consumes; the FusedObservation also carries the
raw evidence so the identity step can read each camera's bumper crop with its
source weight.

SAM2 can occasionally propagate two object masks onto the same robot. Same-camera
detections are therefore also merged when their image boxes overlap strongly.
Requiring both field proximity and image overlap avoids collapsing merely-adjacent
robots."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
from web.field.tracker import FieldDetection


@dataclass
class FusedObservation:
    detection: FieldDetection                 # what the tracker sees
    evidence: list                            # [(source_camera, bumper_crop, weight), ...]
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


def fuse_detections(
    per_camera: dict,
    weights: dict,
    radius_m: float = 0.6,
    same_camera_ios: float = 0.7,
    is_hub_occluded: Optional[Callable[[str, tuple], bool]] = None,
) -> list:
    # flatten to (camera, det, crop, weight)
    items = []
    for cam, dets in per_camera.items():
        w = float(weights.get(cam, 1.0))
        for (det, crop) in dets:
            items.append((cam, det, crop, w))

    used = [False] * len(items)
    out = []
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
        out.append(_merge(cluster, is_hub_occluded=is_hub_occluded))
    return out


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
    evidence = [(cam, crop, w) for (cam, det, crop, w) in cluster]
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
        detection=FieldDetection(pos=(float(avg[0]), float(avg[1])), color=color, box=box, embedding=merged_embedding),
        evidence=evidence,
        members=members,
    )
