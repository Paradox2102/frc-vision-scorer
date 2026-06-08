"""Cross-camera fusion: many cameras' field detections -> merged detections (for
the tracker) + pooled, source-weighted identity evidence (side channel).

Greedy single-link clustering by field distance. Inputs are per-camera lists of
(FieldDetection, bumper_crop); a cluster of detections within radius_m is one
physical robot. The merged FieldDetection (weighted-average position, weighted
color vote) is what the tracker consumes; the FusedObservation also carries the
raw evidence so the identity step can read each camera's bumper crop with its
source weight."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from web.field.tracker import FieldDetection


@dataclass
class FusedObservation:
    detection: FieldDetection                 # what the tracker sees
    evidence: list                            # [(source_camera, bumper_crop, weight), ...]
    members: list = field(default_factory=list)   # [(camera, per_camera_track_id), ...]
    track_id: int | None = None               # stable fused id (assigned by FusedTrackRegistry)


def fuse_detections(per_camera: dict, weights: dict, radius_m: float = 0.6) -> list:
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
            if cam_j == cam_i:
                continue  # one camera shouldn't merge with itself
            if np.linalg.norm(pi - np.array(det_j.pos, float)) <= radius_m:
                cluster.append((cam_j, det_j, crop_j, w_j))
                used[j] = True
        out.append(_merge(cluster))
    return out


def _merge(cluster) -> FusedObservation:
    ws = np.array([c[3] for c in cluster], float)
    pos = np.array([c[1].pos for c in cluster], float)
    # Main is the reference geometry. Corner cameras are deliberately weighted
    # higher for identity evidence, but their low, oblique ground projection is
    # less stable and must not drag a valid main-camera position across the field.
    # Corner geometry is still used when main is genuinely occluded.
    main_members = [c for c in cluster if c[0] == "main"]
    if main_members:
        main_ws = np.array([c[3] for c in main_members], float)
        main_pos = np.array([c[1].pos for c in main_members], float)
        avg = (main_pos * main_ws[:, None]).sum(axis=0) / main_ws.sum()
    else:
        avg = (pos * ws[:, None]).sum(axis=0) / ws.sum()
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
    members = [(cam, det.track_id) for (cam, det, crop, w) in cluster]

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
