"""Cross-camera fusion: many cameras' field detections -> merged detections (for
the tracker) + pooled, source-weighted identity evidence (side channel).

Greedy single-link clustering by field distance. Inputs are per-camera lists of
(FieldDetection, bumper_crop); a cluster of detections within radius_m is one
physical robot. The merged FieldDetection (weighted-average position, weighted
color vote) is what the tracker consumes; the FusedObservation also carries the
raw evidence so the identity step can read each camera's bumper crop with its
source weight."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from web.field.tracker import FieldDetection


@dataclass
class FusedObservation:
    detection: FieldDetection                 # what the tracker sees
    evidence: list                            # [(source_camera, bumper_crop, weight), ...]


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
    avg = (pos * ws[:, None]).sum(axis=0) / ws.sum()
    # weighted color vote, ignoring "unknown"
    votes = {}
    for cam, det, crop, w in cluster:
        if det.color and det.color != "unknown":
            votes[det.color] = votes.get(det.color, 0.0) + w
    color = max(votes, key=votes.get) if votes else "unknown"
    # representative box: highest-weight cluster member
    box = max(cluster, key=lambda c: c[3])[1].box
    evidence = [(cam, crop, w) for (cam, det, crop, w) in cluster]
    return FusedObservation(
        detection=FieldDetection(pos=(float(avg[0]), float(avg[1])), color=color, box=box),
        evidence=evidence,
    )
