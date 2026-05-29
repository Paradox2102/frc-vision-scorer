"""Occlusion geometry (pure NumPy + a CameraCalibration for projection).

An occluder lies between the camera and the robot. Hubs are static prisms derived
from the tallest tag z-ring; persons/robots are depth-ordered boxes. Produces, per
track, view_occluded (drives coasting) and bumper_occluded (gates VLM reads)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Prism:
    footprint: np.ndarray   # (N,2) convex polygon, field meters
    z_top: float
    z_bottom: float = 0.0


def _zgroups(zs, tol=0.05):
    zs = np.sort(np.unique(np.round(zs, 3)))
    groups = [[zs[0]]]
    for z in zs[1:]:
        (groups[-1].append(z) if z - groups[-1][-1] <= tol else groups.append([z]))
    return groups


def _convex_hull(pts):
    pts = np.asarray(pts, float)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(pts) <= 2:
        return pts
    def _cross2d(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(ps):
        h = []
        for p in ps:
            while len(h) >= 2 and _cross2d(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]
    return np.array(half(pts) + half(pts[::-1]))


def hub_prisms(field_model):
    """Two hub prisms from the tallest tag z-group, split at the field midline."""
    ids = list(field_model.tags)
    ts = np.array([field_model.tag_translation(i) for i in ids])  # (N,3)
    top = _zgroups(ts[:, 2])[-1]
    mask = np.isin(np.round(ts[:, 2], 3), np.round(top, 3))
    hub_xy = ts[mask, :2]; hub_z = ts[mask, 2].max()
    mid = field_model.length_m / 2.0
    prisms = []
    for sel in (hub_xy[:, 0] < mid, hub_xy[:, 0] >= mid):
        if sel.sum() >= 3:
            prisms.append(Prism(footprint=_convex_hull(hub_xy[sel]), z_top=float(hub_z) + 0.2))
    return prisms


def _point_in_poly(pt, poly):
    x, y = pt; inside = False; n = len(poly); j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _expand_polygon(poly, margin):
    """Expand a convex polygon outward from its centroid by `margin` meters."""
    centroid = poly.mean(axis=0)
    vecs = poly - centroid
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return poly + vecs / (norms + 1e-9) * margin


_HUB_MARGIN_M = 0.35  # footprint expansion: tag ring sits at inner hub edge; accounts for surrounding frame/netting


def occluded_by_hub(cam_center, prism, track_xy, n=100, margin=_HUB_MARGIN_M):
    """Does the hub prism block the line-of-sight from cam_center to track_xy?

    Uses a 2-D ground-plane shadow check on a slightly expanded footprint.
    We also verify the hub is closer to the camera than the target so we don't
    flag robots that are on the near side.
    """
    cam_xy = np.asarray(cam_center, float)[:2]
    tgt_xy = np.array([track_xy[0], track_xy[1]], float)

    # Hub must be closer to the camera than the target
    hub_center = prism.footprint.mean(axis=0)
    d_hub = np.linalg.norm(cam_xy - hub_center)
    d_target = np.linalg.norm(cam_xy - tgt_xy)
    if d_hub >= d_target:
        return False

    expanded = _expand_polygon(prism.footprint, margin)
    for t in np.linspace(0.0, 1.0, n):
        p = cam_xy + (tgt_xy - cam_xy) * t
        if _point_in_poly(p, expanded):
            return True
    return False


def camera_distance(cam_center, ground_xy):
    return float(np.linalg.norm(np.asarray(cam_center, float)
                                - np.array([ground_xy[0], ground_xy[1], 0.0])))


def box_overlap(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def _bumper_region(box):
    x1, y1, x2, y2 = box
    h = y2 - y1
    return (x1, y1 + 0.6 * h, x2, y2)


def resolve_occlusion(cam, tracks, person_boxes, hub_prisms, robot_h=0.5):
    """tracks: list of {slot_id, pos:(x,y), box:(x1,y1,x2,y2)}.
    Returns {slot_id: {view_occluded, bumper_occluded, cause}}.

    robot_h is reserved for future 3-D hub occlusion; the current 2-D shadow
    model inside occluded_by_hub does not use robot height.
    """
    cc = cam.camera_center
    out = {}
    tdist = {t["slot_id"]: camera_distance(cc, t["pos"]) for t in tracks}
    for t in tracks:
        sid = t["slot_id"]; box = t["box"]; view = False; bump = False; cause = None
        area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        bregion = _bumper_region(box)
        barea = max(1.0, (bregion[2] - bregion[0]) * (bregion[3] - bregion[1]))
        for p in hub_prisms:
            if occluded_by_hub(cc, p, t["pos"]):
                view = True; cause = "hub"
        for pb in person_boxes:
            if box_overlap(box, pb) / area > 0.5:
                view = True; cause = cause or "person"
            if box_overlap(bregion, pb) / barea > 0.3:
                bump = True
        for o in tracks:
            if o["slot_id"] == sid:
                continue
            if tdist[o["slot_id"]] < tdist[sid]:
                if box_overlap(box, o["box"]) / area > 0.6:
                    view = True; cause = cause or "robot"
                if box_overlap(bregion, o["box"]) / barea > 0.3:
                    bump = True
        out[sid] = {"view_occluded": view, "bumper_occluded": bump or view, "cause": cause}
    return out
