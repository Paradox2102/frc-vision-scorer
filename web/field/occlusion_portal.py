"""Occlusion exit portals: predict where a hub-occluded track re-emerges.

A slot that goes into coast behind a hub prism keeps a constant-velocity
estimate. Intersecting that ray with the (expanded) footprint gives the exit
point and ETA; the tracker relaxes its association gate around the portal so
re-acquisition on the far side does not wait for the generic coast-gate to
widen (or lose the robot to a fresh seed under a new id)."""
from __future__ import annotations
import numpy as np

from web.field.occlusion import _expand_polygon, _point_in_poly


def predict_exit(pos, vel, footprint, margin_m=0.35, max_eta_s=6.0,
                 min_speed_mps=0.3):
    """(exit_xy, eta_s) where the constant-velocity ray from ``pos`` last
    leaves the expanded ``footprint``, or None (too slow / never inside /
    exits beyond ``max_eta_s``). Sampled at 0.05s steps — exact enough for a
    gate that is meters wide, and robust to concave expansion artifacts."""
    v = np.asarray(vel, float)
    speed = float(np.linalg.norm(v))
    if speed < min_speed_mps:
        return None
    poly = _expand_polygon(np.asarray(footprint, float), margin_m)
    p = np.asarray(pos, float)
    inside_seen = False
    last_inside_t = None
    for t in np.arange(0.0, max_eta_s + 1e-9, 0.05):
        q = p + v * t
        if _point_in_poly((q[0], q[1]), poly):
            inside_seen = True
            last_inside_t = t
        elif inside_seen:
            exit_t = t
            q = p + v * exit_t
            return (float(q[0]), float(q[1])), float(exit_t)
    del last_inside_t
    return None


def path_crosses_footprint(a, b, footprint, margin_m=0.35, n=50):
    """True if the straight segment a->b passes through the expanded
    footprint. Used to flag associations that 'teleport through' a hub
    without the occlusion gap geometry says they must have had."""
    poly = _expand_polygon(np.asarray(footprint, float), margin_m)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    for t in np.linspace(0.0, 1.0, n):
        q = a + (b - a) * t
        if _point_in_poly((q[0], q[1]), poly):
            return True
    return False
