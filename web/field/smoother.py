"""Offline RTS smoother over FieldTracker measurement history.

The live tracker (web/field/tracker.py) is a causal constant-velocity Kalman
filter with display-oriented coasting hacks (velocity decay while missed).
Match jobs are offline, so exported 2D paths can do better: re-run an exact
forward CV-KF over each recorded track segment — same F/Q/R as the live _KF,
same per-measurement r_scale (corner-only inflation, over-speed downweight),
but no decay — then a Rauch-Tung-Striebel backward pass. Occlusion gaps become
two-sided interpolations instead of coast-and-stall. The causal estimates
still drive the live HUD and every tracking decision; this module only feeds
the exported path rewrite (apply_smoothed_paths), gated on the rts_smooth
knob."""
from __future__ import annotations

import numpy as np

Q_ACCEL = 2.0   # matches _KF q
R_MEAS = 0.05   # matches _KF r
_H = np.array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])
_I4 = np.eye(4)


def _fq(dt: float, q: float):
    dt = max(float(dt), 1e-3)
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    G = np.array([[0.5 * dt * dt, 0.0], [0.0, 0.5 * dt * dt],
                  [dt, 0.0], [0.0, dt]])
    return F, G @ (q * np.eye(2)) @ G.T


def rts_smooth_segment(records, q=Q_ACCEL, r=R_MEAS):
    """Forward filter + RTS backward over one track segment.

    Output covers the seed through the LAST measured record; trailing
    coast-only records are dropped (no future measurement to smooth against —
    the causal path already covers that display case)."""
    if not records:
        return []
    measured = [i for i, rec in enumerate(records) if rec["z"] is not None]
    if not measured:
        # Defensive: producer invariant guarantees the seed record carries z,
        # but an all-coast segment has nothing to smooth against.
        return []
    last = measured[-1]
    z0 = records[0]["z"]
    x = np.array([z0[0], z0[1], 0.0, 0.0], float)
    P = np.diag([r, r, 4.0, 4.0])          # mirrors _KF.__init__
    xs_f, Ps_f = [x], [P]
    xs_p, Ps_p = [x.copy()], [P.copy()]    # k=0 prediction unused by backward pass
    for rec in records[1:]:
        F, Q = _fq(rec["dt"], q)
        xp = F @ x
        Pp = F @ P @ F.T + Q
        if rec["z"] is not None:
            r_scale = rec["r_scale"] if rec["r_scale"] is not None else 1.0
            R = (r * float(r_scale)) * np.eye(2)
            S = _H @ Pp @ _H.T + R
            K = Pp @ _H.T @ np.linalg.inv(S)
            x = xp + K @ (np.asarray(rec["z"], float) - _H @ xp)
            P = (_I4 - K @ _H) @ Pp
        else:
            x, P = xp, Pp
        xs_p.append(xp); Ps_p.append(Pp)
        xs_f.append(x); Ps_f.append(P)

    n = last + 1
    xs_s = [None] * n
    xs_s[-1] = xs_f[last]
    for k in range(n - 2, -1, -1):
        F, _ = _fq(records[k + 1]["dt"], q)
        # C = P_f F^T inv(P_p);  solve for stability instead of inv()
        C = np.linalg.solve(Ps_p[k + 1].T, (Ps_f[k] @ F.T).T).T
        xs_s[k] = xs_f[k] + C @ (xs_s[k + 1] - xs_p[k + 1])

    return [{
        "frame_num": records[k]["frame_num"],
        "pos": (float(xs_s[k][0]), float(xs_s[k][1])),
        "vel": (float(xs_s[k][2]), float(xs_s[k][3])),
        "measured": records[k]["z"] is not None,
        "corner_only": bool(records[k].get("corner_only", False)),
    } for k in range(n)]


def smooth_history(history, q=Q_ACCEL, r=R_MEAS):
    """slot_id -> {frame_num -> {pos, measured, corner_only}} over all segments."""
    out: dict[int, dict[int, dict]] = {}
    for slot_id, segments in (history or {}).items():
        frames = out.setdefault(slot_id, {})
        for seg in segments:
            for o in rts_smooth_segment(seg, q=q, r=r):
                if o["frame_num"] is None:
                    continue
                frames[int(o["frame_num"])] = {
                    "pos": o["pos"],
                    "measured": o["measured"],
                    "corner_only": o["corner_only"],
                }
    return out


def apply_smoothed_paths(paths, smoothed, label_for_slot, fps):
    """Rewrite exported autonomous_paths in place with smoothed positions.

    Existing samples get their field_pos replaced; smoothed frames strictly
    inside a label's existing frame range that have no sample (occlusion /
    corner-only stretches, which the export loop skips) are inserted as
    synthetic samples flagged interpolated=True with tracker_prob / det_conf
    None, so
    downstream consumers can tell inference from measurement. Returns
    {label: (n_replaced, n_inserted)} for the job log."""
    stats: dict[str, tuple[int, int]] = {}
    fps = float(fps) if fps else 1.0
    for slot_id, frames in (smoothed or {}).items():
        label = label_for_slot(slot_id)
        samples = paths.get(label)
        if not samples:
            continue
        by_frame = {s["frame_num"]: s for s in samples}
        fmin, fmax = min(by_frame), max(by_frame)
        replaced = inserted = 0
        for fn, info in frames.items():
            px = round(float(info["pos"][0]), 4)
            py = round(float(info["pos"][1]), 4)
            s = by_frame.get(fn)
            if s is not None:
                s["field_pos"] = [px, py]
                replaced += 1
            elif fmin < fn < fmax:
                samples.append({
                    "frame_num": fn,
                    "timestamp": round(fn / fps, 4),
                    "field_pos": [px, py],
                    "corner_only": bool(info["corner_only"]),
                    "occluded": None,
                    "tracker_prob": None,
                    "det_conf": None,
                    "interpolated": True,
                })
                inserted += 1
        samples.sort(key=lambda s: s["frame_num"])
        stats[label] = (replaced, inserted)
    return stats
