"""Offline replay of the field tracker (web/field/tracker.py) from a TRACKER_TRACE dump.

A ``TRACKER_TRACE=1`` job writes, per processed frame, the fused observations the
tracker consumed (pos / cams / has_main_box / color / fused track id), the raw
per-camera detections, the tracker's output tracks and ``det_to_slot``. This
script rebuilds the ``FieldTracker.update`` inputs from those rows and replays the
tracker frame by frame, so tracker variants (knobs) can be A/B'd on a real match
in seconds without re-running the pipeline.

Fidelity. The unmodified tracker must reproduce the logged ``det_to_slot`` and
tracks; ``--fidelity`` reports the match rate. Inputs the old trace format lacks:

- occlusion flags (``view_flags``: decided from track boxes + per-camera person
  boxes, which are not traced). Newer traces carry them per frame (``occ``) and
  replay exactly. For older traces the replay uses ``--occ oracle`` for the
  fidelity check (a slot is "occluded" iff the LOG kept it alive at that frame:
  occlusion only changes the coast budget, so this reproduces every logged
  expiry/survival while the replay state matches the log) and ``--occ position``
  for variant A/Bs (a slot is occluded iff it sits within 1.5 m of a logged slot
  in an occluded coast, i.e. one that outlived ``max_coast_open``) — occlusion
  is a property of the field position, not of the slot that happens to hold it.
- ``dt`` / occlusion portals: traced per frame in new traces; old ones use
  frame_skip / fps (``--dt``, default 8/60) and no portals (knob was off).
- tracker config (static zones, sticky gate): new traces carry it on the first
  frame (``tracker_cfg``) plus the untraced early-OCR update (``early``); old
  ones use the 2026cur hub zones (DEFAULT_ZONES) and the job defaults, and
  start without the early update (its two seeded slots converge within ~15
  frames; qm8: 99.3% of frames still reproduce det_to_slot exactly).
- input precision: old traces round positions to 1 cm, which flips the rare
  exact assignment tie (the first qm8 divergence: 0.454 vs 0.456 m); new
  traces log 0.1 mm.
- detection boxes: only presence matters to the tracker (corner_only, r_scale);
  a placeholder box stands in for a main-camera box.

Reads. A ``vlm_read`` row names an identity (robot_id) at its crop frame. Its crop
came from the fused observation that updated that identity's slot at that frame,
so a read maps to ``(frame, fused det index)`` through the identity log
(``identity_offline_check.reconstruct_identity_log``) and the LOGGED det_to_slot,
and then to a replayed slot through the replay's det_to_slot.

Metrics (``evaluate``):

- robot-change proxies: far reseeds (seed > 4 m from the slot's last
  measurement), implausible re-acquisitions (coast re-acquire implying > 5 m/s
  from the last measurement, jump > 1 m), duplicate pair-frames (two
  same-alliance slots within 1 m), corner-shadow frames (a corner-only slot
  within 2.5 m of a visible main-backed same-alliance slot), and main-camera
  object steals: a slot's measuring main-camera SAM object changes while the
  previous object is still detected where the slot left it (<= 1.5 m) and
  > 1 m from the new detection — the slot left a robot that is still there
  (excludes second SAM masks on one robot and SAM mask hijacks);
- read purity: consecutive confident (conf >= 0.65) same-slot reads that agree,
  overall / within a segment / across a discontinuity (seed, or a re-acquire
  jump > 2 m), distinct teams per segment, and per-slot majority share;
- coverage: per-slot visible / coasting fraction, fused-det utilisation, reads
  that land on no slot ("lost"; a read whose corner det was suppressed as a
  duplicate counts on its main-backed parent's slot, reported as "+Ndup").

Usage:
  .venv/bin/python scripts/replay_field_tracker.py TRACE.json --job 91411eb8 \\
      --rosters 3128,6773,9066/6497,2724,2637 [--fidelity] [--variants all] \\
      [--set phys_reach_slack_m=0.5] [--jitter 0.005 --seeds 8] [--json out.json]

``--jitter/--seeds`` replays each arm on N position-jittered copies and prints
mean +- sd of the key metrics: the tracker is chaotic (one tie flip re-routes
the rest of the match), so compare arm differences against that noise floor.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from web.field.tracker import FieldDetection, FieldTracker  # noqa: E402

LABELS = ["B1", "B2", "B3", "R1", "R2", "R3"]
PLACEHOLDER_BOX = (0.0, 0.0, 1.0, 1.0)
DEFAULT_DT = 8 / 60.0
# 2026cur hub no-seed zones (the only static zones on the qm8/qm11 jobs; logged
# at job start as "Static-structure zones"). New traces carry their own.
DEFAULT_ZONES = [
    (3.8218613999999995, 3.2312351999999995, 5.429174199999999, 4.8380399999999995),
    (11.1118646, 3.2312351999999995, 12.719177399999998, 4.8380399999999995),
]
# Tracker kwargs the job used (job_config defaults: sticky_max_dist_m 2.5).
BASE_TRACKER_KW = {"sticky_max_dist_m": 2.5}


# --------------------------------------------------------------------------
# Trace -> tracker inputs
# --------------------------------------------------------------------------

def load_trace(path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):          # tolerate {"frames": [...]} wrappers
        data = data.get("frames", [])
    return data


def trace_detections(rec: dict, rng=None, jitter_m: float = 0.0) -> list[FieldDetection]:
    """The fused observations of one traced frame, as the tracker saw them
    (optionally with uniform +-jitter_m position noise — the noise-floor runs:
    the tracker is chaotic, so one rounding-level tie flip re-routes the rest
    of a match; jittered seeds show which arm differences exceed that)."""
    out = []
    for o in rec["fused"]:
        jx = jy = 0.0
        if rng is not None and jitter_m > 0:
            jx, jy = rng.uniform(-jitter_m, jitter_m), rng.uniform(-jitter_m, jitter_m)
        out.append(FieldDetection(
            pos=(float(o["pos"][0]) + jx, float(o["pos"][1]) + jy),
            color=o.get("color") or "unknown",
            box=PLACEHOLDER_BOX if o.get("has_main_box") else None,
            track_id=o.get("tid"),
            cams=tuple(o.get("cams") or ()),
        ))
    return out


def trace_tracker_cfg(trace: list[dict]) -> dict:
    """Tracker construction kwargs recorded by newer traces (first frame)."""
    for rec in trace[:1]:
        cfg = rec.get("tracker_cfg")
        if cfg:
            return dict(cfg)
    return {}


def _logged_present(trace):
    return [{t["slot"]: t for t in rec["tracks"]} for rec in trace]


def occlusion_oracle(trace: list[dict]) -> Callable:
    """Occlusion inferred from the log: at frame k a slot is 'occluded' iff the
    log kept it alive at k. Exact for every logged expiry/survival decision
    while the replay matches the log (occlusion only sets the coast budget and
    occ_coasting, which feeds the default-off portal/guard only)."""
    present = _logged_present(trace)

    def occ(k, prev_out):
        return {s: {"view_occluded": s in present[k]} for s in range(6)}
    return occ


def logged_occluded_positions(trace: list[dict], max_open: int = 8,
                              logged_occ: Optional[list] = None) -> list[list]:
    """Per frame, the positions of logged slots in an OCCLUDED coast: a coast
    run that outlived ``max_open`` missed frames (only occlusion extends the
    budget), or slots whose traced ``occ`` flag is set."""
    n = len(trace)
    out = [[] for _ in range(n)]
    if logged_occ is not None:
        for k, rec in enumerate(trace):
            for t in rec["tracks"]:
                if logged_occ[k].get(t["slot"]):
                    out[k].append(tuple(t["pos"]))
        return out
    runs = defaultdict(list)     # slot -> current coast run [(k, pos)]
    for k, rec in enumerate(trace):
        present = {t["slot"]: t for t in rec["tracks"]}
        for s in range(6):
            t = present.get(s)
            if t is not None and t["state"] == "coasting":
                runs[s].append((k, tuple(t["pos"])))
                continue
            run = runs.pop(s, [])
            if len(run) > max_open:
                for kk, p in run:
                    out[kk].append(p)
    for s, run in runs.items():
        if len(run) > max_open:
            for kk, p in run:
                out[kk].append(p)
    return out


def occlusion_position(trace: list[dict], radius_m: float = 1.5,
                       max_open: int = 8) -> Callable:
    """Occlusion transferred by position: a replayed slot is occluded iff its
    last output position is within ``radius_m`` of a logged occluded-coast slot
    at that frame. Arm-independent, so A/B arms see the same occluders."""
    logged_occ = None
    if all("occ" in rec for rec in trace):
        logged_occ = [{int(s): bool(v) for s, v in rec["occ"].items()} for rec in trace]
    occ_pos = logged_occluded_positions(trace, max_open, logged_occ)

    def occ(k, prev_out):
        res = {}
        for t in prev_out or []:
            p = t.pos
            res[t.slot_id] = {"view_occluded": any(
                math.hypot(p[0] - q[0], p[1] - q[1]) <= radius_m for q in occ_pos[k])}
        return res
    return occ


def occlusion_traced(trace: list[dict]) -> Callable:
    """Exact occlusion flags from a trace that recorded them (``occ``)."""
    flags = [{int(s): {"view_occluded": bool(v)} for s, v in rec["occ"].items()}
             for rec in trace]

    def occ(k, prev_out):
        return flags[k]
    return occ


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------

@dataclass
class ReplayFrame:
    frame: int
    tracks: list            # TrackOut
    det_to_slot: dict       # fused det index -> slot
    slot_events: list
    n_dets: int


def make_tracker(field_length: float = 16.541, zones=None, **kw) -> FieldTracker:
    args = dict(BASE_TRACKER_KW)
    args.update(kw)
    return FieldTracker(field_length, zones=DEFAULT_ZONES if zones is None else zones,
                        **args)


def replay(trace: list[dict], tracker: FieldTracker, occ_fn: Callable,
           dt: float = DEFAULT_DT, early: bool = True, jitter_m: float = 0.0,
           seed: int = 0) -> list[ReplayFrame]:
    import random
    rng = random.Random(seed) if jitter_m > 0 else None
    out = []
    prev = []
    if early and trace:
        first = trace[0]
        if first.get("early") is not None:
            e = first["early"]
            edets = [FieldDetection(pos=tuple(o["pos"]), color=o.get("color") or "unknown",
                                    box=PLACEHOLDER_BOX if o.get("has_main_box") else None,
                                    track_id=o.get("tid")) for o in e["dets"]]
            tracker.update(edets, {}, float(e.get("dt", dt)), frame_num=e.get("frame"))
    for k, rec in enumerate(trace):
        dets = trace_detections(rec, rng, jitter_m)
        step_dt = float(rec.get("dt", dt))
        portals = ({int(sid): tuple(v) for sid, v in rec["portals"].items()}
                   if rec.get("portals") else None)
        tracks = tracker.update(dets, occ_fn(k, prev), step_dt,
                                portals=portals, frame_num=rec["frame"])
        out.append(ReplayFrame(rec["frame"], tracks, dict(tracker.last_det_to_slot),
                               list(tracker.slot_events), len(dets)))
        prev = tracks
    return out


def fidelity(trace: list[dict], rep: list[ReplayFrame], pos_tol: float = 0.011) -> dict:
    """How exactly the replay reproduces the logged tracker output."""
    n = len(trace)
    d2s_eq = tracks_eq = pos_eq = 0
    pairs = pairs_eq = 0
    first_div = None
    for rec, r in zip(trace, rep):
        logged = {int(k): int(v) for k, v in rec["det_to_slot"].items()}
        ok_d = logged == r.det_to_slot
        d2s_eq += ok_d
        for k, v in logged.items():
            pairs += 1
            pairs_eq += r.det_to_slot.get(k) == v
        lt = {(t["slot"], t["state"]) for t in rec["tracks"]}
        rt = {(t.slot_id, t.state) for t in r.tracks}
        ok_t = lt == rt
        tracks_eq += ok_t
        lp = {t["slot"]: t["pos"] for t in rec["tracks"]}
        ok_p = ok_t and all(
            abs(round(t.pos[0], 2) - lp[t.slot_id][0]) <= pos_tol
            and abs(round(t.pos[1], 2) - lp[t.slot_id][1]) <= pos_tol
            for t in r.tracks)
        pos_eq += ok_p
        if first_div is None and not (ok_d and ok_t):
            first_div = rec["frame"]
    return {"frames": n, "det_to_slot_frames": d2s_eq / n,
            "assignments": pairs_eq / max(1, pairs), "track_state_frames": tracks_eq / n,
            "track_pos_frames": pos_eq / n, "first_divergence_frame": first_div}


# --------------------------------------------------------------------------
# Reads -> (frame, det index)
# --------------------------------------------------------------------------

def load_job_reads(job_prefix: str, data_root: Path, trace: list[dict]) -> list[dict]:
    """Team-naming reads of a job mapped to (frame, fused det index) via the
    identity log and the LOGGED det_to_slot. Rows that cannot be mapped keep
    ``det=None`` (counted as unmappable)."""
    from identity_offline_check import load_job, reconstruct_identity_log
    d = load_job(data_root, job_prefix)
    ilog = reconstruct_identity_log(d["bindings"], d["frame_nums"])
    key = {(r["frame_num"], r["identity_id"]): r["slot_id"] for r in ilog}
    byf = {rec["frame"]: rec for rec in trace}
    out = []
    for r in d["reads"]:
        team = r.get("matched_team")
        if team is None:
            continue
        slot = key.get((r["frame"], r["robot_id"]))
        rec = byf.get(r["frame"])
        det = None
        if slot is not None and rec is not None:
            inv = {int(v): int(k) for k, v in rec["det_to_slot"].items()}
            det = inv.get(slot)
        out.append({"frame": r["frame"], "team": str(team),
                    "conf": float(r.get("confidence") or 0.0),
                    "logged_slot": slot, "det": det,
                    "evicted": bool(r.get("evicted"))})
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _main_oid(rec: dict, o: dict) -> Optional[int]:
    """Main-camera SAM object id behind a main-backed fused det, recovered by
    exact position match (fused pos == the main member's pos), or None."""
    if not o.get("has_main_box"):
        return None
    hits = [d["tid"] for d in rec.get("raw", {}).get("main", [])
            if abs(d["pos"][0] - o["pos"][0]) < 0.006
            and abs(d["pos"][1] - o["pos"][1]) < 0.006]
    return hits[0] if len(hits) == 1 else None


def _dup_parent_slot(rec: dict, r: "ReplayFrame", j: int, radius_m: float = 2.0):
    """Slot of the main-backed det that corner-only det ``j`` was suppressed
    as a duplicate of (corner_dup_suppress): the read's crop shows that robot,
    whose main view carries the slot. None if ``j`` wasn't suppressed."""
    if not any(e.get("event") == "corner_dup" and j in e.get("det_idx", ())
               for e in r.slot_events):
        return None
    o = rec["fused"][j]
    best = None
    for i, m in enumerate(rec["fused"]):
        if not m.get("has_main_box") or i not in r.det_to_slot:
            continue
        if m.get("color") != o.get("color") or set(o.get("cams") or ()) & set(m.get("cams") or ()):
            continue
        d = math.hypot(m["pos"][0] - o["pos"][0], m["pos"][1] - o["pos"][1])
        if d <= radius_m and (best is None or d < best[0]):
            best = (d, r.det_to_slot[i])
    return best[1] if best else None


def evaluate(trace: list[dict], rep: list[ReplayFrame], reads: list[dict],
             dt: float = DEFAULT_DT, conf_min: float = 0.65, far_seed_m: float = 4.0,
             reach_speed: float = 5.0, dup_m: float = 1.0, disc_jump_m: float = 2.0,
             steal_sep_m: float = 1.0, steal_stay_m: float = 1.5,
             corner_dup_m: float = 2.5,
             rosters: Optional[dict] = None) -> dict:
    n = len(rep)
    m: dict = {}
    # ---- (a) robot-change proxies ----
    last_meas: dict = {}          # slot -> (k, pos)
    far_seeds = seeds = reacq = bad_reacq = 0
    disc = defaultdict(list)      # slot -> frames of discontinuities
    for k, r in enumerate(rep):
        for e in r.slot_events:
            s = e["slot_id"]
            if e["event"] == "seed":
                seeds += 1
                lm = last_meas.get(s)
                if lm is not None:
                    dist = math.hypot(e["pos"][0] - lm[1][0], e["pos"][1] - lm[1][1])
                    if dist > far_seed_m:
                        far_seeds += 1
                disc[s].append(r.frame)
            elif e["event"] == "reacquire":
                reacq += 1
                jump = e.get("jump_m") or 0.0
                sp = e.get("speed_mps") or 0.0
                if jump > 1.0 and sp > reach_speed:
                    bad_reacq += 1
                if jump > disc_jump_m:
                    disc[s].append(r.frame)
        meas = {v: kk for kk, v in r.det_to_slot.items()}
        for s, j in meas.items():
            o = trace[k]["fused"][j]
            last_meas[s] = (k, o["pos"])
    dup = cdup = 0
    for r in rep:
        for a in ("blue", "red"):
            ts = [t for t in r.tracks if t.alliance == a]
            for i in range(len(ts)):
                for j in range(i + 1, len(ts)):
                    d = math.hypot(ts[i].pos[0] - ts[j].pos[0], ts[i].pos[1] - ts[j].pos[1])
                    if d < dup_m:
                        dup += 1
            # corner-only slot shadowing a visible main-backed slot
            for t in ts:
                if t.corner_only and any(
                        u is not t and not u.corner_only and u.state == "visible"
                        and math.hypot(t.pos[0] - u.pos[0], t.pos[1] - u.pos[1]) < corner_dup_m
                        for u in ts):
                    cdup += 1
    # main-camera object steals
    steals = 0
    last_oid: dict = {}           # slot -> (main object id, its position then)
    for k, r in enumerate(rep):
        rec = trace[k]
        main_pos = defaultdict(list)
        for d in rec.get("raw", {}).get("main", []):
            main_pos[d["tid"]].append(d["pos"])
        for j, s in r.det_to_slot.items():
            oid = _main_oid(rec, rec["fused"][j])
            if oid is None:
                continue
            prev = last_oid.get(s)
            p = rec["fused"][j]["pos"]
            if prev is not None and prev[0] != oid and any(
                    math.hypot(q[0] - p[0], q[1] - p[1]) > steal_sep_m
                    and math.hypot(q[0] - prev[1][0], q[1] - prev[1][1]) <= steal_stay_m
                    for q in main_pos.get(prev[0], [])):
                # The slot's previous main-camera object is still detected
                # where the slot left it, well apart from the det now
                # measuring the slot: the slot moved to another robot. (An old
                # object near the new det is a second SAM mask on the same
                # robot; an old object that itself jumped is a SAM hijack.)
                steals += 1
            last_oid[s] = (oid, p)
        for e in r.slot_events:
            if e["event"] == "seed":
                last_oid.pop(e["slot_id"], None)
    m["seeds"] = seeds
    m["far_seeds"] = far_seeds
    m["reacquires"] = reacq
    m["implausible_reacq"] = bad_reacq
    m["dup_pair_frames"] = dup
    m["corner_shadow_frames"] = cdup
    m["oid_steals"] = steals
    # ---- (c) coverage ----
    vis = Counter(); coast = Counter()
    for r in rep:
        for t in r.tracks:
            (vis if t.state == "visible" else coast)[t.slot_id] += 1
    m["visible_frac"] = {LABELS[s]: round(vis[s] / n, 3) for s in range(6)}
    m["coast_frac"] = {LABELS[s]: round(coast[s] / n, 3) for s in range(6)}
    m["mean_visible"] = sum(vis.values()) / (6 * n)
    m["mean_active"] = (sum(vis.values()) + sum(coast.values())) / (6 * n)
    tot_dets = sum(r.n_dets for r in rep)
    m["det_utilisation"] = sum(len(r.det_to_slot) for r in rep) / max(1, tot_dets)
    # ---- (b) read purity ----
    k_of = {r.frame: k for k, r in enumerate(rep)}
    conf_reads = [x for x in reads if x["conf"] >= conf_min]
    by_slot = defaultdict(list)
    lost = unmappable = wrong_alliance = dup_reattributed = 0
    for x in conf_reads:
        if x["det"] is None:
            unmappable += 1
            continue
        k = k_of.get(x["frame"])
        s = rep[k].det_to_slot.get(x["det"]) if k is not None else None
        if s is None and k is not None:
            s = _dup_parent_slot(trace[k], rep[k], x["det"])
            dup_reattributed += s is not None
        if s is None:
            lost += 1
            continue
        if rosters and rosters.get(x["team"]) and rosters[x["team"]] != ("blue" if s < 3 else "red"):
            wrong_alliance += 1
        by_slot[s].append(x)
    pairs = agree = w_pairs = w_agree = x_pairs = x_agree = 0
    seg_teams = []
    maj = tot = 0
    for s, xs in by_slot.items():
        xs.sort(key=lambda x: x["frame"])
        bnd = sorted(disc.get(s, []))
        for a, b in zip(xs, xs[1:]):
            across = bisect.bisect_right(bnd, b["frame"]) > bisect.bisect_right(bnd, a["frame"])
            same = a["team"] == b["team"]
            pairs += 1; agree += same
            if across:
                x_pairs += 1; x_agree += same
            else:
                w_pairs += 1; w_agree += same
        segs = defaultdict(set)
        for x in xs:
            segs[bisect.bisect_right(bnd, x["frame"])].add(x["team"])
        seg_teams.extend(len(v) for v in segs.values())
        c = Counter(x["team"] for x in xs)
        maj += c.most_common(1)[0][1]; tot += len(xs)
    m["conf_reads"] = len(conf_reads)
    m["reads_on_slot"] = tot
    m["reads_lost"] = lost
    m["reads_via_dup_parent"] = dup_reattributed
    m["reads_unmappable"] = unmappable
    m["reads_wrong_alliance"] = wrong_alliance
    m["pair_agree"] = (agree, pairs)
    m["within_agree"] = (w_agree, w_pairs)
    m["across_agree"] = (x_agree, x_pairs)
    m["segments_with_reads"] = len(seg_teams)
    m["segments_multi_team"] = sum(1 for v in seg_teams if v > 1)
    m["slot_majority_share"] = maj / max(1, tot)
    m["teams_per_slot"] = {LABELS[s]: dict(Counter(x["team"] for x in xs))
                           for s, xs in sorted(by_slot.items())}
    return m


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------

VARIANTS: dict[str, dict] = {
    "baseline": {},
}


def register_variants():
    """Variant arms -> FieldTracker kwargs. Kept in one place so the CLI and
    the docs table agree."""
    fix = {"assign_partial_fix": True}
    VARIANTS.update({
        "assign_fix": dict(fix),
        "coast_gate": {"coast_gate_phys": True},
        "reseed": {"reseed_identity": True},
        "corner_dup": {"corner_dup_suppress": True},
        "all3": {"coast_gate_phys": True, "reseed_identity": True,
                 "corner_dup_suppress": True},
        "fix+coast": {**fix, "coast_gate_phys": True},
        "fix+reseed": {**fix, "reseed_identity": True},
        "fix+dup": {**fix, "corner_dup_suppress": True},
        "fix+coast+dup": {**fix, "coast_gate_phys": True, "corner_dup_suppress": True},
        "all4": {**fix, "coast_gate_phys": True, "reseed_identity": True,
                 "corner_dup_suppress": True},
    })


NOISE_KEYS = ("oid_steals", "implausible_reacq", "corner_shadow_frames", "far_seeds",
              "dup_pair_frames", "pair_agree_frac", "within_agree_frac",
              "slot_majority_share", "reads_lost", "mean_visible", "mean_active")


def noise_summary(runs: list[dict]) -> dict:
    """Mean and sample sd of the key metrics over jittered replays."""
    import statistics
    out = {}
    for k in NOISE_KEYS:
        vals = []
        for m in runs:
            if k == "pair_agree_frac":
                a, n = m["pair_agree"]; vals.append(a / n if n else 0.0)
            elif k == "within_agree_frac":
                a, n = m["within_agree"]; vals.append(a / n if n else 0.0)
            else:
                vals.append(float(m[k]))
        out[k] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return out


def fmt_row(name: str, m: dict) -> str:
    pa = m["pair_agree"]; wa = m["within_agree"]; xa = m["across_agree"]
    return (f"{name:<14} seeds {m['seeds']:>3} far {m['far_seeds']:>3} | reacq {m['reacquires']:>4} "
            f"implaus {m['implausible_reacq']:>3} | dup {m['dup_pair_frames']:>4} cshadow {m['corner_shadow_frames']:>4} "
            f"steal {m['oid_steals']:>3} | "
            f"reads agree {pa[0]}/{pa[1]} within {wa[0]}/{wa[1]} across {xa[0]}/{xa[1]} "
            f"segMulti {m['segments_multi_team']}/{m['segments_with_reads']} maj {m['slot_majority_share']:.2f} "
            f"lost {m['reads_lost']}+{m['reads_via_dup_parent']}dup | vis {m['mean_visible']:.3f} act {m['mean_active']:.3f} "
            f"util {m['det_utilisation']:.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace")
    ap.add_argument("--job", help="job id prefix (for vlm_read rows + identity log)")
    ap.add_argument("--data-root", default=str(Path("/home/nick/AI/frc-vision-scorer")))
    ap.add_argument("--dt", type=float, default=DEFAULT_DT)
    ap.add_argument("--fidelity", action="store_true")
    ap.add_argument("--occ", choices=("oracle", "position", "traced", "none"), default="position",
                    help="occlusion model for the variant arms")
    ap.add_argument("--variants", default="all",
                    help="comma list of arms or 'all'")
    ap.add_argument("--set", action="append", default=[],
                    help="extra tracker kwarg for every non-baseline arm, k=v")
    ap.add_argument("--rosters", default="",
                    help="blue teams/red teams, e.g. 3128,6773,9066/6497,2724,2637")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="uniform +- position noise (m) for noise-floor seeds")
    ap.add_argument("--seeds", type=int, default=0,
                    help="with --jitter: also run N jittered seeds per arm and "
                         "print mean/sd of the key metrics")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    register_variants()
    trace = load_trace(args.trace)
    cfg = trace_tracker_cfg(trace)
    field_length = float(cfg.pop("field_length", 16.541))
    zones = cfg.pop("zones", None)
    if zones is not None:
        zones = [tuple(z) for z in zones]
    if "suspect_zones" in cfg:
        cfg["suspect_zones"] = [tuple(z) for z in cfg["suspect_zones"]]
    # The traced job's own tracker config (variant knobs as the job ran them).
    base_kw = dict(cfg)
    has_occ = all("occ" in rec for rec in trace)

    def occ_fn(kind):
        if kind == "traced" or (kind == "oracle" and has_occ):
            return occlusion_traced(trace)
        if kind == "oracle":
            return occlusion_oracle(trace)
        if kind == "position":
            return occlusion_position(trace)
        return lambda k, prev: {}

    result: dict = {"trace": args.trace, "frames": len(trace), "has_occ": has_occ}
    if args.fidelity:
        for kind in (("traced",) if has_occ else ("oracle", "position", "none")):
            rep = replay(trace, make_tracker(field_length, zones=zones, **base_kw), occ_fn(kind), args.dt)
            f = fidelity(trace, rep)
            result[f"fidelity_{kind}"] = f
            print(f"fidelity[{kind}]: det_to_slot frames {f['det_to_slot_frames']:.4f}  "
                  f"assignments {f['assignments']:.4f}  track-state frames "
                  f"{f['track_state_frames']:.4f}  pos frames {f['track_pos_frames']:.4f}  "
                  f"first divergence @ {f['first_divergence_frame']}")
    rosters = {}
    if args.rosters:
        b, r = args.rosters.split("/")
        rosters.update({t: "blue" for t in b.split(",")})
        rosters.update({t: "red" for t in r.split(",")})
    reads = load_job_reads(args.job, Path(args.data_root), trace) if args.job else []
    extra = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            extra[k] = json.loads(v)
        except ValueError:
            extra[k] = v
    names = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
    result["arms"] = {}
    for name in names:
        kw = dict(base_kw)
        kw.update(VARIANTS[name])
        if name != "baseline":
            kw.update(extra)
        rep = replay(trace, make_tracker(field_length, zones=zones, **kw), occ_fn(args.occ), args.dt)
        mm = evaluate(trace, rep, reads, dt=args.dt, rosters=rosters or None)
        result["arms"][name] = mm
        print(fmt_row(name, mm))
        if args.seeds and args.jitter > 0:
            runs = []
            for sd in range(args.seeds):
                rj = replay(trace, make_tracker(field_length, zones=zones, **kw),
                            occ_fn(args.occ), args.dt, jitter_m=args.jitter, seed=sd + 1)
                runs.append(evaluate(trace, rj, reads, dt=args.dt, rosters=rosters or None))
            summ = noise_summary(runs)
            result.setdefault("noise", {})[name] = summ
            print("   " + "  ".join(f"{k} {v[0]:.3g}+-{v[1]:.2g}" for k, v in summ.items()))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
