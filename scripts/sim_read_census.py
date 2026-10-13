"""Offline read-supply simulation: per-slot bumper reads with read_census OFF vs ON.

Replays a stored full match's identity timeline through the REAL read
scheduler and read pass of web/app.py — ``_collect_due_vlm_reads`` (with
``census_read_decision``), ``_schedule_vlm_pass`` (single-flight background
task), ``read_fused_bumper_numbers`` (stamps, backoff, consecutive lock,
audits, trigger telemetry) and ``_request_immediate_read`` for the
re-attachment escalation — with only the reader faked.

Inputs (committed fixture, built once from job 1798f296 = qm11 full match,
identity_continuity ON, 326 identities):
  * the identity -> slot binding timeline (match_robot_binding rows, DB read
    read-only): births and rebinds drive the re-attachment escalation that
    queued 92% of real immediate reads, exactly as the identity sync does;
  * per processed frame (stride 8 @ 59.94 fps), each slot's field position
    from the replay meta.json (novelty gate). Crop availability is not
    recoverable (the replay draws a subset of slots), so by default every
    bound slot has a crop every frame (upper bound, same for every arm);
    --visibility replay restricts crops to drawn slots (lower bound);
  * per slot, the fraction of real reads that named a team (the fake
    reader's legibility: blue slots 5-21%, red 19-53%).

Reader latency is modelled in video seconds (the loop ran ~real-time: 166.9 s
wall for the match): a pass of k reads holds the single-flight slot for
``base + per_read * (k - 1)`` s (defaults 0.8 / 0.08: the real job spent
119 s in passes for 145 mostly-single reads, ~0.8 s each; Clef p50 0.5 s,
a k-way concurrent pass waits for its slowest request).

Arms: census OFF (today), census ON (default knobs), census ON with
read_census_honor_backoff. Per slot: vlm_read rows, reads per bound minute,
p90 / max gap between attempts while bound, named reads; plus reads paid
(reader items, incl. reads whose identity was evicted while in flight and
which therefore leave no row), passes, Clef cost and trigger counts.
Absolute counts run ~2x the real job's (no occlusion gate / crop failures
modelled); compare arms, not the real column.

Usage:
  .venv/bin/python scripts/sim_read_census.py               # 3-arm table
  .venv/bin/python scripts/sim_read_census.py --interval 3 --seed 1
  .venv/bin/python scripts/sim_read_census.py --visibility replay
  .venv/bin/python scripts/sim_read_census.py --build-fixture   # rebuild
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import math
import random
import sqlite3
import sys
import threading
import types
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

JOB_ID = "1798f296-a15e-4393-ac7b-7e92477eb63b"
FIXTURE = REPO / "data" / "eval" / "read_census" / "qm11_1798f296_timeline.json"
LABELS = ["B1", "B2", "B3", "R1", "R2", "R3"]
# Clef cost per request on the real job: 0.69 cents / 174 requests.
CLEF_USD_PER_READ = 0.0069 / 174


# --------------------------------------------------------------------------- #
# Fixture (built once from the stored job; read-only on every source)
# --------------------------------------------------------------------------- #

def _slot_of_label(label: str) -> int:
    return (0 if label[0] == "B" else 3) + int(label[1]) - 1


def build_fixture(db: Path, diagnostic: Path, meta: Path, out: Path = FIXTURE,
                  job_id: str = JOB_ID) -> dict:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT identity_id, slot, start_frame, end_frame FROM match_robot_binding "
        "WHERE job_id = ? ORDER BY start_frame, id", (job_id,)).fetchall()
    fps = 59.94
    m = json.loads(meta.read_text())
    frames = [int(f["frame_num"]) for f in m["frames"]]
    pos = {lab: [] for lab in LABELS}
    for f in m["frames"]:
        here = {r["slot"]: r.get("field_pos") for r in f["robots"]}
        for lab in LABELS:
            p = here.get(lab)
            pos[lab].append(None if p is None else [round(float(p[0]), 2),
                                                     round(float(p[1]), 2)])
    # Real reads, credited to the slot the read identity held at its frame.
    owner_at = _binding_segments(rows, frames[-1] + 1)
    reads = []
    with diagnostic.open() as fh:
        for line in fh:
            if '"vlm_read"' not in line:
                continue
            r = json.loads(line)
            if r.get("type") != "vlm_read":
                continue
            slot = next((s for s, iid, a, b in owner_at
                         if iid == r["robot_id"] and a <= r["frame"] < b), None)
            reads.append([int(r["frame"]), slot, r.get("matched_team")])
    stats = {}
    for s in range(6):
        mine = [r for r in reads if r[1] == s]
        named = [r[2] for r in mine if r[2]]
        top = max(set(named), key=named.count) if named else None
        stats[LABELS[s]] = {"attempts": len(mine), "named": len(named), "top_team": top}
    roster = None
    for line in diagnostic.open():
        if '"snapshot"' in line:
            roster = json.loads(line).get("roster")
            break
    fx = {
        "job_id": job_id, "fps": fps, "frames": frames, "pos": pos,
        "bindings": [list(r) for r in rows], "roster": roster,
        "real_reads": reads, "real_stats": stats,
        "source": {"db": "frc_scorer.db match_robot_binding (read-only)",
                   "diagnostic": diagnostic.name, "meta": "replay meta.json frames"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fx, separators=(",", ":")))
    return fx


def _binding_segments(rows, end_frame):
    """[(slot, identity, start, end)] from ordered binding rows: a new row
    moves its identity (vacating its old slot) and evicts the slot's owner."""
    owner, where, opened, segs = {}, {}, {}, []
    for iid, slot, sf, ef in rows:
        old = where.get(iid)
        if old is not None and owner.get(old) == iid:
            segs.append((old, iid, opened.pop(old), sf))
            owner.pop(old)
        prev = owner.get(slot)
        if prev is not None:
            segs.append((slot, prev, opened.pop(slot), sf))
            where.pop(prev, None)
        owner[slot], where[iid], opened[slot] = iid, slot, sf
    for slot, sf in opened.items():
        segs.append((slot, owner[slot], sf, end_frame))
    return segs


def load_fixture(path: Path = FIXTURE) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #

class _GatedReader:
    """Fake _vision_read_bumpers: blocks (on the worker thread) until the sim
    clock reaches the pass's completion frame; a crop of slot s at frame f
    names the slot's team with the slot's real named-read probability
    (deterministic per (seed, slot, frame))."""

    def __init__(self, p_named: dict, team_of: dict, seed: int):
        self.p_named, self.team_of, self.seed = p_named, team_of, seed
        self.entered = threading.Event()
        self.release = threading.Event()
        self.n_items = 0
        self.items_total = 0           # reads paid for (reader items)

    def __call__(self, crops, teams, blue, red, expected_alliances=None,
                 provider="local", **_kw):
        self.n_items = len(crops)
        self.items_total += len(crops)
        self.entered.set()
        self.release.wait()
        out = []
        for c in crops:
            _tag, s, f = c.split(":")
            s, f = int(s), int(f)
            rng = random.Random(self.seed * 1_000_003 + s * 100_003 + f)
            if rng.random() < self.p_named.get(s, 0.0) and self.team_of.get(s):
                t = self.team_of[s]
                out.append({"team_number": t, "confidence": 0.95, "raw_read": t,
                            "alliance": "blue" if s < 3 else "red"})
            else:
                out.append({"team_number": None, "confidence": 0.0, "raw_read": "none",
                            "alliance": "unknown"})
        return out


class _RowSink:
    """DiagnosticRecorder stand-in: keeps vlm_read rows (sans crop)."""

    def __init__(self):
        self.rows = []

    def record_vlm_read(self, **kw):
        kw.pop("crop_b64", None)
        self.rows.append(kw)

    def __getattr__(self, name):
        if name.startswith("record_"):
            return lambda *a, **k: None
        raise AttributeError(name)


class _Quiet:
    """identity_diagnostics stand-in (counts scheduler reasons)."""

    def __init__(self):
        self.counts = {}

    def increment(self, key, n=1):
        self.counts[key] = self.counts.get(key, 0) + n

    def record(self, *a, **k):
        pass

    def gauge(self, *a, **k):
        pass


@dataclass
class SimResult:
    census: bool
    interval: float
    honor_backoff: bool
    reader_items: int                 # reads sent to the reader (paid)
    rows: list                        # vlm_read rows: frame, robot_id, trigger, matched_team
    reads: list                       # (frame, slot, identity, trigger, team)
    bound: dict                       # slot -> list[(start_s, end_s)] bound spans
    span_s: float
    passes: int
    scheduled: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


def _make_job(fx, census: bool, interval: float, reader_name: str,
              honor_backoff: bool = False):
    import web.app as app_module
    from web.app import InferenceJob
    from web.field.identity import IdentityRegistry
    from web.field.identity_confidence import ConfidenceConfig
    from web.job_config import JobConfig

    roster = fx.get("roster") or {}
    job = InferenceJob.__new__(InferenceJob)
    job.config = JobConfig({"read_census": census, "read_census_interval_s": interval,
                            "read_census_honor_backoff": honor_backoff,
                            "team_reader": reader_name})
    job.team_reader = reader_name
    job._camera_centers_xy = {}
    job.match_blue_teams = list(roster.get("blue") or [])
    job.match_red_teams = list(roster.get("red") or [])
    job.match_teams = job.match_blue_teams + job.match_red_teams
    job.vlm_status = {}
    job._vlm_task = None
    job._claude_vision_calls = 0
    job._current_frame = 0
    job._evidence_frame = 0
    job.video_fps = fx["fps"]
    job.field_tracker = types.SimpleNamespace(last_det_to_slot={}, _id_to_slot={},
                                              binding_events=[])
    job.identity_registry = IdentityRegistry()
    job.robot_tracks = {}
    job._last_bindings = {}
    job._slot_source = {}
    job._slot_crops = {}
    job.scoring_events = []
    job.user_seed_zones_raw = []
    job._occ = {}
    job._occ_by_cam = {}
    job.confidence_config = ConfidenceConfig()
    job.gallery = None
    job.co_occlusion_tracker = types.SimpleNamespace(get_active_events=lambda: [])
    job.apply_gallery_matches = lambda crops: None
    job.identity_diagnostics = _Quiet()
    job.diagnostic = _RowSink()
    job.timings = {}
    return app_module, job


async def _run(fx, job, app_module, reader, *, base_latency_s, per_read_latency_s,
               visibility="bound"):
    from web.field.identity import RobotIdentity
    from web.field.identity_confidence import (
        IMMEDIATE_RE_ATTACHMENT, should_queue_immediate_read)

    fps = fx["fps"]
    frames = fx["frames"]
    rows = [tuple(r) for r in fx["bindings"]]
    pos = {_slot_of_label(lab): v for lab, v in fx["pos"].items()}
    reg = job.identity_registry
    owner, where = {}, {}          # slot -> identity, identity -> slot
    ends = {}                      # identity -> end_frame (explicit unbinds)
    last_seen = {}
    last_pos = {}                  # slot -> last replay position
    frame_slot_of = {}             # frame -> {identity: slot} (read attribution)
    bound_spans = {s: [] for s in range(6)}
    open_since = {}
    ri = 0
    pending_done_at = None         # sim time at which the in-flight pass completes
    passes = 0

    def alliance(slot):
        return "blue" if slot < 3 else "red"

    def evict_for_birth(all_):
        # IdentityRegistry caps: <= 3 identities per alliance; a birth evicts
        # the oldest unbound, unconfirmed identities down to 2.
        same = [i for i in reg.identities.values() if i.alliance == all_]
        while len(same) >= 3:
            victims = [i for i in same
                       if i.identity_id not in where and not i.is_confirmed]
            if not victims:
                break
            victims.sort(key=lambda i: (last_seen.get(i.identity_id, 0), i.confidence))
            v = victims[0]
            reg.identities.pop(v.identity_id)
            same.remove(v)

    for idx, frame in enumerate(frames):
        t = frame / fps
        # ---- apply the binding timeline up to this frame
        while ri < len(rows) and rows[ri][2] <= frame:
            iid, slot, _sf, ef = rows[ri]
            ri += 1
            old = where.get(iid)
            if old is not None and owner.get(old) == iid:
                owner.pop(old)
            prev = owner.get(slot)
            if prev is not None:
                where.pop(prev, None)
            if iid not in reg.identities:
                evict_for_birth(alliance(slot))
                reg.identities[iid] = RobotIdentity(identity_id=iid, alliance=alliance(slot),
                                                    born_frame=frame)
            owner[slot], where[iid] = iid, slot
            if ef is not None:
                ends[iid] = ef
        for iid, ef in list(ends.items()):
            if ef <= frame and iid in where:
                slot = where.pop(iid)
                if owner.get(slot) == iid:
                    owner.pop(slot)
                ends.pop(iid)
        for ident in reg.identities.values():
            ident.active_slot = where.get(ident.identity_id)
        for s in range(6):
            if s in owner and s not in open_since:
                open_since[s] = t
            elif s not in owner and s in open_since:
                bound_spans[s].append((open_since.pop(s), t))
        # ---- positions (novelty gate) + re-attachment escalation
        for slot, iid in owner.items():
            ident = reg.identities.get(iid)
            if ident is None:
                continue
            last_seen[iid] = frame
            p = pos[slot][idx] if pos[slot][idx] is not None else last_pos.get(slot)
            if p is not None:
                job._get_or_create_identity_track(iid, ident.alliance).field_position = tuple(p)
            if job._last_bindings.get(iid) is None and should_queue_immediate_read(
                    t, ident.last_vlm_time, ident.last_immediate_queue_time,
                    job.confidence_config.immediate_read_cooldown_sec):
                job._request_immediate_read(ident, IMMEDIATE_RE_ATTACHMENT, renew=True)
                ident.last_immediate_queue_time = t
        job._last_bindings = {iid: s for s, iid in owner.items()}
        frame_slot_of[frame] = dict(where)
        # ---- complete the in-flight pass when its latency has elapsed
        task = job._vlm_task
        if task is not None and not task.done() and pending_done_at is not None \
                and t >= pending_done_at:
            reader.release.set()
            await task
            reader.release.clear()
            pending_done_at = None
        # ---- this frame's crops: one main-camera crop per bound slot. Crop
        # availability is not reconstructible (the replay draws a subset of
        # slots: R1 is absent for 82 s while the real job read it 17 times
        # then). visibility="bound" (default, upper bound shared by every
        # arm): every bound slot has a crop, a slot the replay omits holds
        # its last position; "replay" (lower bound): only drawn slots.
        fused, det_to_slot = [], {}
        for slot in sorted(owner):
            p = pos[slot][idx]
            if p is None:
                if visibility == "replay":
                    continue           # lower bound: only slots the replay drew
                p = last_pos.get(slot)
            else:
                last_pos[slot] = p
            if p is None:
                continue
            det_to_slot[len(fused)] = slot
            fused.append(types.SimpleNamespace(
                evidence=[("main", f"crop:{slot}:{frame}", 1.0, tuple(p))],
                detection=types.SimpleNamespace(pos=tuple(p)), track_id=slot))
        job.field_tracker.last_det_to_slot = det_to_slot
        # ---- the main loop's VLM stage
        clock = job._read_clock_sec(frame, idx)
        due = job._collect_due_vlm_reads(clock)
        if fused and due:
            before = job._vlm_task
            job._schedule_vlm_pass(fused, due, crop_frame=frame)
            if job._vlm_task is not before:
                passes += 1
                reader.entered.clear()
                new = job._vlm_task
                while not new.done() and not reader.entered.is_set():
                    await asyncio.sleep(0)
                    if not reader.entered.is_set():
                        await asyncio.sleep(0.0002)
                if not new.done():
                    k = max(1, reader.n_items)
                    pending_done_at = t + base_latency_s + per_read_latency_s * (k - 1)
        job._current_frame = frame
    if job._vlm_task is not None and not job._vlm_task.done():
        reader.release.set()
        await job._vlm_task
    end_t = frames[-1] / fps
    for s, t0 in open_since.items():
        bound_spans[s].append((t0, end_t))
    return bound_spans, passes, frame_slot_of


def simulate(fx: dict, *, census: bool, interval: float = 2.0, seed: int = 0,
             honor_backoff: bool = False, reader_name: str = "clef",
             base_latency_s: float = 0.8, per_read_latency_s: float = 0.08,
             visibility: str = "bound") -> SimResult:
    real = fx["real_stats"]
    p_named = {_slot_of_label(lab): (st["named"] / st["attempts"] if st["attempts"] else 0.0)
               for lab, st in real.items()}
    team_of = {_slot_of_label(lab): st["top_team"] for lab, st in real.items()}
    app_module, job = _make_job(fx, census, interval, reader_name, honor_backoff)
    reader = _GatedReader(p_named, team_of, seed)
    saved = app_module._vision_read_bumpers
    app_module._vision_read_bumpers = reader
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            bound, passes, frame_slot_of = asyncio.run(_run(
                fx, job, app_module, reader, visibility=visibility,
                base_latency_s=base_latency_s,
                per_read_latency_s=per_read_latency_s))
            job._finalize_read_triggers()
    finally:
        app_module._vision_read_bumpers = saved
    fps = fx["fps"]
    # Credit each row to the slot its identity held at the crop frame (the
    # sim's own binding state at that processed frame).
    reads = []
    for r in job.diagnostic.rows:
        slot = (frame_slot_of.get(r["frame"]) or {}).get(r["robot_id"])
        reads.append((r["frame"], slot, r["robot_id"], r.get("trigger"), r.get("matched_team")))
    span = (fx["frames"][-1] - fx["frames"][0]) / fps
    return SimResult(census=census, interval=interval, honor_backoff=honor_backoff,
                     reader_items=reader.items_total, rows=job.diagnostic.rows,
                     reads=reads, bound=bound, span_s=span, passes=passes,
                     scheduled=dict(job.identity_diagnostics.counts),
                     timings=dict(job.timings))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _gaps_while_bound(times, spans):
    """Longest / 90th-pct gap (s) between consecutive read attempts of a slot,
    counting the stretch from bind to first read and last read to unbind."""
    gaps = []
    for a, b in spans:
        ts = sorted(t for t in times if a <= t <= b)
        pts = [a] + ts + [b]
        gaps += [y - x for x, y in zip(pts, pts[1:]) if y > x]
    if not gaps:
        return 0.0, 0.0
    gaps.sort()
    return gaps[-1], gaps[min(len(gaps) - 1, int(math.ceil(0.9 * len(gaps))) - 1)]


def summarize(res: SimResult, fps: float) -> dict:
    per = {}
    for s in range(6):
        times = [f / fps for f, slot, *_ in res.reads if slot == s]
        named = sum(1 for _f, slot, _i, _t, team in res.reads if slot == s and team)
        bound_s = sum(b - a for a, b in res.bound[s])
        mx, p90 = _gaps_while_bound(times, res.bound[s])
        per[LABELS[s]] = {"reads": len(times), "named": named,
                          "per_min": len(times) / bound_s * 60 if bound_s else 0.0,
                          "max_gap_s": mx, "p90_gap_s": p90, "bound_s": bound_s}
    trig = {}
    for _f, _s, _i, t, _team in res.reads:
        trig[t] = trig.get(t, 0) + 1
    total = len(res.reads)
    return {"per_slot": per, "total": total,
            "named": sum(1 for r in res.reads if r[4]),
            "per_s": total / res.span_s if res.span_s else 0.0,
            "paid": res.reader_items, "passes": res.passes, "triggers": trig,
            "usd": res.reader_items * CLEF_USD_PER_READ}


def format_report(fx, arms: list, interval: float) -> str:
    """arms: [(name, summary)]; the first is the baseline."""
    real = fx["real_stats"]
    span = (fx["frames"][-1] - fx["frames"][0]) / fx["fps"]
    out = [f"qm11 full match {fx['job_id'][:8]}: {span:.0f} s, {len(fx['frames'])} processed "
           f"frames, {len({b[0] for b in fx['bindings']})} identities; census interval "
           f"{interval:g} s",
           "per slot: rows (reads/min bound) p90/max gap s between attempts while bound, named",
           f"{'slot':4} {'real':>4} | " + " | ".join(f"{n:^27}" for n, _ in arms),
           "-" * (12 + 30 * len(arms))]
    for lab in LABELS:
        cells = []
        for _n, a in arms:
            p = a["per_slot"][lab]
            cells.append(f"{p['reads']:>3} ({p['per_min']:>4.1f}/m) "
                         f"{p['p90_gap_s']:>4.1f}/{p['max_gap_s']:<4.1f} {p['named']:>3}")
        out.append(f"{lab:4} {real[lab]['attempts']:>4} | " + " | ".join(cells))
    real_total = sum(st["attempts"] for st in real.values())
    out.append(f"{'all':4} {real_total:>4} | " + " | ".join(
        f"{a['total']:>3} ({a['per_s']:.2f}/s) named {a['named']:<3}      " for _n, a in arms))
    for n, a in arms:
        trig = ", ".join(f"{k}={v}" for k, v in sorted(a["triggers"].items(),
                                                        key=lambda kv: -kv[1]))
        out.append(f"{n}: {a['paid']} reads paid ({a['paid'] - a['total']} dropped: identity "
                   f"evicted in flight), {a['passes']} passes, Clef ~${a['usd']:.4f}; "
                   f"triggers {trig}")
    return "\n".join(out)


def run_arms(fx, interval=2.0, seed=0, **kw):
    arms = [("OFF", dict(census=False)),
            ("census ON", dict(census=True)),
            ("census ON + honor_backoff", dict(census=True, honor_backoff=True))]
    return [(n, summarize(simulate(fx, interval=interval, seed=seed, **a, **kw), fx["fps"]))
            for n, a in arms]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixture", type=Path, default=FIXTURE)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reader", default="clef")
    ap.add_argument("--base-latency", type=float, default=0.8)
    ap.add_argument("--per-read-latency", type=float, default=0.08)
    ap.add_argument("--visibility", choices=("bound", "replay"), default="bound",
                    help="bound: every bound slot has a crop every frame (upper "
                         "bound); replay: only slots the replay drew (lower bound)")
    ap.add_argument("--build-fixture", action="store_true")
    ap.add_argument("--db", type=Path, default=Path("/home/nick/AI/frc-vision-scorer/frc_scorer.db"))
    ap.add_argument("--diagnostic", type=Path, default=Path(
        f"/home/nick/AI/frc-vision-scorer/web/uploads/diagnostic_{JOB_ID}.jsonl"))
    ap.add_argument("--meta", type=Path, default=Path(
        f"/home/nick/AI/frc-vision-scorer/.worktrees/main-server/web/replays/{JOB_ID}/meta.json"))
    args = ap.parse_args(argv)
    if args.build_fixture:
        fx = build_fixture(args.db, args.diagnostic, args.meta, args.fixture)
        print(f"wrote {args.fixture} ({len(fx['frames'])} frames, "
              f"{len(fx['bindings'])} binding rows, {len(fx['real_reads'])} real reads)")
        return 0
    fx = load_fixture(args.fixture)
    arms = run_arms(fx, interval=args.interval, seed=args.seed, reader_name=args.reader,
                    base_latency_s=args.base_latency,
                    per_read_latency_s=args.per_read_latency,
                    visibility=args.visibility)
    print(format_report(fx, arms, args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
