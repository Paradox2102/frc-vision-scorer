"""Live staggered dispatch: each new match video -> the idle pod, as it appears.

    swarm live --name N --event EV [--local] [--poll-s 60] [--auto-import] [--reprocess-replayed]
    swarm live --name N --event EV --rehearse [--speed 1.0] [--matches qm1-qm30] [--local]

A poller watches TBA (``/event/{key}/matches``) for match videos. A new video is
downloaded HERE (home IP: YouTube blocks datacenter IPs) and added to an open
run whose ``live`` policy gives it to the pod idle longest (never two queued on
one pod while another is idle; with every pod busy, the pod that frees first
takes it). After every job the pod's new calibration / layout / zones /
season files are copied to the other ready pods (atomic tmp + mv), so a
camera move solved on one pod is known to all. Per match the run records:
video seen -> downloaded -> dispatched -> final replay; ``--auto-import``
pulls + imports each finished match into the local app (one DB backup per
session; skipped while a local job runs).

``--rehearse`` replays a PAST event: arrival times are the event's actual TBA
match times rebased to now (``--speed`` x faster), videos come from local
files instead of polling; everything else is identical. Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import swarm

cli = swarm.cli

TBA = "https://www.thebluealliance.com/api/v3"
UA = "frc-swarm-live/1.0"
SYNC_PUSH_TRIES = 3


# --------------------------------------------------------------------------
# Arrival sources (pure state + injectable fetch / clock)
# --------------------------------------------------------------------------
def tba_fetcher(api_key: str | None = None, timeout: float = 20):
    """fetch(path) -> parsed JSON from TBA v3 (stdlib urllib)."""
    key = api_key or os.environ.get("TBA_API_KEY") or cli.local_env().get("TBA_API_KEY")
    if not key:
        raise SystemExit("TBA_API_KEY missing (main checkout .env)")

    def fetch(path: str):
        req = urllib.request.Request(f"{TBA}{path}", headers={"X-TBA-Auth-Key": key, "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    return fetch


def parse_matches(rows: list) -> list:
    """TBA match rows -> [{match_key, comp_level, rank, videos, actual_time,
    post_result_time}] in schedule order (videos = YouTube ids)."""
    out = []
    for r in rows or []:
        lvl = r.get("comp_level")
        if not r.get("key") or lvl not in swarm.LEVEL_RANK:
            continue
        vids = [v.get("key") for v in (r.get("videos") or []) if v.get("type") == "youtube" and v.get("key")]
        out.append({"match_key": r["key"], "comp_level": lvl,
                    "rank": swarm.match_rank(lvl, r.get("set_number"), r.get("match_number") or 0),
                    "videos": vids, "actual_time": r.get("actual_time"),
                    "post_result_time": r.get("post_result_time")})
    out.sort(key=lambda m: m["rank"])
    return out


class TbaSource:
    """Diffs successive TBA polls. ``seen`` (persisted in the run) maps
    match_key -> {"videos": [ids], "first_seen", "n"}. A match whose video
    list gains a NEW id (a replay re-posted under the same key) is a new
    arrival "<mk>@r<n>" only with ``reprocess_replayed``; otherwise it is
    logged once and ignored."""

    def __init__(self, event: str, fetch, seen: dict, *, reprocess_replayed: bool = False, log=print):
        self.event, self.fetch, self.seen = event, fetch, seen
        self.reprocess = reprocess_replayed
        self.log = log
        self.exhausted = False

    def poll(self, now: float) -> list:
        try:
            rows = parse_matches(self.fetch(f"/event/{self.event}/matches"))
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.log(f"[tba] poll failed (retried next time): {e}")
            return []
        new = []
        for r in rows:
            if not r["videos"]:
                continue
            mk = r["match_key"]
            s = self.seen.get(mk)
            if s is None:
                self.seen[mk] = {"videos": list(r["videos"]), "first_seen": now, "n": 1}
                new.append({"key": mk, "match_key": mk, "video_key": r["videos"][0], "available_at": now,
                            "actual_time": r["actual_time"], "post_result_time": r["post_result_time"]})
                continue
            fresh = [v for v in r["videos"] if v not in s["videos"]]
            if not fresh:
                continue
            s["videos"] += fresh
            if not self.reprocess:
                if not s.get("ignored"):
                    self.log(f"[tba] {mk}: new video {fresh[0]} (replayed match?) ignored; "
                             f"--reprocess-replayed to run it")
                s["ignored"] = True
                continue
            s["n"] = s.get("n", 1) + 1
            key = f"{mk}@r{s['n']}"
            new.append({"key": key, "match_key": mk, "video_key": fresh[-1], "available_at": now,
                        "actual_time": r["actual_time"], "post_result_time": r["post_result_time"],
                        "replay_of": mk})
        return new


class RehearsalSource:
    """A past event at its real cadence: match i arrives at
    ``t0_real + (actual_time_i - actual_time_first) / speed``."""

    def __init__(self, schedule: list, t0_real: float, speed: float, emitted: list):
        if speed <= 0:
            raise SystemExit("--speed must be > 0")
        self.sched = sorted((m for m in schedule if m.get("actual_time")), key=lambda m: m["actual_time"])
        self.t0_real, self.speed = t0_real, speed
        self.t0_event = self.sched[0]["actual_time"] if self.sched else 0
        self.emitted = emitted              # persisted in the run: keys already released

    def due_at(self, m: dict) -> float:
        return self.t0_real + (m["actual_time"] - self.t0_event) / self.speed

    def poll(self, now: float) -> list:
        out = []
        for m in self.sched:
            if m["match_key"] in self.emitted or self.due_at(m) > now:
                continue
            self.emitted.append(m["match_key"])
            out.append({"key": m["match_key"], "match_key": m["match_key"], "video": m["video"],
                        "available_at": self.due_at(m), "actual_time": m["actual_time"]})
        return out

    @property
    def exhausted(self) -> bool:
        return len(self.emitted) >= len(self.sched)

    def next_due(self) -> float | None:
        left = [self.due_at(m) for m in self.sched if m["match_key"] not in self.emitted]
        return min(left) if left else None


def rehearsal_schedule(rows: list, spec: str, video_for) -> tuple[list, list]:
    """Matches of ``spec`` that have an actual time and a local video ->
    (schedule, missing)."""
    chosen = swarm.select_matches(parse_matches(rows), spec)
    sched, missing = [], []
    for m in chosen:
        if not m.get("actual_time"):
            missing.append({"match_key": m["match_key"], "why": "no TBA actual_time"})
            continue
        v = video_for(m["match_key"])
        if v is None:
            missing.append({"match_key": m["match_key"], "why": "no local video"})
            continue
        sched.append({"match_key": m["match_key"], "actual_time": m["actual_time"], "video": str(v)})
    return sched, missing


# --------------------------------------------------------------------------
# Arrival loop
# --------------------------------------------------------------------------
def arrival_step(ctx: swarm.RunCtx, source, *, now: float, cache_dir: Path, downloader=None) -> list:
    """One poll: new arrivals -> local video (download when needed) ->
    add_items. Timestamps go into the item: t_available (video first seen /
    rehearsal due time), t_seen (poll), t_downloaded. Returns added keys."""
    sw, run = ctx.sw, ctx.run
    added = []
    for a in source.poll(now):
        key = a["key"]
        if key in run["items"]:
            continue
        video = a.get("video")
        t_dl0 = swarm._now()
        if not video:
            dest = Path(cache_dir) / f"{swarm.safe_key(key)}.mp4"
            if not (dest.is_file() and dest.stat().st_size > 0):
                ctx.event(f"[live] {key}: downloading youtube {a['video_key']} ...")
                try:
                    dest = Path((downloader or swarm.default_downloader)(a["video_key"], dest))
                except Exception as e:
                    ctx.event(f"[live] {key}: download FAILED ({str(e)[:160]}); will retry next poll")
                    _forget(source, a)
                    continue
            video = str(dest)
        p = Path(video)
        if not p.is_file():
            ctx.event(f"[live] {key}: video {video} missing; skipped")
            continue
        item = {"key": key, "match_key": a["match_key"], "video": str(p.resolve()), "size": p.stat().st_size,
                "event": run.get("event"), "video_key": a.get("video_key")}
        with sw.lock:
            got = swarm.add_items(run, [item])
            if got:
                run["items"][key].update(t_available=a["available_at"], t_seen=now, t_download_start=t_dl0,
                                         t_downloaded=swarm._now(), actual_time=a.get("actual_time"),
                                         replay_of=a.get("replay_of"))
            sw.save()
        if got:
            added.append(key)
            ctx.event(f"[live] {key}: queued (video ready {swarm._fmt_min(swarm._now() - a['available_at'])} "
                      f"after it appeared)")
    return added


def _forget(source, a: dict) -> None:
    """A failed download: let the next poll see the match again."""
    if isinstance(source, TbaSource):
        s = source.seen.get(a["match_key"])
        if s and a.get("replay_of") is None:
            source.seen.pop(a["match_key"], None)
        elif s:
            s["videos"] = [v for v in s["videos"] if v != a.get("video_key")]
            s["n"] -= 1
    elif isinstance(source, RehearsalSource) and a["match_key"] in source.emitted:
        source.emitted.remove(a["match_key"])


def arrival_loop(ctx: swarm.RunCtx, source, *, poll_s: float, cache_dir: Path, until: float | None,
                 downloader=None, clock=None) -> None:
    now_fn = clock.now if clock else (lambda: swarm._now())
    sleep = clock.sleep if clock else ctx.sleep
    run, sw = ctx.run, ctx.sw
    while not ctx.stop.is_set():
        now = now_fn()
        try:
            arrival_step(ctx, source, now=now, cache_dir=cache_dir, downloader=downloader)
        except Exception as e:                  # the poller must not die mid-event
            ctx.event(f"[live] arrival poll error: {type(e).__name__}: {str(e)[:200]}")
        if source.exhausted or (until is not None and now_fn() >= until):
            with sw.lock:
                run["open"] = False
                sw.save()
            ctx.event("[live] no more arrivals: the run closes once the queued matches finish")
            return
        wait = poll_s
        nd = getattr(source, "next_due", lambda: None)()
        if nd is not None:
            wait = max(0.5, min(poll_s, nd - now_fn()))
        sleep(wait)


# --------------------------------------------------------------------------
# Calibration sync
# --------------------------------------------------------------------------
def local_event_state(uploads: Path) -> dict:
    """{name: sha1} of the local syncable event-state files: what every pod
    started from (the bundle ships them), so only later changes sync."""
    out = {}
    if Path(uploads).is_dir():
        for p in sorted(Path(uploads).iterdir()):
            if p.is_file() and swarm.sync_selectable(p.name):
                out[p.name] = hashlib.sha1(p.read_bytes()).hexdigest()
    return out


class EventStateSync:
    """After a job on member ``m``: fetch the event-state files it changed
    (vs the known canonical set) and push every file that differs from the
    starting state to each other ready pod that doesn't have that version
    yet (a pod added mid-event gets the backlog; a failed push is retried
    after the next job). Last writer wins on a file two pods both changed."""

    def __init__(self, sw: swarm.Swarm, run: dict, stage: Path):
        self.sw, self.run, self.stage = sw, run, Path(stage)
        self.lock = threading.Lock()
        self.live = run.setdefault("live", {})
        self.live.setdefault("baseline", dict(self.live.get("event_state") or {}))
        self.live.setdefault("event_state", dict(self.live["baseline"]))
        self.live.setdefault("delivered", {})
        self.live.setdefault("sync_log", [])

    def targets(self, src: str) -> list:
        gone = set(self.run["dead"]) | set(self.run["retired"])
        out = []
        for p in self.run["queues"]:
            if p == src or p in gone:
                continue
            try:
                m = self.sw.member(p)
            except KeyError:
                continue
            if m.get("status") == "ready" and swarm.is_pod(m):
                out.append(m)
        return out

    def __call__(self, ctx, m: dict, key: str, ok: bool, state: dict) -> list:
        if not swarm.is_pod(m):
            return []
        with self.lock:
            src = m["pod_id"]
            T = swarm.transport(m)
            try:
                have = T.list_event_state(m)
            except (swarm.Transport, swarm.RemoteError) as e:
                ctx.event(f"[sync] {src}: could not list event state ({e}); skipped")
                return []
            known, base = self.live["event_state"], self.live["baseline"]
            delivered = self.live["delivered"]
            changed = sorted(n for n, h in have.items() if known.get(n) != h)
            if changed:
                try:
                    got = T.fetch_files(m, changed, self.stage)
                except (swarm.Transport, swarm.RemoteError) as e:
                    ctx.event(f"[sync] {src}: fetch failed ({e}); retried after its next job")
                    got = []
                for n in got:
                    known[n] = have[n]
                changed = got
            delivered.setdefault(src, {}).update(have)
            synced = sorted(n for n, h in known.items() if base.get(n) != h and (self.stage / n).is_file())
            pushed = []
            for t in self.targets(src):
                d = delivered.setdefault(t["pod_id"], {})
                names = [n for n in synced if d.get(n) != known[n]]
                if not names:
                    continue
                err = None
                for _ in range(SYNC_PUSH_TRIES):
                    try:
                        swarm.transport(t).push_files(t, [self.stage / n for n in names])
                        err = None
                        break
                    except (swarm.Transport, swarm.RemoteError) as e:
                        err = e
                if err is None:
                    for n in names:
                        d[n] = known[n]
                    pushed.append(t["pod_id"])
                else:
                    ctx.event(f"[sync] push to {t['pod_id']} failed ({err}); retried after the next job")
            if changed or pushed:
                self.live["sync_log"].append({"at": swarm._now(), "from": src, "key": key, "files": changed,
                                              "to": pushed})
                ctx.event(f"[sync] {src} ({key}) changed {', '.join(changed) or 'nothing new'} -> pushed to "
                          f"{', '.join(pushed) or 'no other pod'}")
            with self.sw.lock:
                self.sw.save()
            return changed


# --------------------------------------------------------------------------
# Auto-import
# --------------------------------------------------------------------------
class AutoImporter:
    """Pull + import each finished match into the local app. One DB backup
    per live session; while a local job runs the import waits (retried after
    the next finished match)."""

    def __init__(self, sw: swarm.Swarm, *, active_job=swarm.local_active_job, pull=None, out=print):
        main = cli.main_checkout()
        self.sw = sw
        self.local_db, self.replays = main / "frc_scorer.db", main / "web" / "replays"
        self.uploads = main / "web" / "uploads"
        self.active_job, self.pull = active_job, pull or swarm.pull_member
        self.backup = None
        self.lock = threading.Lock()

    def __call__(self, ctx, m: dict, key: str, ok: bool, state: dict) -> int:
        if not ok or not swarm.is_pod(m):
            return 0
        with self.lock:
            try:
                self.pull(self.sw, m)
            except (Exception, SystemExit) as e:
                ctx.event(f"[import] pull {m['pod_id']} failed: {e}")
                return 0
            imp = swarm.plan_import(self.sw, local_db=self.local_db, replays_dir=self.replays,
                                    pull_root=cli.PULL_ROOT, uploads_dir=self.uploads)
            add = [e for e in imp["entries"] if e["action"] == "add"]
            if not add:
                return 0
            jid = self.active_job()
            if jid:
                ctx.event(f"[import] local app busy with job {jid}: {len(add)} replay(s) wait for the next match")
                return 0
            if self.backup is None:
                self.backup = swarm.backup_db(self.local_db, f"{self.sw.name}-live")
                ctx.event(f"[import] backup (once per session): {self.backup}")
            n = swarm.apply_import(imp["entries"], local_db=self.local_db, replays_dir=self.replays)
            ctx.event(f"[import] {n} replay(s) imported ({', '.join(e.get('match_key') or e['id'] for e in add)})")
            return n


# --------------------------------------------------------------------------
# Latency report (pure)
# --------------------------------------------------------------------------
def _q(xs: list, q: float):
    xs = sorted(xs)
    if not xs:
        return None
    i = (len(xs) - 1) * q
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def latency_rows(run: dict) -> list:
    rows = []
    for key, it in run["items"].items():
        t0 = it.get("t_available")
        if t0 is None:
            continue

        def d(k):
            v = it.get(k)
            return None if v is None else v - t0
        rows.append({"key": key, "pod": it.get("pod"), "status": it["status"], "t_available": t0,
                     "download_s": d("t_downloaded"), "dispatch_s": d("assigned_at"),
                     "launch_s": d("launched_at"),
                     "final_s": d("finished_at") if it["status"] == "done" else None,
                     "job_s": it.get("wall_final_s")})
    rows.sort(key=lambda r: r["t_available"])
    return rows


def latency_summary(run: dict, now: float | None = None) -> dict:
    now = swarm._now() if now is None else now
    rows = latency_rows(run)
    fin = [r["final_s"] for r in rows if r["final_s"] is not None]
    disp = [r["dispatch_s"] for r in rows if r["dispatch_s"] is not None]
    t_start = now if run.get("started_at") is None else run["started_at"]
    span = max(1e-9, now - t_start)
    util = {}
    for p in run["queues"]:
        busy = 0.0
        for it in run["items"].values():
            if it.get("pod") != p or it.get("assigned_at") is None:
                continue
            end = it.get("finished_at") if it["status"] in ("done", "failed", "retry") else now
            if end is None or end < it["assigned_at"]:
                end = now
            busy += max(0.0, min(end, now) - max(it["assigned_at"], t_start))
        util[p] = busy / span
    return {"n": len(rows), "n_done": len(fin), "latency_p50_s": _q(fin, 0.5), "latency_p90_s": _q(fin, 0.9),
            "latency_max_s": max(fin) if fin else None, "wait_p50_s": _q(disp, 0.5),
            "wait_p90_s": _q(disp, 0.9), "utilization": util, "span_s": span}


def live_table(sw: swarm.Swarm, now: float | None = None) -> str:
    run = sw.data["run"]
    now = swarm._now() if now is None else now
    fm = swarm._fmt_min
    lines = [f"{'match':<22} {'pod':<16} {'status':<8} {'dl':>5} {'disp':>5} {'final':>6} {'job':>5}"]
    for r in latency_rows(run)[-25:]:
        lines.append(f"{r['key']:<22} {str(r['pod'] or '-'):<16} {r['status']:<8} {fm(r['download_s']):>5} "
                     f"{fm(r['dispatch_s']):>5} {fm(r['final_s']):>6} {fm(r['job_s']):>5}")
    s = latency_summary(run, now)
    util = " ".join(f"{p}:{100 * u:.0f}%" for p, u in s["utilization"].items())
    nxt = swarm.predict_free(run, now)
    lines.append(f"latency (video seen -> final) p50 {fm(s['latency_p50_s'])} p90 {fm(s['latency_p90_s'])} "
                 f"max {fm(s['latency_max_s'])} | wait to dispatch p50 {fm(s['wait_p50_s'])} | "
                 f"{s['n_done']}/{s['n']} done | waiting {len(run.get('unassigned') or [])} | "
                 f"next free {nxt or '-'} | util {util}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
def cmd_live(args, *, fetch=None, downloader=None) -> int:
    sw = swarm.Swarm.load(swarm.check_name(args.name), create=True)
    with swarm.run_lock(sw):
        return _cmd_live(args, sw, fetch=fetch, downloader=downloader)


def _cmd_live(args, sw: swarm.Swarm, *, fetch=None, downloader=None) -> int:
    fetch = fetch or tba_fetcher()
    pods = swarm.run_pods(sw, args)
    if not pods:
        raise SystemExit("no ready members (`swarm up`), or --local")
    main = cli.main_checkout()
    cfg = json.loads(args.config_json) if args.config_json else {}
    mode = "rehearse" if args.rehearse else "tba"
    run = sw.data.get("run")
    plan = sw.data.get("plan") or {}
    resume = (run and run.get("policy") == "live" and plan.get("kind") == "live"
              and plan.get("event") == args.event and (run.get("live") or {}).get("mode") == mode
              and not swarm.run_finished(run))
    with sw.lock:
        if resume:
            for p in pods:
                swarm.add_pod(run, p)
            print(f"resuming live run: {json.dumps(swarm.run_counts(run))}")
        else:
            if swarm._run_active(run):
                raise SystemExit("the previous run still has jobs in flight; finish it first")
            if run:
                sw.data.setdefault("run_history", []).append(run)
            plan = {"id": f"live-{args.event}-{int(swarm._now())}", "kind": "live", "event": args.event,
                    "spec": mode, "items": [], "skipped": [], "missing": [], "created_at": swarm._now()}
            restart = bool(args.restart_each)
            run = swarm.new_run(plan, pods, cfg, policy="live", restart_each=restart)
            run["open"] = True
            base = local_event_state(main / "web" / "uploads")
            run["live"] = {"mode": mode, "speed": args.speed, "seen": {}, "emitted": [],
                           "t0_real": swarm._now(), "baseline": base, "event_state": dict(base)}
            sw.data["plan"], sw.data["run"] = plan, run
            print(f"new live run on {len(pods)} member(s) ({', '.join(pods)}); "
                  f"{'rehearsal x' + str(args.speed) if args.rehearse else 'polling TBA every ' + str(args.poll_s) + ' s'}")
        sw.save()
    live = run["live"]
    if args.rehearse:
        dirs = swarm.video_dirs([args.videos] if args.videos else [])
        planned = {}            # videos of earlier (batch) plans / runs of this swarm
        for i in (sw.data.get("plan") or {}).get("items") or []:
            planned.setdefault(i["match_key"], i.get("video"))
        for hist in sw.data.get("run_history") or []:
            for k, it in (hist.get("items") or {}).items():
                planned.setdefault(it.get("match_key") or k, it.get("video"))

        def video_for(mk):
            v = planned.get(mk)
            if v and Path(v).is_file():
                return Path(v)
            return swarm.find_video(mk, dirs)
        sched, missing = rehearsal_schedule(fetch(f"/event/{args.event}/matches"), args.matches, video_for)
        for m in missing:
            print(f"  rehearsal skips {m['match_key']}: {m['why']}")
        if not sched:
            raise SystemExit("rehearsal: no match with an actual time and a local video")
        source = RehearsalSource(sched, live["t0_real"], args.speed, live["emitted"])
        span = (sched[-1]["actual_time"] - sched[0]["actual_time"]) / args.speed
        print(f"rehearsal: {len(sched)} match(es) over {swarm._fmt_min(span)} (x{args.speed})")
    else:
        source = TbaSource(args.event, fetch, live["seen"], reprocess_replayed=args.reprocess_replayed)
    wargs = argparse.Namespace(**vars(args))
    wargs.poll_s = args.job_poll_s
    wargs.terminate_when_done = False
    ctx = swarm.RunCtx(sw, wargs)
    hooks = [EventStateSync(sw, run, swarm.SWARM_DIR / sw.name / "event_state")]
    if args.auto_import and not args.local:
        hooks.append(AutoImporter(sw))
    elif args.auto_import:
        print("--auto-import ignored with --local (replays are already in the local DB)")

    def on_done(c, m, key, ok, state):
        for h in hooks:
            try:
                h(c, m, key, ok, state)
            except Exception as e:
                c.event(f"[live] {type(h).__name__} failed: {type(e).__name__}: {str(e)[:200]}")
    ctx.on_done = on_done
    until = swarm._now() + args.hours * 3600 if args.hours else None
    cache = Path(args.cache_dir) if args.cache_dir else main / "data" / "videos"
    poller = threading.Thread(target=arrival_loop, args=(ctx, source),
                              kwargs={"poll_s": args.poll_s, "cache_dir": cache, "until": until,
                                      "downloader": downloader}, daemon=True)
    rc = swarm.drive_run(sw, wargs, run, pods, f"swarm live --name {sw.name} --event {args.event}"
                         + (" --rehearse" if args.rehearse else ""), ctx=ctx,
                         table=lambda: live_table(sw), side_threads=[poller])
    rep = {"summary": latency_summary(run), "rows": latency_rows(run), "sync": live.get("sync_log", [])}
    outd = cli.PULL_ROOT / "live" / sw.name
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "latency.json").write_text(json.dumps(rep, indent=1))
    print(f"latency report -> {outd / 'latency.json'}")
    return rc


def add_parser(ss, named) -> None:
    q = named("live", "LIVE staggered dispatch: TBA video poller -> idle pod (or --rehearse a past event)")
    q.add_argument("--event", required=True)
    q.add_argument("--poll-s", type=float, default=60, help="TBA poll interval (s)")
    q.add_argument("--job-poll-s", type=float, default=20, help="job status poll interval (s)")
    q.add_argument("--config-json", help="knobs for every job")
    q.add_argument("--reprocess-replayed", action="store_true",
                   help="a match re-posted with a NEW video id (replayed) runs again as <mk>@r2")
    q.add_argument("--auto-import", action="store_true",
                   help="pull + import each finished match into the local app (one DB backup per session)")
    q.add_argument("--hours", type=float, default=None, help="stop polling after this long (then drain)")
    q.add_argument("--cache-dir", help="download dir (default main checkout data/videos)")
    q.add_argument("--rehearse", action="store_true", help="replay a PAST event at its real match cadence")
    q.add_argument("--speed", type=float, default=1.0, help="rehearsal speed factor")
    q.add_argument("--matches", default="all", help="rehearsal: match spec (e.g. qm1-qm30)")
    q.add_argument("--videos", help="rehearsal: dir of <match_key>.mp4 (else plan / data/videos)")
    swarm.add_member_flags(q)
    q.set_defaults(fn=cmd_live)


