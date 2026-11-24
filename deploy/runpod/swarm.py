"""Swarm: rent N cheap GPU pods and batch-process every match of a past event.

Why: a full match is serial-Python bound (Nov 13, RTX PRO 6000: GPU ~27% busy,
~3.5 CPU cores, two jobs in one server only 1.2x), so the scaling answer is
one job per server and many cheap servers. Each pod is a normal single-pod
deployment (same bundle, secrets, bootstrap), driven by the same detached
``pod_job.py`` as ``tune``. Stdlib only; wired as ``runpod_cli.py swarm <sub>``.

    swarm up        --name N --count K [--gpu G ...] [--dry-run]   rent + ship + bootstrap
    swarm plan      --name N --event EV [--matches SPEC] [--download]
    swarm run       --name N [--config-json JSON] [--terminate-when-done]
    swarm status    --name N
    swarm pull      --name N [--with-video]
    swarm import    --name N [--apply] [--render] [--gdrive] [--gdrive-video]
    swarm stop|terminate --name N [--force]
    swarm sweep evalset|plan|run|score|report ...   quality sweep (sweep.py)
    swarm live --name N --event EV [--rehearse] [--local]   live staggered dispatch (live.py)

Every dispatching command takes --local (the local :8004 instead of pods)
and --restart-each (fresh app server before every job).

State: deploy/runpod/.swarm/<name>.json (gitignored), written atomically after
every transition; member logs in deploy/runpod/.swarm/<name>/<podId>.log. A
member dict is a valid single-pod ``st`` for the runpod_cli helpers, but
nothing here ever reads or writes the single-pod .state.json.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import runpod_cli as cli

SWARM_DIR = cli.HERE / ".swarm"
POD_BATCH = f"{cli.POD_WS}/batch"
POD_VIDEOS = f"{POD_BATCH}/videos"
POD_JOBS = f"{POD_BATCH}/jobs"
LOCAL_APP = "http://127.0.0.1:8004"

# Cheap 48 GB cards: the pipeline is CPU / serial-Python bound, so a bigger
# GPU barely helps (README "Swarm"). gpuTypeIds from the RunPod REST enum.
DEFAULT_SWARM_GPUS = ["NVIDIA L40S", "NVIDIA RTX A6000", "NVIDIA A40",
                      "NVIDIA RTX 6000 Ada Generation"]
# Secure-cloud $/GPU-h used only when the live price query fails (README, Nov 12).
FALLBACK_PRICES = {"NVIDIA A40": 0.59, "NVIDIA RTX A6000": 0.69, "NVIDIA L40S": 1.09,
                   "NVIDIA RTX 6000 Ada Generation": 0.99, "NVIDIA A100 80GB PCIe": 2.09,
                   "NVIDIA RTX PRO 6000 Blackwell Server Edition": 2.69,
                   "NVIDIA H100 PCIe": 2.89, "NVIDIA H100 80GB HBM3": 4.39}
MAX_ATTEMPTS = 2                 # a failed match is retried once (on another pod if possible)
FINAL_TIMEOUT_S = 2400           # pod_job --final-timeout (covers a ws drop mid-job)
LOST_AFTER_S = 600               # ssh failing this long -> ask the API whether the pod is gone
DEFAULT_JOB_MIN = 7.5            # ETA guess before any job finished (Nov 13 qm8: 6-7.4 min)
LEVEL_RANK = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,40}$")
_MATCH_RE = re.compile(r"^(?:[A-Za-z0-9]+_)?(qm|ef|qf|sf|f)(\d+)(?:m(\d+))?$")
LIVE_STALE_S = 300               # live policy: an idle pod that hasn't asked for work this long is skipped
LIVE_STATUSES = ("created", "ssh_wait", "pushing", "secrets", "bootstrapping", "ready", "setup_failed")
BILLING_STATUSES = LIVE_STATUSES + ("lost",)     # "lost" may still bill: include it in stop/pull/burn


def _now() -> float:
    return time.time()


def check_name(name: str) -> str:
    if not name or not _NAME_RE.match(name):
        raise SystemExit(f"bad swarm name {name!r} (letters, digits, - and _)")
    return name


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
class Swarm:
    """The swarm state file + a lock shared by the member threads."""

    def __init__(self, data: dict, path: Path):
        self.data = data
        self.path = Path(path)
        self.lock = threading.RLock()

    @classmethod
    def path_for(cls, name: str) -> Path:
        return SWARM_DIR / f"{check_name(name)}.json"

    @classmethod
    def load(cls, name: str, *, create: bool = False) -> "Swarm":
        p = cls.path_for(name)
        if p.exists():
            return cls(json.loads(p.read_text()), p)
        if not create:
            raise SystemExit(f"no swarm {name!r} ({p}); start one with `swarm up --name {name} --count K`")
        return cls({"name": name, "created_at": _now(), "gpu_priority": None, "cloud": None,
                    "members": [], "create_errors": [], "plan": None, "run": None}, p)

    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def members(self) -> list:
        return self.data["members"]

    def member(self, pod_id: str) -> dict:
        for m in self.members:
            if m["pod_id"] == pod_id:
                return m
        raise KeyError(pod_id)

    def save(self) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            for _ in range(5):          # a member thread may be mid-update of its dict
                try:
                    text = json.dumps(self.data, indent=1)
                    break
                except RuntimeError:
                    time.sleep(0.01)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(text)
            os.replace(tmp, self.path)

    def update(self, m: dict, **kw) -> None:
        with self.lock:
            m.update(kw)
            self.save()

    def saver(self, _m: dict = None):
        """``save`` callback for runpod_cli.refresh_endpoint (members are
        dicts inside self.data, so saving the swarm saves them)."""
        return lambda _st: self.save()

    def log_dir(self) -> Path:
        return self.path.parent / self.name

    def logger(self, m: dict):
        path = self.log_dir() / f"{m['pod_id']}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = threading.Lock()

        def out(msg: str) -> None:
            with lock, open(path, "a") as f:
                f.write(f"{_dt.datetime.now().strftime('%H:%M:%S')} {msg}\n")
        out.path = path
        return out


def live_members(sw: Swarm) -> list:
    """Pod members that exist (the local backend member is never one)."""
    return [m for m in sw.members if m.get("status") in LIVE_STATUSES and is_pod(m)]


def ready_members(sw: Swarm) -> list:
    return [m for m in sw.members if m.get("status") == "ready" and is_pod(m)]


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------
def gpu_prices(gpus: list, cloud: str, fetch=None) -> dict:
    """gpuTypeId -> $/GPU-h for ``cloud`` (live gpu_types, else FALLBACK_PRICES)."""
    rows = None
    try:
        rows = (fetch or cli.gpu_types)()
    except (Exception, SystemExit):
        rows = None
    out = {}
    by = {r.get("id"): r for r in (rows or [])}
    for g in gpus:
        r = by.get(g)
        price = None
        if r:
            price = r.get("securePrice") if cloud == "SECURE" else r.get("communityPrice")
        out[g] = float(price) if price else FALLBACK_PRICES.get(g)
    return out


def cost_estimate(count: int, prices: dict) -> tuple[str, float, float]:
    known = [p for p in prices.values() if p]
    if not known:
        return f"{count} pods x $?/h (no price for {', '.join(prices)})", 0.0, 0.0
    first = next((p for p in prices.values() if p), None)
    lo, hi = min(known), max(known)
    rng = f"${lo:.2f}" if lo == hi else f"${lo:.2f}-{hi:.2f}"
    return (f"{count} pods x {rng}/h (first choice ${first:.2f}/h) ~= ${count * first:.2f}/h total "
            f"(up to ${count * hi:.2f}/h)", count * first, count * hi)


def member_cost(m: dict, now: float | None = None) -> float:
    """GPU $ since creation (until terminated); storage not included."""
    rate = float(m.get("cost_per_hr") or 0)
    now = _now() if now is None else now
    t0 = now if m.get("created_at") is None else m["created_at"]
    t1 = now if m.get("terminated_at") is None else m["terminated_at"]
    return max(0.0, t1 - t0) / 3600 * rate


# --------------------------------------------------------------------------
# up
# --------------------------------------------------------------------------
def create_pods(sw: Swarm, bodies: list, request=None, out=print) -> list:
    """POST one pod per body; keep the ones created (state saved after each,
    so a crash never forgets a billing pod), report the failures."""
    request = request or cli._request
    errors = []
    for body in bodies:
        try:
            pod = request("POST", f"{cli.REST}/pods", body)
            if not pod or not pod.get("id"):
                raise RuntimeError(f"no pod id in response: {str(pod)[:200]}")
        except (Exception, SystemExit) as e:
            msg = f"{body['name']}: {str(e)[:300]}"
            errors.append(msg)
            sw.data.setdefault("create_errors", []).append({"at": _now(), "error": msg})
            sw.save()
            out(f"  create FAILED {msg}")
            continue
        m = {"pod_id": pod["id"], "name": body["name"], "idx": body.get("_idx"),
             "created_at": _now(), "gpu_priority": body["gpuTypeIds"],
             "cost_per_hr": pod.get("costPerHr"), "image": body["imageName"],
             "status": "created"}
        with sw.lock:
            sw.members.append(m)
            sw.save()
        out(f"  created pod {pod['id']} ({body['name']}) at ${pod.get('costPerHr')}/h")
    return errors


def setup_member(sw: Swarm, m: dict, tar_path: Path | None, secrets: dict | None,
                 boot_args: str, ref: str, *, wait_s: float = 900) -> None:
    """ssh -> bundle -> secrets -> bootstrap for one member, resuming from
    the last completed step. Output goes to the member's log file."""
    out = sw.logger(m)
    try:
        sw.update(m, status="ssh_wait", error=None)
        cli.wait_for_ssh(m, wait_s, save=sw.saver(m), out=out)
        if not m.get("pushed"):
            if tar_path is None:
                raise RuntimeError("bundle not built")
            sw.update(m, status="pushing")
            cli.push_tar(m, tar_path, replace_code=True, out=out)
            sw.update(m, pushed=True, pushed_ref=ref)
        if not m.get("secrets_installed"):
            if secrets is None:
                raise RuntimeError("secrets not gathered")
            sw.update(m, status="secrets")
            cli.install_secrets(m, secrets, out=out)
            sw.update(m, secrets_installed=True)
        sw.update(m, status="bootstrapping")
        cli.run_bootstrap(m, boot_args=boot_args, out=out)
        sw.update(m, status="ready", ready_at=_now())
        out("member ready")
    except BaseException as e:   # SystemExit from ssh helpers ends only this member
        sw.update(m, status="setup_failed", error=f"{type(e).__name__}: {str(e)[:300]}")
        out(f"setup FAILED: {type(e).__name__}: {e}")
        if isinstance(e, KeyboardInterrupt):
            raise


def member_table(sw: Swarm) -> str:
    rows = [f"{'#':>3} {'pod':<16} {'gpu':<22} {'$/h':>5} {'status':<14} note"]
    for m in sw.members:
        note = m.get("error") or (f"ssh {m['ssh_host']}:{m['ssh_port']}" if m.get("ssh_host") else "")
        rows.append(f"{m.get('idx', '-')!s:>3} {m['pod_id']:<16} {str(m.get('gpu') or '-')[:22]:<22} "
                    f"{m.get('cost_per_hr') or '-'!s:>5} {m.get('status', '-'):<14} {str(note)[:70]}")
    return "\n".join(rows)


def _pubkey() -> str:
    p = Path(str(cli.ssh_key_path()) + ".pub")
    return p.read_text() if p.exists() else "ssh-ed25519 <missing ~/.ssh/id_ed25519.pub>"


def cmd_up(args) -> int:
    name = check_name(args.name)
    sw = Swarm.load(name, create=True)
    # a swarm made by `swarm plan` has no members yet: it is still "new" here
    exists = Swarm.path_for(name).exists() and bool(sw.members)
    gpus = args.gpu or sw.data.get("gpu_priority") or DEFAULT_SWARM_GPUS
    cloud = args.cloud or sw.data.get("cloud") or "SECURE"
    live = live_members(sw)
    if args.add:
        to_create = args.add
    elif args.count is not None:
        to_create = max(0, args.count - len(live))
    elif not exists:
        raise SystemExit("a new swarm needs --count K")
    else:
        to_create = 0
    next_idx = max([m.get("idx") or 0 for m in sw.members] + [0]) + 1
    bodies = []
    for i in range(to_create):
        b = cli.build_create_body(name=f"{name}-{next_idx + i}", gpus=gpus, cloud=cloud, image=args.image,
                                  volume_gb=args.volume_gb, container_gb=args.container_gb,
                                  pubkey=_pubkey(), min_vcpu=args.min_vcpu, min_ram=args.min_ram,
                                  data_centers=args.data_center)
        bodies.append(b)
    prices = gpu_prices(gpus, cloud)
    cost_line, _, _ = cost_estimate(to_create, prices)
    boot_args = "" if args.with_vllm else "--no-vllm"
    pending = [m for m in live if m.get("status") != "ready"]
    cli.warn_dirty(args.ref)
    if args.dry_run:
        print(f"DRY RUN: nothing is created. swarm {name!r}: {len(live)} live member(s), "
              f"{to_create} pod(s) would be created ({cloud}; priority {', '.join(gpus)}).")
        for b in bodies:
            print(f"POST {cli.REST}/pods")
            print(json.dumps(cli.redact_body(b), indent=2))
        print("\nprices: " + ", ".join(f"{g} ${p}/h" if p else f"{g} ?" for g, p in prices.items()))
        print(f"cost: {cost_line}")
        with tempfile.TemporaryDirectory(dir=args.scratch) as td:
            plan = cli.plan_bundle(args.ref, include_db=True, include_event_state=True,
                                   include_sam2=not args.no_sam2_fallback, scratch=Path(td))
            cli.print_plan(plan)
            print(f"(built ONCE, shipped to every new member: {to_create + len(pending)} upload(s) of "
                  f"{cli.fmt_mb(plan.total_size())} from this machine)")
        secrets = cli.select_secrets(cli.local_env(), interactive=False) if args.secrets_from_env else {}
        print("\nsecrets: " + (", ".join(f"{k}={cli.mask(v)}" for k, v in sorted(secrets.items()))
                               if secrets else "prompted once (or --secrets-from-env), installed on every member"))
        print(f"bootstrap: bootstrap.sh {boot_args or '(with the ~9 GB vLLM venv)'}")
        return 0
    if not exists:
        sw.data.update(gpu_priority=gpus, cloud=cloud)
        sw.save()
    if not bodies and not pending:
        print(f"swarm {name}: {len(live)} member(s), all ready; nothing to do (--add K grows it)")
        print(member_table(sw))
        return 0
    need_secrets = bool(bodies) or any(not m.get("secrets_installed") for m in pending)
    secrets = cli.gather_secrets(args) if need_secrets else None
    with tempfile.TemporaryDirectory(dir=args.scratch) as td:
        tar_path = None
        if bodies or any(not m.get("pushed") for m in pending):
            plan = cli.plan_bundle(args.ref, include_db=True, include_event_state=True,
                                   include_sam2=not args.no_sam2_fallback, scratch=Path(td))
            cli.print_plan(plan)
            tar_path = Path(td) / "bundle.tar"
            cli.write_bundle(plan, tar_path)
        if bodies:
            print(f"\n{cost_line}")
            if not args.yes and not cli.confirm(
                    f"This RENTS {len(bodies)} GPU pod(s) ({', '.join(gpus)}; {cloud}) billed per hour "
                    f"until you stop/terminate them (`swarm terminate --name {name}`).", "rent"):
                return 1
            errors = create_pods(sw, [{**b, "_idx": next_idx + i} for i, b in enumerate(bodies)])
            for b in bodies:
                b.pop("_idx", None)
            if errors:
                print(f"{len(errors)} of {len(bodies)} pod(s) were NOT created (kept the rest); "
                      f"re-run `swarm up --name {name} --count K` or `--add N` to retry")
        todo = [m for m in live_members(sw) if m.get("status") != "ready"]
        print(f"\nsetting up {len(todo)} member(s) in parallel (logs: {sw.log_dir()}/<podId>.log)...")
        threads = [threading.Thread(target=setup_member, args=(sw, m, tar_path, secrets, boot_args, args.ref),
                                    daemon=True) for m in todo]
        for t in threads:
            t.start()
        last = None
        try:
            while any(t.is_alive() for t in threads):
                time.sleep(args.poll_s)
                tbl = member_table(sw)
                if tbl != last:
                    print("\n" + tbl, flush=True)
                    last = tbl
        except KeyboardInterrupt:
            print("\ninterrupted: pods keep running (and billing). Re-run the same `swarm up` to resume setup.")
            return 130
    print("\n" + member_table(sw))
    ready = len(ready_members(sw))
    print(f"\n{ready}/{len(live_members(sw))} member(s) ready. Next: "
          f"`swarm plan --name {name} --event <event> --download` then `swarm run --name {name}`")
    return 0 if ready == len(live_members(sw)) else 1


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
def match_rank(comp_level: str, set_number, match_number) -> tuple:
    """Schedule order: quals by number, then playoffs by level / set / match."""
    lvl = LEVEL_RANK.get(comp_level, 9)
    if lvl == 0:
        return (0, 0, int(match_number))
    return (lvl, int(set_number or 1), int(match_number))


def key_rank(match_key: str) -> tuple | None:
    m = _MATCH_RE.match(match_key or "")
    if not m:
        return None
    lvl, a, b = m.group(1), int(m.group(2)), m.group(3)
    if lvl == "qm":
        return (0, 0, a)
    return (LEVEL_RANK[lvl], a, int(b) if b is not None else None)


def schedule_from_db(db_path: Path, event: str) -> list:
    """The event's matches from the local DB matches table, in schedule order."""
    if not Path(db_path).exists():
        raise SystemExit(f"local DB not found: {db_path}")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT m.tba_match_key, m.comp_level, m.set_number, m.match_number, m.tba_video_key "
            "FROM matches m JOIN events e ON e.id = m.event_id WHERE e.event_key = ?", (event,)).fetchall()
    finally:
        con.close()
    out = [{"match_key": k, "comp_level": lvl, "rank": match_rank(lvl, s, n), "video_key": vk}
           for k, lvl, s, n, vk in rows]
    out.sort(key=lambda r: r["rank"])
    if not out:
        raise SystemExit(f"no matches for event {event!r} in {db_path} (load the event in the app first)")
    return out


def _spec_selector(tok: str):
    """One SPEC token -> predicate over a schedule row."""
    t = tok.strip().lower()
    if t in ("all", "*"):
        return lambda r: True
    if t in ("qm", "quals", "qual"):
        return lambda r: r["comp_level"] == "qm"
    if t in ("playoffs", "elims"):
        return lambda r: r["comp_level"] != "qm"
    if t in LEVEL_RANK:
        return lambda r, t=t: r["comp_level"] == t
    if "-" in t:
        a, b = t.split("-", 1)
        ra, rb = key_rank(a), key_rank(b)
        if ra is None or rb is None:
            raise SystemExit(f"bad match range {tok!r} (e.g. qm1-qm40, sf1m1-sf5m1)")
        lo = (ra[0], ra[1], ra[2] if ra[2] is not None else 0)
        hi = (rb[0], rb[1], rb[2] if rb[2] is not None else 10 ** 6)
        if lo > hi:
            raise SystemExit(f"empty match range {tok!r}")
        return lambda r, lo=lo, hi=hi: lo <= tuple(r["rank"]) <= hi
    rk = key_rank(t)
    if rk is None:
        raise SystemExit(f"bad match spec token {tok!r} (all, qm, playoffs, qm8, qm1-qm40, sf1m1, f1m2)")
    if rk[2] is None:          # "sf3" = every match of set 3
        return lambda r, rk=rk: tuple(r["rank"][:2]) == rk[:2]
    return lambda r, rk=rk: tuple(r["rank"]) == rk


def select_matches(schedule: list, spec: str) -> list:
    sels = [_spec_selector(t) for t in (spec or "all").split(",") if t.strip()]
    return [r for r in schedule if any(s(r) for s in sels)]


def _complete_keys(db_path: Path, keys: set, only_ids: set | None = None) -> set:
    if not keys or not Path(db_path).exists():
        return set()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT id, match_key FROM replays WHERE status = 'complete'").fetchall()
    except sqlite3.Error:
        return set()
    finally:
        con.close()
    return {mk for rid, mk in rows if mk in keys and (only_ids is None or rid in only_ids)}


def existing_results(keys: set, local_db: Path, pull_root: Path, run: dict | None) -> dict:
    """match_key -> where a finished result already exists: the local DB, a
    previous pull (a pulled replay dir + its row in that pull's DB) or this
    swarm's run."""
    found = {k: "local DB" for k in _complete_keys(local_db, keys)}
    if Path(pull_root).is_dir():
        for d in sorted(Path(pull_root).iterdir()):
            rdir = d / "web" / "replays"
            if not (d / "frc_scorer.db").exists() or not rdir.is_dir():
                continue
            ids = {p.name for p in rdir.iterdir() if cli.replay_pulled(p)}
            for k in _complete_keys(d / "frc_scorer.db", keys, ids):
                found.setdefault(k, f"pull {d.name}")
    for key, it in ((run or {}).get("items") or {}).items():
        mk = item_match_key(run, key)
        if it.get("status") == "done" and mk in keys and not it.get("config_id"):
            found.setdefault(mk, "this swarm's run")
    return found


def video_dirs(extra: list | None = None) -> list:
    main = cli.main_checkout()
    dirs = [Path(d) for d in (extra or []) if d]
    return dirs + [main / "data" / "videos", main / "data" / "test_footage"]


def find_video(match_key: str, dirs: list) -> Path | None:
    for d in dirs:
        p = Path(d) / f"{match_key}.mp4"
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def default_downloader(video_key: str, out_path: Path) -> Path:
    """YouTube download from THIS machine (home IP: no datacenter bot block),
    via the app's own recipe (web/field/auto_calibrate.download_match_video)."""
    if str(cli.REPO) not in sys.path:
        sys.path.insert(0, str(cli.REPO))
    from web.field.auto_calibrate import download_match_video
    return download_match_video(video_key, out_path)


def build_plan(event: str, spec: str, *, local_db: Path, pull_root: Path, run: dict | None,
               videos: str | None = None, download: bool = False, cache_dir: Path | None = None,
               redo: bool = False, downloader=None, out=print) -> dict:
    schedule = schedule_from_db(local_db, event)
    chosen = select_matches(schedule, spec)
    if not chosen:
        raise SystemExit(f"match spec {spec!r} selects nothing for {event}")
    have = {} if redo else existing_results({r["match_key"] for r in chosen}, local_db, pull_root, run)
    cache_dir = Path(cache_dir or cli.main_checkout() / "data" / "videos")
    dirs = video_dirs([videos] if videos else []) + [cache_dir]
    items, skipped, missing = [], [], []
    for r in chosen:
        mk = r["match_key"]
        if mk in have:
            skipped.append({"match_key": mk, "why": f"already processed ({have[mk]}); --redo to repeat"})
            continue
        p = find_video(mk, dirs)
        if p is None and download:
            if not r.get("video_key"):
                missing.append({"match_key": mk, "why": "no TBA video key"})
                continue
            out(f"  downloading {mk} (youtube {r['video_key']}) -> {cache_dir}/{mk}.mp4 ...")
            try:
                p = Path((downloader or default_downloader)(r["video_key"], cache_dir / f"{mk}.mp4"))
            except Exception as e:
                missing.append({"match_key": mk, "why": f"download failed: {str(e)[:200]}"})
                out(f"    FAILED: {str(e)[:200]}")
                continue
        if p is None or not p.is_file():
            missing.append({"match_key": mk, "why": "no local video (--videos DIR or --download)"})
            continue
        items.append({"match_key": mk, "video": str(p.resolve()), "size": p.stat().st_size,
                      "video_key": r.get("video_key")})
    return {"id": f"{event}-{int(_now())}", "event": event, "spec": spec, "created_at": _now(),
            "items": items, "skipped": skipped, "missing": missing}


def _run_active(run: dict | None) -> bool:
    return bool(run) and any(it.get("status") == "running" for it in run.get("items", {}).values())


def cmd_plan(args) -> int:
    # Planning (incl. --download) comes BEFORE renting: a swarm with no pods yet
    # is created here so pods never bill while match videos download.
    sw = Swarm.load(check_name(args.name), create=True)
    if _run_active(sw.data.get("run")):
        raise SystemExit("a run has jobs in flight: finish it (`swarm run`) before re-planning")
    main = cli.main_checkout()
    plan = build_plan(args.event, args.matches, local_db=main / "frc_scorer.db", pull_root=cli.PULL_ROOT,
                      run=sw.data.get("run"), videos=args.videos, download=args.download,
                      cache_dir=Path(args.cache_dir) if args.cache_dir else None, redo=args.redo)
    with sw.lock:
        sw.data["plan"] = plan
        sw.save()
    print_plan_summary(plan)
    return 0


def print_plan_summary(plan: dict) -> None:
    if plan.get("kind") == "sweep":
        import sweep
        return sweep.print_sweep_plan(plan)
    if plan.get("kind") == "live":
        print(f"live plan {plan['event']}: matches arrive while the run goes (policy live)")
        return
    items = plan["items"]
    size = sum(i["size"] for i in items)
    print(f"plan {plan['event']} [{plan['spec']}]: {len(items)} match(es) to run "
          f"({cli.fmt_mb(size)} of video), {len(plan['skipped'])} skipped, "
          f"{len(plan['missing'])} without video")
    if items:
        print("  order: " + " ".join(i["match_key"].split("_", 1)[-1] for i in items))
    for s in plan["skipped"]:
        print(f"  skip {s['match_key']}: {s['why']}")
    for s in plan["missing"]:
        print(f"  MISSING {s['match_key']}: {s['why']}")


# --------------------------------------------------------------------------
# Assignment (pure; unit-tested)
# --------------------------------------------------------------------------
def split_blocks(items: list, pods: list) -> dict:
    """Contiguous schedule blocks, as even as possible (earlier pods get the
    extra item). Consecutive matches stay on one server because the app's
    camera-move auto-recalibration and per-match corner calibration carry
    state from one match to the next."""
    out = {p: [] for p in pods}
    if not pods:
        return out
    n, k = len(items), len(pods)
    base, extra = divmod(n, k)
    i = 0
    for j, p in enumerate(pods):
        size = base + (1 if j < extra else 0)
        out[p] = list(items[i:i + size])
        i += size
    return out


# Item keys. A batch run's items are keyed by match key ("2026cur_qm8"), as
# before. A quality sweep runs the same match under several configs, so its
# items are keyed "<config_id>/<match_key>" ("Q3/2026cur_qm8"), and a live
# run's re-posted video of a match is "<match_key>@r2". Every new item records
# its real ``match_key`` (+ ``config_id`` / ``knobs`` / ``event`` when it has
# them); items of runs saved before this (batch state files) have none of
# these fields, so ``item_match_key`` falls back to the key itself.
ITEM_FIELDS = ("match_key", "event", "config_id", "knobs", "video_key")


def item_key(i: dict) -> str:
    """The run key of a plan item: its explicit ``key``, else its match key."""
    return i.get("key") or i["match_key"]


def item_match_key(run: dict | None, key: str) -> str:
    it = ((run or {}).get("items") or {}).get(key) or {}
    return it.get("match_key") or key


def safe_key(key: str) -> str:
    """An item key as a file-name component ("Q3/2026cur_qm8" -> "Q3__2026cur_qm8")."""
    return re.sub(r"[^A-Za-z0-9_.@+-]", "_", key.replace("/", "__"))


def video_name(key: str) -> str:
    """The pod-side video file stem of an item: the key without its sweep
    config prefix, so every config of one match shares one upload
    ("Q3/2026cur_qm8" -> "2026cur_qm8"), while a live re-posted video keeps
    its own file ("2026cur_qm8@r2")."""
    return safe_key(key.split("/", 1)[-1])


def _new_item(i: dict) -> dict:
    it = {"video": i["video"], "size": i.get("size"), "status": "pending",
          "attempts": 0, "avoid": [], "pod": None}
    for f in ITEM_FIELDS:
        if i.get(f) is not None:
            it[f] = i[f]
    return it


def new_run(plan: dict, pods: list, config: dict | None = None, policy: str = "blocks",
            pod_config: dict | None = None, restart_each: bool = False) -> dict:
    """Run state. ``policy`` names the assignment policy (POLICIES);
    ``pod_config`` overrides the run config per pod; ``open`` = more items
    may still arrive (live sources), so idle pods wait; ``restart_each`` =
    restart the member's app server before every job (the ~0.7 GB/job CUDA
    leak would otherwise make job order a confounder)."""
    run = {"plan_id": plan["id"], "event": plan.get("event"), "config": config or {},
           "pod_config": dict(pod_config or {}), "policy": policy, "open": False,
           "restart_each": bool(restart_each),
           "started_at": _now(), "queues": {p: [] for p in pods}, "unassigned": [],
           "current": {p: None for p in pods}, "retry": [], "dead": [], "retired": [],
           "steals": [], "items": {}}
    for i in plan["items"]:
        run["items"][item_key(i)] = _new_item(i)
    get_policy(run).initial(run, [item_key(i) for i in plan["items"]], list(pods))
    return run


def job_config(run: dict, pod: str, key: str | None = None) -> dict:
    """The knobs one job runs with: the run config, then the item's own
    knobs (a sweep config), then the pod's override."""
    it = ((run.get("items") or {}).get(key) or {}) if key else {}
    return {**(run.get("config") or {}), **(it.get("knobs") or {}),
            **((run.get("pod_config") or {}).get(pod) or {})}


def item_event(run: dict, key: str) -> str:
    return (run["items"].get(key) or {}).get("event") or run.get("event")


def add_items(run: dict, items: list) -> list:
    """Append newly arrived items ({match_key, video, size} [+ key, ...]) to
    a running run; the policy places them. Returns the keys actually added.
    The run loop picks them up on its next poll (a live arrival source)."""
    new = []
    for i in items:
        key = item_key(i)
        if key in run["items"]:
            continue
        run["items"][key] = _new_item(i)
        new.append(key)
    if new:
        get_policy(run).place(run, new)
    return new


def add_pod(run: dict, pod: str) -> None:
    run["queues"].setdefault(pod, [])
    run["current"].setdefault(pod, None)
    if pod in run["dead"] and pod not in run["retired"]:
        run["dead"].remove(pod)


def _live_pods(run: dict) -> list:
    gone = set(run["dead"]) | set(run["retired"])
    return [p for p in run["queues"] if p not in gone]


def next_item(run: dict, pod: str) -> str | None:
    """What ``pod`` should process next: its in-flight item (resume), else
    whatever the run's policy picks (mutates ``run``). None = nothing now."""
    if pod in run["dead"] or pod in run["retired"]:
        return None
    add_pod(run, pod)
    cur = run["current"].get(pod)
    if cur:
        return cur
    mk = get_policy(run).pick(run, pod)
    if mk is not None:
        it = run["items"][mk]
        it.update(status="running", pod=pod, attempts=it.get("attempts", 0) + 1, progress=None,
                  assigned_at=_now())
        run["current"][pod] = mk
    return mk


def _take_retry(run: dict, pod: str, allow_avoided: bool = False) -> str | None:
    """A retry this pod hasn't failed; with ``allow_avoided`` also one it
    has, but only when no other live pod is left to take it."""
    items = run["items"]
    for r in run["retry"]:
        if pod not in items[r]["avoid"]:
            run["retry"].remove(r)
            return r
    if allow_avoided and run["retry"] and not [p for p in _live_pods(run) if p != pod]:
        return run["retry"].pop(0)
    return None


class BlockPolicy:
    """Past-event batch: contiguous schedule blocks per pod + work stealing.

    pick(): 1. the head of its own block, 2. newly arrived unassigned items,
    3. a retry it hasn't failed, 4. STEAL the tail half of the largest other
    block (a dead pod's whole block), 5. a retry it already failed, only when
    no other pod is left. None = nothing for it right now.

    A policy is three methods over the JSON run state: ``initial(run, keys,
    pods)`` (first assignment), ``place(run, keys)`` (items arriving while
    the run is going) and ``pick(run, pod)`` (an idle pod asks for work).
    A LIVE staggered policy (each arriving match -> the idle / least-loaded
    pod) only needs these three; the run loop stays the same."""
    name = "blocks"

    def initial(self, run: dict, keys: list, pods: list) -> None:
        run["queues"].update(split_blocks(keys, pods))

    def place(self, run: dict, keys: list) -> None:
        run["unassigned"].extend(keys)       # first idle pod takes them, in order

    def pick(self, run: dict, pod: str) -> str | None:
        q = run["queues"][pod]
        if q:
            return q.pop(0)
        if run.get("unassigned"):
            return run["unassigned"].pop(0)
        mk = _take_retry(run, pod)
        if mk is None:
            mk = self._steal(run, pod)
        if mk is None:
            mk = _take_retry(run, pod, allow_avoided=True)
        return mk

    @staticmethod
    def _steal(run: dict, pod: str) -> str | None:
        victims = [(len(v), p) for p, v in run["queues"].items() if p != pod and v]
        if not victims:
            return None
        # largest block; ties -> earliest pod in the queues order
        order = list(run["queues"])
        _, victim = max(victims, key=lambda t: (t[0], -order.index(t[1])))
        vq = run["queues"][victim]
        if victim in run["dead"] or victim in run["retired"]:
            take, keep = vq[:], []
        else:
            cut = len(vq) // 2
            take, keep = vq[cut:], vq[:cut]
        run["queues"][victim] = keep
        run["queues"][pod] = take[1:]
        run["steals"].append({"thief": pod, "victim": victim, "items": take, "at": _now()})
        return take[0]


class PinnedPolicy:
    """Quality sweep: every item of one config runs on ONE pod (config i ->
    pod i), so pods' different speeds/hosts never mix inside a config. With
    fewer pods than configs a pod that finishes takes the next whole config
    not started yet; a dead pod's remaining config is taken over whole by the
    next pod that runs dry. A failed item is retried on its config's pod
    (config consistency beats avoiding the pod)."""
    name = "pinned"

    @staticmethod
    def _cfg(run: dict, key: str) -> str:
        return run["items"][key].get("config_id") or "-"

    def _groups(self, run: dict, keys: list) -> list:
        groups: dict = {}
        for k in keys:
            groups.setdefault(self._cfg(run, k), []).append(k)
        return list(groups.items())

    def initial(self, run: dict, keys: list, pods: list) -> None:
        run.setdefault("pin", {})
        run.setdefault("unassigned_groups", [])
        for j, (cid, ks) in enumerate(self._groups(run, keys)):
            if j < len(pods):
                run["pin"][cid] = pods[j]
                run["queues"][pods[j]] = list(ks)
            else:
                run["unassigned_groups"].append([cid, list(ks)])

    def place(self, run: dict, keys: list) -> None:
        run.setdefault("pin", {})
        run.setdefault("unassigned_groups", [])
        for cid, ks in self._groups(run, keys):
            pod = run["pin"].get(cid)
            if pod is not None:
                run["queues"].setdefault(pod, []).extend(ks)
                continue
            for g in run["unassigned_groups"]:
                if g[0] == cid:
                    g[1].extend(ks)
                    break
            else:
                run["unassigned_groups"].append([cid, list(ks)])

    def pick(self, run: dict, pod: str) -> str | None:
        pins = run.setdefault("pin", {})
        q = run["queues"][pod]
        if q:
            return q.pop(0)
        mine = {c for c, p in pins.items() if p == pod}
        for r in run["retry"]:
            if self._cfg(run, r) in mine:
                run["retry"].remove(r)
                return r
        gone = set(run["dead"]) | set(run["retired"])
        for cid, owner in list(pins.items()):        # take over a lost pod's config whole
            if owner in gone and owner != pod and (
                    run["queues"].get(owner) or any(self._cfg(run, r) == cid for r in run["retry"])):
                pins[cid] = pod
                run["queues"][pod] = [k for k in run["queues"].get(owner, []) if self._cfg(run, k) == cid]
                run["queues"][owner] = [k for k in run["queues"].get(owner, []) if self._cfg(run, k) != cid]
                run["steals"].append({"thief": pod, "victim": owner, "config": cid, "at": _now()})
                return self.pick(run, pod)
        groups = run.setdefault("unassigned_groups", [])
        if groups:
            cid, ks = groups.pop(0)
            pins[cid] = pod
            run["queues"][pod] = list(ks)
            return self.pick(run, pod)
        if run.get("unassigned"):
            return run["unassigned"].pop(0)
        return None

    def peek(self, run: dict, pod: str) -> str | None:
        q = run["queues"].get(pod) or []
        return q[0] if q else None


class LivePolicy:
    """Live staggered dispatch: matches arrive one by one (``place`` in
    arrival order); each goes to an IDLE pod, the one idle longest. Two
    matches are never queued on one pod while another pod is idle: nothing
    is pre-assigned, so when every pod is busy the waiting match goes to
    whichever pod frees first (``predict_free`` names the likely one from
    elapsed vs median job time, for prefetching its video and for the ETA).
    Retries (another pod) go before new arrivals."""
    name = "live"

    def initial(self, run: dict, keys: list, pods: list) -> None:
        run.setdefault("idle_since", {})
        for p in pods:
            run["idle_since"].setdefault(p, _now())
        self.place(run, keys)

    def place(self, run: dict, keys: list) -> None:
        run["unassigned"].extend(keys)

    @staticmethod
    def idle_pods(run: dict) -> list:
        return [p for p in _live_pods(run) if not run["current"].get(p)]

    def pick(self, run: dict, pod: str) -> str | None:
        idle_since = run.setdefault("idle_since", {})
        idle_since.setdefault(pod, _now())
        polled = run.setdefault("polled", {})
        polled[pod] = _now()
        # an idle pod counts only while its worker keeps asking (a member
        # whose worker isn't running must not hold arrivals back)
        idle = [p for p in self.idle_pods(run) if _now() - polled.get(p, -1e18) <= LIVE_STALE_S]
        mk = _take_retry(run, pod, allow_avoided=False)
        if mk is None and run.get("unassigned"):
            # only the longest-idle pod takes the next arrival (ties: pod order)
            order = list(run["queues"])
            first = min(idle, key=lambda p: (idle_since.get(p, _now()), order.index(p))) if idle else pod
            if first not in order:
                first = pod
            if first == pod:
                mk = run["unassigned"].pop(0)
        if mk is None:
            mk = _take_retry(run, pod, allow_avoided=True)
        if mk is not None:
            idle_since.pop(pod, None)
        return mk

    def peek(self, run: dict, pod: str) -> str | None:
        """The waiting match this pod should prefetch: the head of the
        arrival queue when this pod is predicted to free up first."""
        if not run.get("unassigned"):
            return None
        return run["unassigned"][0] if predict_free(run) == pod else None


def median_job_s(run: dict, default: float = DEFAULT_JOB_MIN * 60) -> float:
    w = sorted(x for x in ((it.get("wall_final_s") or 0) for it in run["items"].values()
                           if it.get("status") == "done") if x > 0)
    if not w:
        return default
    n = len(w)
    return w[n // 2] if n % 2 else (w[n // 2 - 1] + w[n // 2]) / 2


def predict_free(run: dict, now: float | None = None) -> str | None:
    """The pod expected to free up first: an idle pod (longest idle), else
    the busy pod with the least (median job time - elapsed) left."""
    now = _now() if now is None else now
    live = _live_pods(run)
    idle = [p for p in live if not run["current"].get(p)]
    if idle:
        since = run.get("idle_since") or {}
        return min(idle, key=lambda p: (since.get(p, now), live.index(p)))
    if not live:
        return None
    med = median_job_s(run)

    def left(p):
        it = run["items"][run["current"][p]]
        t0 = now if it.get("assigned_at") is None else it["assigned_at"]
        return med - (now - t0)
    return min(live, key=lambda p: (left(p), live.index(p)))


POLICIES = {BlockPolicy.name: BlockPolicy(), PinnedPolicy.name: PinnedPolicy(),
            LivePolicy.name: LivePolicy()}


def get_policy(run: dict):
    name = run.get("policy") or "blocks"
    if name not in POLICIES:
        raise SystemExit(f"unknown dispatch policy {name!r} (have: {', '.join(POLICIES)})")
    return POLICIES[name]


def peek_next(run: dict, pod: str) -> str | None:
    """The item ``pod`` will most likely run next (its video is prefetched)."""
    pol = get_policy(run)
    if hasattr(pol, "peek"):
        return pol.peek(run, pod)
    q = run["queues"].get(pod) or []
    return q[0] if q else None


def finish_item(run: dict, mk: str, pod: str, ok: bool, max_attempts: int = MAX_ATTEMPTS, **info) -> str:
    """Record ``mk``'s result on ``pod``; a failure is retried (avoiding this
    pod) until ``max_attempts``. Returns the item's new status."""
    it = run["items"][mk]
    it.update(info)
    if run["current"].get(pod) == mk:
        run["current"][pod] = None
    it["finished_at"] = _now()
    if ok:
        it["status"] = "done"
        return "done"
    it.setdefault("errors", []).append({"pod": pod, "error": info.get("error"), "at": _now()})
    if pod not in it["avoid"]:
        it["avoid"].append(pod)
    if it["attempts"] >= max_attempts:
        it["status"] = "failed"
    else:
        it["status"] = "retry"
        if mk not in run["retry"]:
            run["retry"].append(mk)
    return it["status"]


def mark_pod_dead(run: dict, pod: str, why: str = "pod lost") -> None:
    """The pod is gone: its in-flight item counts as a failed attempt and its
    whole remaining block becomes stealable."""
    if pod not in run["dead"]:
        run["dead"].append(pod)
    cur = run["current"].get(pod)
    if cur:
        finish_item(run, cur, pod, False, error=why)


def retire_pod(run: dict, pod: str) -> None:
    if pod not in run["retired"]:
        run["retired"].append(pod)


def all_terminal(run: dict) -> bool:
    return all(it["status"] in ("done", "failed") for it in run["items"].values())


def run_finished(run: dict) -> bool:
    """Every item is terminal AND no more can arrive (``open`` is False)."""
    return all_terminal(run) and not run.get("open")


def run_counts(run: dict) -> dict:
    c = {"pending": 0, "running": 0, "retry": 0, "done": 0, "failed": 0}
    for it in run["items"].values():
        c[it["status"]] = c.get(it["status"], 0) + 1
    c["total"] = len(run["items"])
    return c


# --------------------------------------------------------------------------
# run (I/O)
# --------------------------------------------------------------------------
class PodLost(Exception):
    """The RunPod API says the pod is gone or not RUNNING (never raised just
    because the API or ssh can't be reached)."""


class Transport(Exception):
    """ssh itself failed (exit 255, timeout, no endpoint): network trouble,
    laptop sleep. Retried without limit (with an API check now and then)."""


class RemoteError(Exception):
    """The remote command ran and exited non-zero (disk full, missing venv,
    server down...). Retried REMOTE_TRIES times, then RemoteFailed."""


class RemoteFailed(Exception):
    """A remote command kept failing: the item fails instead of the pod
    retrying forever while it bills."""


REMOTE_TRIES = 3
SSH_TIMEOUT_S = 120
UPLOAD_TIMEOUT_S = 3600


def remote(m: dict, cmd: str, *, timeout: float = SSH_TIMEOUT_S, stdin_path=None) -> str:
    """Run ``cmd`` on member ``m``: stdout, or Transport (ssh failed: exit 255
    or timeout) / RemoteError (the command exited non-zero)."""
    try:
        base = cli.ssh_base(m)
    except SystemExit as e:                 # no ssh endpoint yet
        raise Transport(str(e))
    f = open(stdin_path, "rb") if stdin_path else None
    try:
        r = subprocess.run(base + [cmd], stdin=f if f else subprocess.DEVNULL,
                           capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise Transport(f"ssh timed out after {timeout:.0f}s: {cmd[:60]}")
    except OSError as e:
        raise Transport(f"ssh: {e}")
    finally:
        if f:
            f.close()
    err = (r.stderr or b"").decode(errors="replace").strip()[-300:]
    if r.returncode == 255:
        raise Transport(f"ssh exit 255: {err}")
    if r.returncode != 0:
        raise RemoteError(f"exit {r.returncode} from `{cmd[:80]}`: {err}")
    return (r.stdout or b"").decode(errors="replace")


def read_job_state(m: dict, outp: str) -> tuple[bool, dict | None]:
    """(launched, state) for one attempt: ``launched`` = its launch marker
    exists; ``state`` = the driver's out JSON, None when missing (or caught
    mid-write). ssh failures raise; they never read as "missing"."""
    q = shlex.quote(outp)
    txt = remote(m, f"if [ -d {q}.launch ]; then echo L; else echo N; fi; "
                    f"if [ -f {q} ]; then cat {q}; fi", timeout=60)
    head, _, body = txt.partition("\n")
    state = None
    if body.strip():
        try:
            state = json.loads(body)
        except ValueError:
            state = None
    return head.strip() == "L", state


def driver_alive(m: dict, state: dict | None, outp: str) -> bool:
    """Is this attempt's pod_job.py running? By pid once the state has one,
    else by its --out argument ("[-]-out" so pgrep can't match its own shell)."""
    if state and state.get("pid"):
        cmd = f"kill -0 {int(state['pid'])} 2>/dev/null && echo yes || echo no"
    else:
        cmd = f"pgrep -f -- {shlex.quote('[-]-out ' + outp)} >/dev/null && echo yes || echo no"
    return remote(m, cmd, timeout=60).strip() == "yes"


def server_active_job(m: dict) -> str | None:
    """job_id the pod's server is running, None when idle. Unlike
    runpod_cli.remote_job_active, an ssh failure or a server that doesn't
    answer is an error (unknown), never "idle"."""
    txt = remote(m, f"curl -s -m 10 http://127.0.0.1:{cli.POD_PORT}/api/job/active", timeout=60)
    try:
        return json.loads(txt).get("job_id")
    except (ValueError, AttributeError):
        raise RemoteError(f"/api/job/active did not answer JSON: {txt[:120]!r}")


def upload_video(m: dict, local: Path, remote_path: str, out, what: str = "video") -> bool:
    """Upload unless a same-size file is there (.part + mv). True if uploaded."""
    q = shlex.quote(remote_path)
    have = remote(m, f"stat -c %s {q} 2>/dev/null || echo 0", timeout=60).strip()
    size = Path(local).stat().st_size
    if have == str(size):
        return False
    out(f"uploading {what} {Path(local).name} ({cli.fmt_mb(size)}) -> {remote_path}")
    parent = shlex.quote(remote_path.rsplit("/", 1)[0])
    remote(m, f"mkdir -p {parent} && cat > {q}.part && mv {q}.part {q}",
           timeout=UPLOAD_TIMEOUT_S, stdin_path=local)
    return True


def launch_once(m: dict, outp: str, cmd: str) -> str:
    """Launch at most once per attempt: the pod creates ``<out>.launch``
    atomically (mkdir) before starting the driver, so a retry after an ssh
    drop that hid a successful launch never starts a second pod_job.py."""
    jobs = shlex.quote(outp.rsplit("/", 1)[0])
    q = shlex.quote(outp + ".launch")
    txt = remote(m, f"mkdir -p {jobs} && if mkdir {q} 2>/dev/null; then {cmd}; else echo already; fi")
    return "already" if "already" in txt else "started"


def pod_api_status(m: dict, saver=None) -> str | None:
    """RunPod's view of the pod: its desiredStatus ("RUNNING", "EXITED", ...),
    "GONE" when the API says it doesn't exist, None when the API can't be
    reached (unknown: NOT a reason to give up on the pod)."""
    try:
        pod = cli.refresh_endpoint(m, save=saver)
    except SystemExit as e:
        return "GONE" if "HTTP 404" in str(e) else None
    except Exception:
        return None
    return pod.get("desiredStatus") or None


# --------------------------------------------------------------------------
# Members: one transport per kind (RunPod pod over ssh, or the local app)
# --------------------------------------------------------------------------
SYNC_EXCLUDE_PREFIXES = ("clockscan_",)       # per-video caches: never synced between pods


def sync_selectable(name: str) -> bool:
    """Event-state files live mode copies between pods: calibrations,
    camera layouts, zones, season overrides; never clock-scan caches."""
    return cli.event_state_ok(name) and not name.startswith(SYNC_EXCLUDE_PREFIXES)


def is_pod(m: dict) -> bool:
    return (m.get("backend") or "pod") == "pod"


def vllm_model_for(cfg: dict) -> str | None:
    """vlm_model a job config needs served by a resident vLLM reader."""
    return (cfg.get("vlm_model") or "qwen3-vl-4b") if cfg.get("team_reader") == "vllm" else None


class PodMember:
    """A RunPod member over ssh. Each method calls the module-level helper
    by name at call time (the tests patch those)."""
    kind = "pod"

    def video_path(self, m: dict, mk: str, local: Path) -> str:
        return f"{POD_VIDEOS}/{mk}.mp4"

    def attempt_paths(self, m: dict, key: str, attempt: int) -> tuple[str, str]:
        return attempt_paths(key, attempt)

    def read_job_state(self, m, outp):
        return read_job_state(m, outp)

    def driver_alive(self, m, state, outp):
        return driver_alive(m, state, outp)

    def active_job(self, m):
        return server_active_job(m)

    def upload(self, m, local, remote_path, out, what="video"):
        return upload_video(m, local, remote_path, out, what=what)

    def launch(self, m, outp, logp, *, video, match_key, event, label, config) -> str:
        cmd = cli.pod_job_command(video=video, match_key=match_key, event=event, label=label,
                                  config=config, out=outp, log=logp, profile=False,
                                  extra=f"--final-timeout {FINAL_TIMEOUT_S:g}")
        return launch_once(m, outp, cmd)

    def restart(self, m, out) -> None:
        out("restarting the app server before the job (serve.sh restart; waits for :8004)")
        remote(m, f"bash {cli.POD_APP}/deploy/runpod/serve.sh restart", timeout=600)

    def prepare(self, m, cfg: dict, out) -> None:
        """Before a job: a config with a resident vLLM reader gets it started
        first (cold start outside the measured wall; it stays up)."""
        model = vllm_model_for(cfg)
        if not model:
            return
        txt = remote(m, f"cd {cli.POD_APP} && source deploy/runpod/pod_env.sh && set -a && "
                        f"{{ [ -f profile.env ] && . ./profile.env || true; }} && set +a && "
                        f".venv/bin/python deploy/runpod/vllm_ctl.py start {shlex.quote(model)}", timeout=1500)
        res = None
        for line in reversed(txt.strip().splitlines()):
            try:
                res = json.loads(line)
                break
            except ValueError:
                continue
        if not (res or {}).get("ok"):
            raise RemoteError(f"vLLM pre-start of {model} failed: {txt[-200:]}")
        out(f"vLLM reader {model} up")

    def tail_log(self, m, logp) -> str:
        return remote(m, f"tail -n 20 {shlex.quote(logp)} 2>/dev/null || true", timeout=60)

    def list_event_state(self, m) -> dict:
        """{file name: sha1} of the pod's syncable event-state files."""
        d = shlex.quote(f"{cli.POD_APP}/web/uploads")
        txt = remote(m, f"cd {d} 2>/dev/null && ls -1 | while read -r f; do "
                        f"[ -f \"$f\" ] && sha1sum \"$f\"; done; true", timeout=120)
        out = {}
        for line in txt.splitlines():
            h, _, name = line.strip().partition("  ")
            if name and sync_selectable(name):
                out[name] = h
        return out

    def fetch_files(self, m, names: list, dest: Path) -> list:
        """Copy event-state files from the pod into ``dest`` (base64 tar over ssh)."""
        import base64
        import io
        import tarfile
        if not names:
            return []
        d = shlex.quote(f"{cli.POD_APP}/web/uploads")
        txt = remote(m, f"cd {d} && tar -cf - {' '.join(shlex.quote(n) for n in names)} | base64 -w0",
                     timeout=300)
        got = []
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(txt.strip())), mode="r:") as tf:
            for ti in tf:
                if ti.isfile() and "/" not in ti.name and sync_selectable(ti.name):
                    data = tf.extractfile(ti).read()
                    tmp = dest / f".{ti.name}.tmp"
                    tmp.write_bytes(data)
                    os.replace(tmp, dest / ti.name)
                    got.append(ti.name)
        return got

    def push_files(self, m, files: list) -> None:
        """Install files into the pod's web/uploads atomically: extract into
        a temp dir on the same volume, then mv each into place."""
        import tarfile
        if not files:
            return
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tf_:
            tar_path = tf_.name
        try:
            with tarfile.open(tar_path, "w") as tf:
                for f in files:
                    tf.add(str(f), arcname=Path(f).name)
            up = f"{cli.POD_APP}/web/uploads"
            tmp = f"{up}/.swarm-sync-{os.getpid()}-{int(_now() * 1000)}"
            names = " ".join(shlex.quote(Path(f).name) for f in files)
            remote(m, f"mkdir -p {shlex.quote(tmp)} && tar -xf - -C {shlex.quote(tmp)} && "
                      f"for f in {names}; do mv -f {shlex.quote(tmp)}/\"$f\" {shlex.quote(up)}/\"$f\"; done && "
                      f"rmdir {shlex.quote(tmp)}", timeout=300, stdin_path=tar_path)
        finally:
            os.unlink(tar_path)


LOCAL_PIDFILE = "/tmp/frc-vision-dev.pid"
LOCAL_POD_JOB = None             # pod_job.py the local backend launches (None: deploy/runpod/pod_job.py)
_LOCAL_PROCS: dict = {}          # out path -> Popen of a local pod_job.py we started


def local_start_cmd() -> str:
    """How the local :8004 is (re)started: the main-server worktree (README
    memory: it serves main), else the main checkout's start_all.sh."""
    main = cli.main_checkout()
    for p in (main / ".worktrees" / "main-server" / "web" / "start_all.sh", main / "web" / "start_all.sh"):
        if p.exists():
            return str(p)
    return str(main / "web" / "start_all.sh")


def local_member(name: str, *, app_url: str = LOCAL_APP, app_dir: Path | None = None,
                 python: str | None = None) -> dict:
    """The member dict of the LOCAL backend: jobs run against the local app
    (``app_url``) via pod_job.py as a local subprocess; ``app_dir`` holds the
    DB + web/replays the app writes (main-server symlinks them to the main
    checkout)."""
    main = cli.main_checkout()
    py = python or str(main / ".venv" / "bin" / "python")
    if not Path(py).exists():
        py = sys.executable
    return {"pod_id": "local", "name": "local", "backend": "local", "status": "ready",
            "app_url": app_url, "app_dir": str(app_dir or main), "python": py,
            "pod_job": str(LOCAL_POD_JOB or cli.HERE / "pod_job.py"),
            "jobs_dir": str(SWARM_DIR / name / "local_jobs"),
            "pidfile": LOCAL_PIDFILE, "start_cmd": local_start_cmd(),
            "cost_per_hr": 0, "created_at": _now()}


class LocalMember:
    """The local app as a swarm member: no upload (the video is already
    here), pod_job.py launched as a local subprocess with the same driver
    contract, results read from local files, the local DB holds the replay."""
    kind = "local"

    def video_path(self, m, mk, local: Path) -> str:
        return str(Path(local).resolve())

    def attempt_paths(self, m, key, attempt):
        d = m["jobs_dir"]
        return f"{d}/{safe_key(key)}.a{attempt}.json", f"{d}/{safe_key(key)}.a{attempt}.log"

    def read_job_state(self, m, outp):
        launched = os.path.isdir(outp + ".launch")
        state = None
        try:
            txt = Path(outp).read_text()
            state = json.loads(txt) if txt.strip() else None
        except (OSError, ValueError):
            state = None
        return launched, state

    def driver_alive(self, m, state, outp):
        p = _LOCAL_PROCS.get(outp)
        if p is not None:
            return p.poll() is None
        pid = int((state or {}).get("pid") or 0)
        if not pid:
            return False
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            return stat.rsplit(")", 1)[-1].split()[0] != "Z"
        except OSError:
            return False

    def active_job(self, m):
        try:
            with urllib.request.urlopen(f"{m['app_url']}/api/job/active", timeout=10) as r:
                return json.loads(r.read() or b"{}").get("job_id")
        except (urllib.error.URLError, OSError) as e:
            raise Transport(f"local app {m['app_url']} not answering: {e}")
        except ValueError as e:
            raise RemoteError(f"/api/job/active did not answer JSON: {e}")

    def upload(self, m, local, remote_path, out, what="video"):
        return False

    def launch(self, m, outp, logp, *, video, match_key, event, label, config) -> str:
        Path(outp).parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(outp + ".launch")          # same at-most-once guard as on a pod
        except FileExistsError:
            return "already"
        hostport = m["app_url"].split("://", 1)[-1].rstrip("/")
        cmd = [m["python"], m["pod_job"], "--video", video, "--match-key", match_key,
               "--event", event, "--label", label, "--config-json", json.dumps(config),
               "--host", hostport, "--no-profile", "--final-timeout", f"{FINAL_TIMEOUT_S:g}",
               "--app-dir", m["app_dir"], "--out", outp]
        with open(logp, "ab") as lf:
            _LOCAL_PROCS[outp] = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                                  stdin=subprocess.DEVNULL, start_new_session=True,
                                                  cwd=m["app_dir"])
        return "started"

    def restart(self, m, out) -> None:
        """kill the local app (pidfile), start it again detached, wait for :8004."""
        out(f"restarting the LOCAL app server ({m['start_cmd']})")
        pidf = Path(m.get("pidfile") or LOCAL_PIDFILE)
        pid = 0
        try:
            pid = int(pidf.read_text().strip())
        except (OSError, ValueError):
            pid = 0
        if pid:
            try:
                os.kill(pid, 15)
                for _ in range(60):
                    os.kill(pid, 0)
                    time.sleep(0.5)
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        subprocess.run(["setsid", "-f", "bash", m["start_cmd"]], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                if self.active_job(m) is None:
                    out("local app back up")
                    return
            except (Transport, RemoteError):
                pass
            time.sleep(2)
        raise RemoteError("local app did not come back on restart within 300 s")

    def prepare(self, m, cfg, out) -> None:
        return None          # the local app starts its own vLLM on demand

    def tail_log(self, m, logp) -> str:
        try:
            return "\n".join(Path(logp).read_text(errors="replace").splitlines()[-20:])
        except OSError:
            return ""

    def list_event_state(self, m) -> dict:
        return {}

    def fetch_files(self, m, names, dest):
        return []

    def push_files(self, m, files) -> None:
        return None


TRANSPORTS = {"pod": PodMember(), "local": LocalMember()}


def transport(m: dict):
    return TRANSPORTS[(m.get("backend") or "pod")]


def _ssh_retry(fn, ctx: "RunCtx", m: dict, out, what: str, tries: int = REMOTE_TRIES):
    """Call ``fn`` until it succeeds. Transport failures retry without limit;
    after LOST_AFTER_S of them the API is asked, and only an API answer of
    "not RUNNING / gone" raises PodLost (an unreachable API keeps retrying).
    A remote command's real non-zero exit retries ``tries`` times, then
    RemoteFailed (the item fails; the pod is never stuck on it)."""
    t0, fails = _now(), 0
    while True:
        try:
            return fn()
        except RemoteError as e:
            fails += 1
            out(f"{what}: {e} ({fails}/{tries})")
            if fails >= tries:
                raise RemoteFailed(f"{what}: {e}")
        except Transport as e:
            out(f"{what}: ssh: {e}")
            if _now() - t0 > LOST_AFTER_S and is_pod(m):
                status = pod_api_status(m, ctx.sw.saver(m))
                if status is not None and status != "RUNNING":
                    raise PodLost(f"pod is {status}")
                out(f"{what}: API says {status or 'nothing (unreachable)'}; still retrying")
                t0 = _now()
        if ctx.stop.is_set():
            raise KeyboardInterrupt
        ctx.sleep(min(30, max(1, ctx.poll_s)))


class RunCtx:
    def __init__(self, sw: Swarm, args, sleep=time.sleep):
        self.sw = sw
        self.run = sw.data["run"]
        self.poll_s = args.poll_s
        self.terminate_when_done = args.terminate_when_done
        self.with_video = getattr(args, "with_video", False)
        self.stop = threading.Event()
        self.sleep = sleep
        self.events: list = []
        # live mode: called (outside the lock) after each finished item as
        # on_done(ctx, member, key, ok, state) -- calibration sync, auto-import
        self.on_done = None

    def event(self, msg: str) -> None:
        with self.sw.lock:
            self.events.append(f"{_dt.datetime.now().strftime('%H:%M:%S')} {msg}")
        print(msg, flush=True)


NOTHING_AFTER_S = 300      # a launched driver that never wrote its out file


def attempt_paths(mk: str, attempt: int) -> tuple[str, str]:
    """Per-attempt out/log on the pod (a retry never reuses a file)."""
    k = safe_key(mk)       # sweep keys "Q3/<mk>" -> "Q3__<mk>" (batch keys unchanged)
    return f"{POD_JOBS}/{k}.a{attempt}.json", f"{POD_JOBS}/{k}.a{attempt}.log"


def process_item(ctx: RunCtx, m: dict, mk: str) -> tuple[bool, dict]:
    """Upload, launch (or re-attach to) pod_job.py for ``mk`` on member ``m``,
    prefetch the pod's next video, then poll until done/error."""
    try:
        return _process_item(ctx, m, mk)
    except RemoteFailed as e:
        return False, {"error": str(e)[:400]}


def _process_item(ctx: RunCtx, m: dict, key: str) -> tuple[bool, dict]:
    run, sw, pod = ctx.run, ctx.sw, m["pod_id"]
    out = sw.logger(m)
    T = transport(m)
    it = run["items"][key]
    mk = item_match_key(run, key)
    label = f"{safe_key(key)}.a{it['attempts']}"
    local = Path(it["video"])
    remote_video = T.video_path(m, video_name(key), local)
    outp, logp = T.attempt_paths(m, key, it["attempts"])

    def R(fn, what):
        return _ssh_retry(fn, ctx, m, out, what)
    launched, state = R(lambda: T.read_job_state(m, outp), "read job state")
    if state and state.get("status") in ("done", "error"):
        return state.get("status") == "done", state
    if launched:
        out(f"[{key}] attempt already launched; re-attaching ({label})")
    else:
        if not local.is_file():
            return False, {"error": f"local video missing: {local}"}
        R(lambda: T.upload(m, local, remote_video, out), "upload video")
        waited = _now()
        while True:   # never start a second job on a busy server (ssh failure = unknown, not idle)
            jid = R(lambda: T.active_job(m), "job/active")
            if not jid:
                break
            if _now() - waited > FINAL_TIMEOUT_S:
                return False, {"error": f"server stayed busy with job {jid}"}
            out(f"[{key}] server busy with job {jid}; waiting")
            ctx.sleep(ctx.poll_s)
        cfg = job_config(run, pod, key)
        if run.get("restart_each"):
            # fresh server per job: the ~0.7 GB/job CUDA leak shifts tracking
            # after a few jobs, so job order would confound a config sweep
            R(lambda: T.restart(m, out), "restart server")
            with sw.lock:
                it["restarts"] = it.get("restarts", 0) + 1
        R(lambda: T.prepare(m, cfg, out), "prepare")
        with sw.lock:
            it["config"] = cfg
        how = R(lambda: T.launch(m, outp, logp, video=remote_video, match_key=mk,
                                 event=item_event(run, key), label=label, config=cfg), "launch")
        out(f"[{key}] launch: {how} ({label})")
    with sw.lock:
        it["launched_at"] = it.get("launched_at") or _now()
        sw.save()
    # prefetch the next video of this pod's block while the job runs
    with sw.lock:
        nxt = peek_next(run, pod)
    if nxt:
        nloc = Path(run["items"][nxt]["video"])
        try:
            if nloc.is_file():
                T.upload(m, nloc, T.video_path(m, video_name(nxt), nloc), out, what="next video")
        except (Transport, RemoteError, OSError) as e:
            out(f"prefetch {nxt} failed (retried at launch): {e}")
    seen = state is not None
    window = last_ok = _now()           # "wrote nothing" clock: only while no state was ever seen
    polls = 0
    while not ctx.stop.is_set():
        ctx.sleep(ctx.poll_s)
        polls += 1
        _, new = R(lambda: T.read_job_state(m, outp), "poll")
        now = _now()
        if now - last_ok > 3 * ctx.poll_s + 60:
            window = now                # we were offline (ssh retries / laptop sleep): fresh window
        last_ok = now
        if new is None or new.get("label") not in (None, label):
            if not seen and now - window > NOTHING_AFTER_S:
                if R(lambda: T.driver_alive(m, None, outp), "driver check"):
                    window = now        # alive but slow to write: keep waiting
                    continue
                log = ""
                try:
                    log = T.tail_log(m, logp)
                except (Transport, RemoteError):
                    pass
                return False, {"error": f"pod_job.py wrote nothing in {NOTHING_AFTER_S}s and is not "
                                        f"running: {log[-300:]}"}
            continue
        state, seen = new, True
        with sw.lock:
            prog = 100 if state.get("status") == "done" else state.get("progress", it.get("progress"))
            it.update(progress=prog, message=str(state.get("message") or "")[:80],
                      job_id=state.get("job_id") or it.get("job_id"), driver_status=state.get("status"))
            sw.save()
        if state.get("status") in ("done", "error"):
            return state.get("status") == "done", state
        if polls % 10 == 0 and not R(lambda: T.driver_alive(m, state, outp), "alive"):
            _, fin = R(lambda: T.read_job_state(m, outp), "final read")   # it may have just finished
            if fin and fin.get("status") in ("done", "error"):
                return fin.get("status") == "done", fin
            return False, {"error": "pod_job.py driver died", "job_id": state.get("job_id")}
    raise KeyboardInterrupt


def pull_member(sw: Swarm, m: dict, with_video: bool = False, out=None) -> dict:
    out = out or sw.logger(m)
    cli.refresh_endpoint(m, save=sw.saver(m))
    res = cli.pull_pod(m, with_video, out=out)
    dest = Path(res["dest"])
    run = sw.data.get("run") or {}
    jobs = [it.get("job_id") for it in run.get("items", {}).values()
            if it.get("pod") == m["pod_id"] and it.get("status") == "done" and it.get("job_id")]
    covered = [j for j in jobs if cli.replay_pulled(dest / "web" / "replays" / j)]
    sw.update(m, pulled_at=_now(), pulled_jobs=len(covered), pulled_job_ids=covered,
              pulled_replays_on_pod=len(res["replays_on_pod"]))
    return res


def pull_covers(sw: Swarm, m: dict) -> tuple[bool, str]:
    """Is the member's last finished job covered by its latest pull?"""
    run = sw.data.get("run") or {}
    done = [it for it in run.get("items", {}).values()
            if it.get("pod") == m["pod_id"] and it.get("status") == "done" and it.get("job_id")]
    running = [mk for mk, it in run.get("items", {}).items()
               if it.get("pod") == m["pod_id"] and it.get("status") == "running"]
    if running:
        return False, f"job running ({running[0]})"
    if not done:
        return True, "no finished jobs"
    last = max(done, key=lambda it: it.get("finished_at") or 0)
    pulled = m.get("pulled_at")
    have = cli.replay_pulled(cli.PULL_ROOT / m["pod_id"] / "web" / "replays" / last["job_id"])
    if pulled and pulled >= (last.get("finished_at") or 0) and have:
        return True, f"pulled {len(m.get('pulled_job_ids') or [])} job(s)"
    return False, f"last job {last['job_id'][:8]} not pulled"


def terminate_member(sw: Swarm, m: dict, request=None) -> None:
    (request or cli._request)("DELETE", f"{cli.REST}/pods/{m['pod_id']}")
    sw.update(m, status="terminated", terminated_at=_now())


def pod_worker(ctx: RunCtx, m: dict) -> None:
    run, sw, pod = ctx.run, ctx.sw, m["pod_id"]
    out = sw.logger(m)
    while not ctx.stop.is_set():
        with sw.lock:
            mk = next_item(run, pod)
            retiring = finished = False
            if mk is None:
                finished = run_finished(run)
                # Retire (atomically with the empty pick, so another pod
                # never waits on one that is leaving) only when nothing can
                # come back to it: no retry pending, no live arrivals.
                retiring = (ctx.terminate_when_done and not run.get("open")
                            and not run["retry"])
                if retiring:
                    retire_pod(run, pod)
            sw.save()
        if mk is None:
            if finished or retiring:
                break
            # others may still fail an item into the retry pool, or (open
            # run) new matches may arrive via add_items
            ctx.sleep(ctx.poll_s)
            continue
        ctx.event(f"[{pod}] {mk}: start (attempt {run['items'][mk]['attempts']})")
        try:
            ok, state = process_item(ctx, m, mk)
        except PodLost as e:
            with sw.lock:
                mark_pod_dead(run, pod, str(e))
                sw.update(m, status="lost", error=str(e)[:200])
            ctx.event(f"[{pod}] LOST: {e}; its block is up for stealing")
            return
        except KeyboardInterrupt:
            return
        except Exception as e:           # unexpected: count as a failed attempt
            ok, state = False, {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        info = {k: state.get(k) for k in ("job_id", "wall_job_s", "wall_final_s", "error", "replay",
                                          "peak_vram_mib")
                if state.get(k) is not None}
        with sw.lock:
            st = finish_item(run, mk, pod, ok, **info)
            sw.save()
        wall = state.get("wall_final_s")
        ctx.event(f"[{pod}] {mk}: {st}" + (f" in {wall / 60:.1f} min" if wall else "")
                  + (f" ({str(state.get('error'))[:120]})" if not ok else ""))
        if ctx.on_done is not None:
            try:
                ctx.on_done(ctx, m, mk, ok, state)
            except Exception as e:        # a hook failure never stops the worker
                ctx.event(f"[{pod}] after-job hook failed: {type(e).__name__}: {str(e)[:200]}")
    if ctx.stop.is_set() or not ctx.terminate_when_done or pod not in run["retired"]:
        return
    if not is_pod(m):
        ctx.event(f"[{pod}] no work left")
        return
    # --terminate-when-done: nothing left for this pod and nothing to steal
    ctx.event(f"[{pod}] no work left: pulling, then terminating (--terminate-when-done)")
    try:
        pull_member(sw, m, ctx.with_video, out=out)
        ok, why = pull_covers(sw, m)
        if not ok:
            ctx.event(f"[{pod}] NOT terminated: {why}")
            return
        terminate_member(sw, m)
        ctx.event(f"[{pod}] terminated (${member_cost(m):.2f} GPU total)")
    except (Exception, SystemExit) as e:
        ctx.event(f"[{pod}] pull/terminate failed, pod left running: {e}")


def _fmt_min(s: float | None) -> str:
    if s is None:
        return "-"
    s = max(0, s)
    return f"{s / 3600:.0f}h{(s % 3600) / 60:02.0f}m" if s >= 3600 else f"{s / 60:.0f}m"


def _mean_wall(items: list) -> float | None:
    w = [it.get("wall_final_s") or ((it.get("finished_at") or 0) - (it.get("launched_at") or 0))
         for it in items if it.get("status") == "done"]
    w = [x for x in w if x and x > 0]
    return sum(w) / len(w) if w else None


def run_table(sw: Swarm, now: float | None = None) -> str:
    run = sw.data.get("run")
    if not run:
        return "(no run yet)"
    now = now or _now()
    items = run["items"]
    overall = _mean_wall(list(items.values())) or DEFAULT_JOB_MIN * 60
    rows = [f"{'pod':<16} {'gpu':<14} {'match':<16} {'prog':>5} {'done':>4} {'fail':>4} {'left':>4} "
            f"{'ETA':>6} {'$':>7}  state"]
    by_id = {m["pod_id"]: m for m in sw.members}
    for pod in run["queues"]:
        m = by_id.get(pod, {})
        mine = [it for it in items.values() if it.get("pod") == pod]
        cur = run["current"].get(pod)
        prog = (items[cur].get("progress") or 0) if cur else None
        left = len(run["queues"][pod])
        mean = _mean_wall(mine) or overall
        eta = (left + (1 - (prog or 0) / 100.0 if cur else left)) * mean if (cur or left) else 0
        state = ("dead" if pod in run["dead"] else "retired" if pod in run["retired"]
                 else m.get("status", "?"))
        rows.append(f"{pod:<16} {str(m.get('gpu') or '-')[:14]:<14} {(cur or '-'):<16} "
                    f"{(f'{prog:.0f}%' if prog is not None else '-'):>5} "
                    f"{sum(1 for it in mine if it['status'] == 'done'):>4} "
                    f"{sum(1 for it in mine if it['status'] in ('failed', 'retry')):>4} {left:>4} "
                    f"{_fmt_min(eta):>6} {member_cost(m, now) if m else 0:>7.2f}  {state}")
    c = run_counts(run)
    live = _live_pods(run)
    remaining = c["pending"] + c["retry"] + sum(
        1 - (items[mk].get("progress") or 0) / 100.0 for mk in run["current"].values() if mk)
    eta = remaining * overall / max(1, len(live)) if live else None
    burn = sum(float(m.get("cost_per_hr") or 0) for m in sw.members
               if m.get("status") in BILLING_STATUSES and is_pod(m))
    total = sum(member_cost(m, now) for m in sw.members)
    rows.append(f"total: done {c['done']}/{c['total']}  failed {c['failed']}  running {c['running']}  "
                f"retry {c['retry']}  | ETA {_fmt_min(eta)}  | ${total:.2f} so far, burning ${burn:.2f}/h")
    return "\n".join(rows)


def parse_pod_configs(specs) -> dict:
    """["POD_ID={json}", ...] -> {pod_id: dict}."""
    out = {}
    for spec in specs or []:
        pod_id, sep, js = spec.partition("=")
        try:
            val = json.loads(js) if sep else None
        except ValueError:
            val = None
        if not pod_id.strip() or not isinstance(val, dict):
            raise SystemExit(f"bad --pod-config {spec!r} (POD_ID='{{\"knob\": value}}')")
        out[pod_id.strip()] = val
    return out


class run_lock:
    """Exclusive flock on .swarm/<name>.run.lock: one `swarm run` per swarm
    (two dispatchers would launch every match twice)."""

    def __init__(self, sw: Swarm):
        self.path = sw.path.with_name(sw.path.name.replace(".json", "") + ".run.lock")
        self.f = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a+")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.f.seek(0)
            holder = self.f.read().strip()
            self.f.close()
            raise SystemExit(f"another `swarm run` is already driving this swarm ({holder or 'unknown pid'}; "
                             f"lock {self.path})")
        self.f.seek(0)
        self.f.truncate()
        self.f.write(f"pid {os.getpid()} since {_dt.datetime.now().isoformat(timespec='seconds')}")
        self.f.flush()
        return self

    def __exit__(self, *exc):
        try:
            fcntl.flock(self.f, fcntl.LOCK_UN)
        finally:
            self.f.close()
        return False


def revive_lost(sw: Swarm, status_fn=None) -> list:
    """Members marked lost whose pod the API now reports RUNNING go back to
    ready (an API we can't reach changes nothing)."""
    status_fn = status_fn or pod_api_status
    back = []
    for m in sw.members:
        if m.get("status") == "lost" and is_pod(m) and status_fn(m, sw.saver(m)) == "RUNNING":
            sw.update(m, status="ready", error=None)
            back.append(m["pod_id"])
    return back


def cmd_run(args) -> int:
    sw = Swarm.load(args.name)
    with run_lock(sw):
        return _cmd_run(args, sw)


def ensure_local_member(sw: Swarm, args) -> dict:
    """The swarm's LOCAL backend member (created on first use; app URL /
    dir refreshed from the flags each time)."""
    want = local_member(sw.name, app_url=getattr(args, "local_url", None) or LOCAL_APP,
                        app_dir=Path(args.local_app_dir) if getattr(args, "local_app_dir", None) else None)
    with sw.lock:
        for m in sw.members:
            if m["pod_id"] == "local":
                m.update({k: want[k] for k in ("app_url", "app_dir", "python", "pod_job", "jobs_dir",
                                               "start_cmd", "backend")}, status="ready")
                sw.save()
                return m
        sw.members.append(want)
        sw.save()
    return want


def run_pods(sw: Swarm, args) -> list:
    """The members a run dispatches to: the local app (--local) or the
    ready pods."""
    if getattr(args, "local", False):
        return [ensure_local_member(sw, args)["pod_id"]]
    return [m["pod_id"] for m in ready_members(sw)]


def warn_local_restart(args, restart_each: bool) -> None:
    if getattr(args, "local", False) and not restart_each:
        print("WARNING: --local without --restart-each: the local app keeps its ~0.7 GB/job CUDA leak "
              "between jobs, so job ORDER may confound config comparisons (pass --restart-each to "
              "restart :8004 before every job)")


def drive_run(sw: Swarm, args, run: dict, pods: list, rerun: str, *, ctx: "RunCtx | None" = None,
              table=None, side_threads: list | None = None) -> int:
    """One worker thread per member until the run is finished (or Ctrl-C:
    130). ``side_threads`` (e.g. the live arrival poller) are started too;
    ``table`` renders the periodic status (default run_table)."""
    ctx = ctx or RunCtx(sw, args)
    workers = [threading.Thread(target=pod_worker, args=(ctx, sw.member(p)), daemon=True)
               for p in run["queues"] if p in pods and p not in run["dead"] and p not in run["retired"]]
    if not workers and not run_finished(run):
        raise SystemExit("no ready pod can take the remaining work (`swarm status`; `swarm up --add N`)")
    for t in workers + list(side_threads or []):
        t.start()
    render = table or (lambda: run_table(sw))

    def table() -> str:          # arrivals / workers mutate the run: render under the lock
        with sw.lock:
            return render()
    try:
        last_print = 0.0
        while any(t.is_alive() for t in workers):
            time.sleep(1)
            if time.time() - last_print >= args.poll_s:
                print("\n" + table(), flush=True)
                last_print = time.time()
    except KeyboardInterrupt:
        ctx.stop.set()
        sw.save()
        print("\ninterrupted: jobs already launched keep running (pods keep billing). "
              f"Re-run `{rerun}` to re-attach.")
        return 130
    ctx.stop.set()                 # side threads (pollers) end with the workers
    print("\n" + table())
    return 0


def _cmd_run(args, sw: Swarm) -> int:
    plan = sw.data.get("plan")
    if not plan or not plan.get("items"):
        raise SystemExit(f"no plan with work: `swarm plan --name {args.name} --event EV ...` first")
    cfg = json.loads(args.config_json) if args.config_json else None
    if cfg is None and plan.get("config") is not None:
        cfg = plan["config"]                    # sweep plans carry their base config
    pod_cfg = parse_pod_configs(getattr(args, "pod_config", None))
    for p in revive_lost(sw):
        print(f"{p}: the API reports it RUNNING again; back in the run")
    run = sw.data.get("run")
    pods = run_pods(sw, args)
    with sw.lock:
        if not run or run.get("plan_id") != plan["id"]:
            if _run_active(run):
                raise SystemExit("the previous run still has jobs in flight; re-plan after it ends")
            if run:
                sw.data.setdefault("run_history", []).append(run)
            if not pods:
                raise SystemExit("no ready members (`swarm up` / `swarm status`; or --local)")
            unknown = set(pod_cfg) - set(pods)
            if unknown:
                raise SystemExit(f"--pod-config for pods not ready in this swarm: {', '.join(sorted(unknown))}")
            policy = plan.get("policy") or getattr(args, "policy", None) or "blocks"
            restart_each = getattr(args, "restart_each", None)
            if restart_each is None:
                restart_each = bool(plan.get("restart_each")) and not getattr(args, "local", False)
            warn_local_restart(args, restart_each)
            run = sw.data["run"] = new_run(plan, pods, cfg, policy=policy, pod_config=pod_cfg,
                                           restart_each=restart_each)
            if not getattr(args, "local", False) and run_wants_gdrive(run):
                print("NOTE: " + GDRIVE_REMOTE_NOTE.format(name=args.name))
            print(f"new run ({policy}{', restart before each job' if restart_each else ''}): "
                  f"{len(plan['items'])} item(s) on {len(pods)} member(s): "
                  + "; ".join(f"{p}: {len(q)}" for p, q in run["queues"].items())
                  + (f"; {len(run.get('unassigned_groups') or [])} config(s) waiting for a free pod"
                     if run.get("unassigned_groups") else ""))
        else:
            if getattr(args, "restart_each", None) is not None and args.restart_each != run.get("restart_each"):
                run["restart_each"] = bool(args.restart_each)
                print(f"restart before each job now {'ON' if run['restart_each'] else 'OFF'}")
            if cfg is not None and cfg != run.get("config"):
                print(f"NOTE: this run's config is fixed at {json.dumps(run.get('config'))}; "
                      f"--config-json ignored (re-plan to start a new run)")
            if pod_cfg:
                run.setdefault("pod_config", {}).update(pod_cfg)   # jobs launched from now on
                print(f"per-pod config overrides now: {json.dumps(run['pod_config'])}")
            for p in pods:
                add_pod(run, p)
            gone = {m["pod_id"] for m in sw.members if m.get("status") in ("terminated", "lost", "stopped")}
            for p in list(run["queues"]):
                if p in gone and p not in run["dead"] and p not in run["retired"]:
                    mark_pod_dead(run, p, "member not running")
            print(f"resuming run: {json.dumps(run_counts(run))}")
        sw.save()
    rc = drive_run(sw, args, run, pods, f"swarm run --name {args.name}")
    if rc == 130:
        return rc
    c = run_counts(run)
    for mk, it in run["items"].items():
        if it["status"] == "failed":
            print(f"  FAILED {mk}: {(it.get('errors') or [{}])[-1].get('error')}")
    print(f"\nrun finished: {c['done']} done, {c['failed']} failed of {c['total']}.")
    pull = "" if getattr(args, "local", False) else f"swarm pull --name {args.name}   (if not terminated)\n      "
    if plan.get("kind") == "sweep":
        print(f"next: {pull}swarm sweep score --name {args.name}\n      swarm sweep report --name {args.name}")
    elif getattr(args, "local", False):
        print("results are in the local DB (local backend: nothing to pull or import)")
    else:
        print(f"next: {pull}swarm import --name {args.name}            (dry run)\n"
              f"      swarm import --name {args.name} --apply")
        if run_wants_gdrive(run):
            print("      (" + GDRIVE_REMOTE_NOTE.format(name=args.name) + ")")
    return 0 if c["failed"] == 0 and run_finished(run) else 1


# --------------------------------------------------------------------------
# status / pull
# --------------------------------------------------------------------------
def cmd_status(args) -> int:
    sw = Swarm.load(args.name)
    now = _now()
    print(f"swarm {sw.name}: {len(sw.members)} member(s), gpu priority "
          f"{', '.join(sw.data.get('gpu_priority') or [])} ({sw.data.get('cloud')})")
    print(f"{'#':>3} {'pod':<16} {'local':<13} {'runpod':<9} {'gpu':<22} {'$/h':>5} {'up h':>5} {'$':>7}  pulled")
    for m in sw.members:
        rp = "-"
        up_h = None
        if m.get("status") != "terminated" and is_pod(m):
            try:
                pod = cli.refresh_endpoint(m, save=sw.saver(m))
                rp = pod.get("desiredStatus") or "?"
                t = cli.parse_ts(pod.get("lastStartedAt"))
                if t:
                    up_h = (_dt.datetime.now(_dt.timezone.utc) - t).total_seconds() / 3600
            except (Exception, SystemExit) as e:
                rp = f"? ({str(e)[:20]})"
        pulled = (_dt.datetime.fromtimestamp(m["pulled_at"]).strftime("%m-%d %H:%M")
                  + f" ({m.get('pulled_jobs', 0)} jobs)") if m.get("pulled_at") else "-"
        print(f"{m.get('idx', '-')!s:>3} {m['pod_id']:<16} {m.get('status', '-'):<13} {rp:<9} "
              f"{str(m.get('gpu') or '-')[:22]:<22} {m.get('cost_per_hr') or '-'!s:>5} "
              f"{(f'{up_h:.1f}' if up_h is not None else '-'):>5} {member_cost(m, now):>7.2f}  {pulled}")
    for e in sw.data.get("create_errors") or []:
        print(f"  create error: {e['error'][:150]}")
    plan = sw.data.get("plan")
    if plan:
        print()
        print_plan_summary(plan)
    if sw.data.get("run"):
        print("\n" + run_table(sw, now))
    return 0


def cmd_pull(args) -> int:
    sw = Swarm.load(args.name)
    todo = [m for m in sw.members if m.get("status") in BILLING_STATUSES + ("stopped",)
            and m.get("status") != "created" and is_pod(m)]
    if not todo:
        print("no reachable members to pull")
        return 0
    results = {}

    def one(m):
        try:
            results[m["pod_id"]] = pull_member(sw, m, args.with_video)
        except (Exception, SystemExit) as e:
            results[m["pod_id"]] = e
    threads = [threading.Thread(target=one, args=(m,), daemon=True) for m in todo]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rc = 0
    for m in todo:
        r = results.get(m["pod_id"])
        if isinstance(r, BaseException) or r is None:
            rc = 1
            print(f"  {m['pod_id']}: pull FAILED: {r}")
        else:
            print(f"  {m['pod_id']}: {len(r['new_replays'])} new replay(s), {len(r['replays_on_pod'])} on pod, "
                  f"{m.get('pulled_jobs', 0)} run job(s) covered -> {r['dest']}")
    print(f"next: swarm import --name {args.name}   (dry run; --apply to merge into the local app)")
    return rc


# --------------------------------------------------------------------------
# import
# --------------------------------------------------------------------------
# Rows a finished job writes, keyed by the replay/job id (grep of
# web/app.py + web/database.py, Nov 13 2026):
#   replays                       insert_replay at finalize (id = job id)
#   match_robot_identity          upsert_match_robot_identities (job_id)
#   match_robot_crop              insert_match_robot_crops (job_id)
#   match_robot_identity_segment  replace_match_identity_segments (job_id)
#   match_robot_binding           insert_match_robot_binding, live (job_id)
#   appearance_gallery            insert_gallery_exemplars (job_id)
#   replay_slot_corrections       manual corrections only (replay_id)
# NOT merged: event_anchors (event_lineup knob, default OFF; its tiles live in
# data/event_anchors on the pod and are not pulled) and the event-load tables
# events / matches / match_robots (TBA; local already has them). scoring_events
# and match_robots fuel counts are written only by the UI's /api/match/save.
JOB_TABLES = (("match_robot_identity", "job_id"), ("match_robot_crop", "job_id"),
              ("match_robot_identity_segment", "job_id"), ("match_robot_binding", "job_id"),
              ("appearance_gallery", "job_id"), ("replay_slot_corrections", "replay_id"))
REPORT_ONLY_TABLES = (("event_anchors", "replay_id"),)
REPLAY_FILE_COLS = {"video_path": "hud.mp4", "meta_path": "meta.json", "thumb_path": "thumb.jpg"}


def _columns(con, table: str) -> list:
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def _rows(con, table: str, key: str, val: str) -> list:
    cols = _columns(con, table)
    if not cols or key not in cols:
        return []
    cur = con.execute(f"SELECT * FROM {table} WHERE {key} = ?", (val,))
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


def plan_import(sw: Swarm, *, local_db: Path, replays_dir: Path, pull_root: Path,
                uploads_dir: Path | None = None) -> dict:
    """What `import` would add: one entry per pulled replay of every member."""
    run = sw.data.get("run") or {}
    run_videos = {item_match_key(run, k): it.get("video") for k, it in (run.get("items") or {}).items()}
    for i in (sw.data.get("plan") or {}).get("items") or []:
        run_videos.setdefault(i["match_key"], i["video"])
    lcon = sqlite3.connect(f"file:{local_db}?mode=ro", uri=True)
    entries, seen = [], set()
    try:
        for m in sw.members:
            if not is_pod(m):
                continue          # the local backend wrote straight into the local DB
            dest = Path(pull_root) / m["pod_id"]
            rdir, pdb = dest / "web" / "replays", dest / "frc_scorer.db"
            if not rdir.is_dir() or not pdb.exists():
                continue
            pcon = sqlite3.connect(f"file:{pdb}?mode=ro", uri=True)
            try:
                for d in sorted(rdir.iterdir()):
                    if not (d / "meta.json").exists() or d.name in seen:
                        continue
                    rid = d.name
                    seen.add(rid)
                    e = {"id": rid, "pod_id": m["pod_id"], "src": str(d), "tables": {}, "notes": []}
                    if not cli.replay_pulled(d):
                        e["action"] = "skip"
                        e["notes"].append("pull not marked complete (old or interrupted pull): `swarm pull` again")
                        entries.append(e)
                        continue
                    rows = _rows(pcon, "replays", "id", rid)
                    if not rows:
                        e["action"] = "skip"
                        e["notes"].append("no replays row in the pulled DB")
                        entries.append(e)
                        continue
                    row = rows[0]
                    e["match_key"] = row.get("match_key")
                    e["status"] = row.get("status")
                    if row.get("status") != "complete":
                        e["action"] = "skip"
                        e["notes"].append(f"status {row.get('status')!r}: only complete replays are imported")
                        entries.append(e)
                        continue
                    if lcon.execute("SELECT 1 FROM replays WHERE id = ?", (rid,)).fetchone():
                        e["action"] = "skip"
                        e["notes"].append("id already in the local DB (never overwritten)")
                        entries.append(e)
                        continue
                    if (Path(replays_dir) / rid).exists():
                        e["action"] = "skip"
                        e["notes"].append(f"{Path(replays_dir) / rid} already exists (never overwritten)")
                        entries.append(e)
                        continue
                    e["action"] = "add"
                    local_dir = Path(replays_dir) / rid
                    new = dict(row)
                    for col, fname in REPLAY_FILE_COLS.items():
                        if col in new and new[col]:
                            new[col] = str(local_dir / Path(new[col]).name)
                        elif col in new:
                            new[col] = str(local_dir / fname)
                    vid = run_videos.get(row.get("match_key"))
                    if not (vid and Path(vid).is_file()):
                        # same file name in local uploads (UI uploads), else <match_key>.mp4
                        src = Path(row.get("source_video_path") or "")
                        cand = Path(uploads_dir) / src.name if uploads_dir and src.name else None
                        if cand is not None and cand.is_file():
                            vid = str(cand)
                        elif row.get("match_key"):
                            found = find_video(row["match_key"], video_dirs())
                            vid = str(found) if found else None
                    if vid and Path(vid).is_file():
                        new["source_video_path"] = str(Path(vid).resolve())
                    else:
                        e["notes"].append(f"source video not local (kept {row.get('source_video_path')})")
                    e["event_key"] = row.get("event_key") or str(row.get("match_key") or "").split("_")[0] or None
                    e["knobs"] = replay_knobs(d / "meta.json")
                    e["hud_pulled"] = (d / Path(new.get("video_path") or "hud.mp4").name).exists()
                    if not e["hud_pulled"]:
                        if vid and Path(vid).is_file() and "render_status" in new:
                            new["render_status"], new["render_error"] = "queued", None
                            e["notes"].append("no HUD mp4 pulled: render queued (local server renders it)")
                            e["render"] = True
                        else:
                            e["notes"].append("no HUD mp4 (pull --with-video, or re-render once the video is local)")
                    e["replay"] = new
                    for t, key in JOB_TABLES:
                        e["tables"][t] = _rows(pcon, t, key, rid)
                    for t, key in REPORT_ONLY_TABLES:
                        n = len(_rows(pcon, t, key, rid))
                        if n:
                            e["notes"].append(f"{n} {t} row(s) NOT imported (event_lineup tiles are not pulled)")
                    n_local = lcon.execute("SELECT COUNT(*) FROM replays WHERE match_key = ?",
                                           (row.get("match_key"),)).fetchone()[0]
                    if n_local:
                        e["notes"].append(f"local already has {n_local} other replay(s) of this match")
                    entries.append(e)
            finally:
                pcon.close()
    finally:
        lcon.close()
    return {"entries": entries}


def event_state_diff(sw: Swarm, *, uploads_dir: Path, pull_root: Path) -> list:
    """Pulled event-state files (calibrations etc.) that differ from local."""
    out = []
    for m in sw.members:
        if not is_pod(m):
            continue
        up = Path(pull_root) / m["pod_id"] / "web" / "uploads"
        if not up.is_dir():
            continue
        for p in sorted(up.iterdir()):
            if not p.is_file() or not cli.event_state_ok(p.name):
                continue
            loc = Path(uploads_dir) / p.name
            if not loc.exists():
                out.append((m["pod_id"], p.name, "new on pod"))
            elif loc.read_bytes() != p.read_bytes():
                out.append((m["pod_id"], p.name, "differs from local"))
    return out


def local_active_job(url: str = LOCAL_APP, timeout: float = 5):
    """job_id the local app reports as running; None when idle or not running."""
    try:
        with urllib.request.urlopen(f"{url}/api/job/active", timeout=timeout) as r:
            return json.loads(r.read() or b"{}").get("job_id")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def backup_db(local_db: Path, tag: str) -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = Path(local_db).with_name(f"{Path(local_db).name}.bak-swarm-{tag}-{ts}")
    src = sqlite3.connect(f"file:{local_db}?mode=ro", uri=True)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return dest


def apply_import(entries: list, *, local_db: Path, replays_dir: Path) -> int:
    """Copy replay dirs + insert rows, all-or-nothing (one transaction; dirs
    staged under a temp name and moved into place before the commit)."""
    add = [e for e in entries if e.get("action") == "add"]
    if not add:
        return 0
    replays_dir = Path(replays_dir)
    replays_dir.mkdir(parents=True, exist_ok=True)
    staged, placed = [], []
    con = sqlite3.connect(local_db, timeout=60)
    try:
        for e in add:
            tmp = replays_dir / f".swarm-import-{e['id']}.tmp"
            if tmp.exists():
                shutil.rmtree(tmp)
            staged.append(tmp)            # before the copy: a copy that dies half-way is cleaned up too
            shutil.copytree(e["src"], tmp, ignore=shutil.ignore_patterns(cli.PULL_MARKER))
        con.execute("BEGIN IMMEDIATE")
        lcols = {t: set(_columns(con, t)) for t in ("replays",) + tuple(t for t, _ in JOB_TABLES)}
        for e in add:
            if con.execute("SELECT 1 FROM replays WHERE id = ?", (e["id"],)).fetchone():
                raise RuntimeError(f"replay {e['id']} appeared locally during import")
            row = {k: v for k, v in e["replay"].items() if k in lcols["replays"]}
            con.execute(f"INSERT INTO replays ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                        list(row.values()))
            for t, rows in e["tables"].items():
                for r in rows:
                    r = {k: v for k, v in r.items() if k in lcols[t] and not (k == "id" and t != "replays")}
                    if not r:
                        continue
                    con.execute(f"INSERT INTO {t} ({', '.join(r)}) VALUES ({', '.join('?' * len(r))})",
                                list(r.values()))
        for e, tmp in zip(add, staged):
            final = replays_dir / e["id"]
            if final.exists():
                raise RuntimeError(f"{final} appeared during import")
            os.replace(tmp, final)
            placed.append(final)
        con.commit()
    except BaseException:
        con.rollback()
        for p in staged + placed:
            shutil.rmtree(p, ignore_errors=True)
        raise
    finally:
        con.close()
    return len(add)


def cmd_import(args, *, local_db: Path | None = None, replays_dir: Path | None = None,
               uploads_dir: Path | None = None, pull_root: Path | None = None,
               active_job=local_active_job, post=None, upload=None) -> int:
    sw = Swarm.load(args.name)
    main = cli.main_checkout()
    local_db = Path(local_db or main / "frc_scorer.db")
    replays_dir = Path(replays_dir or main / "web" / "replays")
    uploads_dir = Path(uploads_dir or main / "web" / "uploads")
    pull_root = Path(pull_root or cli.PULL_ROOT)
    imp = plan_import(sw, local_db=local_db, replays_dir=replays_dir, pull_root=pull_root,
                      uploads_dir=uploads_dir)
    entries = imp["entries"]
    add = [e for e in entries if e["action"] == "add"]
    print(f"{'APPLY' if args.apply else 'DRY RUN'}: {len(add)} replay(s) to add, "
          f"{len(entries) - len(add)} skipped -> {local_db} + {replays_dir}")
    for e in entries:
        n = {t: len(r) for t, r in e["tables"].items() if r}
        print(f"  {e['action']:<4} {e['id']} {e.get('match_key') or '-':<16} {e.get('status') or '':<9} "
              f"pod {e['pod_id']}  " + (", ".join(f"{t} {c}" for t, c in n.items()) or ""))
        for note in e["notes"]:
            print(f"         - {note}")
    diff = event_state_diff(sw, uploads_dir=uploads_dir, pull_root=pull_root)
    if diff:
        print("\nevent-state files NOT merged (calibrations / layouts made on the pods; copy by hand if wanted):")
        for pod, fname, why in diff:
            print(f"  {pod}: {fname} ({why}) <- {pull_root / pod / 'web' / 'uploads' / fname}")
    gplan = gdrive_import_plan(add, force=getattr(args, "gdrive", False),
                               force_video=getattr(args, "gdrive_video", False))
    if not args.apply:
        if gplan:
            print("\nGoogle Drive upload after --apply:")
            print_gdrive_plan(gplan, main, render=getattr(args, "render", False), dry=True)
        print("\nnothing changed (dry run). --apply to merge.")
        return 0
    if not add:
        return 0
    jid = active_job()
    if jid:
        raise SystemExit(f"the local app is running job {jid}; import after it finishes")
    bak = backup_db(local_db, sw.name)
    print(f"backup: {bak}")
    n = apply_import(entries, local_db=local_db, replays_dir=replays_dir)
    print(f"imported {n} replay(s)")
    queued = [e["id"] for e in add if e.get("render")]
    if queued:
        if getattr(args, "render", False):
            post = post or _post_render
            for rid in queued:
                print(f"  render {rid}: {post(rid)}")
        else:
            print(f"{len(queued)} replay(s) have render_status=queued: restart :8004 (its startup sweep "
                  f"renders them) or re-run with --render while it is up")
    if gplan:
        print("\nGoogle Drive upload:")
        print_gdrive_plan(gplan, main, render=getattr(args, "render", False), dry=False)
        if not upload_gdrive(gplan, main, upload or run_gdrive_upload):
            return 1
    return 0


# --------------------------------------------------------------------------
# Google Drive upload at import. Pods have no rclone and no Drive credentials
# (GDRIVE_* is never shipped: runpod_cli.NEVER_SEND_PREFIXES), so a remote
# run's gdrive_upload knob is honoured here, from the local checkout, after
# the import (web/gdrive_upload.py has the Drive layout).
# --------------------------------------------------------------------------
GDRIVE_REMOTE_NOTE = ("pods have no Google Drive access: the gdrive_upload knob uploads at "
                      "`swarm import --name {name} --apply` (add --render for the HUD videos)")


def replay_knobs(meta_path) -> dict:
    """The job's non-default knobs a pulled replay recorded (meta.json
    ``summary.config``); {} for a missing / unreadable file. Stdlib twin of
    web.gdrive_upload.replay_knobs."""
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        cfg = (meta.get("summary") or {}).get("config") if isinstance(meta, dict) else None
        return dict(cfg) if isinstance(cfg, dict) else {}
    except (OSError, ValueError, AttributeError):
        return {}


def run_wants_gdrive(run: dict | None) -> bool:
    """Whether a run's config (or a per-pod override) turns gdrive_upload on."""
    run = run or {}
    cfgs = [run.get("config") or {}, *((run.get("pod_config") or {}).values())]
    return any(bool((c or {}).get("gdrive_upload")) for c in cfgs)


def gdrive_import_plan(entries: list, *, force: bool = False, force_video: bool = False) -> dict:
    """What to upload for imported entries, per event:
    {event: {"matches": [mk], "video": [mk], "after_render": [mk], "notes": [str]}}.
    A match goes up when its job's knobs had gdrive_upload (``force`` /
    ``force_video``: every imported complete match). Its video: ``video``
    when the HUD mp4 is already here; ``after_render`` when the local render
    is queued and the knobs ask for it (the :8004 render hook uploads it);
    else a note."""
    out: dict = {}
    for e in entries:
        mk, ev = e.get("match_key"), e.get("event_key")
        if e.get("action") != "add" or e.get("status") != "complete" or not mk or not ev:
            continue
        knobs = e.get("knobs") or {}
        knob_video = bool(knobs.get("gdrive_upload")) and bool(knobs.get("gdrive_upload_video"))
        if not (force or force_video or knobs.get("gdrive_upload")):
            continue
        p = out.setdefault(ev, {"matches": [], "video": [], "after_render": [], "notes": []})
        p["matches"].append(mk)
        if not (force_video or knob_video):
            continue
        if e.get("hud_pulled"):
            p["video"].append(mk)
        elif e.get("render") and knob_video:
            p["after_render"].append(mk)
        elif e.get("render"):
            p["notes"].append(f"{mk}: HUD render queued and its knobs don't ask for the video: once "
                              f"rendered, `scripts/upload_gdrive.py {ev} --match {mk} --video --no-event-export`")
        else:
            p["notes"].append(f"{mk}: no HUD video here (pull --with-video, or re-render once the "
                              f"source video is local), video not uploaded")
    return out


def gdrive_command(main: Path, event: str, p: dict) -> list:
    """One scripts/upload_gdrive.py call for an event: every match file, the
    listed videos, the event export once."""
    py = Path(main) / ".venv" / "bin" / "python"
    cmd = [str(py if py.exists() else Path(sys.executable)), str(Path(main) / "scripts" / "upload_gdrive.py"), event]
    for mk in p["matches"]:
        cmd += ["--video-match", mk] if mk in p["video"] else ["--match", mk]
    return cmd


def run_gdrive_upload(cmd: list, cwd: Path) -> int:
    """Run the upload script (its output streams to this terminal); its exit code."""
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def print_gdrive_plan(gplan: dict, main: Path, *, render: bool, dry: bool) -> None:
    for ev, p in gplan.items():
        print(f"  {ev}: {len(p['matches'])} match file(s) + event export"
              + (f", {len(p['video'])} video(s)" if p["video"] else ""))
        if dry:
            print(f"    would run: {shlex.join(gdrive_command(main, ev, p))}")
        if p["after_render"]:
            how = ("--render asked :8004 to render them" if render and not dry else
                   "with --render while :8004 is up, or when a :8004 restart's sweep renders them")
            print(f"    {len(p['after_render'])} video(s) go up from :8004 after their HUD render "
                  f"(their knobs ask for it; renders {how}): {', '.join(p['after_render'])}")
        for n in p["notes"]:
            print(f"    - {n}")


def upload_gdrive(gplan: dict, main: Path, runner) -> bool:
    """One upload subprocess per event. A failure is reported, never rolls
    back the import. True when every event uploaded."""
    ok = True
    for ev, p in gplan.items():
        cmd = gdrive_command(main, ev, p)
        try:
            rc = runner(cmd, main)
            err = None if rc == 0 else f"exit {rc}"
        except Exception as e:          # noqa: BLE001 - report, keep the import
            err = f"{type(e).__name__}: {e}"
        if err:
            ok = False
            print(f"  gdrive upload FAILED for {ev} ({err}); the import is kept. Re-send with: "
                  f"{shlex.join(cmd)}")
        else:
            print(f"  gdrive upload done for {ev}")
    return ok


def _post_render(rid: str) -> str:
    req = urllib.request.Request(f"{LOCAL_APP}/api/replays/{rid}/render", data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode()[:120]
    except (urllib.error.URLError, OSError) as e:
        return f"failed: {e}"


# --------------------------------------------------------------------------
# stop / terminate
# --------------------------------------------------------------------------
def cmd_stop(args, request=None) -> int:
    sw = Swarm.load(args.name)
    todo = [m for m in sw.members if m.get("status") in BILLING_STATUSES and is_pod(m)]
    if not todo:
        print("no live members")
        return 0
    running = [mk for mk, it in ((sw.data.get("run") or {}).get("items") or {}).items()
               if it.get("status") == "running"]
    msg = (f"STOP {len(todo)} pod(s) of swarm {sw.name}: {', '.join(m['pod_id'] for m in todo)}. "
           f"GPU billing stops; volumes are kept (and billed for storage).")
    if running:
        msg += f"\n{len(running)} job(s) are RUNNING and will be killed: {', '.join(running)}"
    if not cli.confirm(msg, f"stop swarm {sw.name}"):
        return 1
    request = request or cli._request
    for m in todo:
        try:
            request("POST", f"{cli.REST}/pods/{m['pod_id']}/stop")
            sw.update(m, status="stopped", stopped_at=_now())
            print(f"  {m['pod_id']}: stop requested")
        except (Exception, SystemExit) as e:
            print(f"  {m['pod_id']}: stop FAILED: {e}")
    return 0


def cmd_terminate(args, request=None) -> int:
    sw = Swarm.load(args.name)
    todo = [m for m in sw.members if m.get("status") not in ("terminated",) and is_pod(m)]
    if not todo:
        print("no members left")
        return 0
    allowed, refused = [], []
    for m in todo:
        ok, why = pull_covers(sw, m)
        (allowed if ok or args.force else refused).append((m, why))
    for m, why in refused:
        hint = (" - marked lost: the API said it was not RUNNING; start it in the RunPod console and "
                "`swarm pull`, --force DELETES its unpulled results" if m.get("status") == "lost" else "")
        print(f"  REFUSED {m['pod_id']}: {why} (`swarm pull --name {sw.name}` first, or --force){hint}")
    if not allowed:
        return 1
    msg = (f"TERMINATE {len(allowed)} pod(s) of swarm {sw.name}: the pods AND their volumes are DELETED.\n"
           + "\n".join(f"  {m['pod_id']}: {why}" for m, why in allowed))
    if not cli.confirm(msg, f"terminate swarm {sw.name}"):
        return 1
    for m, _ in allowed:
        try:
            terminate_member(sw, m, request)
            print(f"  {m['pod_id']}: terminated (${member_cost(m):.2f} GPU total)")
        except (Exception, SystemExit) as e:
            print(f"  {m['pod_id']}: terminate FAILED: {e}")
    return 1 if refused else 0


# --------------------------------------------------------------------------
def add_parser(sub) -> None:
    p = sub.add_parser("swarm", help="N cheap pods batch-processing a past event (swarm <sub> --help)")
    ss = p.add_subparsers(dest="swarm_cmd", required=True)

    def named(name, help_):
        q = ss.add_parser(name, help=help_)
        q.add_argument("--name", required=True, help="swarm name (state: deploy/runpod/.swarm/<name>.json)")
        return q

    q = named("up", "create K pods, ship the bundle once, secrets once, bootstrap in parallel")
    q.add_argument("--count", type=int, help="target number of live pods (new swarm: required)")
    q.add_argument("--add", type=int, help="grow the swarm by this many pods")
    q.add_argument("--gpu", action="append",
                   help=f"gpuTypeId, repeatable priority list (default {', '.join(DEFAULT_SWARM_GPUS)})")
    q.add_argument("--cloud", choices=("SECURE", "COMMUNITY"))
    q.add_argument("--yes", action="store_true", help="skip the rent prompt (keys still need --secrets-from-env)")
    q.add_argument("--secrets-from-env", action="store_true")
    q.add_argument("--dry-run", action="store_true", help="print bodies, bundle manifest, cost; create nothing")
    q.add_argument("--ref", default="main")
    q.add_argument("--with-vllm", action="store_true", help="also build the ~9 GB vLLM reader venv (default skipped)")
    q.add_argument("--no-sam2-fallback", action="store_true")
    q.add_argument("--min-vcpu", type=int, default=8)
    q.add_argument("--min-ram", type=int, default=32)
    q.add_argument("--volume-gb", type=int, default=80)
    q.add_argument("--container-gb", type=int, default=40)
    q.add_argument("--image", default=cli.DEFAULT_IMAGE)
    q.add_argument("--data-center", action="append")
    q.add_argument("--poll-s", type=float, default=15)
    q.add_argument("--scratch", default=None)
    q.set_defaults(fn=cmd_up)

    q = named("plan", "the work list: matches in schedule order + local videos")
    q.add_argument("--event", required=True)
    q.add_argument("--matches", default="all", help='e.g. "all", "qm", "playoffs", "qm1-qm40,sf1m1"')
    q.add_argument("--videos", help="dir of <match_key>.mp4 files")
    q.add_argument("--download", action="store_true", help="fetch missing videos from YouTube on THIS machine")
    q.add_argument("--cache-dir", help="download dir (default: main checkout data/videos)")
    q.add_argument("--redo", action="store_true", help="include matches that already have a result")
    q.set_defaults(fn=cmd_plan)

    q = named("run", "dispatch the plan across the ready pods (re-attachable)")
    q.add_argument("--config-json", help="per-job knobs for every match, e.g. '{\"sam3_track_fps\": 8}'")
    q.add_argument("--pod-config", action="append", metavar="POD_ID=JSON",
                   help="override knobs on one pod, merged over --config-json (repeatable; e.g. a "
                        "quality sweep with a different config per pod)")
    q.add_argument("--policy", default="blocks", choices=sorted(POLICIES),
                   help="assignment policy for a NEW run (blocks = contiguous blocks + work stealing)")
    q.add_argument("--terminate-when-done", action="store_true",
                   help="pull + terminate each pod once it has no work left (the flag is the confirmation)")
    q.add_argument("--with-video", action="store_true", help="with --terminate-when-done: pull HUD mp4s too")
    q.add_argument("--poll-s", type=float, default=20)
    add_member_flags(q)
    q.set_defaults(fn=cmd_run)

    q = named("status", "members (RunPod state, $/h, cost) + run progress")
    q.set_defaults(fn=cmd_status)

    q = named("pull", "pull every member into data/runpod_pull/<podId>/")
    q.add_argument("--with-video", action="store_true")
    q.set_defaults(fn=cmd_pull)

    q = named("import", "merge pulled results into the local app (dry run unless --apply)")
    q.add_argument("--apply", action="store_true")
    q.add_argument("--render", action="store_true",
                   help="ask the local :8004 to render HUD videos for imported replays that have none")
    q.add_argument("--gdrive", action="store_true",
                   help="upload every imported match to Google Drive (default: only matches whose job "
                        "had the gdrive_upload knob)")
    q.add_argument("--gdrive-video", action="store_true",
                   help="like --gdrive, plus each match's HUD video (default: only when the job had "
                        "gdrive_upload_video)")
    q.set_defaults(fn=cmd_import)

    q = named("stop", "stop every member (one typed confirmation)")
    q.set_defaults(fn=cmd_stop)
    q = named("terminate", "terminate every member whose results are pulled")
    q.add_argument("--force", action="store_true", help="also members whose last job is not pulled")
    q.set_defaults(fn=cmd_terminate)

    import sweep        # sibling modules; they import this one (no cycle at import time)
    import live
    sweep.add_parser(ss, named)
    live.add_parser(ss, named)


def add_member_flags(q) -> None:
    """Flags shared by every dispatching command (run, sweep run, live)."""
    q.add_argument("--local", action="store_true",
                   help="dispatch to the LOCAL app (:8004) instead of pods: pod_job.py runs here as a "
                        "subprocess, results land in the local DB (nothing rented, nothing pulled)")
    q.add_argument("--local-url", default=LOCAL_APP, help="local app URL (with --local)")
    q.add_argument("--local-app-dir", default=None,
                   help="dir holding the local app's frc_scorer.db + web/replays (default main checkout)")
    g = q.add_mutually_exclusive_group()
    g.add_argument("--restart-each", dest="restart_each", action="store_true", default=None,
                   help="restart the member's app server before EVERY job (the ~0.7 GB/job CUDA leak "
                        "otherwise shifts tracking after a few jobs). Default: ON for sweeps on pods, "
                        "OFF otherwise (and OFF for --local: it restarts :8004 itself)")
    g.add_argument("--no-restart-each", dest="restart_each", action="store_false")
