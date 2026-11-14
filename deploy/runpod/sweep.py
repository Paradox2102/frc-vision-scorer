"""Quality sweep: K configs x M eval matches, one config per pod, scored locally.

Why one config per pod: pods differ in speed and host, so mixing configs on
a pod (or one config over several pods) would mix those differences into the
comparison. Every item is a (config, match) pair keyed "<config_id>/<match_key>";
the ``pinned`` policy gives config i to pod i (with fewer pods than configs, a
pod that finishes takes the next whole config). The ~0.7 GB/job CUDA leak in
the app server shifts tracking after a few jobs, so sweep runs restart the
server before every job by default (``--no-restart-each`` to skip).

    swarm sweep evalset --out FILE [--auto] [--add MATCH_KEY ...]   eval matches + videos + truths
    swarm sweep plan    --name N --evalset FILE --configs FILE       validate, write the work list
    swarm sweep run     --name N [--local] [--terminate-when-done]   dispatch (= swarm run)
    swarm sweep score   --name N [--audit [--yes]]                   score every finished item locally
    swarm sweep report  --name N [--baselines Q0a,Q0b]               rank configs, identity first

Output: data/runpod_pull/sweeps/<name>/ (scores/, audits/, report.{txt,json,md}).
Stdlib only (scoring runs deploy/runpod/score_pulled.py under the main venv).
"""
from __future__ import annotations

import datetime as _dt
import difflib
import hashlib
import json
import re
import sqlite3
import statistics
import subprocess
import sys
from pathlib import Path

import swarm

cli = swarm.cli

AUDIT_USD_PER_MATCH = 0.25        # Jev label audit of one replay (plan, Nov 13)
WRONG_WEIGHT = 5.0                # primary = right - 5 x wrong (a wrong team is worse than none)
# veto floors: a config that otherwise WINS is vetoed when it regresses one of
# these by more than max(baseline repeat spread, floor) vs the WORSE baseline
VETO_FLOORS = {"cov": 0.03, "err": 0.05, "manta": 0.10}
# metric -> +1 higher is better, -1 lower is better
METRICS = {"primary": 1, "right": 1, "wrong": -1, "unl": -1, "misid_s": -1, "cov": 1, "err": -1,
           "manta": -1, "audit_misid": -1}
_KNOB_RE = re.compile(r'Knob\(\s*"([A-Za-z0-9_]+)"')


def sweep_dir(name: str) -> Path:
    return cli.PULL_ROOT / "sweeps" / swarm.check_name(name)


def _repo_on_path() -> None:
    if str(cli.REPO) not in sys.path:
        sys.path.insert(0, str(cli.REPO))


# --------------------------------------------------------------------------
# Eval set
# --------------------------------------------------------------------------
def fingerprint(path) -> str:
    """web.field.ground_truth.video_fingerprint (size + sha1 of the first 4 MB)."""
    _repo_on_path()
    from web.field.ground_truth import video_fingerprint
    return video_fingerprint(path)


def gt_docs(gt_dir: Path, match_key: str) -> list:
    """The match's hand-GT docs (web.field.ground_truth.list_gt)."""
    _repo_on_path()
    from web.field.ground_truth import list_gt
    return [d for d in list_gt(match_key, Path(gt_dir)) if isinstance(d, dict) and d.get("team")]


def gt_matches(gt_dir: Path) -> list:
    _repo_on_path()
    from web.field.ground_truth import list_gt_matches
    return list_gt_matches(Path(gt_dir)) if Path(gt_dir).is_dir() else []


def manta_cache_file(manta_dir: Path, event: str, match_key: str) -> Path | None:
    """Manta truth exists for quals only: <event>_qual_<n>.json (scripts/manta_client.py)."""
    rk = swarm.key_rank(match_key)
    if rk is None or rk[0] != 0:
        return None
    p = Path(manta_dir) / f"{event}_qual_{rk[2]}.json"
    return p if p.is_file() else None


def _size_index(dirs: list) -> dict:
    idx: dict = {}
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() == ".mp4" and p.is_file():
                try:
                    idx.setdefault(p.stat().st_size, []).append(p)
                except OSError:
                    continue
    return idx


def find_match_video(match_key: str, docs: list, dirs: list, index: dict, fp=fingerprint) -> tuple:
    """(path, how) for a match's local video: a file whose fingerprint equals
    a GT doc's (only same-size files are hashed), else the GT doc's file name,
    else <match_key>.mp4. (None, None) when nothing is found."""
    want = {(d.get("video") or {}).get("fingerprint") for d in docs} - {None}
    for f in sorted(want):
        try:
            size = int(f.split(":", 1)[0])
        except ValueError:
            continue
        for cand in index.get(size, []):
            try:
                if fp(cand) == f:
                    return cand, "fingerprint"
            except OSError:
                continue
    names = {(d.get("video") or {}).get("name") for d in docs} - {None}
    for d in dirs:
        for n in sorted(names):
            p = Path(d) / n
            if p.is_file():
                return p, "gt video name"
    p = swarm.find_video(match_key, dirs)
    return (p, "match key") if p else (None, None)


def discover_evalset(*, gt_dir: Path, dirs: list, manta_dir: Path, auto: bool, add: list,
                     existing: dict | None = None, fp=fingerprint) -> dict:
    """The eval set: every hand-GT match (``auto``) + ``add``, each with its
    local video and the truths that exist for it. Entries already in
    ``existing`` (a human-edited file) are kept as they are; only their
    truths are refreshed."""
    old = {m["match_key"]: m for m in (existing or {}).get("matches") or []}
    keys = (gt_matches(gt_dir) if auto else []) + list(add or [])
    keys += [k for k in old if k not in keys]
    index = None
    matches, missing = [], []
    for mk in dict.fromkeys(keys):
        docs = gt_docs(gt_dir, mk) if (Path(gt_dir) / mk).is_dir() else []
        event = next((d.get("event_key") for d in docs if d.get("event_key")), None) or mk.split("_")[0]
        e = dict(old.get(mk) or {})
        if not (e.get("video") and Path(e["video"]).is_file()):
            if index is None:
                index = _size_index(dirs)
            path, how = find_match_video(mk, docs, dirs, index, fp)
            if path is None:
                missing.append({"match_key": mk, "why": "no local video (fingerprint / name / <match_key>.mp4 in "
                                                         + ", ".join(str(d) for d in dirs) + ")"})
                continue
            e.update(video=str(Path(path).resolve()), found_by=how)
        manta = manta_cache_file(manta_dir, event, mk)
        truths = (["hand_gt"] if docs else []) + (["manta"] if manta else [])
        fps = sorted({(d.get("video") or {}).get("fingerprint") for d in docs} - {None})
        e.update(match_key=mk, event=e.get("event") or event, size=Path(e["video"]).stat().st_size,
                 truths=truths or ["none"], gt_teams=sorted(str(d["team"]) for d in docs),
                 gt_fingerprints=fps, manta_cache=str(manta) if manta else None)
        if docs and e.get("found_by") not in (None, "fingerprint"):
            e["note"] = ("video is not the GT's file (fingerprint differs): scored by match time "
                         "alignment (match_t0_frame)")
        matches.append(e)
    return {"version": 1, "created_at": (existing or {}).get("created_at") or _dt.datetime.now().isoformat(
            timespec="seconds"), "updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "gt_dir": str(gt_dir), "matches": matches, "missing": missing}


def cmd_evalset(args) -> int:
    main = cli.main_checkout()
    out = Path(args.out)
    existing = json.loads(out.read_text()) if out.exists() else None
    if not args.auto and not args.add and not existing:
        raise SystemExit("nothing to do: --auto (every hand-GT match) and/or --add MATCH_KEY")
    gt_dir = Path(args.gt_dir) if args.gt_dir else main / "data" / "eval" / "ground_truth"
    dirs = [Path(d) for d in (args.videos or [])] + [main / "web" / "uploads", main / "data" / "videos",
                                                       main / "data" / "test_footage"]
    ev = discover_evalset(gt_dir=gt_dir, dirs=dirs, manta_dir=main / "data" / "manta_cache",
                          auto=args.auto, add=args.add or [], existing=existing)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ev, indent=1) + "\n")
    print(f"eval set -> {out}: {len(ev['matches'])} match(es), {len(ev['missing'])} without a local video")
    for m in ev["matches"]:
        print(f"  {m['match_key']:<18} {'+'.join(m['truths']):<14} {m.get('found_by') or 'kept':<14} "
              f"{m['video']}" + (f"   NOTE {m['note']}" if m.get("note") else ""))
    for m in ev["missing"]:
        print(f"  MISSING {m['match_key']}: {m['why']}")
    events = sorted({m["event"] for m in ev["matches"]})
    print(f"events: {', '.join(events) or '-'}")
    return 0 if ev["matches"] else 1


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------
def knob_registry():
    """(names, check(name, value) -> error or None). The app's Knob registry
    (web/job_config.py); without the app's deps importable, the knob names
    are parsed from its source and values are not checked."""
    try:
        _repo_on_path()
        from web import job_config as jc
    except Exception:
        src = (cli.REPO / "web" / "job_config.py").read_text()
        names = set(_KNOB_RE.findall(src))
        return names, (lambda n, v: None), False
    names = set(jc._BY_NAME) | set(jc._ALIASES)

    def check(n, v):
        knob = jc._BY_NAME[jc.canonical_knob_name(n)]
        try:
            jc._validate(knob, jc._coerce(knob, v))
        except ValueError as e:
            return str(e)
        return None
    return names, check, True


def validate_knobs(knobs: dict, where: str, registry=None) -> list:
    names, check, _ = registry or knob_registry()
    errs = []
    if not isinstance(knobs, dict):
        return [f"{where}: knobs must be an object"]
    for k, v in knobs.items():
        if k not in names:
            near = difflib.get_close_matches(k, sorted(names), n=1)
            errs.append(f"{where}: unknown knob {k!r}" + (f" (did you mean {near[0]!r}?)" if near else ""))
            continue
        e = check(k, v)
        if e:
            errs.append(f"{where}: {e}")
    return errs


def load_configs(path: Path, registry=None) -> dict:
    """{"base": {...}, "baselines": [...], "configs": [{"id", "knobs", ...}]}
    (a bare list = configs). Fails fast on typos / bad values / duplicate ids."""
    raw = json.loads(Path(path).read_text())
    doc = {"configs": raw} if isinstance(raw, list) else dict(raw)
    configs = doc.get("configs") or []
    base = doc.get("base") or {}
    reg = registry or knob_registry()
    errs = validate_knobs(base, "base", reg)
    seen = set()
    for i, c in enumerate(configs):
        cid = c.get("id") if isinstance(c, dict) else None
        if not cid or not re.match(r"^[A-Za-z0-9_.+-]{1,32}$", str(cid)):
            errs.append(f"configs[{i}]: bad or missing id {cid!r} (letters, digits, _ . + -)")
            continue
        if cid in seen:
            errs.append(f"configs[{i}]: duplicate id {cid!r}")
        seen.add(cid)
        errs += validate_knobs(c.get("knobs") or {}, cid, reg)
    baselines = doc.get("baselines") or [c["id"] for c in configs if str(c.get("id", "")).startswith("Q0")]
    for b in baselines:
        if b not in seen:
            errs.append(f"baselines: {b!r} is not a config id")
    if not configs:
        errs.append("no configs")
    if errs:
        raise SystemExit("bad sweep config file " + str(path) + ":\n  " + "\n  ".join(errs))
    if not reg[2]:
        print("NOTE: web.job_config not importable with this python: knob NAMES checked, values not "
              "(run with the main checkout's .venv/bin/python for full validation)")
    return {"base": base, "baselines": list(baselines),
            "configs": [{"id": c["id"], "knobs": dict(c.get("knobs") or {}), "why": c.get("why", "")}
                        for c in configs]}


def build_sweep_plan(evalset: dict, cfgs: dict, *, name: str, restart_each: bool = True) -> dict:
    matches = evalset.get("matches") or []
    if not matches:
        raise SystemExit("the eval set has no matches with a local video")
    items = []
    for c in cfgs["configs"]:
        for m in matches:
            items.append({"key": f"{c['id']}/{m['match_key']}", "match_key": m["match_key"],
                          "event": m["event"], "video": m["video"], "size": m.get("size"),
                          "config_id": c["id"], "knobs": c["knobs"]})
    h = hashlib.sha1(json.dumps([cfgs, [m["match_key"] for m in matches]], sort_keys=True).encode())
    return {"id": f"sweep-{name}-{h.hexdigest()[:10]}", "kind": "sweep", "policy": "pinned",
            "restart_each": restart_each, "event": matches[0]["event"], "spec": "sweep",
            "config": cfgs["base"], "configs": cfgs["configs"], "baselines": cfgs["baselines"],
            "matches": {m["match_key"]: {k: m.get(k) for k in ("event", "video", "truths", "gt_teams",
                                                               "manta_cache")} for m in matches},
            "created_at": swarm._now(), "items": items, "skipped": [],
            "missing": list(evalset.get("missing") or [])}


def cmd_plan(args) -> int:
    sw = swarm.Swarm.load(swarm.check_name(args.name), create=True)
    if swarm._run_active(sw.data.get("run")):
        raise SystemExit("a run has jobs in flight: finish it before re-planning")
    ev = json.loads(Path(args.evalset).read_text())
    cfgs = load_configs(Path(args.configs))
    plan = build_sweep_plan(ev, cfgs, name=sw.name, restart_each=not args.no_restart_each)
    plan.update(evalset=str(Path(args.evalset).resolve()), configs_file=str(Path(args.configs).resolve()))
    with sw.lock:
        sw.data["plan"] = plan
        sw.save()
    print_sweep_plan(plan)
    return 0


def print_sweep_plan(plan: dict) -> None:
    nm, nc = len(plan["matches"]), len(plan["configs"])
    print(f"sweep plan {plan['id']}: {nc} config(s) x {nm} match(es) = {len(plan['items'])} job(s); "
          f"base {json.dumps(plan['config'])}; baselines {', '.join(plan['baselines']) or '-'}")
    for c in plan["configs"]:
        print(f"  {c['id']:<6} {json.dumps(c['knobs'])}")
    print("  matches: " + ", ".join(f"{mk} ({'+'.join(m.get('truths') or [])})" for mk, m in plan["matches"].items()))
    for s in plan.get("missing") or []:
        print(f"  MISSING {s['match_key']}: {s['why']}")
    restart = (" + ~1.5 min server restart" if plan.get("restart_each") else "")
    print(f"pods: {nc} (one per config; fewer pods take configs in turn). Per pod ~{nm} x (job{restart}); "
          f"budget before renting: `swarm up --name <N> --count {nc} --dry-run` prints $/h. "
          + ("A config with team_reader=vllm needs `swarm up --with-vllm`."
             if any(c["knobs"].get("team_reader") == "vllm" for c in plan["configs"]) else ""))


def cmd_run(args) -> int:
    sw = swarm.Swarm.load(args.name)
    plan = sw.data.get("plan") or {}
    if plan.get("kind") != "sweep":
        raise SystemExit(f"swarm {args.name} has no sweep plan (`swarm sweep plan --name {args.name} ...`)")
    args.config_json = None
    args.pod_config = None
    args.policy = "pinned"
    return swarm.cmd_run(args)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def _member(sw: swarm.Swarm, pod: str) -> dict:
    try:
        return sw.member(pod)
    except KeyError:
        return {"pod_id": pod}


def locate_replay(sw: swarm.Swarm, it: dict) -> tuple:
    """(replay row, meta.json path) of a finished item, or (None, why): the
    local backend's replay is in the local app's DB; a pod's in its pull."""
    job = it.get("job_id")
    if not job:
        return None, "no job id"
    m = _member(sw, it.get("pod"))
    if swarm.is_pod(m):
        dest = cli.PULL_ROOT / it["pod"]
        rdir = dest / "web" / "replays" / job
        if not cli.replay_pulled(rdir):
            return None, f"not pulled yet (`swarm pull --name {sw.name}`)"
        db, meta = dest / "frc_scorer.db", rdir / "meta.json"
    else:
        app = Path(m.get("app_dir") or cli.main_checkout())
        db, meta = app / "frc_scorer.db", app / "web" / "replays" / job / "meta.json"
        if not meta.is_file():
            return None, f"no {meta}"
    if not db.exists():
        return None, f"no DB {db}"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute("SELECT * FROM replays WHERE id = ?", (job,)).fetchone()
    finally:
        con.close()
    if row is None:
        return None, f"replay {job} not in {db}"
    row = {k: row[k] for k in row.keys() if k in ("id", "match_key", "event_key", "created_at",
                                                  "blue_teams", "red_teams", "status")}
    return row, meta


def run_score_pulled(argv: list) -> dict:
    """deploy/runpod/score_pulled.py under the main checkout's venv; its
    last stdout line is the JSON result."""
    py = cli.main_checkout() / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    r = subprocess.run([py, str(cli.HERE / "score_pulled.py")] + argv, capture_output=True, text=True,
                       cwd=str(cli.REPO))
    if r.returncode != 0:
        return {"error": (r.stderr or r.stdout)[-600:]}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": r.stdout[-400:]}


def score_args(row_json: Path, meta: Path, video: str, match_key: str, event: str, cfg: dict,
               has_manta: bool, audit_dir: Path | None) -> list:
    rk = swarm.key_rank(match_key)
    argv = ["--replay-json", str(row_json), "--meta", str(meta), "--video", str(video),
            "--match", str(rk[2] if rk and rk[0] == 0 else 0), "--event", event,
            "--period", "auto" if cfg.get("auto_period_only") else "full"]
    if not has_manta:
        argv.append("--no-manta")
    if audit_dir is not None:
        argv += ["--audit-dir", str(audit_dir)]
        if cfg.get("ground_center_offset_m"):
            argv += ["--ground-center-offset", str(cfg["ground_center_offset_m"])]
    return argv


def score_sweep(sw: swarm.Swarm, *, audit: bool = False, rescore: bool = False, runner=run_score_pulled,
                out=print) -> dict:
    """Score every finished item (cached per item by job id). Returns
    {key: cache record}."""
    plan, run = sw.data.get("plan") or {}, sw.data.get("run") or {}
    root = sweep_dir(sw.name)
    (root / "scores").mkdir(parents=True, exist_ok=True)
    recs = {}
    for key, it in (run.get("items") or {}).items():
        if it.get("status") != "done":
            continue
        mk = swarm.item_match_key(run, key)
        minfo = (plan.get("matches") or {}).get(mk) or {}
        truths = minfo.get("truths") or []
        cache = root / "scores" / f"{swarm.safe_key(key)}.json"
        want_audit = audit and "hand_gt" not in truths
        if cache.exists() and not rescore:
            rec = json.loads(cache.read_text())
            if rec.get("job_id") == it.get("job_id") and (not want_audit or (rec.get("score") or {}).get("audit")):
                recs[key] = rec
                continue
        row, meta = locate_replay(sw, it)
        if row is None:
            out(f"  {key}: not scored ({meta})")
            continue
        rowf = root / "scores" / f"{swarm.safe_key(key)}.row.json"
        rowf.write_text(json.dumps(row))
        cfg = it.get("config") or swarm.job_config(run, it.get("pod") or "", key)
        adir = root / "audits" / swarm.safe_key(key) if want_audit else None
        out(f"  {key}: scoring replay {it['job_id'][:8]}{' + Jev audit' if adir else ''} ...")
        score = runner(score_args(rowf, meta, minfo.get("video") or it["video"], mk,
                                  minfo.get("event") or swarm.item_event(run, key) or mk.split("_")[0],
                                  cfg, "manta" in truths, adir))
        rec = {"key": key, "config_id": it.get("config_id"), "match_key": mk, "job_id": it["job_id"],
               "pod": it.get("pod"), "scored_at": swarm._now(), "wall_final_s": it.get("wall_final_s"),
               "peak_vram_mib": it.get("peak_vram_mib"), "truths": truths, "score": score}
        cache.write_text(json.dumps(rec, indent=1))
        recs[key] = rec
    return recs


def cmd_score(args) -> int:
    sw = swarm.Swarm.load(args.name)
    plan, run = sw.data.get("plan") or {}, sw.data.get("run") or {}
    if plan.get("kind") != "sweep" or not run:
        raise SystemExit(f"swarm {args.name}: no sweep run to score")
    if args.audit:
        todo = [k for k, it in run["items"].items() if it.get("status") == "done"
                and "hand_gt" not in ((plan.get("matches") or {}).get(swarm.item_match_key(run, k)) or {})
                .get("truths", [])]
        cached = [k for k in todo if (sweep_dir(sw.name) / "audits" / swarm.safe_key(k) / "audit_score.json").exists()]
        n = len(todo) - len(cached)
        print(f"--audit: Jev label audit of {n} replay(s) without hand GT ({len(cached)} cached) "
              f"~= ${n * AUDIT_USD_PER_MATCH:.2f} (~${AUDIT_USD_PER_MATCH:.2f}/match; CALLS THE JEV API)")
        if n and not args.yes and not cli.confirm("Run the Jev audit at that cost?", "audit"):
            return 1
    recs = score_sweep(sw, audit=args.audit, rescore=args.rescore)
    print(f"scored {len(recs)} item(s) -> {sweep_dir(sw.name) / 'scores'}; next: swarm sweep report --name {sw.name}")
    return 0


# --------------------------------------------------------------------------
# Report (pure)
# --------------------------------------------------------------------------
def item_metrics(rec: dict) -> dict:
    sc = rec.get("score") or {}
    gt, manta, audit = sc.get("gt") or {}, sc.get("manta") or {}, sc.get("audit") or {}
    m = {"key": rec.get("key"), "config_id": rec.get("config_id"), "match_key": rec.get("match_key"),
         "wall_s": rec.get("wall_final_s"), "vram_mib": rec.get("peak_vram_mib")}
    if "error" not in gt and gt.get("n_processed") and gt.get("label_correct") is not None:
        m.update(gt_n=gt["n_processed"], right=gt["label_correct"], wrong=gt.get("label_wrong") or 0.0,
                 unl=gt.get("unlabeled"), misid_s=gt.get("misid_s"), cov=gt.get("coverage"),
                 err=gt.get("err_median_m"))
    if manta.get("agg_meanNN") is not None:
        m["manta"] = manta["agg_meanNN"]
    if audit.get("misid") is not None:
        m["audit_misid"] = audit["misid"]
    return m


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(rows: list, wrong_weight: float = WRONG_WEIGHT) -> dict:
    """One config over its matches. Label rates are pooled over GT robot-time
    (weighted by each match's processed GT samples); primary = 100 x
    (right - wrong_weight x wrong) in points; misID s is the mean per GT match."""
    g = [r for r in rows if r.get("gt_n")]
    n = sum(r["gt_n"] for r in g)

    def pooled(k):
        vals = [(r[k], r["gt_n"]) for r in g if r.get(k) is not None]
        w = sum(nn for _, nn in vals)
        return sum(v * nn for v, nn in vals) / w if w else None
    right, wrong = pooled("right"), pooled("wrong")
    errs = [r["err"] for r in g if r.get("err") is not None]
    walls = [r["wall_s"] for r in rows if r.get("wall_s")]
    vram = [r["vram_mib"] for r in rows if r.get("vram_mib")]
    return {
        "n_items": len(rows), "gt_matches": sorted(r["match_key"] for r in g), "gt_weight": n,
        "right": right, "wrong": wrong, "unl": pooled("unl"),
        "primary": None if right is None else 100 * (right - wrong_weight * (wrong or 0.0)),
        "misid_s": _mean([r.get("misid_s") for r in g]), "cov": pooled("cov"),
        "err": statistics.median(errs) if errs else None,
        "manta": _mean([r.get("manta") for r in rows]),
        "audit_misid": _mean([r.get("audit_misid") for r in rows]),
        "wall_s": _mean(walls), "wall_max_s": max(walls) if walls else None,
        "vram_max_mib": max(vram) if vram else None,
    }


def noise_bands(aggs: dict, baselines: list) -> dict:
    """Per metric over the baseline repeats: spread (max - min), best and
    worst value (direction-aware). Spread is None with fewer than two."""
    out = {}
    for k, sign in METRICS.items():
        vals = [aggs[b][k] for b in baselines if b in aggs and aggs[b].get(k) is not None]
        if not vals:
            out[k] = {"spread": None, "best": None, "worst": None}
            continue
        hi, lo = max(vals), min(vals)
        out[k] = {"spread": (hi - lo) if len(vals) >= 2 else None,
                  "best": hi if sign > 0 else lo, "worst": lo if sign > 0 else hi}
    return out


def verdict(cid: str, agg: dict, bands: dict, baselines: list, base_matches: set,
            min_band: float = 0.0) -> dict:
    """WIN only when primary beats the BETTER baseline by more than the
    baselines' own spread (and nothing is vetoed); LOSS when below the worse
    one by more than it; else noise. Vetoes: coverage / position error /
    Manta auto regressing beyond max(spread, floor) vs the worse baseline."""
    if cid in baselines:
        return {"verdict": "baseline", "vetoes": [], "delta": None}
    p, b = agg.get("primary"), bands["primary"]
    if p is None or b["best"] is None:
        return {"verdict": "no GT" if p is None else "no baseline", "vetoes": [], "delta": None}
    band = max(b["spread"] or 0.0, min_band)
    delta = p - b["best"]
    vetoes = []
    for k, floor in VETO_FLOORS.items():
        v, bk = agg.get(k), bands[k]
        if v is None or bk["worst"] is None:
            continue
        lim = max(bk["spread"] or 0.0, floor)
        if (METRICS[k] > 0 and v < bk["worst"] - lim) or (METRICS[k] < 0 and v > bk["worst"] + lim):
            vetoes.append(k)
    if set(agg.get("gt_matches") or []) != set(base_matches):
        v = "PARTIAL"           # different GT matches than the baseline: not comparable
    elif delta > band:
        v = "WIN" if not vetoes else "VETOED"
    elif p < b["worst"] - band:
        v = "LOSS"
    else:
        v = "noise"
    return {"verdict": v, "vetoes": vetoes, "delta": delta, "band": band,
            "band_known": b["spread"] is not None}


def rank_key(agg: dict) -> tuple:
    inf = float("inf")
    p = agg.get("primary")
    return (-(p if p is not None else -inf), agg.get("misid_s") if agg.get("misid_s") is not None else inf,
            -(agg.get("cov") or 0.0), agg.get("err") if agg.get("err") is not None else inf,
            agg.get("manta") if agg.get("manta") is not None else inf)


def build_report(recs: list, configs: list, baselines: list, *, expected: dict | None = None,
                 min_band: float = 0.0, wrong_weight: float = WRONG_WEIGHT) -> dict:
    """recs: score cache records. configs: ordered config ids. expected:
    {config_id: [match keys planned]} (to list what is missing)."""
    rows = [item_metrics(r) for r in recs]
    by_cfg = {c: [r for r in rows if r["config_id"] == c] for c in configs}
    aggs = {c: aggregate(by_cfg[c], wrong_weight) for c in configs}
    bands = noise_bands(aggs, baselines)
    present = [b for b in baselines if b in aggs and aggs[b].get("gt_matches")]
    base_matches = set(aggs[present[0]]["gt_matches"]) if present else set()
    order = sorted(configs, key=lambda c: rank_key(aggs[c]))
    table = []
    for i, c in enumerate(order, 1):
        v = verdict(c, aggs[c], bands, baselines, base_matches, min_band)
        done = {r["match_key"] for r in by_cfg[c]}
        miss = sorted(set((expected or {}).get(c) or []) - done)
        table.append({"rank": i, "config_id": c, **aggs[c], **v, "missing": miss})
    matches = sorted({r["match_key"] for r in rows}, key=lambda k: (k.split("_")[0], swarm.key_rank(k) or (9,)))
    per_match = {mk: {c: next((r for r in by_cfg[c] if r["match_key"] == mk), None) for c in configs}
                 for mk in matches}
    return {"configs": table, "bands": bands, "baselines": baselines, "wrong_weight": wrong_weight,
            "min_band": min_band, "per_match": per_match, "matches": matches}


def _pct(v, nd=0):
    return "-" if v is None else f"{100 * v:.{nd}f}%"


def _num(v, fmt="{:.2f}"):
    return "-" if v is None else fmt.format(v)


def _cell(r: dict | None) -> str:
    if not r:
        return "-"
    parts = []
    if r.get("gt_n"):
        parts.append(f"{100 * r['right']:.0f}/{100 * (r['wrong'] or 0):.0f}/{_num(r.get('misid_s'), '{:.0f}')}s")
    if r.get("manta") is not None:
        parts.append(f"M{r['manta']:.2f}")
    if r.get("audit_misid") is not None:
        parts.append(f"A{100 * r['audit_misid']:.0f}%")
    return " ".join(parts) or "scored"


def report_rows(rep: dict) -> tuple:
    hdr = ["#", "config", "verdict", "primary", "dPrim", "right", "wrong", "unl", "misID s", "cov",
           "err m", "Manta m", "audit", "wall min", "VRAM GB", "GT n"]
    rows = []
    for t in rep["configs"]:
        verd = t["verdict"] + (f" ({','.join(t['vetoes'])})" if t.get("vetoes") else "")
        rows.append([str(t["rank"]), t["config_id"], verd, _num(t["primary"], "{:.1f}"),
                     _num(t.get("delta"), "{:+.1f}"), _pct(t["right"]), _pct(t["wrong"], 1), _pct(t["unl"]),
                     _num(t["misid_s"], "{:.1f}"), _pct(t["cov"]), _num(t["err"]), _num(t["manta"]),
                     _pct(t["audit_misid"]), _num(t["wall_s"] / 60 if t["wall_s"] else None, "{:.1f}"),
                     _num(t["vram_max_mib"] / 1024 if t["vram_max_mib"] else None, "{:.1f}"),
                     f"{len(t['gt_matches'])}" + (f" (missing {len(t['missing'])})" if t["missing"] else "")])
    return hdr, rows


def _fmt_table(hdr, rows) -> str:
    w = [max(len(str(x)) for x in col) for col in zip(hdr, *rows)] if rows else [len(h) for h in hdr]
    lines = ["  ".join(str(h).ljust(n) for h, n in zip(hdr, w))]
    lines += ["  ".join(str(c).ljust(n) for c, n in zip(r, w)) for r in rows]
    return "\n".join(lines)


def bands_line(rep: dict) -> str:
    b = rep["bands"]
    if all(v["spread"] is None for v in b.values()):
        return (f"noise band: unknown (needs two baseline repeats with results: {', '.join(rep['baselines'])}); "
                f"no config can be a WIN")
    fm = {"primary": "{:.1f} pts", "misid_s": "{:.1f} s", "cov": None, "err": "{:.2f} m", "manta": "{:.2f} m"}
    parts = []
    for k, f in fm.items():
        s = b[k]["spread"]
        if s is not None:
            parts.append(f"{k} {_pct(s, 1) if f is None else f.format(s)}")
    return f"noise band (spread of baselines {', '.join(rep['baselines'])}): " + ", ".join(parts)


def render_text(rep: dict) -> str:
    hdr, rows = report_rows(rep)
    out = [f"ranking: primary = right - {rep['wrong_weight']:g} x wrong (points of GT robot-time), then misID s; "
           f"cov / err / Manta break ties and veto", bands_line(rep), "", _fmt_table(hdr, rows), ""]
    cfgs = [t["config_id"] for t in rep["configs"]]
    mh = ["match"] + cfgs
    mrows = [[mk] + [_cell(rep["per_match"][mk].get(c)) for c in cfgs] for mk in rep["matches"]]
    out += ["per match (GT right/wrong/misID s; M = Manta auto meanNN m; A = audit misID):",
            _fmt_table(mh, mrows)]
    for t in rep["configs"]:
        if t["missing"]:
            out.append(f"  {t['config_id']}: no score for {', '.join(t['missing'])}")
    return "\n".join(out)


def render_md(rep: dict, title: str) -> str:
    hdr, rows = report_rows(rep)
    md = [f"# Quality sweep {title}", "", f"_generated {_dt.datetime.now().isoformat(timespec='seconds')}_", "",
          f"Ranking: primary = right − {rep['wrong_weight']:g} × wrong (points of hand-GT robot-time), then "
          f"misID s/match; coverage, position error and Manta auto meanNN break ties and veto.", "",
          bands_line(rep), "",
          "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    cfgs = [t["config_id"] for t in rep["configs"]]
    md += ["", "## Per match", "", "GT right/wrong/misID s · M = Manta auto meanNN (m) · A = audit misID", "",
           "| match | " + " | ".join(cfgs) + " |", "|" + "---|" * (len(cfgs) + 1)]
    md += ["| " + mk + " | " + " | ".join(_cell(rep["per_match"][mk].get(c)) for c in cfgs) + " |"
           for mk in rep["matches"]]
    return "\n".join(md) + "\n"


def cmd_report(args) -> int:
    sw = swarm.Swarm.load(args.name)
    plan, run = sw.data.get("plan") or {}, sw.data.get("run") or {}
    if plan.get("kind") != "sweep":
        raise SystemExit(f"swarm {args.name}: no sweep plan")
    root = sweep_dir(sw.name)
    recs = [json.loads(p.read_text()) for p in sorted((root / "scores").glob("*.json"))
            if not p.name.endswith(".row.json")] if (root / "scores").is_dir() else []
    # only this run's items (a re-planned sweep keeps old score files around)
    live = {k: it.get("job_id") for k, it in (run.get("items") or {}).items()}
    recs = [r for r in recs if live.get(r.get("key")) == r.get("job_id")]
    configs = [c["id"] for c in plan["configs"]]
    baselines = args.baselines.split(",") if args.baselines else plan.get("baselines") or []
    expected = {c: [i["match_key"] for i in plan["items"] if i["config_id"] == c] for c in configs}
    rep = build_report(recs, configs, baselines, expected=expected, min_band=args.min_band,
                       wrong_weight=args.wrong_weight)
    txt = render_text(rep)
    print(txt)
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.txt").write_text(txt + "\n")
    (root / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    (root / "report.md").write_text(render_md(rep, sw.name))
    print(f"\n-> {root / 'report.txt'}, report.json, report.md")
    return 0


# --------------------------------------------------------------------------
def add_parser(ss, named) -> None:
    p = ss.add_parser("sweep", help="quality sweep: K configs x M eval matches, one config per pod")
    sp = p.add_subparsers(dest="sweep_cmd", required=True)

    q = sp.add_parser("evalset", help="build / refresh the eval-set file (matches, local videos, truths)")
    q.add_argument("--out", required=True)
    q.add_argument("--auto", action="store_true", help="every match with hand GT (data/eval/ground_truth/<mk>/)")
    q.add_argument("--add", action="append", metavar="MATCH_KEY", help="also this match (repeatable)")
    q.add_argument("--videos", action="append", help="extra dir to search for videos (repeatable)")
    q.add_argument("--gt-dir", default=None)
    q.set_defaults(fn=cmd_evalset)

    def sub(name, help_):
        q = sp.add_parser(name, help=help_)
        q.add_argument("--name", required=True, help="swarm name (state: deploy/runpod/.swarm/<name>.json)")
        return q

    q = sub("plan", "validate configs (knob registry) and write the sweep work list (no renting)")
    q.add_argument("--evalset", required=True)
    q.add_argument("--configs", required=True, help="JSON: see deploy/runpod/sweep_configs.example.json")
    q.add_argument("--no-restart-each", action="store_true",
                   help="do NOT restart the server before each job (job order then confounds configs)")
    q.set_defaults(fn=cmd_plan)

    q = sub("run", "dispatch the sweep (pinned: config i -> pod i); re-attachable")
    q.add_argument("--terminate-when-done", action="store_true")
    q.add_argument("--with-video", action="store_true")
    q.add_argument("--poll-s", type=float, default=20)
    swarm.add_member_flags(q)
    q.set_defaults(fn=cmd_run)

    q = sub("score", "score every finished item locally (cached per item)")
    q.add_argument("--audit", action="store_true",
                   help=f"also the Jev label audit for matches WITHOUT hand GT (~${AUDIT_USD_PER_MATCH}/match)")
    q.add_argument("--yes", action="store_true", help="skip the audit cost prompt")
    q.add_argument("--rescore", action="store_true")
    q.set_defaults(fn=cmd_score)

    q = sub("report", "rank configs (identity first) with the baseline noise band")
    q.add_argument("--baselines", default=None, help="comma list of baseline repeat ids (default from the configs file)")
    q.add_argument("--min-band", type=float, default=0.0, help="floor for the primary noise band (points)")
    q.add_argument("--wrong-weight", type=float, default=WRONG_WEIGHT)
    q.set_defaults(fn=cmd_report)
