"""deploy/runpod/sweep.py + live.py + the swarm generalizations they need
(item keys, pinned / live policies, restart-each, member transports).
Offline: no pods, no RunPod API, no real ssh, no TBA, no YouTube, no Jev."""
from __future__ import annotations

import argparse
import hashlib
import http.server
import io
import json
import sqlite3
import sys
import tarfile
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNPOD = REPO / "deploy" / "runpod"
if str(RUNPOD) not in sys.path:
    sys.path.insert(0, str(RUNPOD))

import runpod_cli as cli  # noqa: E402
import swarm  # noqa: E402
import sweep  # noqa: E402
import live  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """State, pulls and the 'main checkout' in tmp; every network/ssh path is a tripwire."""
    monkeypatch.setattr(swarm, "SWARM_DIR", tmp_path / ".swarm")
    monkeypatch.setattr(cli, "PULL_ROOT", tmp_path / "pull")
    monkeypatch.setattr(cli, "STATE_PATH", tmp_path / "single.state.json")
    main = tmp_path / "main"
    (main / "web" / "uploads").mkdir(parents=True)
    monkeypatch.setattr(cli, "main_checkout", lambda *a, **k: main)

    def no_network(*a, **k):
        raise AssertionError(f"network call in an offline test: {a[:2]}")
    for mod, name in ((cli, "_request"), (cli, "ssh_run"), (cli, "ssh_out"), (swarm, "default_downloader"),
                      (swarm, "remote"), (live, "tba_fetcher")):
        monkeypatch.setattr(mod, name, no_network)
    return tmp_path


def _video(path: Path, size: int = 2000, seed: bytes = b"a") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((seed * size)[:size])
    return path


# ----------------------------------------------------------------- eval set
def _gt_doc(gt_dir: Path, mk: str, team: str, video: Path | None, name="orig.mp4", fp=None):
    d = gt_dir / mk
    d.mkdir(parents=True, exist_ok=True)
    fp = fp or (sweep.fingerprint(video) if video else "1:deadbeef")
    (d / f"{team}.json").write_text(json.dumps({
        "schema": 1, "match_key": mk, "event_key": mk.split("_")[0], "team": team, "alliance": "red",
        "video": {"name": name, "fingerprint": fp, "fps": 60.0}, "keyframes": []}))


def test_evalset_discovery_fingerprint_name_and_truths(tmp_path):
    gt, up, vids, manta = tmp_path / "gt", tmp_path / "up", tmp_path / "videos", tmp_path / "manta"
    real = _video(up / "uuid-1.mp4", 3000, b"x")
    _video(up / "uuid-0decoy.mp4", 3000, b"y")          # same size, other content: hashed, rejected
    _video(up / "uuid-other.mp4", 999, b"z")            # other size: never hashed
    _gt_doc(gt, "2026aaa_qm8", "254", real)
    _gt_doc(gt, "2026aaa_qm8", "1678", real)
    _gt_doc(gt, "2025bbb_qm3", "111", None)              # GT, but its video is only found by match key
    _video(vids / "2025bbb_qm3.mp4", 500)
    _video(vids / "2026aaa_qm9.mp4", 600)                # --add, no GT
    manta.mkdir()
    (manta / "2026aaa_qual_8.json").write_text("{}")
    hashed = []

    def fp(p):
        hashed.append(Path(p).name)
        return sweep.fingerprint(p)
    ev = sweep.discover_evalset(gt_dir=gt, dirs=[up, vids], manta_dir=manta, auto=True,
                                add=["2026aaa_qm9", "2026ccc_qm1"], fp=fp)
    by = {m["match_key"]: m for m in ev["matches"]}
    assert set(by) == {"2026aaa_qm8", "2025bbb_qm3", "2026aaa_qm9"}
    assert by["2026aaa_qm8"]["video"] == str(real.resolve()) and by["2026aaa_qm8"]["found_by"] == "fingerprint"
    assert by["2026aaa_qm8"]["truths"] == ["hand_gt", "manta"] and by["2026aaa_qm8"]["gt_teams"] == ["1678", "254"]
    assert "uuid-other.mp4" not in hashed and "uuid-0decoy.mp4" in hashed
    assert by["2025bbb_qm3"]["found_by"] == "match key" and "note" in by["2025bbb_qm3"]
    assert by["2025bbb_qm3"]["event"] == "2025bbb" and by["2025bbb_qm3"]["truths"] == ["hand_gt"]
    assert by["2026aaa_qm9"]["truths"] == ["none"]
    assert [m["match_key"] for m in ev["missing"]] == ["2026ccc_qm1"]
    # a human-edited entry is kept on refresh (its video path wins)
    by["2026aaa_qm9"]["video"] = str(_video(tmp_path / "elsewhere" / "q9.mp4", 600))
    ev2 = sweep.discover_evalset(gt_dir=gt, dirs=[up, vids], manta_dir=manta, auto=True, add=[], existing=ev)
    by2 = {m["match_key"]: m for m in ev2["matches"]}
    assert by2["2026aaa_qm9"]["video"].endswith("elsewhere/q9.mp4")


def test_cmd_evalset_writes_editable_json(tmp_path, capsys):
    main = cli.main_checkout()
    v = _video(main / "web" / "uploads" / "x.mp4", 1234)
    _gt_doc(main / "data" / "eval" / "ground_truth", "2026aaa_qm8", "254", v)
    out = tmp_path / "evalset.json"
    assert sweep.cmd_evalset(argparse.Namespace(out=str(out), auto=True, add=None, videos=None, gt_dir=None)) == 0
    doc = json.loads(out.read_text())
    assert doc["matches"][0]["match_key"] == "2026aaa_qm8" and "events: 2026aaa" in capsys.readouterr().out


# ----------------------------------------------------------------- configs
def test_config_validation_against_the_knob_registry(tmp_path):
    reg = sweep.knob_registry()
    assert reg[2], "web.job_config should import under the app venv"
    ok = sweep.load_configs(RUNPOD / "sweep_configs.example.json")
    assert [c["id"] for c in ok["configs"]][:2] == ["Q0a", "Q0b"] and ok["baselines"] == ["Q0a", "Q0b"]
    assert ok["base"]["corner_track_stride"] == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"base": {"corner_track_strid": 2},
                               "configs": [{"id": "A", "knobs": {"sam3_track_fps": 1000}},
                                           {"id": "A", "knobs": {}}, {"id": "bad id", "knobs": {}}]}))
    with pytest.raises(SystemExit) as e:
        sweep.load_configs(bad)
    msg = str(e.value)
    assert "unknown knob 'corner_track_strid' (did you mean 'corner_track_stride'?)" in msg
    assert "above maximum" in msg and "duplicate id 'A'" in msg and "bad or missing id" in msg
    # a bare list works; baselines default to Q0*
    lst = tmp_path / "list.json"
    lst.write_text(json.dumps([{"id": "Q0a", "knobs": {}}, {"id": "Q1", "knobs": {"event_lineup": True}}]))
    assert sweep.load_configs(lst)["baselines"] == ["Q0a"]
    # legacy alias names are accepted (sam2_* -> sam3_*)
    assert sweep.validate_knobs({"sam2_track_fps": 8}, "x", reg) == []


def test_validation_falls_back_to_source_names_without_the_app(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "web" or name.startswith("web."):
            raise ImportError("no app deps")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    names, check, full = sweep.knob_registry()
    assert not full and "sam3_track_fps" in names and check("sam3_track_fps", 1e9) is None


# ----------------------------------------------------------------- item keys + pinned policy
def _evalset(tmp_path, keys):
    return {"matches": [{"match_key": k, "event": k.split("_")[0], "video": str(_video(tmp_path / f"{k}.mp4")),
                         "size": 2000, "truths": ["hand_gt"]} for k in keys], "missing": []}


CFGS = {"base": {"corner_track_stride": 2}, "baselines": ["Q0a", "Q0b"],
        "configs": [{"id": "Q0a", "knobs": {}}, {"id": "Q0b", "knobs": {}},
                    {"id": "Q1", "knobs": {"sam3_track_fps": 10, "corner_track_stride": 1}}]}


def test_sweep_items_keys_and_per_item_config(tmp_path):
    plan = sweep.build_sweep_plan(_evalset(tmp_path, ["2026a_qm1", "2025b_qm2"]), CFGS, name="s")
    assert [i["key"] for i in plan["items"]][:3] == ["Q0a/2026a_qm1", "Q0a/2025b_qm2", "Q0b/2026a_qm1"]
    run = swarm.new_run(plan, ["p1", "p2", "p3"], plan["config"], policy="pinned",
                        pod_config={"p3": {"sam3_chunk_frames": 500}})
    assert swarm.item_match_key(run, "Q1/2025b_qm2") == "2025b_qm2"
    assert swarm.item_event(run, "Q1/2025b_qm2") == "2025b"            # per-item event (multi-event eval set)
    assert swarm.job_config(run, "p3", "Q1/2026a_qm1") == {"corner_track_stride": 1, "sam3_track_fps": 10,
                                                           "sam3_chunk_frames": 500}
    assert swarm.job_config(run, "p1", "Q0a/2026a_qm1") == {"corner_track_stride": 2}
    assert swarm.safe_key("Q1/2026a_qm1") == "Q1__2026a_qm1" and swarm.video_name("Q1/2026a_qm1") == "2026a_qm1"
    assert swarm.video_name("2026a_qm1@r2") == "2026a_qm1@r2"
    # sweep items never count as "already processed" for a batch plan of the same match
    run["items"]["Q0a/2026a_qm1"]["status"] = "done"
    assert swarm.existing_results({"2026a_qm1"}, tmp_path / "nodb", tmp_path / "nopull", run) == {}


def test_pinned_one_config_per_pod_whole_config_takeover_and_retry_on_its_pod(tmp_path):
    cfgs = dict(CFGS, configs=CFGS["configs"] + [{"id": "Q2", "knobs": {}}])
    plan = sweep.build_sweep_plan(_evalset(tmp_path, ["e_qm1", "e_qm2"]), cfgs, name="s")
    run = swarm.new_run(plan, ["a", "b"], policy="pinned")
    assert run["pin"] == {"Q0a": "a", "Q0b": "b"}
    assert [g[0] for g in run["unassigned_groups"]] == ["Q1", "Q2"]
    seen = {"a": [], "b": []}
    # a is fast: runs Q0a, then takes Q1 whole
    for _ in range(4):
        k = swarm.next_item(run, "a")
        seen["a"].append(k)
        swarm.finish_item(run, k, "a", k != "Q1/e_qm1" or run["items"][k]["attempts"] > 1)
    assert seen["a"] == ["Q0a/e_qm1", "Q0a/e_qm2", "Q1/e_qm1", "Q1/e_qm2"]
    assert run["pin"]["Q1"] == "a"
    # the failed Q1 item is retried on a (its config's pod), not on b
    assert swarm.next_item(run, "b") == "Q0b/e_qm1"
    assert swarm.next_item(run, "a") == "Q1/e_qm1"
    swarm.finish_item(run, "Q1/e_qm1", "a", True)
    # b dies mid Q0b: a takes over Q0b whole (incl. the in-flight item's retry)
    swarm.mark_pod_dead(run, "b", "gone")
    got = [swarm.next_item(run, "a")]
    swarm.finish_item(run, got[-1], "a", True)
    got.append(swarm.next_item(run, "a"))
    assert set(got) == {"Q0b/e_qm1", "Q0b/e_qm2"} and run["pin"]["Q0b"] == "a"
    swarm.finish_item(run, got[-1], "a", True)
    k = swarm.next_item(run, "a")                       # last config Q2
    assert k.startswith("Q2/") and run["pin"]["Q2"] == "a"


# ----------------------------------------------------------------- old state compatibility
def test_old_batch_state_file_loads_and_resumes(tmp_path, monkeypatch, capsys):
    """A .swarm file written by the batch-only code (items keyed by match key,
    no match_key/config fields, no restart_each) still loads, shows and resumes."""
    keys = ["2026tst_qm1", "2026tst_qm2"]
    vids = {k: str(_video(tmp_path / f"{k}.mp4")) for k in keys}
    old = {"name": "old", "created_at": 1.0, "gpu_priority": ["NVIDIA A40"], "cloud": "SECURE",
           "members": [{"pod_id": "a", "status": "ready", "cost_per_hr": 0.5, "created_at": 1.0}],
           "create_errors": [],
           "plan": {"id": "p1", "event": "2026tst", "spec": "all", "created_at": 1.0,
                    "items": [{"match_key": k, "video": vids[k], "size": 1, "video_key": None} for k in keys],
                    "skipped": [], "missing": []},
           "run": {"plan_id": "p1", "event": "2026tst", "config": {"x": 1}, "pod_config": {}, "policy": "blocks",
                   "open": False, "started_at": 1.0, "queues": {"a": [keys[1]]}, "unassigned": [],
                   "current": {"a": None}, "retry": [], "dead": [], "retired": [], "steals": [],
                   "items": {keys[0]: {"video": vids[keys[0]], "size": 1, "status": "done", "attempts": 1,
                                       "avoid": [], "pod": "a", "job_id": "J1"},
                             keys[1]: {"video": vids[keys[1]], "size": 1, "status": "pending", "attempts": 0,
                                       "avoid": [], "pod": None}}}}
    swarm.SWARM_DIR.mkdir(parents=True)
    (swarm.SWARM_DIR / "old.json").write_text(json.dumps(old))
    seen = []

    def fake_process(ctx, m, key):
        seen.append((key, swarm.item_match_key(ctx.run, key), swarm.job_config(ctx.run, m["pod_id"], key)))
        return True, {"job_id": "J2", "wall_final_s": 5}
    monkeypatch.setattr(swarm, "process_item", fake_process)
    monkeypatch.setattr(swarm.time, "sleep", lambda s: None)
    monkeypatch.setattr(swarm, "revive_lost", lambda sw: [])
    assert cli.main(["swarm", "run", "--name", "old", "--poll-s", "0"]) == 0
    assert seen == [("2026tst_qm2", "2026tst_qm2", {"x": 1})]
    run = swarm.Swarm.load("old").data["run"]
    assert swarm.run_counts(run)["done"] == 2 and not run.get("restart_each")
    # plan_import / existing_results read the old items too
    assert swarm.existing_results(set(keys), tmp_path / "nodb", tmp_path / "nopull", run) == \
        {k: "this swarm's run" for k in keys}


# ----------------------------------------------------------------- restart-each + transports
class Pod:
    """Fake pod side behind swarm's module-level ssh helpers + remote()."""

    def __init__(self):
        self.log, self.launched = [], set()

    def install(self, monkeypatch):
        def read(m, outp):
            if outp not in self.launched:
                return False, None
            label = outp.rsplit("/", 1)[-1][:-len(".json")]
            return True, {"label": label, "status": "done", "job_id": "J-" + label, "pid": 1,
                          "wall_final_s": 60.0, "peak_vram_mib": 9000}
        monkeypatch.setattr(swarm, "read_job_state", read)
        monkeypatch.setattr(swarm, "upload_video", lambda m, l, r, out, what="video": self.log.append(("upload", r)))
        monkeypatch.setattr(swarm, "server_active_job", lambda m: None)

        def launch(m, outp, cmd):
            self.log.append(("launch", outp.rsplit("/", 1)[-1], cmd))
            self.launched.add(outp)
            return "started"
        monkeypatch.setattr(swarm, "launch_once", launch)
        monkeypatch.setattr(swarm, "driver_alive", lambda m, s, o: True)

        def remote(m, cmd, timeout=0, stdin_path=None):
            if "serve.sh restart" in cmd:
                self.log.append(("restart",))
                return "server up"
            if "vllm_ctl.py start" in cmd:
                self.log.append(("vllm", cmd.rsplit(" ", 1)[-1]))
                return 'loading...\n{"ok": true}'
            raise AssertionError(cmd)
        monkeypatch.setattr(swarm, "remote", remote)
        return self


def _sweep_swarm(tmp_path, name, pods, cfgs=CFGS, keys=("e_qm1", "e_qm2")):
    sw = swarm.Swarm.load(name, create=True)
    for p in pods:
        sw.members.append({"pod_id": p, "status": "ready", "cost_per_hr": 0.5, "created_at": swarm._now()})
    sw.data["plan"] = sweep.build_sweep_plan(_evalset(tmp_path, list(keys)), cfgs, name=name)
    sw.save()
    return sw


def _run_args(name, **kw):
    a = dict(name=name, terminate_when_done=False, with_video=False, poll_s=0, local=False, local_url=None,
             local_app_dir=None, restart_each=None)
    a.update(kw)
    return argparse.Namespace(**a)


def test_restart_each_restarts_before_every_launch_never_on_reattach(tmp_path, monkeypatch):
    pod = Pod().install(monkeypatch)
    monkeypatch.setattr(swarm.time, "sleep", lambda s: None)
    q4 = {"configs": [{"id": "Q4", "knobs": {"team_reader": "vllm", "vlm_model": "qwen3.5-4b"}}],
          "base": {}, "baselines": []}
    _sweep_swarm(tmp_path, "rs", ["a"], q4)
    assert sweep.cmd_run(_run_args("rs")) == 0              # sweep plan default: restart ON for pods
    kinds = [e[0] for e in pod.log]
    # item 1, the prefetch of item 2 while item 1 runs, then item 2 (its upload is skipped by size on a real pod)
    assert kinds == ["upload", "restart", "vllm", "launch", "upload", "upload", "restart", "vllm", "launch"]
    assert pod.log[2] == ("vllm", "qwen3.5-4b")
    assert "--label Q4__e_qm1.a1" in pod.log[3][2] and "--match-key e_qm1" in pod.log[3][2]
    assert pod.log[3][1] == "Q4__e_qm1.a1.json"
    assert all(e[1] == f"{swarm.POD_VIDEOS}/e_qm{i}.mp4" for i, e in ((1, pod.log[0]), (2, pod.log[4]), (2, pod.log[5])))
    run = swarm.Swarm.load("rs").data["run"]
    assert run["restart_each"] and all(it["restarts"] == 1 for it in run["items"].values())
    assert {it["peak_vram_mib"] for it in run["items"].values()} == {9000}
    # re-attach (launch marker exists): no restart, no relaunch
    pod.log.clear()
    sw = swarm.Swarm.load("rs")
    ctx = swarm.RunCtx(sw, _run_args("rs"), sleep=lambda s: None)
    it = ctx.run["items"]["Q4/e_qm1"]
    it.update(status="running", attempts=1)
    assert swarm.process_item(ctx, sw.member("a"), "Q4/e_qm1")[0]
    assert [e[0] for e in pod.log] == []


def test_restart_each_off_and_batch_default_off(tmp_path, monkeypatch):
    pod = Pod().install(monkeypatch)
    monkeypatch.setattr(swarm.time, "sleep", lambda s: None)
    _sweep_swarm(tmp_path, "nr", ["a", "b", "c"])
    assert sweep.cmd_run(_run_args("nr", restart_each=False)) == 0
    assert "restart" not in [e[0] for e in pod.log]
    run = swarm.Swarm.load("nr").data["run"]
    # pinned: every item of a config ran on one pod
    by_cfg = {}
    for k, it in run["items"].items():
        by_cfg.setdefault(it["config_id"], set()).add(it["pod"])
    assert all(len(p) == 1 for p in by_cfg.values()) and len({next(iter(p)) for p in by_cfg.values()}) == 3


# ----------------------------------------------------------------- local backend
class _App(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"job_id": None}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


FAKE_POD_JOB = r'''
import argparse, json, os, sqlite3, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
for a in ("--video", "--match-key", "--event", "--label", "--config-json", "--host", "--final-timeout",
          "--app-dir", "--out"):
    ap.add_argument(a)
ap.add_argument("--no-profile", action="store_true")
a = ap.parse_args()
jid = "job-" + a.label.replace(".", "-")
app = Path(a.app_dir)
con = sqlite3.connect(app / "frc_scorer.db")
con.execute("CREATE TABLE IF NOT EXISTS replays (id TEXT PRIMARY KEY, match_key TEXT, event_key TEXT, "
            "created_at TEXT, blue_teams TEXT, red_teams TEXT, status TEXT, config TEXT)")
con.execute("INSERT INTO replays VALUES (?,?,?,?,?,?,?,?)", (jid, a.match_key, a.event, "t", "1,2,3", "4,5,6",
            "complete", a.config_json))
con.commit(); con.close()
(app / "web" / "replays" / jid).mkdir(parents=True)
(app / "web" / "replays" / jid / "meta.json").write_text("{}")
out = Path(a.out)
tmp = out.with_suffix(".tmp")
tmp.write_text(json.dumps({"label": a.label, "status": "done", "job_id": jid, "pid": os.getpid(),
                           "wall_final_s": 2.5, "peak_vram_mib": 1234, "config": json.loads(a.config_json),
                           "host": a.host, "no_profile": a.no_profile}))
os.replace(tmp, out)
'''


def test_local_backend_runs_sweep_with_fake_pod_job_then_scores(tmp_path, monkeypatch, capsys):
    srv = http.server.HTTPServer(("127.0.0.1", 0), _App)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    fake = tmp_path / "fake_pod_job.py"
    fake.write_text(FAKE_POD_JOB)
    monkeypatch.setattr(swarm, "LOCAL_POD_JOB", fake)
    try:
        _sweep_swarm(tmp_path, "loc", [], keys=("2026cur_qm8",))   # no pods at all: local only
        rc = sweep.cmd_run(_run_args("loc", local=True, local_url=url, poll_s=0.05))
        out = capsys.readouterr().out
    finally:
        srv.shutdown()
    assert rc == 0 and "WARNING: --local without --restart-each" in out
    sw = swarm.Swarm.load("loc")
    run = sw.data["run"]
    assert swarm.run_counts(run)["done"] == 3 and not run["restart_each"]
    m = sw.member("local")
    assert m["backend"] == "local" and m["app_dir"] == str(cli.main_checkout())
    st = json.loads(Path(m["jobs_dir"], "Q1__2026cur_qm8.a1.json").read_text())
    assert st["no_profile"] and st["config"] == {"corner_track_stride": 1, "sam3_track_fps": 10}
    assert st["host"] == url.split("://")[1]
    # local items are scored straight from the local DB (no pull)
    calls = []

    def runner(argv):
        calls.append(argv)
        return {"gt": {"n_processed": 100, "label_correct": 0.7, "label_wrong": 0.0, "unlabeled": 0.3,
                       "misid_s": 1.0, "coverage": 0.8, "err_median_m": 0.3}, "manta": {"skipped": "x"}}
    recs = sweep.score_sweep(sw, runner=runner, out=lambda *a: None)
    assert len(recs) == 3 and len(calls) == 3
    argv = calls[0]
    assert argv[argv.index("--match") + 1] == "8" and "--no-manta" in argv
    assert argv[argv.index("--period") + 1] == "full" and argv[argv.index("--event") + 1] == "2026cur"
    assert Path(argv[argv.index("--meta") + 1]).is_file()
    # cached by job id: a second pass runs nothing
    assert len(sweep.score_sweep(sw, runner=runner, out=lambda *a: None)) == 3 and len(calls) == 3
    # the local member is never pulled / terminated / billed
    assert swarm.ready_members(sw) == [] and swarm.live_members(sw) == []


def test_local_transport_launch_guard_and_driver_alive(tmp_path):
    m = swarm.local_member("x", python=sys.executable)
    T = swarm.transport(m)
    outp = str(tmp_path / "j" / "k.a1.json")
    fake = tmp_path / "sleep.py"
    fake.write_text("import time; time.sleep(30)\n")
    m["pod_job"] = str(fake)
    assert T.launch(m, outp, outp[:-5] + ".log", video="/v.mp4", match_key="k", event="e", label="k.a1",
                    config={}) == "started"
    assert T.launch(m, outp, outp[:-5] + ".log", video="/v.mp4", match_key="k", event="e", label="k.a1",
                    config={}) == "already"
    assert T.read_job_state(m, outp) == (True, None) and T.driver_alive(m, None, outp)
    swarm._LOCAL_PROCS[outp].kill()
    swarm._LOCAL_PROCS[outp].wait()
    assert not T.driver_alive(m, None, outp)


def test_pod_job_has_app_dir_flag():
    import subprocess
    r = subprocess.run([sys.executable, str(RUNPOD / "pod_job.py"), "--help"], capture_output=True, text=True)
    assert "--app-dir" in r.stdout


# ----------------------------------------------------------------- scoring: pod replays need a marked pull
def test_score_locates_pulled_replay_only_with_marker(tmp_path):
    sw = _sweep_swarm(tmp_path, "pp", ["podA"], keys=("e_qm1",))
    run = sw.data["run"] = swarm.new_run(sw.data["plan"], ["podA"], policy="pinned")
    it = run["items"]["Q0a/e_qm1"]
    it.update(status="done", pod="podA", job_id="J1")
    row, why = sweep.locate_replay(sw, it)
    assert row is None and "swarm pull" in why
    dest = cli.PULL_ROOT / "podA"
    rdir = dest / "web" / "replays" / "J1"
    rdir.mkdir(parents=True)
    (rdir / "meta.json").write_text("{}")
    (rdir / cli.PULL_MARKER).write_text("{}")
    con = sqlite3.connect(dest / "frc_scorer.db")
    con.execute("CREATE TABLE replays (id TEXT, match_key TEXT, event_key TEXT, created_at TEXT, "
                "blue_teams TEXT, red_teams TEXT, status TEXT, video_path TEXT)")
    con.execute("INSERT INTO replays VALUES ('J1','e_qm1','e','t','1','2','complete','/x')")
    con.commit()
    con.close()
    row, meta = sweep.locate_replay(sw, it)
    assert row["id"] == "J1" and "video_path" not in row and meta == rdir / "meta.json"


# ----------------------------------------------------------------- report math
def _rec(cfg, mk, right, wrong, n=100, misid=0.0, cov=0.8, err=0.3, manta=None, audit=None, wall=600):
    sc = {"gt": {"n_processed": n, "label_correct": right, "label_wrong": wrong, "unlabeled": 1 - right - wrong,
                 "misid_s": misid, "coverage": cov, "err_median_m": err}}
    if manta is not None:
        sc["manta"] = {"agg_meanNN": manta}
    if audit is not None:
        sc["audit"] = {"misid": audit}
    return {"key": f"{cfg}/{mk}", "config_id": cfg, "match_key": mk, "score": sc, "wall_final_s": wall,
            "peak_vram_mib": 12000}


def test_aggregate_weights_by_gt_robot_time_and_primary():
    rows = [sweep.item_metrics(_rec("A", "m1", 0.6, 0.02, n=100, misid=4)),
            sweep.item_metrics(_rec("A", "m2", 0.9, 0.0, n=300, misid=0)),
            sweep.item_metrics({"key": "A/m3", "config_id": "A", "match_key": "m3",
                                "score": {"gt": {"error": "no ground truth"}, "manta": {"agg_meanNN": 0.5},
                                          "audit": {"misid": 0.04}}})]
    a = sweep.aggregate(rows)
    assert a["right"] == pytest.approx((0.6 * 100 + 0.9 * 300) / 400)
    assert a["wrong"] == pytest.approx(0.02 * 100 / 400)
    assert a["primary"] == pytest.approx(100 * (a["right"] - 5 * a["wrong"]))
    assert a["misid_s"] == 2.0 and a["gt_matches"] == ["m1", "m2"] and a["manta"] == 0.5
    assert a["audit_misid"] == 0.04 and a["n_items"] == 3 and a["vram_max_mib"] == 12000


def test_report_noise_band_win_logic_and_identity_first_rank():
    recs = []
    for mk in ("m1", "m2"):
        recs += [_rec("Q0a", mk, 0.70, 0.00, misid=3), _rec("Q0b", mk, 0.66, 0.00, misid=4),   # band 4 pts
                 _rec("WINNER", mk, 0.76, 0.00, misid=5),        # +6 over best base > 4 band
                 _rec("NOISE", mk, 0.73, 0.00, misid=1),         # +3: inside the band
                 _rec("WRONGY", mk, 0.80, 0.02, misid=1),        # 80 - 10 = 70: wrong costs 5x
                 _rec("LOSER", mk, 0.55, 0.00),
                 _rec("VETO", mk, 0.80, 0.00, cov=0.60)]         # wins on labels, loses 20 pts coverage
    recs.append(_rec("PART", "m1", 0.95, 0.0))                   # only one of the GT matches
    cfgs = ["Q0a", "Q0b", "WINNER", "NOISE", "WRONGY", "LOSER", "VETO", "PART"]
    rep = sweep.build_report(recs, cfgs, ["Q0a", "Q0b"], expected={c: ["m1", "m2"] for c in cfgs})
    by = {t["config_id"]: t for t in rep["configs"]}
    assert rep["bands"]["primary"]["spread"] == pytest.approx(4.0)
    assert rep["bands"]["primary"]["best"] == pytest.approx(70.0)
    assert by["WINNER"]["verdict"] == "WIN" and by["WINNER"]["delta"] == pytest.approx(6.0)
    assert by["NOISE"]["verdict"] == "noise"
    assert by["WRONGY"]["primary"] == pytest.approx(70.0) and by["WRONGY"]["verdict"] == "noise"
    assert by["LOSER"]["verdict"] == "LOSS"
    assert by["VETO"]["verdict"] == "VETOED" and by["VETO"]["vetoes"] == ["cov"]
    assert by["PART"]["verdict"] == "PARTIAL" and by["PART"]["missing"] == ["m2"]
    assert by["Q0a"]["verdict"] == "baseline"
    order = [t["config_id"] for t in rep["configs"]]
    assert order[0] == "PART" and order.index("VETO") < order.index("WINNER") < order.index("NOISE")
    # WRONGY and Q0a tie on primary (70): misID s breaks it
    assert order.index("WRONGY") < order.index("Q0a")
    txt = sweep.render_text(rep)
    assert "noise band" in txt and "WIN" in txt and "per match" in txt
    assert "| WINNER |" in sweep.render_md(rep, "t") or "WINNER" in sweep.render_md(rep, "t")


def test_report_without_two_baselines_never_wins():
    recs = [_rec("Q0a", "m1", 0.5, 0.0), _rec("X", "m1", 0.9, 0.0)]
    rep = sweep.build_report(recs, ["Q0a", "Q0b", "X"], ["Q0a", "Q0b"])
    by = {t["config_id"]: t for t in rep["configs"]}
    assert by["X"]["verdict"] == "better?" and not by["X"]["band_known"]
    assert "noise band: unknown" in sweep.bands_line(rep)
    rep2 = sweep.build_report(recs, ["Q0a", "X"], ["Q0a"], min_band=50)
    assert {t["config_id"]: t for t in rep2["configs"]}["X"]["verdict"] == "noise"


def test_cmd_report_writes_files_for_this_runs_items_only(tmp_path, capsys):
    sw = _sweep_swarm(tmp_path, "rep", ["a"], keys=("m_qm1",))
    run = sw.data["run"] = swarm.new_run(sw.data["plan"], ["a"], policy="pinned")
    for k, it in run["items"].items():
        it.update(status="done", job_id=f"J-{k}")
    sw.save()
    root = sweep.sweep_dir("rep") / "scores"
    root.mkdir(parents=True)
    for k in run["items"]:
        cfg = k.split("/")[0]
        r = _rec(cfg, "m_qm1", {"Q0a": 0.7, "Q0b": 0.68, "Q1": 0.8}[cfg], 0.0)
        r.update(key=k, job_id=f"J-{k}")
        (root / f"{swarm.safe_key(k)}.json").write_text(json.dumps(r))
    stale = _rec("Q1", "m_qm1", 0.1, 0.5)
    stale.update(key="Q1/m_qm1", job_id="OLD")
    (root / "zz_stale.json").write_text(json.dumps(stale))
    assert sweep.cmd_report(argparse.Namespace(name="rep", baselines=None, min_band=0.0, wrong_weight=5.0)) == 0
    d = sweep.sweep_dir("rep")
    rep = json.loads((d / "report.json").read_text())
    assert rep["configs"][0]["config_id"] == "Q1" and rep["configs"][0]["verdict"] == "WIN"
    assert (d / "report.md").read_text().startswith("# Quality sweep rep")
    assert "Q1" in (d / "report.txt").read_text()


def test_score_audit_prints_estimate_and_needs_confirmation(tmp_path, monkeypatch, capsys):
    sw = _sweep_swarm(tmp_path, "aud", ["a"], keys=("m_qm1",))
    sw.data["plan"]["matches"]["m_qm1"]["truths"] = ["none"]
    run = sw.data["run"] = swarm.new_run(sw.data["plan"], ["a"], policy="pinned")
    for it in run["items"].values():
        it.update(status="done", job_id="J")
    sw.save()
    monkeypatch.setattr(cli, "confirm", lambda *a, **k: False)
    assert sweep.cmd_score(argparse.Namespace(name="aud", audit=True, yes=False, rescore=False)) == 1
    assert "~= $0.75" in capsys.readouterr().out


# ----------------------------------------------------------------- live policy
def test_live_policy_idle_longest_and_never_two_on_one_pod(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(swarm, "_now", lambda: clock[0])
    run = swarm.new_run({"id": "l", "event": "e", "items": []}, ["a", "b", "c"], policy="live")
    run["open"] = True
    run["idle_since"] = {"a": 50.0, "b": 10.0, "c": 30.0}       # b idle longest
    for p in "abc":
        assert swarm.next_item(run, p) is None                   # register polls
    swarm.add_items(run, [{"match_key": "e_qm1", "video": "/v1"}, {"match_key": "e_qm2", "video": "/v2"}])
    assert swarm.next_item(run, "a") is None                     # not the longest idle
    assert swarm.next_item(run, "b") == "e_qm1"
    assert swarm.next_item(run, "a") is None                     # c idle longer than a
    assert swarm.next_item(run, "c") == "e_qm2"
    assert swarm.next_item(run, "a") is None and run["current"]["a"] is None
    # all busy: the waiting match goes to whichever frees first
    swarm.next_item(run, "a")
    swarm.add_items(run, [{"match_key": "e_qm3", "video": "/v3"}])
    assert swarm.next_item(run, "a") == "e_qm3"
    swarm.add_items(run, [{"match_key": "e_qm4", "video": "/v4"}])
    run["items"]["e_qm1"]["assigned_at"] = 0.0                    # b has been busy longest
    assert swarm.predict_free(run, now=200.0) == "b"
    assert swarm.peek_next(run, "b") == "e_qm4" and swarm.peek_next(run, "c") is None
    clock[0] = 300.0
    swarm.finish_item(run, "e_qm2", "c", True, wall_final_s=150.0)
    assert swarm.next_item(run, "c") == "e_qm4"                  # c freed first: it gets it
    # a pod whose worker stopped asking is not "idle" for the policy
    swarm.finish_item(run, "e_qm4", "c", True)
    swarm.finish_item(run, "e_qm3", "a", True)
    run["idle_since"]["a"] = 0.0
    clock[0] = 300.0 + swarm.LIVE_STALE_S + 1
    swarm.add_items(run, [{"match_key": "e_qm5", "video": "/v5"}])
    assert swarm.next_item(run, "c") == "e_qm5"


def test_median_and_retry_first():
    run = swarm.new_run({"id": "l", "event": "e", "items": []}, ["a", "b"], policy="live")
    run["open"] = True
    assert swarm.median_job_s(run) == swarm.DEFAULT_JOB_MIN * 60
    swarm.add_items(run, [{"match_key": "e_qm1", "video": "/v"}, {"match_key": "e_qm2", "video": "/v"}])
    k = swarm.next_item(run, "a")
    swarm.finish_item(run, k, "a", False, error="x")
    assert swarm.next_item(run, "b") == "e_qm1"                  # the retry before the new arrival


# ----------------------------------------------------------------- TBA poller
def _tba_rows(videos: dict, times=None):
    rows = []
    for mk, vids in videos.items():
        n = int(mk.split("qm")[1])
        rows.append({"key": mk, "comp_level": "qm", "set_number": 1, "match_number": n,
                     "videos": [{"type": "youtube", "key": v} for v in vids] + [{"type": "tba", "key": "x"}],
                     "actual_time": (times or {}).get(mk, 1000 + 420 * n), "post_result_time": 2000 + n})
    return rows


def test_tba_poller_diffs_and_replays():
    state = {"rows": _tba_rows({"2026x_qm2": ["v2"], "2026x_qm1": ["v1"], "2026x_qm3": []})}
    logs = []
    src = live.TbaSource("2026x", lambda path: state["rows"], {}, log=logs.append)
    a = src.poll(10.0)
    assert [x["key"] for x in a] == ["2026x_qm1", "2026x_qm2"] and a[0]["video_key"] == "v1"
    assert a[0]["available_at"] == 10.0
    assert src.poll(20.0) == []                                  # nothing new
    state["rows"] = _tba_rows({"2026x_qm1": ["v1", "v1b"], "2026x_qm2": ["v2"], "2026x_qm3": ["v3"]})
    a = src.poll(30.0)
    assert [x["key"] for x in a] == ["2026x_qm3"] and any("replayed" in m for m in logs)
    src2 = live.TbaSource("2026x", lambda path: state["rows"],
                          {"2026x_qm1": {"videos": ["v1"], "first_seen": 1, "n": 1}}, reprocess_replayed=True)
    a = src2.poll(40.0)
    keys = {x["key"]: x for x in a}
    assert keys["2026x_qm1@r2"]["video_key"] == "v1b" and keys["2026x_qm1@r2"]["match_key"] == "2026x_qm1"

    def boom(path):
        raise live.urllib.error.URLError("down")
    assert live.TbaSource("2026x", boom, {}, log=logs.append).poll(1.0) == [] and "poll failed" in logs[-1]


def test_arrival_step_downloads_queues_and_retries_failed_download(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(swarm, "_now", lambda: clock[0])
    sw = swarm.Swarm.load("lv", create=True)
    sw.data["plan"] = {"id": "l", "kind": "live", "event": "2026x", "items": []}
    run = sw.data["run"] = swarm.new_run(sw.data["plan"], ["a"], policy="live")
    run["open"] = True
    ctx = swarm.RunCtx(sw, argparse.Namespace(poll_s=0, terminate_when_done=False), sleep=lambda s: None)
    rows = {"rows": _tba_rows({"2026x_qm1": ["v1"], "2026x_qm2": ["v2"]})}
    src = live.TbaSource("2026x", lambda p: rows["rows"], {}, log=lambda *a: None)
    fail = {"v2": 1}

    def dl(video_key, dest):
        if fail.get(video_key):
            fail[video_key] -= 1
            raise RuntimeError("yt 403")
        clock[0] += 30
        return _video(Path(dest), 100)
    added = live.arrival_step(ctx, src, now=1000.0, cache_dir=tmp_path / "dl", downloader=dl)
    assert added == ["2026x_qm1"]
    it = run["items"]["2026x_qm1"]
    assert it["t_available"] == 1000.0 and it["t_downloaded"] == 1030.0 and it["video_key"] == "v1"
    assert run["unassigned"] == ["2026x_qm1"]
    assert live.arrival_step(ctx, src, now=1100.0, cache_dir=tmp_path / "dl", downloader=dl) == ["2026x_qm2"]


# ----------------------------------------------------------------- calibration sync
def test_sync_file_selection_and_listing_parse(monkeypatch):
    assert swarm.sync_selectable("calibration_2026x_main__20261113.json")
    assert swarm.sync_selectable("composite_layout_2026x.json") and swarm.sync_selectable("zones_2026x.json")
    assert swarm.sync_selectable("field_season_2026x.json")
    assert not swarm.sync_selectable("clockscan_abc.json")         # per-video cache: never synced
    assert not swarm.sync_selectable("detections_J.jsonl") and not swarm.sync_selectable("calibration_x.mp4")

    def remote(m, cmd, timeout=0, stdin_path=None):
        assert "sha1sum" in cmd
        return ("aa  calibration_e_main__1.json\nbb  clockscan_v.json\ncc  zones_e.json\n"
                "dd  video.mp4\n")
    monkeypatch.setattr(swarm, "remote", remote)
    assert swarm.PodMember().list_event_state({"pod_id": "p"}) == {"calibration_e_main__1.json": "aa",
                                                                   "zones_e.json": "cc"}


def test_push_files_is_atomic_tmp_then_mv_and_fetch_extracts(tmp_path, monkeypatch):
    f1 = tmp_path / "calibration_e_main__2.json"
    f1.write_text('{"v": 2}')
    seen = {}

    def remote(m, cmd, timeout=0, stdin_path=None):
        seen["cmd"] = cmd
        with tarfile.open(stdin_path) as tf:
            seen["names"] = tf.getnames()
        return ""
    monkeypatch.setattr(swarm, "remote", remote)
    swarm.PodMember().push_files({"pod_id": "p"}, [f1])
    cmd = seen["cmd"]
    assert seen["names"] == ["calibration_e_main__2.json"]
    assert "tar -xf - -C" in cmd and ".swarm-sync-" in cmd and "mv -f" in cmd and cmd.index("tar") < cmd.index("mv")
    # fetch: base64 tar from the pod -> files written via tmp + replace
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        data = b'{"v": 3}'
        ti = tarfile.TarInfo("calibration_e_main__3.json")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
    import base64
    monkeypatch.setattr(swarm, "remote", lambda m, cmd, timeout=0, stdin_path=None: base64.b64encode(buf.getvalue()).decode())
    got = swarm.PodMember().fetch_files({"pod_id": "p"}, ["calibration_e_main__3.json"], tmp_path / "stage")
    assert got == ["calibration_e_main__3.json"]
    assert (tmp_path / "stage" / "calibration_e_main__3.json").read_bytes() == data


class FakeT:
    """Per-pod event-state store for the sync tests."""

    def __init__(self, pods: dict, fail_push=()):
        self.pods, self.fail_push, self.pushes = pods, set(fail_push), []

    def list_event_state(self, m):
        return {n: hashlib.sha1(b).hexdigest() for n, b in self.pods[m["pod_id"]].items()
                if swarm.sync_selectable(n)}

    def fetch_files(self, m, names, dest):
        dest.mkdir(parents=True, exist_ok=True)
        for n in names:
            (dest / n).write_bytes(self.pods[m["pod_id"]][n])
        return list(names)

    def push_files(self, m, files):
        if m["pod_id"] in self.fail_push:
            raise swarm.Transport("ssh down")
        self.pushes.append((m["pod_id"], sorted(Path(f).name for f in files)))
        for f in files:
            self.pods[m["pod_id"]][Path(f).name] = Path(f).read_bytes()


def test_event_state_sync_after_job(tmp_path, monkeypatch):
    base = {"calibration_e_main__1.json": b"v1", "zones_e.json": b"z"}
    pods = {p: dict(base) for p in ("a", "b", "c")}
    T = FakeT(pods, fail_push={"c"})
    monkeypatch.setattr(swarm, "transport", lambda m: T)
    sw = swarm.Swarm.load("sy", create=True)
    for p in "abc":
        sw.members.append({"pod_id": p, "status": "ready"})
    run = sw.data["run"] = swarm.new_run({"id": "l", "event": "e", "items": []}, ["a", "b", "c"], policy="live")
    run["live"] = {"baseline": {n: hashlib.sha1(b).hexdigest() for n, b in base.items()}}
    ctx = swarm.RunCtx(sw, argparse.Namespace(poll_s=0, terminate_when_done=False))
    sync = live.EventStateSync(sw, run, tmp_path / "stage")
    # a's job re-solved main (camera moved) and wrote a clock-scan cache
    pods["a"]["calibration_e_main__2.json"] = b"v2"
    pods["a"]["clockscan_vid.json"] = b"cache"
    assert sync(ctx, {"pod_id": "a", "status": "ready"}, "e_qm7", True, {}) == ["calibration_e_main__2.json"]
    assert T.pushes == [("b", ["calibration_e_main__2.json"])]
    assert "clockscan_vid.json" not in pods["b"] and pods["b"]["calibration_e_main__2.json"] == b"v2"
    assert "c" not in [p for p, _ in T.pushes]
    # nothing new on b: no fetch; c (push failed) is retried; a pod added later gets the backlog
    T.fail_push.clear()
    sw.members.append({"pod_id": "d", "status": "ready"})
    pods["d"] = dict(base)
    swarm.add_pod(run, "d")
    assert sync(ctx, {"pod_id": "b", "status": "ready"}, "e_qm8", True, {}) == []
    assert sorted(p for p, _ in T.pushes[1:]) == ["c", "d"]
    assert pods["d"]["calibration_e_main__2.json"] == b"v2"
    # b now changes zones: everyone else (incl. a) gets it, b does not get its own file back
    T.pushes.clear()
    pods["b"]["zones_e.json"] = b"z2"
    assert sync(ctx, {"pod_id": "b", "status": "ready"}, "e_qm9", True, {}) == ["zones_e.json"]
    assert sorted(p for p, _ in T.pushes) == ["a", "c", "d"] and all(n == ["zones_e.json"] for _, n in T.pushes)
    assert [e["from"] for e in run["live"]["sync_log"]] == ["a", "b", "b"]
    # the local backend never syncs
    assert sync(ctx, {"pod_id": "local", "backend": "local"}, "k", True, {}) == []


# ----------------------------------------------------------------- latency + rehearsal
def test_latency_accounting():
    run = swarm.new_run({"id": "l", "event": "e", "items": []}, ["a", "b"], policy="live")
    run["started_at"] = 0.0
    items = {"m1": dict(t_available=0, t_downloaded=60, assigned_at=70, launched_at=80, finished_at=1270,
                        status="done", pod="a", wall_final_s=1180),
             "m2": dict(t_available=420, t_downloaded=470, assigned_at=480, launched_at=490, finished_at=1700,
                        status="done", pod="b", wall_final_s=1200),
             "m3": dict(t_available=840, t_downloaded=900, assigned_at=1270, launched_at=1280, status="running",
                        pod="a")}
    for k, v in items.items():
        run["items"][k] = {"video": "/v", "attempts": 1, "avoid": [], **v}
    rows = live.latency_rows(run)
    assert [r["key"] for r in rows] == ["m1", "m2", "m3"]
    assert rows[0]["download_s"] == 60 and rows[0]["dispatch_s"] == 70 and rows[0]["final_s"] == 1270
    assert rows[2]["final_s"] is None and rows[2]["dispatch_s"] == 430
    s = live.latency_summary(run, now=2000.0)
    assert s["n_done"] == 2 and s["latency_p50_s"] == pytest.approx(1275.0)
    assert s["latency_p90_s"] == pytest.approx(1270 + 0.9 * 10)
    assert s["utilization"]["a"] == pytest.approx((1200 + 730) / 2000)
    assert s["utilization"]["b"] == pytest.approx(1220 / 2000)


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_rehearsal_clock_rebases_actual_times(tmp_path, monkeypatch):
    rows = _tba_rows({"2026x_qm1": [], "2026x_qm2": [], "2026x_qm3": [], "2026x_qm4": []},
                     times={"2026x_qm1": 5000, "2026x_qm2": 5420, "2026x_qm3": 5840, "2026x_qm4": None})
    vids = {k: _video(tmp_path / f"{k}.mp4") for k in ("2026x_qm1", "2026x_qm2")}
    sched, missing = live.rehearsal_schedule(rows, "qm1-qm4", lambda mk: vids.get(mk))
    assert [m["match_key"] for m in sched] == ["2026x_qm1", "2026x_qm2"]
    assert {m["match_key"]: m["why"] for m in missing} == {"2026x_qm3": "no local video",
                                                          "2026x_qm4": "no TBA actual_time"}
    src = live.RehearsalSource(sched, t0_real=100.0, speed=2.0, emitted=[])
    assert [a["key"] for a in src.poll(100.0)] == ["2026x_qm1"]
    assert src.poll(309.0) == [] and src.next_due() == 310.0       # 420 s of event time at x2
    a = src.poll(310.0)
    assert a[0]["available_at"] == 310.0 and a[0]["video"] == str(vids["2026x_qm2"]) and src.exhausted
    # the arrival loop on a fake clock: sleeps exactly to the next due time, then closes the run
    clock = Clock(100.0)
    monkeypatch.setattr(swarm, "_now", clock.now)
    sw = swarm.Swarm.load("rh", create=True)
    sw.data["plan"] = {"id": "l", "kind": "live", "event": "2026x", "items": []}
    run = sw.data["run"] = swarm.new_run(sw.data["plan"], ["a"], policy="live")
    run["open"] = True
    ctx = swarm.RunCtx(sw, argparse.Namespace(poll_s=0, terminate_when_done=False))
    src = live.RehearsalSource(sched, t0_real=100.0, speed=2.0, emitted=[])
    live.arrival_loop(ctx, src, poll_s=60, cache_dir=tmp_path, until=None, clock=clock)
    assert run["open"] is False and clock.t == 310.0
    assert [run["items"][k]["t_available"] for k in ("2026x_qm1", "2026x_qm2")] == [100.0, 310.0]
    with pytest.raises(SystemExit):
        live.RehearsalSource(sched, 0, 0, [])


def test_live_rehearsal_end_to_end_with_fake_members(tmp_path, monkeypatch, capsys):
    """swarm live --rehearse on two fake pods: arrivals at the (sped-up) event
    cadence, idle-pod dispatch, calibration sync after each job, latency file."""
    clock = Clock(1000.0)
    monkeypatch.setattr(swarm, "_now", clock.now)
    keys = [f"2026x_qm{i}" for i in range(1, 5)]
    for k in keys:
        _video(tmp_path / "vids" / f"{k}.mp4")
    rows = _tba_rows({k: [] for k in keys})
    sw = swarm.Swarm.load("lr", create=True)
    for p in ("a", "b"):
        sw.members.append({"pod_id": p, "status": "ready", "cost_per_hr": 1, "created_at": 0})
    sw.save()
    ran, synced = [], []

    def fake_process(ctx, m, key):
        ran.append((m["pod_id"], key))
        clock.t += 100
        return True, {"job_id": "J" + key, "wall_final_s": 100.0}
    monkeypatch.setattr(swarm, "process_item", fake_process)
    monkeypatch.setattr(live.EventStateSync, "__call__",
                        lambda self, ctx, m, key, ok, st: synced.append((m["pod_id"], key)) or [])
    monkeypatch.setattr(swarm.time, "sleep", lambda s: None)
    args = argparse.Namespace(name="lr", event="2026x", poll_s=5, job_poll_s=0, config_json=None,
                              reprocess_replayed=False, auto_import=False, hours=None, cache_dir=None,
                              rehearse=True, speed=100.0, matches="all", videos=str(tmp_path / "vids"),
                              local=False, local_url=None, local_app_dir=None, restart_each=None)
    assert live.cmd_live(args, fetch=lambda path: rows) == 0
    run = swarm.Swarm.load("lr").data["run"]
    assert swarm.run_counts(run)["done"] == 4 and run["open"] is False
    assert sorted(k for _, k in ran) == keys and sorted(k for _, k in synced) == keys
    assert {p for p, _ in ran} == {"a", "b"}
    rep = json.loads((cli.PULL_ROOT / "live" / "lr" / "latency.json").read_text())
    assert rep["summary"]["n_done"] == 4 and all(r["final_s"] is not None for r in rep["rows"])
    assert "latency (video seen -> final)" in capsys.readouterr().out


def test_auto_import_backs_up_once_and_waits_for_a_local_job(tmp_path, monkeypatch):
    sw = swarm.Swarm.load("ai", create=True)
    calls = {"backup": 0, "apply": 0, "pull": 0}
    monkeypatch.setattr(swarm, "plan_import", lambda sw_, **k: {"entries": [{"action": "add", "id": "R",
                                                                              "match_key": "m"}]})
    monkeypatch.setattr(swarm, "backup_db", lambda db, tag: calls.__setitem__("backup", calls["backup"] + 1) or "bak")
    monkeypatch.setattr(swarm, "apply_import", lambda e, **k: calls.__setitem__("apply", calls["apply"] + 1) or 1)
    busy = ["J"]
    imp = live.AutoImporter(sw, active_job=lambda: busy[0],
                            pull=lambda sw_, m: calls.__setitem__("pull", calls["pull"] + 1))
    ctx = swarm.RunCtx(sw, argparse.Namespace(poll_s=0, terminate_when_done=False))
    m = {"pod_id": "p"}
    assert imp(ctx, m, "k1", True, {}) == 0 and calls == {"backup": 0, "apply": 0, "pull": 1}
    busy[0] = None
    assert imp(ctx, m, "k2", True, {}) == 1 and imp(ctx, m, "k3", True, {}) == 1
    assert calls == {"backup": 1, "apply": 2, "pull": 3}
    assert imp(ctx, m, "k4", False, {}) == 0 and imp(ctx, {"pod_id": "local", "backend": "local"}, "k", True, {}) == 0
