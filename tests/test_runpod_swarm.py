"""deploy/runpod/swarm.py: assignment, plan, state, import, dry run (offline:
no pods, no RunPod API, no ssh, no YouTube)."""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNPOD = REPO / "deploy" / "runpod"
if str(RUNPOD) not in sys.path:
    sys.path.insert(0, str(RUNPOD))

import runpod_cli as cli  # noqa: E402
import swarm  # noqa: E402
from web.database import SCHEMA  # noqa: E402
from web.field.event_anchors import DDL as ANCHOR_DDL  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every test: swarm state + pulls in tmp, and any real network/ssh call fails loudly."""
    monkeypatch.setattr(swarm, "SWARM_DIR", tmp_path / ".swarm")
    monkeypatch.setattr(cli, "PULL_ROOT", tmp_path / "pull")
    monkeypatch.setattr(cli, "STATE_PATH", tmp_path / "single.state.json")

    def no_network(*a, **k):
        raise AssertionError(f"network call in an offline test: {a[:2]}")
    monkeypatch.setattr(cli, "_request", no_network)
    monkeypatch.setattr(cli, "ssh_run", no_network)
    monkeypatch.setattr(cli, "ssh_out", no_network)
    monkeypatch.setattr(swarm, "default_downloader", no_network)
    return tmp_path


def _plan(keys, tmp_path=None):
    items = []
    for k in keys:
        v = (tmp_path / f"{k}.mp4") if tmp_path else Path(f"/v/{k}.mp4")
        items.append({"match_key": k, "video": str(v), "size": 10})
    return {"id": "p1", "event": "2026tst", "items": items}


KEYS = [f"2026tst_qm{i}" for i in range(1, 11)]


# ----------------------------------------------------------------- assignment
def test_split_blocks_contiguous_and_even():
    b = swarm.split_blocks(KEYS, ["a", "b", "c"])
    assert [len(b[p]) for p in "abc"] == [4, 3, 3]
    assert b["a"] + b["b"] + b["c"] == KEYS            # contiguous, schedule order kept
    assert swarm.split_blocks(KEYS[:2], ["a", "b", "c"]) == {"a": [KEYS[0]], "b": [KEYS[1]], "c": []}
    assert swarm.split_blocks([], ["a"]) == {"a": []}


def test_next_item_steals_tail_half_of_largest_block():
    run = swarm.new_run(_plan(KEYS), ["a", "b"])
    assert run["queues"]["a"] == KEYS[:5] and run["queues"]["b"] == KEYS[5:]
    # b is fast: it drains its own block, then steals from a
    assert swarm.next_item(run, "a") == KEYS[0]           # a busy with qm1
    for k in KEYS[5:]:
        assert swarm.next_item(run, "b") == k
        swarm.finish_item(run, k, "b", True)
    # a's remaining queue = qm2..qm5 -> b takes the tail half (qm4, qm5)
    assert swarm.next_item(run, "b") == KEYS[3]
    assert run["queues"]["a"] == KEYS[1:3] and run["queues"]["b"] == [KEYS[4]]
    assert run["steals"][-1]["items"] == KEYS[3:5]
    # resume: the in-flight item comes back, nothing new is assigned
    assert swarm.next_item(run, "b") == KEYS[3]
    assert run["items"][KEYS[3]]["attempts"] == 1


def _simulate(pods_speed: dict, keys: list, fail=lambda mk, pod, attempt: False, die: dict | None = None):
    """Event-driven sim of the dispatcher with the pure functions only."""
    run = swarm.new_run(_plan(keys), list(pods_speed))
    t, busy, log = 0.0, {}, []          # pod -> (finish time, mk)
    die = dict(die or {})
    for _ in range(10_000):
        for p in pods_speed:
            if p in busy or p in run["dead"]:
                continue
            mk = swarm.next_item(run, p)
            if mk:
                busy[p] = (t + pods_speed[p], mk)
        if not busy:
            break
        p, (t, mk) = min(busy.items(), key=lambda kv: kv[1][0])
        del busy[p]
        if p in die and t >= die[p]:
            swarm.mark_pod_dead(run, p, "gone")
            log.append((p, mk, "lost"))
            continue
        ok = not fail(mk, p, run["items"][mk]["attempts"])
        swarm.finish_item(run, mk, p, ok)
        log.append((p, mk, ok))
    return run, log


def test_sim_uneven_speeds_all_done_once_and_blocks_stay_contiguous():
    keys = [f"2026tst_qm{i}" for i in range(1, 41)]
    run, log = _simulate({"fast": 1.0, "mid": 2.0, "slow": 5.0}, keys)
    assert swarm.all_terminal(run) and swarm.run_counts(run)["done"] == 40
    assert sorted(mk for _, mk, ok in log if ok) == sorted(keys)        # exactly once
    per = {p: [mk for q, mk, _ in log if q == p] for p in ("fast", "mid", "slow")}
    assert len(per["fast"]) > len(per["mid"]) > len(per["slow"])      # stealing balanced the load
    # every pod's work is a few contiguous runs of the schedule, not round-robin
    for p, mks in per.items():
        idx = [keys.index(m) for m in mks]
        runs = 1 + sum(1 for a, b in zip(idx, idx[1:]) if b != a + 1)
        assert runs <= 6, (p, idx)


def test_sim_failure_retried_once_on_another_pod_then_failed():
    keys = KEYS[:6]
    bad = {KEYS[1]: "always", KEYS[4]: "once"}

    def fail(mk, pod, attempt):
        return bad.get(mk) == "always" or (bad.get(mk) == "once" and attempt == 1)
    run, log = _simulate({"a": 1.0, "b": 1.0}, keys, fail)
    it1, it4 = run["items"][KEYS[1]], run["items"][KEYS[4]]
    assert it1["status"] == "failed" and it1["attempts"] == 2
    assert len({e["pod"] for e in it1["errors"]}) == 2               # second try on the other pod
    assert it4["status"] == "done" and it4["attempts"] == 2
    tried4 = [p for p, mk, _ in log if mk == KEYS[4]]
    assert tried4[0] != tried4[1]
    assert swarm.run_counts(run)["done"] == 5


def test_retry_falls_back_to_the_same_pod_when_it_is_the_last_one():
    run = swarm.new_run(_plan(KEYS[:2]), ["a", "b"])
    swarm.retire_pod(run, "b")
    run["queues"]["a"] = KEYS[:2]
    run["queues"]["b"] = []
    mk = swarm.next_item(run, "a")
    swarm.finish_item(run, mk, "a", False, error="boom")
    assert run["items"][mk]["status"] == "retry"
    assert swarm.next_item(run, "a") == KEYS[1]                      # own block first
    swarm.finish_item(run, KEYS[1], "a", True)
    assert swarm.next_item(run, "a") == mk                           # nobody else left -> same pod
    swarm.finish_item(run, mk, "a", False, error="boom")
    assert run["items"][mk]["status"] == "failed" and swarm.all_terminal(run)


def test_dead_pod_block_is_stolen_whole_and_in_flight_item_retried():
    run, log = _simulate({"a": 1.0, "b": 1.0, "c": 1.0}, KEYS, die={"c": 0.5})
    assert "c" in run["dead"]
    assert swarm.run_counts(run)["done"] == 10
    lost = [mk for p, mk, s in log if s == "lost"]
    assert lost and run["items"][lost[0]]["attempts"] == 2 and run["items"][lost[0]]["pod"] != "c"


def test_open_run_accepts_new_items_and_waits():
    run = swarm.new_run(_plan(KEYS[:1]), ["a"])
    run["open"] = True
    mk = swarm.next_item(run, "a")
    swarm.finish_item(run, mk, "a", True)
    assert swarm.all_terminal(run) and not swarm.run_finished(run)  # open -> keep waiting
    assert swarm.next_item(run, "a") is None
    assert swarm.add_items(run, [{"match_key": KEYS[1], "video": "/v", "size": 1},
                                 {"match_key": KEYS[0], "video": "/v", "size": 1}]) == [KEYS[1]]
    assert swarm.next_item(run, "a") == KEYS[1]
    swarm.finish_item(run, KEYS[1], "a", True)
    run["open"] = False
    assert swarm.run_finished(run)


def test_policy_interface_and_per_pod_config():
    pol = swarm.POLICIES["blocks"]
    assert all(callable(getattr(pol, f)) for f in ("initial", "place", "pick"))
    run = swarm.new_run(_plan(KEYS[:2]), ["a", "b"], {"sam3_track_fps": 8},
                        pod_config={"b": {"sam3_track_fps": 10, "corner_track_stride": 1}})
    assert swarm.job_config(run, "a") == {"sam3_track_fps": 8}
    assert swarm.job_config(run, "b") == {"sam3_track_fps": 10, "corner_track_stride": 1}
    assert swarm.parse_pod_configs(['p1={"x": 1}']) == {"p1": {"x": 1}}
    with pytest.raises(SystemExit):
        swarm.parse_pod_configs(["p1=notjson"])
    run["policy"] = "nope"
    with pytest.raises(SystemExit):
        swarm.next_item(run, "a")


# ----------------------------------------------------------------- plan
def _make_db(path: Path, event="2026tst"):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executescript(ANCHOR_DDL)
    con.execute("INSERT INTO events (event_key, event_name, year) VALUES (?, 'Test', 2026)", (event,))
    eid = con.execute("SELECT id FROM events").fetchone()[0]
    rows = [("qm", 1, 10), ("qm", 1, 2), ("qm", 1, 1), ("f", 1, 1), ("sf", 2, 1), ("sf", 1, 1), ("qm", 1, 3)]
    for lvl, s, n in rows:
        key = f"{event}_{lvl}{n}" if lvl == "qm" else f"{event}_{lvl}{s}m{n}"
        con.execute("INSERT INTO matches (event_id, tba_match_key, comp_level, match_number, set_number, "
                    "tba_video_key) VALUES (?, ?, ?, ?, ?, ?)", (eid, key, lvl, n, s, f"yt_{key}"))
    con.commit()
    con.close()
    return path


def test_schedule_order_and_spec_parsing(tmp_path):
    db = _make_db(tmp_path / "frc.db")
    sched = swarm.schedule_from_db(db, "2026tst")
    order = [r["match_key"].split("_")[1] for r in sched]
    assert order == ["qm1", "qm2", "qm3", "qm10", "sf1m1", "sf2m1", "f1m1"]

    def sel(spec):
        return [r["match_key"].split("_")[1] for r in swarm.select_matches(sched, spec)]
    assert sel("all") == order
    assert sel("qm2-qm10") == ["qm2", "qm3", "qm10"]
    assert sel("qm1,sf1m1") == ["qm1", "sf1m1"]
    assert sel("2026tst_qm3") == ["qm3"]
    assert sel("playoffs") == ["sf1m1", "sf2m1", "f1m1"]
    assert sel("sf") == ["sf1m1", "sf2m1"] and sel("sf2") == ["sf2m1"]
    assert sel("qm3-f1m1") == ["qm3", "qm10", "sf1m1", "sf2m1", "f1m1"]
    for bad in ("qm", "xyz", "qm5-qm1"):
        if bad == "qm":
            assert sel(bad) == ["qm1", "qm2", "qm3", "qm10"]
            continue
        with pytest.raises(SystemExit):
            sel(bad)
    with pytest.raises(SystemExit):
        swarm.schedule_from_db(db, "2026none")


def test_build_plan_videos_download_and_skip_existing(tmp_path):
    db = _make_db(tmp_path / "frc.db")
    con = sqlite3.connect(db)
    con.execute("INSERT INTO replays (id, match_key, status) VALUES ('r1', '2026tst_qm1', 'complete')")
    con.execute("INSERT INTO replays (id, match_key, status) VALUES ('r2', '2026tst_qm2', 'partial')")
    con.commit()
    con.close()
    # a previous pull has qm3 (dir + row); its baseline row for qm10 has no dir -> not counted
    pull = tmp_path / "pull" / "podA"
    (pull / "web" / "replays" / "r3").mkdir(parents=True)
    (pull / "web" / "replays" / "r3" / "meta.json").write_text("{}")
    _make_db(pull / "frc_scorer.db")
    pc = sqlite3.connect(pull / "frc_scorer.db")
    pc.execute("INSERT INTO replays (id, match_key, status) VALUES ('r3', '2026tst_qm3', 'complete')")
    pc.execute("INSERT INTO replays (id, match_key, status) VALUES ('r9', '2026tst_qm10', 'complete')")
    pc.commit()
    pc.close()
    vids = tmp_path / "vids"
    vids.mkdir()
    (vids / "2026tst_qm2.mp4").write_bytes(b"x" * 100)
    cache = tmp_path / "cache"
    got = []

    def fake_dl(video_key, out):
        got.append(video_key)
        if "sf2m1" in video_key:
            raise RuntimeError("Sign in to confirm you're not a bot")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"v" * 50)
        return out
    run = {"items": {"2026tst_f1m1": {"status": "done"}}}
    plan = swarm.build_plan("2026tst", "all", local_db=db, pull_root=tmp_path / "pull", run=run,
                            videos=str(vids), download=True, cache_dir=cache, downloader=fake_dl,
                            out=lambda m: None)
    assert [i["match_key"] for i in plan["items"]] == ["2026tst_qm2", "2026tst_qm10", "2026tst_sf1m1"]
    assert {s["match_key"] for s in plan["skipped"]} == {"2026tst_qm1", "2026tst_qm3", "2026tst_f1m1"}
    assert plan["missing"] == [{"match_key": "2026tst_sf2m1",
                                "why": "download failed: Sign in to confirm you're not a bot"}]
    assert got == ["yt_2026tst_qm10", "yt_2026tst_sf1m1", "yt_2026tst_sf2m1"]   # qm2 was local
    assert plan["items"][0]["video"] == str((vids / "2026tst_qm2.mp4").resolve())
    # --redo ignores existing results; no --download -> missing
    plan2 = swarm.build_plan("2026tst", "qm1", local_db=db, pull_root=tmp_path / "pull", run=None,
                             cache_dir=cache, redo=True, out=lambda m: None)
    assert plan2["items"] == [] and plan2["missing"][0]["match_key"] == "2026tst_qm1"


# ----------------------------------------------------------------- state
def test_state_roundtrip_atomic_and_resume(tmp_path):
    sw = swarm.Swarm.load("t1", create=True)
    sw.members.append({"pod_id": "p1", "status": "ready", "ssh_host": "1.2.3.4", "ssh_port": 22})
    sw.data["run"] = swarm.new_run(_plan(KEYS[:4]), ["p1"])
    mk = swarm.next_item(sw.data["run"], "p1")
    sw.save()
    assert not list(sw.path.parent.glob("*.tmp"))
    sw2 = swarm.Swarm.load("t1")
    assert sw2.member("p1")["ssh_host"] == "1.2.3.4"
    assert swarm.next_item(sw2.data["run"], "p1") == mk            # re-attach to the in-flight item
    # a member dict is a valid single-pod `st`
    assert cli.ssh_base(sw2.member("p1"))[-1] == "root@1.2.3.4"
    with pytest.raises(SystemExit):
        swarm.Swarm.load("missing")
    with pytest.raises(SystemExit):
        swarm.check_name("../evil")
    assert not cli.STATE_PATH.exists()


def test_refresh_endpoint_member_save_never_touches_single_pod_state(monkeypatch):
    monkeypatch.setattr(cli, "get_pod", lambda pid: {"desiredStatus": "RUNNING", "costPerHr": 0.69,
                                                    "publicIp": "9.9.9.9", "portMappings": {"22": 4022}})
    saved = []
    st = {"pod_id": "p1"}
    cli.refresh_endpoint(st, save=lambda s: saved.append(dict(s)))
    assert saved and saved[0]["ssh_host"] == "9.9.9.9" and saved[0]["ssh_port"] == 4022
    assert not cli.STATE_PATH.exists()
    cli.refresh_endpoint({"pod_id": "p2"})                         # single-pod default unchanged
    assert json.loads(cli.STATE_PATH.read_text())["pod_id"] == "p2"


def test_partial_pod_creation_keeps_created_pods(tmp_path):
    sw = swarm.Swarm.load("t2", create=True)
    calls = []

    def fake_request(method, url, body=None, timeout=60):
        calls.append(body["name"])
        if body["name"].endswith("-2"):
            raise SystemExit("RunPod POST -> HTTP 500: There are no longer any instances available")
        return {"id": f"pod{len(calls)}", "costPerHr": 0.69}
    bodies = [cli.build_create_body(name=f"t2-{i}", gpus=["NVIDIA A40"], cloud="SECURE", image="img",
                                    volume_gb=80, container_gb=40, pubkey="k", min_vcpu=8, min_ram=32)
              for i in (1, 2, 3)]
    errs = swarm.create_pods(sw, bodies, request=fake_request, out=lambda m: None)
    assert calls == ["t2-1", "t2-2", "t2-3"]
    assert [m["pod_id"] for m in sw.members] == ["pod1", "pod3"]
    assert len(errs) == 1 and "no longer any instances" in errs[0]
    disk = json.loads(sw.path.read_text())
    assert [m["pod_id"] for m in disk["members"]] == ["pod1", "pod3"]
    assert disk["create_errors"] and all(m["status"] == "created" for m in disk["members"])


def test_cost_estimate_and_prices():
    prices = swarm.gpu_prices(["NVIDIA L40S", "NVIDIA A40"], "SECURE",
                              fetch=lambda: [{"id": "NVIDIA L40S", "securePrice": 0.86, "communityPrice": 0.79}])
    assert prices == {"NVIDIA L40S": 0.86, "NVIDIA A40": 0.59}        # live, then README fallback
    line, nominal, hi = swarm.cost_estimate(4, prices)
    assert "4 pods x $0.59-0.86/h" in line and nominal == pytest.approx(3.44) and hi == pytest.approx(3.44)

    def boom():
        raise SystemExit("RUNPOD_API not found")
    assert swarm.gpu_prices(["NVIDIA A40"], "SECURE", fetch=boom) == {"NVIDIA A40": 0.59}
    assert swarm.member_cost({"cost_per_hr": 1.0, "created_at": 0, "terminated_at": 7200}) == 2.0


def test_swarm_up_dry_run_creates_nothing(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "gpu_types", lambda: [{"id": "NVIDIA L40S", "securePrice": 0.86}])
    monkeypatch.setattr(cli, "local_env", lambda: {"JEV_TYPESAFE_API_TOKEN": "jev-1234567890",
                                                   "HF_TOKEN": "hf_abcdefgh", "TBA_API_KEY": "tba-xyzxyzxyz",
                                                   "RUNPOD_API": "never"})
    monkeypatch.setattr(cli, "warn_dirty", lambda ref: None)
    monkeypatch.setattr(cli, "plan_bundle", lambda ref, **k: cli.BundlePlan(ref=ref, commit="abc1234",
                                                                            code_bytes=b"x" * 10))
    rc = cli.main(["swarm", "up", "--name", "dry", "--count", "3", "--gpu", "NVIDIA L40S",
                   "--dry-run", "--secrets-from-env"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count('"name": "dry-') == 3 and '"gpuTypeIds": [' in out
    assert "3 pods x $0.86/h" in out and "$2.58/h total" in out
    assert "--no-vllm" in out and "RUNPOD_API" not in out and "jev-1234567890" not in out
    assert not (tmp_path / ".swarm").exists()                       # no state for a dry run
    assert not cli.STATE_PATH.exists()


def test_up_requires_count_for_new_swarm(monkeypatch):
    with pytest.raises(SystemExit, match="--count"):
        cli.main(["swarm", "up", "--name", "fresh"])


def test_setup_member_resumes_and_logs(monkeypatch, tmp_path):
    sw = swarm.Swarm.load("t3", create=True)
    m = {"pod_id": "p1", "status": "created", "pushed": True}
    sw.members.append(m)
    calls = []
    monkeypatch.setattr(cli, "wait_for_ssh", lambda st, t, save=None, out=None: calls.append("ssh"))
    monkeypatch.setattr(cli, "push_tar", lambda *a, **k: calls.append("push"))
    monkeypatch.setattr(cli, "install_secrets", lambda st, s, out=None: calls.append(("secrets", sorted(s))))
    monkeypatch.setattr(cli, "run_bootstrap", lambda st, boot_args="", out=None: calls.append(("boot", boot_args)))
    swarm.setup_member(sw, m, None, {"HF_TOKEN": "x"}, "--no-vllm", "main")
    assert calls == ["ssh", ("secrets", ["HF_TOKEN"]), ("boot", "--no-vllm")]   # push already done
    assert m["status"] == "ready" and m["secrets_installed"]

    def bad_boot(st, boot_args="", out=None):
        raise SystemExit("bootstrap failed (exit 1)")
    m2 = {"pod_id": "p2", "status": "created"}
    sw.members.append(m2)
    monkeypatch.setattr(cli, "run_bootstrap", bad_boot)
    swarm.setup_member(sw, m2, tmp_path / "b.tar", {"HF_TOKEN": "x"}, "", "main")
    assert m2["status"] == "setup_failed" and "bootstrap failed" in m2["error"]
    assert (sw.log_dir() / "p2.log").exists()


# ----------------------------------------------------------------- run loop (fakes)
def _ready_swarm(name, pods, keys, tmp_path):
    sw = swarm.Swarm.load(name, create=True)
    for p in pods:
        sw.members.append({"pod_id": p, "status": "ready", "cost_per_hr": 0.5, "created_at": swarm._now()})
    for k in keys:
        (tmp_path / f"{k}.mp4").write_bytes(b"v")
    sw.data["plan"] = _plan(keys, tmp_path)
    sw.save()
    return sw


def test_cmd_run_end_to_end_with_fake_pods_and_terminate(monkeypatch, tmp_path, capsys):
    sw = _ready_swarm("t4", ["a", "b"], KEYS[:5], tmp_path)
    seen = []

    def fake_process(ctx, m, mk):
        seen.append((m["pod_id"], mk, swarm.job_config(ctx.run, m["pod_id"])))
        if mk == KEYS[2] and ctx.run["items"][mk]["attempts"] == 1:
            return False, {"error": "boom"}
        return True, {"job_id": f"job-{mk}", "wall_final_s": 400}
    monkeypatch.setattr(swarm, "process_item", fake_process)
    pulled, killed = [], []

    def fake_pull(sw_, m, with_video=False, out=None):
        pulled.append(m["pod_id"])
        sw_.update(m, pulled_at=swarm._now() + 1)
        for it in sw_.data["run"]["items"].values():
            if it.get("pod") == m["pod_id"] and it.get("job_id"):
                d = cli.PULL_ROOT / m["pod_id"] / "web" / "replays" / it["job_id"]
                d.mkdir(parents=True, exist_ok=True)
                (d / "meta.json").write_text("{}")
    monkeypatch.setattr(swarm, "pull_member", fake_pull)
    monkeypatch.setattr(swarm, "terminate_member",
                        lambda sw_, m, request=None: (killed.append(m["pod_id"]),
                                                      sw_.update(m, status="terminated")))
    monkeypatch.setattr(swarm.time, "sleep", lambda s: None)
    rc = cli.main(["swarm", "run", "--name", "t4", "--poll-s", "0", "--terminate-when-done",
                   "--config-json", '{"sam3_track_fps": 8}', "--pod-config", 'b={"sam3_track_fps": 10}'])
    sw = swarm.Swarm.load("t4")
    run = sw.data["run"]
    assert rc == 0 and swarm.run_counts(run)["done"] == 5
    assert run["items"][KEYS[2]]["attempts"] == 2
    assert {cfg["sam3_track_fps"] for p, _, cfg in seen if p == "a"} == {8}
    assert {cfg["sam3_track_fps"] for p, _, cfg in seen if p == "b"} == {10}
    assert sorted(killed) == ["a", "b"] and sorted(pulled) == ["a", "b"]
    assert sorted(run["retired"]) == ["a", "b"]
    assert "swarm import --name t4 --apply" in capsys.readouterr().out


def test_process_item_uploads_launches_prefetches_and_polls(monkeypatch, tmp_path):
    sw = _ready_swarm("t5", ["a"], KEYS[:2], tmp_path)
    sw.data["run"] = swarm.new_run(sw.data["plan"], ["a"], {"x": 1})
    run = sw.data["run"]
    m = sw.member("a")
    mk = swarm.next_item(run, "a")
    uploads, launches = [], []
    states = iter([None, {"label": f"{mk}.a1", "status": "running", "progress": 40, "pid": 5, "job_id": "J"},
                   {"label": f"{mk}.a1", "status": "done", "job_id": "J", "wall_final_s": 300.0}])
    monkeypatch.setattr(cli, "read_remote_json", lambda st, path, timeout=120: next(states))
    monkeypatch.setattr(cli, "ensure_remote_file",
                        lambda st, local, remote, out=None, what="": uploads.append(remote) or True)
    monkeypatch.setattr(cli, "remote_job_active", lambda st: None)
    monkeypatch.setattr(cli, "pod_job_alive", lambda st, state: True)
    monkeypatch.setattr(cli, "ssh_out", lambda st, cmd, **k: launches.append(cmd) or "started")
    ctx = swarm.RunCtx(sw, argparse.Namespace(poll_s=0, terminate_when_done=False), sleep=lambda s: None)
    ok, state = swarm.process_item(ctx, m, mk)
    assert ok and state["job_id"] == "J"
    assert uploads == [f"{swarm.POD_VIDEOS}/{mk}.mp4", f"{swarm.POD_VIDEOS}/{KEYS[1]}.mp4"]   # + prefetch
    cmd = launches[0]
    assert cmd.startswith(f"rm -f {swarm.POD_JOBS}/{mk}.json; ") and "setsid -f nohup" in cmd
    assert f"--label {mk}.a1" in cmd and "--no-profile" in cmd and "--final-timeout 2400" in cmd
    assert f"--out {swarm.POD_JOBS}/{mk}.json" in cmd and '{"x": 1}' in cmd
    assert run["items"][mk]["progress"] == 100 and run["items"][mk]["job_id"] == "J"


def test_process_item_reattaches_without_relaunch(monkeypatch, tmp_path):
    sw = _ready_swarm("t6", ["a"], KEYS[:1], tmp_path)
    sw.data["run"] = swarm.new_run(sw.data["plan"], ["a"])
    mk = swarm.next_item(sw.data["run"], "a")
    st = {"label": f"{mk}.a1", "status": "running", "pid": 7}
    states = iter([st, {**st, "status": "done", "job_id": "J"}])
    monkeypatch.setattr(cli, "read_remote_json", lambda *a, **k: next(states))
    monkeypatch.setattr(cli, "pod_job_alive", lambda s, state: True)
    monkeypatch.setattr(cli, "ensure_remote_file", lambda *a, **k: pytest.fail("no upload on re-attach"))
    ctx = swarm.RunCtx(sw, argparse.Namespace(poll_s=0, terminate_when_done=False), sleep=lambda s: None)
    ok, state = swarm.process_item(ctx, sw.member("a"), mk)
    assert ok and state["job_id"] == "J"


# ----------------------------------------------------------------- import
def _replay_row(rid, mk, base="/workspace/frc/web/replays"):
    return {"id": rid, "match_key": mk, "event_key": "2026tst", "status": "complete",
            "video_path": f"{base}/{rid}/hud.mp4", "meta_path": f"{base}/{rid}/meta.json",
            "thumb_path": f"{base}/{rid}/thumb.jpg",
            "source_video_path": f"/workspace/batch/videos/{mk}.mp4", "render_status": "rendered"}


def _insert(con, table, row):
    con.execute(f"INSERT INTO {table} ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})", list(row.values()))


@pytest.fixture
def import_env(tmp_path):
    main = tmp_path / "main"
    (main / "web" / "replays").mkdir(parents=True)
    (main / "web" / "uploads").mkdir(parents=True)
    local_db = _make_db(main / "frc_scorer.db")
    con = sqlite3.connect(local_db)
    _insert(con, "replays", _replay_row("r-dup", "2026tst_qm1", str(main / "web" / "replays")))
    _insert(con, "match_robot_identity", {"job_id": "r-dup", "identity_id": 1, "team_number": "111"})
    _insert(con, "match_robot_crop", {"job_id": "zzz", "identity_id": 1})      # takes id 1 locally
    con.commit()
    con.close()
    (main / "web" / "uploads" / "calibration_2026tst_main.json").write_text('{"v": 1}')
    pod = cli.PULL_ROOT / "podA"
    pdb = _make_db(pod / "frc_scorer.db") if (pod.mkdir(parents=True) or True) else None
    pc = sqlite3.connect(pdb)
    for rid, mk in (("r-new", "2026tst_qm2"), ("r-dup", "2026tst_qm1")):
        _insert(pc, "replays", _replay_row(rid, mk))
        d = pod / "web" / "replays" / rid
        d.mkdir(parents=True)
        (d / "meta.json").write_text("{}")
        (d / "thumb.jpg").write_bytes(b"jpg")
    _insert(pc, "replays", _replay_row("r-base", "2026tst_qm3"))        # baseline row, no dir pulled
    _insert(pc, "match_robot_identity", {"job_id": "r-new", "identity_id": 3, "slot": 0, "team_number": "254",
                                         "label_share": 0.9})
    _insert(pc, "match_robot_identity", {"job_id": "r-dup", "identity_id": 1, "team_number": "999"})
    _insert(pc, "match_robot_crop", {"job_id": "r-new", "identity_id": 3, "legibility": 0.5, "image": b"\x00"})
    _insert(pc, "match_robot_identity_segment", {"job_id": "r-new", "slot": 0, "start_frame": 1, "end_frame": 9,
                                                 "team_number": "254", "source": "window_digits"})
    _insert(pc, "match_robot_binding", {"job_id": "r-new", "identity_id": 3, "slot": 0, "start_frame": 1})
    _insert(pc, "appearance_gallery", {"event_key": "2026tst", "team": "254", "alliance_color": "red",
                                       "embedding": b"e", "job_id": "r-new"})
    _insert(pc, "event_anchors", {"namespace": "live", "event_key": "2026tst", "team": "254",
                                  "match_key": "2026tst_qm2", "replay_id": "r-new", "kind": "digits",
                                  "tile_path": "live/2026tst/1.jpg"})
    pc.commit()
    pc.close()
    (pod / "web" / "uploads").mkdir(parents=True)
    (pod / "web" / "uploads" / "calibration_2026tst_main.json").write_text('{"v": 2}')
    (pod / "web" / "uploads" / "zones_2026tst.json").write_text("{}")
    video = tmp_path / "2026tst_qm2.mp4"
    video.write_bytes(b"v")
    sw = swarm.Swarm.load("imp", create=True)
    sw.members.append({"pod_id": "podA", "status": "terminated"})
    sw.data["plan"] = {"id": "p", "event": "2026tst",
                       "items": [{"match_key": "2026tst_qm2", "video": str(video), "size": 1}]}
    sw.save()
    return types.SimpleNamespace(main=main, db=local_db, replays=main / "web" / "replays",
                                 uploads=main / "web" / "uploads", video=video)


def _dump(db):
    con = sqlite3.connect(db)
    try:
        return {t: con.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall()
                for t in ("replays", "match_robot_identity", "match_robot_crop", "match_robot_identity_segment",
                          "match_robot_binding", "appearance_gallery", "event_anchors")}
    finally:
        con.close()


def _import(env, apply, **kw):
    args = argparse.Namespace(name="imp", apply=apply, render=kw.pop("render", False))
    return swarm.cmd_import(args, local_db=env.db, replays_dir=env.replays, uploads_dir=env.uploads, **kw)


def test_import_dry_run_changes_nothing(import_env, capsys):
    before = _dump(import_env.db)
    assert _import(import_env, False, active_job=lambda: pytest.fail("dry run must not ask")) == 0
    out = capsys.readouterr().out
    assert _dump(import_env.db) == before
    assert not (import_env.replays / "r-new").exists()
    assert not list(import_env.db.parent.glob("*.bak-swarm-*"))
    assert "add  r-new" in out and "skip r-dup" in out and "never overwritten" in out
    assert "r-base" not in out                                       # baseline rows are not results
    assert "calibration_2026tst_main.json (differs from local)" in out and "zones_2026tst.json (new on pod)" in out
    assert "event_anchors row(s) NOT imported" in out


def test_import_apply_merges_rows_rewrites_paths_and_backs_up(import_env, capsys):
    before = _dump(import_env.db)
    assert _import(import_env, True, active_job=lambda: None) == 0
    baks = list(import_env.db.parent.glob("frc_scorer.db.bak-swarm-imp-*"))
    assert len(baks) == 1 and _dump(baks[0]) == before
    con = sqlite3.connect(import_env.db)
    con.row_factory = sqlite3.Row
    r = dict(con.execute("SELECT * FROM replays WHERE id = 'r-new'").fetchone())
    rdir = import_env.replays / "r-new"
    assert r["meta_path"] == str(rdir / "meta.json") and r["video_path"] == str(rdir / "hud.mp4")
    assert r["thumb_path"] == str(rdir / "thumb.jpg")
    assert r["source_video_path"] == str(import_env.video.resolve())
    assert r["render_status"] == "queued" and r["render_error"] is None   # no HUD pulled -> local render
    assert (rdir / "meta.json").exists() and not list(import_env.replays.glob(".swarm-import-*"))
    ident = con.execute("SELECT team_number, label_share FROM match_robot_identity WHERE job_id = 'r-new'").fetchall()
    assert [tuple(x) for x in ident] == [("254", 0.9)]
    crops = con.execute("SELECT id, job_id FROM match_robot_crop ORDER BY id").fetchall()
    assert [tuple(x) for x in crops] == [(1, "zzz"), (2, "r-new")]       # fresh autoincrement ids
    for t in ("match_robot_identity_segment", "match_robot_binding", "appearance_gallery"):
        assert con.execute(f"SELECT COUNT(*) FROM {t} WHERE job_id = 'r-new'").fetchone()[0] == 1, t
    # never overwrite: the local r-dup row and identity are untouched
    assert con.execute("SELECT team_number FROM match_robot_identity WHERE job_id = 'r-dup'").fetchone()[0] == "111"
    assert con.execute("SELECT COUNT(*) FROM event_anchors").fetchone()[0] == 0
    con.close()
    # re-running is a no-op (everything already there)
    capsys.readouterr()
    assert _import(import_env, True, active_job=lambda: None) == 0
    assert "0 replay(s) to add" in capsys.readouterr().out


def test_import_refuses_while_local_job_runs_and_rolls_back_on_error(import_env, monkeypatch):
    before = _dump(import_env.db)
    with pytest.raises(SystemExit, match="running job"):
        _import(import_env, True, active_job=lambda: "job-xyz")
    assert _dump(import_env.db) == before
    real = swarm.os.replace

    def boom(a, b):
        if ".swarm-import-" in str(a):
            raise OSError("disk full")
        return real(a, b)
    monkeypatch.setattr(swarm.os, "replace", boom)
    with pytest.raises(OSError):
        _import(import_env, True, active_job=lambda: None)
    assert _dump(import_env.db) == before
    assert not (import_env.replays / "r-new").exists() and not list(import_env.replays.glob(".swarm-import-*"))


def test_import_render_posts_to_local_app(import_env):
    posted = []
    assert _import(import_env, True, active_job=lambda: None, render=True,
                   post=lambda rid: posted.append(rid) or "queued") == 0
    assert posted == ["r-new"]


# ----------------------------------------------------------------- stop / terminate
def test_terminate_refuses_unpulled_members(monkeypatch, tmp_path, capsys):
    sw = swarm.Swarm.load("t7", create=True)
    sw.members.extend([{"pod_id": "a", "status": "ready"}, {"pod_id": "b", "status": "ready"},
                   {"pod_id": "c", "status": "ready"}])
    sw.data["run"] = swarm.new_run(_plan(KEYS[:3]), ["a", "b", "c"])
    for p, mk in zip("abc", KEYS):
        assert swarm.next_item(sw.data["run"], p) == mk
    swarm.finish_item(sw.data["run"], KEYS[0], "a", True, job_id="ja")
    swarm.finish_item(sw.data["run"], KEYS[1], "b", True, job_id="jb")    # c still running
    d = cli.PULL_ROOT / "a" / "web" / "replays" / "ja"
    d.mkdir(parents=True)
    (d / "meta.json").write_text("{}")
    sw.member("a")["pulled_at"] = swarm._now() + 5
    sw.save()
    assert swarm.pull_covers(sw, sw.member("a"))[0]
    assert not swarm.pull_covers(sw, sw.member("b"))[0]
    assert swarm.pull_covers(sw, sw.member("c")) == (False, f"job running ({KEYS[2]})")
    prompts, deleted = [], []
    monkeypatch.setattr(cli, "confirm", lambda msg, word: prompts.append(word) or True)
    rc = swarm.cmd_terminate(argparse.Namespace(name="t7", force=False),
                             request=lambda method, url, body=None: deleted.append((method, url)))
    assert rc == 1 and prompts == ["terminate swarm t7"]
    assert deleted == [("DELETE", f"{cli.REST}/pods/a")]
    out = capsys.readouterr().out
    assert "REFUSED b" in out and "REFUSED c" in out
    assert swarm.Swarm.load("t7").member("a")["status"] == "terminated"
    # --force takes the rest, still with ONE typed confirmation
    prompts.clear()
    rc = swarm.cmd_terminate(argparse.Namespace(name="t7", force=True),
                             request=lambda method, url, body=None: deleted.append((method, url)))
    assert rc == 0 and prompts == ["terminate swarm t7"] and len(deleted) == 3


def test_stop_one_confirmation_for_all(monkeypatch):
    sw = swarm.Swarm.load("t8", create=True)
    sw.members.extend([{"pod_id": "a", "status": "ready"}, {"pod_id": "b", "status": "setup_failed"},
                   {"pod_id": "c", "status": "terminated"}])
    sw.save()
    words, calls = [], []
    monkeypatch.setattr(cli, "confirm", lambda msg, word: words.append(word) or False)
    assert swarm.cmd_stop(argparse.Namespace(name="t8"), request=lambda *a, **k: calls.append(a)) == 1
    assert words == ["stop swarm t8"] and calls == []
    monkeypatch.setattr(cli, "confirm", lambda msg, word: True)
    assert swarm.cmd_stop(argparse.Namespace(name="t8"), request=lambda *a, **k: calls.append(a)) == 0
    assert [c[1].rsplit("/", 2)[-2] for c in calls] == ["a", "b"]


# ----------------------------------------------------------------- pod_job
def test_pod_job_keeps_waiting_when_websocket_closes(monkeypatch, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("pod_job_t", RUNPOD / "pod_job.py")
    pj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pj)

    def http_json(url, body=None, timeout=300):
        return {"job_id": None} if url.endswith("/api/job/active") else {"event_key": "2026tst"}
    monkeypatch.setattr(pj, "http_json", http_json)
    monkeypatch.setattr(pj.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout='{"job_id": "J1"}',
                                                                                     stderr=""))

    class NoSampler:
        peak, n = 0, 0

        def start(self):
            pass

        def stop(self):
            pass
    monkeypatch.setattr(pj, "VramSampler", NoSampler)

    async def drive(host, job_id, rec):
        return "closed"
    monkeypatch.setattr(pj, "drive", drive)
    checks = {"n": 0}

    def final_ready(job_id, want):
        checks["n"] += 1
        return checks["n"] >= 4          # the job finishes a while after the ws dropped
    monkeypatch.setattr(pj, "final_ready", final_ready)
    monkeypatch.setattr(pj, "replay_row", lambda job_id: {"id": job_id, "timings": "{}"})
    monkeypatch.setattr(pj.time, "sleep", lambda s: None)
    out = tmp_path / "j.json"
    monkeypatch.setattr(sys, "argv", ["pod_job.py", "--video", "v.mp4", "--match-key", "2026tst_qm2",
                                      "--event", "2026tst", "--out", str(out), "--no-profile",
                                      "--final-timeout", "60"])
    assert pj.main() == 0
    res = json.loads(out.read_text())
    assert res["status"] == "done" and res["ws_status"] == "closed" and res["wall_job_s"] is None
    assert res["wall_final_s"] is not None and checks["n"] >= 4
    # never final -> an error that says why
    monkeypatch.setattr(pj, "final_ready", lambda job_id, want: False)
    monkeypatch.setattr(sys, "argv", ["pod_job.py", "--video", "v.mp4", "--match-key", "m", "--event", "e",
                                      "--out", str(out), "--no-profile", "--final-timeout", "0"])
    assert pj.main() == 1
    res = json.loads(out.read_text())
    assert res["status"] == "error" and "websocket closed" in res["error"]
    # a real job error from the websocket still fails immediately
    async def drive_err(host, job_id, rec):
        return "error"
    monkeypatch.setattr(pj, "drive", drive_err)
    assert pj.main() == 1 and json.loads(out.read_text())["status"] == "error"


# ----------------------------------------------------------------- single-pod commands unchanged
def test_single_pod_helpers_keep_their_commands(monkeypatch):
    sent = []
    monkeypatch.setattr(cli, "ssh_run", lambda st, remote, **k: sent.append(remote) or types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli, "ssh_out", lambda st, remote, **k: "BOOTSTRAP_EXIT=0\n")
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    cli.run_bootstrap({"pod_id": "p"}, poll_s=0, out=lambda m: None)
    assert sent[0] == ("mkdir -p /workspace/logs && : > /workspace/logs/bootstrap.log && setsid -f nohup bash -c "
                       "'bash /workspace/frc/deploy/runpod/bootstrap.sh; echo BOOTSTRAP_EXIT=$?' "
                       ">> /workspace/logs/bootstrap.log 2>&1 < /dev/null")
    sent.clear()
    cli.run_bootstrap({"pod_id": "p"}, poll_s=0, boot_args="--no-vllm", out=lambda m: None)
    assert "bootstrap.sh --no-vllm; echo" in sent[0]
    # the tune launch string is byte-identical to the pre-refactor one
    cfg = {"sam3_track_fps": 8}
    old = (f"mkdir -p /workspace/tune/jobs && cd /workspace/frc && setsid -f nohup .venv/bin/python "
           f"deploy/runpod/pod_job.py --video /workspace/tune/2026cur_qm8.mp4 --match-key 2026cur_qm8 "
           f"--event 2026cur --label R2 --config-json '{json.dumps(cfg)}' "
           f"--host 127.0.0.1:8004 --no-profile "
           f"--out /workspace/tune/jobs/R2.json > /workspace/tune/jobs/R2.log 2>&1 < /dev/null; echo started")
    assert cli.pod_job_command(video="/workspace/tune/2026cur_qm8.mp4", match_key="2026cur_qm8", event="2026cur",
                               label="R2", config=cfg, out="/workspace/tune/jobs/R2.json",
                               log="/workspace/tune/jobs/R2.log", profile=False) == old


def test_cmd_pull_delegates_to_pull_pod(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "load_state", lambda: {"pod_id": "p1"})
    monkeypatch.setattr(cli, "refresh_endpoint", lambda st, save=None: seen.append(("refresh", save)) or {})
    monkeypatch.setattr(cli, "pull_pod", lambda st, with_video=False, out=print: seen.append(("pull", with_video)))
    assert cli.main(["pull", "--with-video"]) == 0
    assert seen == [("refresh", None), ("pull", True)]


def test_bootstrap_has_no_vllm_flag():
    text = (RUNPOD / "bootstrap.sh").read_text()
    assert "--no-vllm) VLLM=0" in text


def test_swarm_dir_is_gitignored():
    assert "deploy/runpod/.swarm/" in (REPO / ".gitignore").read_text().splitlines()


def test_import_path_is_one_module():
    # `python runpod_cli.py ...` runs as __main__; swarm must reuse that module
    assert swarm.cli is sys.modules["runpod_cli"]
    assert asyncio  # (imported for pod_job's asyncio.run in the closed-ws test)
