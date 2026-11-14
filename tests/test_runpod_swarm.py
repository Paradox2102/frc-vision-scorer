"""deploy/runpod/swarm.py: assignment, plan, state, import, dry run (offline:
no pods, no RunPod API, no ssh, no YouTube)."""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import types
import urllib.error
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

REAL_REMOTE = swarm.remote      # the autouse fixture replaces it with a tripwire
REAL_ACTIVE = swarm.server_active_job


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
    monkeypatch.setattr(swarm, "remote", no_network)
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
    (pull / "web" / "replays" / "r3" / cli.PULL_MARKER).write_text("{}")
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
                (d / cli.PULL_MARKER).write_text("{}")
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


class Clock:
    """Fake time for the poll loop: sleep() advances now()."""

    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += max(s, 1)


class FakePod:
    """The pod side of the swarm transport: a launch marker, the attempt's
    out JSON (scripted per poll), the server's active job, and failures
    injected per helper. Patched in over swarm's ssh helpers."""

    def __init__(self, states=(), launched=False):
        self.states = list(states)          # successive out-file reads after launch
        self.launched = launched
        self.drivers = 0
        self.uploads, self.active = [], [None]
        self.fail = {}                      # helper name -> list of exceptions to raise first
        self.alive = True

    def _maybe_fail(self, name):
        q = self.fail.get(name)
        if q:
            raise q.pop(0)

    def read(self, m, outp):
        self._maybe_fail("poll" if self.launched else "read")
        if not self.launched:
            return False, None
        st = self.states.pop(0) if len(self.states) > 1 else (self.states[0] if self.states else None)
        return True, st

    def upload(self, m, local, remote_path, out, what="video"):
        self._maybe_fail("upload")
        self.uploads.append(remote_path)
        return True

    def active_job(self, m):
        self._maybe_fail("active")
        return self.active.pop(0) if len(self.active) > 1 else self.active[0]

    def launch(self, m, outp, cmd):
        self.cmd = cmd
        first = not self.launched
        if first:                   # the remote mkdir guard: only the first launch starts a driver
            self.launched = True
            self.drivers += 1
        self._maybe_fail("launch")  # ssh may die AFTER the remote side launched
        return "started" if first else "already"

    def driver_alive(self, m, state, outp):
        self._maybe_fail("alive")
        return self.alive

    def install(self, monkeypatch):
        monkeypatch.setattr(swarm, "read_job_state", self.read)
        monkeypatch.setattr(swarm, "upload_video", self.upload)
        monkeypatch.setattr(swarm, "server_active_job", self.active_job)
        monkeypatch.setattr(swarm, "launch_once", self.launch)
        monkeypatch.setattr(swarm, "driver_alive", self.driver_alive)
        return self


def _item_ctx(monkeypatch, tmp_path, name, keys=None, poll_s=20):
    keys = keys or KEYS[:2]
    sw = _ready_swarm(name, ["a"], keys, tmp_path)
    sw.data["run"] = swarm.new_run(sw.data["plan"], ["a"], {"x": 1})
    clock = Clock()
    monkeypatch.setattr(swarm, "_now", clock.now)
    ctx = swarm.RunCtx(sw, argparse.Namespace(poll_s=poll_s, terminate_when_done=False), sleep=clock.sleep)
    mk = swarm.next_item(sw.data["run"], "a")
    return sw, ctx, mk, clock


def _st(mk, status, **kw):
    return {"label": f"{mk}.a1", "status": status, "pid": 5, **kw}


def test_process_item_uploads_launches_prefetches_and_polls(monkeypatch, tmp_path):
    sw, ctx, mk, _ = _item_ctx(monkeypatch, tmp_path, "t5")
    pod = FakePod([_st(mk, "running", progress=40, job_id="J"),
                   _st(mk, "done", job_id="J", wall_final_s=300.0)]).install(monkeypatch)
    ok, state = swarm.process_item(ctx, sw.member("a"), mk)
    assert ok and state["job_id"] == "J" and pod.drivers == 1
    assert pod.uploads == [f"{swarm.POD_VIDEOS}/{mk}.mp4", f"{swarm.POD_VIDEOS}/{KEYS[1]}.mp4"]   # + prefetch
    out, log = swarm.attempt_paths(mk, 1)
    assert out == f"{swarm.POD_JOBS}/{mk}.a1.json"                    # unique per attempt
    assert f"--label {mk}.a1" in pod.cmd and "--no-profile" in pod.cmd and "--final-timeout 2400" in pod.cmd
    assert f"--out {out} > {log}" in pod.cmd and '{"x": 1}' in pod.cmd
    it = sw.data["run"]["items"][mk]
    assert it["progress"] == 100 and it["job_id"] == "J"


def test_process_item_reattaches_without_relaunch(monkeypatch, tmp_path):
    sw, ctx, mk, _ = _item_ctx(monkeypatch, tmp_path, "t6", KEYS[:1])
    pod = FakePod([_st(mk, "running"), _st(mk, "done", job_id="J")], launched=True).install(monkeypatch)
    pod.upload = lambda *a, **k: pytest.fail("no upload on re-attach")
    monkeypatch.setattr(swarm, "upload_video", pod.upload)
    ok, state = swarm.process_item(ctx, sw.member("a"), mk)
    assert ok and state["job_id"] == "J" and pod.drivers == 0
    # launched (marker exists) but the driver hasn't written yet: still no relaunch
    sw2, ctx2, mk2, _ = _item_ctx(monkeypatch, tmp_path, "t6b", KEYS[:1])
    pod2 = FakePod([None, _st(mk2, "done", job_id="J2")], launched=True).install(monkeypatch)
    assert swarm.process_item(ctx2, sw2.member("a"), mk2)[0] and pod2.drivers == 0


# finding 1: a transient ssh failure in the poll loop must not fail a healthy job
def test_poll_survives_ssh_failures_past_the_nothing_window(monkeypatch, tmp_path):
    sw, ctx, mk, clock = _item_ctx(monkeypatch, tmp_path, "f1")
    pod = FakePod([None, None, _st(mk, "running", progress=10), _st(mk, "done", job_id="J")]).install(monkeypatch)
    # 20 transport failures = 400+ s of ssh trouble right after launch
    pod.fail["poll"] = [swarm.Transport("ssh exit 255")] * 20
    monkeypatch.setattr(swarm, "pod_api_status", lambda m, saver=None: "RUNNING")
    ok, state = swarm.process_item(ctx, sw.member("a"), mk)
    assert ok and state["job_id"] == "J" and pod.drivers == 1


def test_poll_missing_file_rule_only_before_first_state(monkeypatch, tmp_path):
    # never wrote anything, driver gone after the window -> fail (with the log tail)
    sw, ctx, mk, clock = _item_ctx(monkeypatch, tmp_path, "f1b")
    pod = FakePod([None]).install(monkeypatch)
    pod.alive = False
    monkeypatch.setattr(swarm, "remote", lambda m, cmd, **k: "Traceback: venv missing")
    ok, state = swarm.process_item(ctx, sw.member("a"), mk)
    assert not ok and "wrote nothing" in state["error"] and "venv missing" in state["error"]
    assert clock.t - 1000 > swarm.NOTHING_AFTER_S
    # never wrote anything but the driver is alive: keep waiting, then it finishes
    sw, ctx, mk, clock = _item_ctx(monkeypatch, tmp_path, "f1c")
    states = [None] * 40 + [_st(mk, "done", job_id="J")]
    pod = FakePod(states).install(monkeypatch)
    ok, _ = swarm.process_item(ctx, sw.member("a"), mk)
    assert ok
    # state seen once, then the file reads empty for 15 min (mid-write / glitch): no 5-min failure
    sw, ctx, mk, clock = _item_ctx(monkeypatch, tmp_path, "f1d")
    pod = FakePod([_st(mk, "running")] + [None] * 45 + [_st(mk, "done", job_id="J")]).install(monkeypatch)
    pod.alive = True
    ok, _ = swarm.process_item(ctx, sw.member("a"), mk)
    assert ok


def test_read_job_state_raises_on_ssh_failure_instead_of_missing(monkeypatch):
    monkeypatch.setattr(swarm, "remote", REAL_REMOTE)
    m = {"pod_id": "p", "ssh_host": "1.2.3.4", "ssh_port": 22}

    def fake_run(rc, out=b"", err=b""):
        return lambda *a, **k: types.SimpleNamespace(returncode=rc, stdout=out, stderr=err)
    monkeypatch.setattr(swarm.subprocess, "run", fake_run(255, err=b"Connection reset"))
    with pytest.raises(swarm.Transport):
        swarm.read_job_state(m, "/workspace/batch/jobs/x.a1.json")

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("ssh", 60)
    monkeypatch.setattr(swarm.subprocess, "run", timeout)
    with pytest.raises(swarm.Transport):
        swarm.read_job_state(m, "/x")
    monkeypatch.setattr(swarm.subprocess, "run", fake_run(0, b"N\n"))
    assert swarm.read_job_state(m, "/x") == (False, None)
    monkeypatch.setattr(swarm.subprocess, "run", fake_run(0, b'L\n{"status": "running"}\n'))
    assert swarm.read_job_state(m, "/x") == (True, {"status": "running"})
    monkeypatch.setattr(swarm.subprocess, "run", fake_run(0, b'L\n{"status": "runn'))   # mid-write
    assert swarm.read_job_state(m, "/x") == (True, None)
    monkeypatch.setattr(swarm.subprocess, "run", fake_run(1, err=b"No space left on device"))
    with pytest.raises(swarm.RemoteError, match="No space left"):
        swarm.upload_video(m, Path(__file__), "/workspace/batch/videos/x.mp4", lambda s: None)
    with pytest.raises(swarm.Transport):                      # no endpoint yet = transport
        REAL_REMOTE({"pod_id": "p"}, "true")


# finding 2: launch is idempotent and job/active failure is "unknown", not "idle"
def test_launch_retry_after_hidden_success_starts_one_driver(monkeypatch, tmp_path):
    sw, ctx, mk, _ = _item_ctx(monkeypatch, tmp_path, "f2")
    pod = FakePod([_st(mk, "done", job_id="J")]).install(monkeypatch)
    pod.fail["launch"] = [swarm.Transport("ssh died after the remote launched")]
    ok, _ = swarm.process_item(ctx, sw.member("a"), mk)
    assert ok and pod.drivers == 1


def test_launch_once_command_is_guarded_and_never_deletes(monkeypatch):
    sent = []
    monkeypatch.setattr(swarm, "remote", lambda m, cmd, **k: sent.append(cmd) or "already\n")
    out = "/workspace/batch/jobs/2026tst_qm1.a1.json"
    assert swarm.launch_once({}, out, "LAUNCH; echo started") == "already"
    cmd = sent[0]
    assert f"if mkdir {out}.launch 2>/dev/null; then LAUNCH; echo started; else echo already; fi" in cmd
    assert "rm -f" not in cmd


def test_busy_guard_treats_ssh_failure_as_unknown(monkeypatch, tmp_path):
    sw, ctx, mk, _ = _item_ctx(monkeypatch, tmp_path, "f2b")
    pod = FakePod([_st(mk, "done", job_id="J")]).install(monkeypatch)
    pod.fail["active"] = [swarm.Transport("timeout"), swarm.Transport("timeout")]
    pod.active = ["OTHER", None]                       # then busy once, then idle
    seen = []
    orig = pod.launch

    def launch(m, outp, cmd):
        seen.append(len(pod.fail["active"]))
        return orig(m, outp, cmd)
    monkeypatch.setattr(swarm, "launch_once", launch)
    ok, _ = swarm.process_item(ctx, sw.member("a"), mk)
    assert ok and seen == [0] and pod.drivers == 1     # launched only after a real "idle" answer
    monkeypatch.setattr(swarm, "remote", lambda m, cmd, **k: "curl: (7) Failed to connect")
    with pytest.raises(swarm.RemoteError):
        REAL_ACTIVE({})
    monkeypatch.setattr(swarm, "remote", lambda m, cmd, **k: '{"job_id": null}')
    assert REAL_ACTIVE({}) is None


# finding 3: a deterministic remote failure fails the item, not retried forever
def test_remote_error_is_bounded_and_fails_the_item(monkeypatch, tmp_path):
    sw, ctx, mk, _ = _item_ctx(monkeypatch, tmp_path, "f3")
    pod = FakePod().install(monkeypatch)
    pod.fail["upload"] = [swarm.RemoteError("exit 1: No space left on device")] * 50
    ok, state = swarm.process_item(ctx, sw.member("a"), mk)
    assert not ok and "No space left" in state["error"]
    assert len(pod.fail["upload"]) == 50 - swarm.REMOTE_TRIES and pod.drivers == 0


# finding 4: API failure is not "lost"; only an API verdict is
def test_pod_api_status_and_lost_only_on_api_verdict(monkeypatch, tmp_path):
    def raise_(e):
        def f(*a, **k):
            raise e
        return f
    monkeypatch.setattr(cli, "refresh_endpoint", raise_(SystemExit("RunPod GET -> HTTP 404: not found")))
    assert swarm.pod_api_status({"pod_id": "p"}) == "GONE"
    monkeypatch.setattr(cli, "refresh_endpoint", raise_(urllib.error.URLError("offline")))
    assert swarm.pod_api_status({"pod_id": "p"}) is None
    monkeypatch.setattr(cli, "refresh_endpoint", raise_(SystemExit("RunPod GET -> HTTP 502: bad gateway")))
    assert swarm.pod_api_status({"pod_id": "p"}) is None
    # laptop offline: ssh AND the API fail for > LOST_AFTER_S -> keeps retrying, no PodLost
    sw, ctx, mk, clock = _item_ctx(monkeypatch, tmp_path, "f4")
    pod = FakePod([_st(mk, "done", job_id="J")]).install(monkeypatch)
    pod.fail["poll"] = [swarm.Transport("no route")] * 100       # ~50 min at 30 s per retry
    monkeypatch.setattr(swarm, "pod_api_status", lambda m, saver=None: None)
    assert swarm.process_item(ctx, sw.member("a"), mk)[0]
    # ... but the API saying the pod is stopped is PodLost
    sw, ctx, mk, clock = _item_ctx(monkeypatch, tmp_path, "f4b")
    pod = FakePod().install(monkeypatch)
    pod.fail["poll"] = [swarm.Transport("no route")] * 100
    monkeypatch.setattr(swarm, "pod_api_status", lambda m, saver=None: "EXITED")
    with pytest.raises(swarm.PodLost, match="EXITED"):
        swarm.process_item(ctx, sw.member("a"), mk)


def test_lost_members_are_stopped_pulled_billed_and_revived(monkeypatch, tmp_path):
    sw = swarm.Swarm.load("f4c", create=True)
    sw.members.extend([{"pod_id": "a", "status": "lost", "cost_per_hr": 1.0, "created_at": swarm._now()},
                       {"pod_id": "b", "status": "ready", "cost_per_hr": 1.0, "created_at": swarm._now()}])
    sw.data["run"] = swarm.new_run(_plan(KEYS[:2]), ["a", "b"])
    sw.save()
    assert "burning $2.00/h" in swarm.run_table(sw)
    calls = []
    monkeypatch.setattr(cli, "confirm", lambda msg, word: True)
    swarm.cmd_stop(argparse.Namespace(name="f4c"), request=lambda *a, **k: calls.append(a[1]))
    assert any("/pods/a/stop" in c for c in calls)
    pulled = []
    monkeypatch.setattr(swarm, "pull_member", lambda sw_, m, wv=False, out=None: pulled.append(m["pod_id"])
                        or {"dest": "d", "new_replays": [], "replays_on_pod": []})
    swarm.cmd_pull(argparse.Namespace(name="f4c", with_video=False))
    assert "a" in pulled
    sw = swarm.Swarm.load("f4c")
    sw.member("a")["status"] = "lost"
    assert swarm.revive_lost(sw, status_fn=lambda m, saver: None) == []          # API unknown: unchanged
    assert swarm.revive_lost(sw, status_fn=lambda m, saver: "RUNNING") == ["a"]
    assert sw.member("a")["status"] == "ready"


# finding 5: a truncated / failed pull is rejected and nothing is marked complete
def _tar_bytes(files: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def _fake_pull_pod(monkeypatch, tmp_path, listing, stream: bytes, rc: int, db_ids=("r1",)):
    db = tmp_path / "poddb.db"
    if not db.exists():
        con = sqlite3.connect(db)
        con.executescript(SCHEMA)
        for rid in db_ids:
            con.execute("INSERT INTO replays (id, status) VALUES (?, 'complete')", (rid,))
        con.commit()
        con.close()
    monkeypatch.setattr(cli, "ssh_run", lambda *a, **k: types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli, "ssh_out", lambda st, cmd, **k: " ".join(listing) + "\n")
    monkeypatch.setattr(cli, "pull_tune_files", lambda st, dest: 0)

    def fake_run(cmd, stdout=None, check=False, **k):
        stdout.write(db.read_bytes())
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    class P:
        def __init__(self, *a, **k):
            self.stdout = io.BytesIO(stream)

        def wait(self):
            return rc

        def kill(self):
            pass
    monkeypatch.setattr(cli.subprocess, "Popen", P)


def test_pull_pod_rejects_truncated_stream_and_marks_only_clean(monkeypatch, tmp_path):
    st = {"pod_id": "pp", "ssh_host": "h", "ssh_port": 1}
    full = _tar_bytes({"web/replays/r1/meta.json": b"{}", "web/replays/r1/thumb.jpg": b"j" * 5000,
                       "web/replays/r2/meta.json": b"{}"})
    _fake_pull_pod(monkeypatch, tmp_path, ["r1", "r2"], full[:2000], 0)         # truncated mid-file
    with pytest.raises(SystemExit, match="FAILED"):
        cli.pull_pod(st, out=lambda s: None)
    rdir = cli.PULL_ROOT / "pp" / "web" / "replays"
    assert not any(cli.replay_pulled(d) for d in rdir.iterdir()) if rdir.exists() else True
    _fake_pull_pod(monkeypatch, tmp_path, ["r1", "r2"], full, 255)               # ssh died after the data
    with pytest.raises(SystemExit, match="exit 255"):
        cli.pull_pod(st, out=lambda s: None)
    assert not cli.replay_pulled(rdir / "r1")
    _fake_pull_pod(monkeypatch, tmp_path, ["r1", "r2"], full, 0)
    res = cli.pull_pod(st, out=lambda s: None)
    # r2 has no DB row yet (job still finalizing): pulled but not marked, retried next time
    assert res["new_replays"] == ["r1"] and res["pending_replays"] == ["r2"]
    assert cli.replay_pulled(rdir / "r1") and not cli.replay_pulled(rdir / "r2")
    marker = json.loads((rdir / "r1" / cli.PULL_MARKER).read_text())
    assert marker["files"] == ["meta.json", "thumb.jpg"]
    # nothing new and no tar stream at all: fine
    _fake_pull_pod(monkeypatch, tmp_path, ["r1"], b"", 0)
    assert cli.pull_pod(st, out=lambda s: None)["new_replays"] == []
    # nothing new BUT replays expected and an empty stream: failure
    _fake_pull_pod(monkeypatch, tmp_path, ["r1", "r2"], b"", 0)
    with pytest.raises(SystemExit):
        cli.pull_pod(st, out=lambda s: None)


def test_pull_covers_requires_the_complete_marker(tmp_path):
    sw = swarm.Swarm.load("f5", create=True)
    sw.members.append({"pod_id": "a", "status": "ready", "pulled_at": swarm._now() + 10})
    sw.data["run"] = swarm.new_run(_plan(KEYS[:1]), ["a"])
    mk = swarm.next_item(sw.data["run"], "a")
    swarm.finish_item(sw.data["run"], mk, "a", True, job_id="ja")
    d = cli.PULL_ROOT / "a" / "web" / "replays" / "ja"
    d.mkdir(parents=True)
    (d / "meta.json").write_text("{}")                           # meta.json alone: truncated pull
    assert not swarm.pull_covers(sw, sw.member("a"))[0]
    (d / cli.PULL_MARKER).write_text("{}")
    assert swarm.pull_covers(sw, sw.member("a"))[0]


# lower-priority fixes
def test_import_copy_failure_leaves_no_tmp_dir(import_env, monkeypatch):
    real = shutil.copytree

    def half_copy(src, dst, **k):
        Path(dst).mkdir(parents=True)
        (Path(dst) / "meta.json").write_text("{}")
        raise OSError("disk full mid-copy")
    monkeypatch.setattr(swarm.shutil, "copytree", half_copy)
    before = _dump(import_env.db)
    with pytest.raises(OSError):
        _import(import_env, True, active_job=lambda: None)
    assert not list(import_env.replays.glob(".swarm-import-*")) and _dump(import_env.db) == before
    monkeypatch.setattr(swarm.shutil, "copytree", real)


def test_run_lock_blocks_a_second_dispatcher(tmp_path):
    sw = swarm.Swarm.load("lk", create=True)
    sw.save()
    with swarm.run_lock(sw):
        with pytest.raises(SystemExit, match="already driving"):
            with swarm.run_lock(swarm.Swarm.load("lk")):
                pass
    with swarm.run_lock(sw):           # released: usable again
        pass


def test_pod_job_command_quotes_paths():
    cmd = cli.pod_job_command(video="/v/a b.mp4", match_key="k", event="e", label="l", config={},
                              out="/w/jobs dir/x;rm -rf.json", log="/w/l og.log", profile=False)
    assert "'/w/jobs dir/x;rm -rf.json'" in cmd and "'/w/l og.log'" in cmd and "mkdir -p '/w/jobs dir'" in cmd


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
        (d / cli.PULL_MARKER).write_text("{}")
        (d / "thumb.jpg").write_bytes(b"jpg")
    _insert(pc, "replays", _replay_row("r-base", "2026tst_qm3"))        # baseline row, no dir pulled
    # a partial (aborted) replay, and one whose pull never finished (no marker)
    _insert(pc, "replays", {**_replay_row("r-part", "2026tst_qm4"), "status": "partial"})
    _insert(pc, "replays", _replay_row("r-nomark", "2026tst_qm5"))
    for rid, marked in (("r-part", True), ("r-nomark", False)):
        d = pod / "web" / "replays" / rid
        d.mkdir(parents=True)
        (d / "meta.json").write_text("{}")
        if marked:
            (d / cli.PULL_MARKER).write_text("{}")
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
    assert "skip r-part" in out and "only complete replays" in out
    assert "skip r-nomark" in out and "not marked complete" in out
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
    (d / cli.PULL_MARKER).write_text("{}")
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
    # `python runpod_cli.py ...` runs as __main__; swarm must reuse that module.
    # Checked in a fresh interpreter: other test files re-load runpod_cli under
    # the same name at collection time, so an in-process identity check is
    # order-dependent.
    import subprocess
    path = str(RUNPOD / "runpod_cli.py")
    # exec the script inside the real __main__ module, exactly like
    # `python runpod_cli.py swarm --help` (runpy.run_path uses a temp module)
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(RUNPOD)!r})\n"
        f"sys.argv = [{path!r}, 'swarm', '--help']\n"
        f"__file__ = {path!r}\n"
        "try:\n"
        f"    exec(compile(open({path!r}).read(), {path!r}, 'exec'), globals())\n"
        "except SystemExit:\n"
        "    pass\n"
        "main = sys.modules['__main__']\n"
        "import swarm\n"
        "assert sys.modules['runpod_cli'] is main, 'second runpod_cli copy'\n"
        "assert swarm.cli is main, 'swarm bound to a second copy'\n"
        "print('ONE_MODULE_OK')\n")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "ONE_MODULE_OK" in res.stdout, res.stdout[-800:] + res.stderr[-1500:]
    assert asyncio  # (imported for pod_job's asyncio.run in the closed-ws test)
