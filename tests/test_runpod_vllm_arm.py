"""tune arm Q plumbing: vLLM pre-start/stop around reader arms (no pod)."""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNPOD = REPO / "deploy" / "runpod"
if str(RUNPOD) not in sys.path:
    sys.path.insert(0, str(RUNPOD))

import profiles  # noqa: E402
import runpod_cli as cli  # noqa: E402

TIER = profiles.tier_for_vram(97887)


def _arms_on_r1():
    base = profiles.rung_by_name(TIER, "R1")
    return base, {a.tag: profiles.arm_rung(base, a) for a in profiles.ARMS}


def test_only_reader_arms_need_vllm():
    _, arms = _arms_on_r1()
    assert cli.arm_vllm_model(arms["Q"]) == "qwen3.5-4b"
    assert cli.arm_vllm_model(arms["M"]) is None
    assert cli.arm_vllm_model(arms["P"]) is None


def test_vllm_ctl_remote_parses_last_json_line(monkeypatch):
    seen = {}

    def fake_ssh_out(st, remote, timeout=120, check=True):
        seen["remote"] = remote
        return 'noise\n{"ok": true, "seconds": 98.2, "model": "Qwen/Qwen3.5-4B"}\n'

    monkeypatch.setattr(cli, "ssh_out", fake_ssh_out)
    out = cli.vllm_ctl_remote({}, "start qwen3.5-4b")
    assert out == {"ok": True, "seconds": 98.2, "model": "Qwen/Qwen3.5-4B"}
    assert "pod_env.sh" in seen["remote"] and "profile.env" in seen["remote"]
    assert seen["remote"].endswith("vllm_ctl.py start qwen3.5-4b")


def test_vllm_ctl_remote_timeout_and_garbage(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired("ssh", 5)
    monkeypatch.setattr(cli, "ssh_out", boom)
    assert cli.vllm_ctl_remote({}, "start x", timeout=5)["ok"] is False
    monkeypatch.setattr(cli, "ssh_out", lambda *a, **k: "Traceback ...")
    assert cli.vllm_ctl_remote({}, "stop")["ok"] is False


def _run(monkeypatch, prestart_ok=True):
    calls = []
    monkeypatch.setattr(cli, "save_tune", lambda res: None)
    monkeypatch.setattr(cli, "print_tune_table", lambda *a, **k: None)
    monkeypatch.setattr(cli, "score_locally", lambda *a, **k: {})

    def fake_ctl(st, cmd, timeout=1500):
        calls.append(("ctl", cmd))
        if cmd.startswith("start"):
            return {"ok": prestart_ok, "seconds": 90.0, **({} if prestart_ok else {"error": "no venv"})}
        return {"ok": True}

    def fake_run(st, rung, video, profile=True):
        calls.append(("job", rung.name))
        return {"status": "done", "job_id": "abcdef0123", "wall_job_s": 400}

    monkeypatch.setattr(cli, "vllm_ctl_remote", fake_ctl)
    monkeypatch.setattr(cli, "run_rung_remote", fake_run)
    rungs = profiles.ladder(TIER)
    res = {"rungs": {"R0": {"status": "done"}, "R1": {"status": "done"}}}
    args = types.SimpleNamespace(redo=False, no_profile=True, score_shift=0.475,
                                 budget_min=8.0, budget_on="final")
    cli.run_arms({}, res, rungs, "R1", "/v.mp4", Path("/v.mp4"), args)
    return calls, res


def test_reader_arm_prestarts_and_stops_vllm(monkeypatch):
    calls, res = _run(monkeypatch)
    i = calls.index(("job", "R1Q"))
    assert calls[i - 1] == ("ctl", "start qwen3.5-4b")
    assert calls[i + 1] == ("ctl", "stop")
    assert ("job", "R1M") in calls and ("job", "R1P") in calls
    # non-reader arms never touch vLLM
    assert sum(1 for c in calls if c[0] == "ctl") == 2
    assert res["rungs"]["R1Q"]["vllm_prestart"]["ok"] is True


def test_failed_prestart_records_error_and_skips_job(monkeypatch):
    calls, res = _run(monkeypatch, prestart_ok=False)
    assert ("job", "R1Q") not in calls
    row = res["rungs"]["R1Q"]
    assert row["status"] == "error" and "vLLM pre-start failed" in row["error"]
    assert res["rungs"]["R1M"]["status"] == "done"


def test_vllm_ctl_unknown_model_reports_json(capsys):
    sys.path.insert(0, str(REPO))
    import importlib.util
    spec = importlib.util.spec_from_file_location("vllm_ctl", RUNPOD / "vllm_ctl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main(["start", "no-such-model", "--timeout", "1"])
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert rc == 1 and '"ok": false' in out
