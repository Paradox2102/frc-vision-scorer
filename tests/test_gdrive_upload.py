"""Google Drive upload (web/gdrive_upload.py + scripts/upload_gdrive.py).
No network and no real rclone: a fake ``_run`` records every rclone call
(and the staged file's content), and the scouting API is stubbed."""
import importlib.util
import json
import re
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from web import gdrive_upload as gd
from web.scouting import api, export

REPO = Path(__file__).resolve().parents[1]
EV = "2026cur"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeRclone:
    """Stands in for gd._run: records (args, timeout, staged content)."""
    def __init__(self, fail_on=(), hook=None):
        self.calls = []
        self.fail_on = tuple(fail_on)
        self.hook = hook
        self.lock = threading.Lock()

    def __call__(self, args, timeout):
        content = None
        if args and args[0] == "copyto":
            p = Path(args[1])
            content = p.read_text() if p.suffix == ".json" else p.read_bytes()
        with self.lock:
            self.calls.append((list(args), timeout, content))
        if self.hook:
            self.hook(args)
        if any(s in " ".join(args) for s in self.fail_on):
            raise gd.GDriveError("rclone copyto exited 1: quota exceeded")
        return ""

    def remotes(self):
        return [a[2] for a, _, _ in self.calls if a[0] == "copyto"]


def _reset():
    with gd._lock:
        gd._pending.clear()
        gd._running.clear()
        gd._in_flight.clear()
        gd._last.clear()
        gd._last_video.clear()
        gd._video_jobs.clear()
        gd._video_state.update(running=False, current=None)
        gd._expected.clear()


def _wait_idle(timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        with gd._lock:
            if not gd._running and not gd._video_state["running"]:
                return
        time.sleep(0.01)
    raise AssertionError("gdrive workers did not finish")


@pytest.fixture
def env(monkeypatch):
    """Configured, with the .env file and rclone lookup out of the picture."""
    monkeypatch.setattr(export, "_load_dotenv", lambda: None)
    monkeypatch.setenv("GDRIVE_FOLDER_ID", "folder123")
    monkeypatch.delenv("GDRIVE_RCLONE_REMOTE", raising=False)
    monkeypatch.setattr(gd.shutil, "which", lambda name: "/usr/bin/rclone")
    _reset()
    yield
    _wait_idle()
    _reset()


@pytest.fixture
def on(env, monkeypatch):
    """Enabled (pytest guard lifted), fake rclone, stubbed scouting API."""
    monkeypatch.setattr(gd, "_under_pytest", lambda: False)
    fake = FakeRclone()
    monkeypatch.setattr(gd, "_run", fake)
    monkeypatch.setattr(api, "_match_sync",
                        lambda ev, mk, team: {"event_key": ev, "match": {"match_key": mk,
                                                                          "xs": [1, 2, 3]}})
    monkeypatch.setattr(api, "_export_sync",
                        lambda ev, paths=True: {"event_key": ev, "paths": paths, "matches": []})
    return fake


# ----------------------------------------------------------------- paths + rclone command
def test_layout_paths():
    assert gd.event_export_path(EV) == "2026cur/2026cur_scouting_export.json"
    assert gd.match_path(EV, "2026cur_qm8") == "2026cur/2026cur_matches/2026cur_qm8.json"
    assert gd.video_path(EV, "2026cur_qm8") == "2026cur/videos/2026cur_qm8.mp4"


@pytest.mark.parametrize("value,expected", [
    (None, "gdrive:"), ("", "gdrive:"), ("gdrive:", "gdrive:"), ("mydrive", "mydrive:"),
    ("mydrive:", "mydrive:"), ("mydrive:scouting", "mydrive:scouting/"),
])
def test_remote_with_or_without_colon(env, monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("GDRIVE_RCLONE_REMOTE", raising=False)
    else:
        monkeypatch.setenv("GDRIVE_RCLONE_REMOTE", value)
    assert gd.config()["remote"] == expected
    assert gd.remote_path("a/b.json") == expected + "a/b.json"


def test_run_builds_rclone_command_with_root_folder_id(env, monkeypatch, tmp_path):
    monkeypatch.setattr(gd, "_under_pytest", lambda: False)
    monkeypatch.setenv("GDRIVE_RCLONE_REMOTE", "mydrive")
    seen = []

    def fake_run(cmd, **kw):
        seen.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(gd.subprocess, "run", fake_run)
    monkeypatch.setattr(api, "_match_sync", lambda ev, mk, team: {"m": mk})
    out = gd.upload_match(EV, "2026cur_qm8")
    (cmd, kw), = seen
    assert cmd[:2] == ["rclone", "copyto"]
    assert cmd[3] == "mydrive:2026cur/2026cur_matches/2026cur_qm8.json"
    assert cmd[-2:] == ["--drive-root-folder-id", "folder123"]
    assert kw["timeout"] == gd.JSON_TIMEOUT_S and kw["capture_output"] and kw["text"]
    assert out["remote"] == "2026cur/2026cur_matches/2026cur_qm8.json" and out["bytes"] > 0
    assert not Path(cmd[2]).exists()             # staged in a temp dir, cleaned up


def test_run_raises_with_stderr_tail_and_on_timeout(env, monkeypatch):
    monkeypatch.setattr(gd, "_under_pytest", lambda: False)
    monkeypatch.setattr(gd.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 3, stdout="", stderr="line1\nline2\nFailed to copy: 403 Forbidden\n"))
    with pytest.raises(gd.GDriveError, match="exited 3: .*403 Forbidden"):
        gd._run(["lsf", "gdrive:"], 10)

    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(gd.subprocess, "run", slow)
    with pytest.raises(gd.GDriveError, match="timed out"):
        gd._run(["copyto", "a", "b"], 5)


def test_real_run_refuses_under_pytest(env, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("subprocess.run must not be called under pytest")
    monkeypatch.setattr(gd.subprocess, "run", boom)
    with pytest.raises(gd.GDriveError, match="pytest"):
        gd._run(["lsf", "gdrive:"], 10)


# ----------------------------------------------------------------- config
def test_not_configured_without_folder_id_or_rclone(env, monkeypatch, capsys):
    monkeypatch.setattr(gd, "_under_pytest", lambda: False)
    assert gd.config()["configured"] is True
    monkeypatch.setenv("GDRIVE_FOLDER_ID", "")
    cfg = gd.config()
    assert cfg["configured"] is False and cfg["folder_id_set"] is False
    monkeypatch.setenv("GDRIVE_FOLDER_ID", "folder123")
    monkeypatch.setattr(gd.shutil, "which", lambda name: None)
    cfg = gd.config()
    assert cfg["configured"] is False and cfg["rclone"] is None
    monkeypatch.setenv("GDRIVE_FOLDER_ID", "")
    assert gd.schedule_match(EV, "2026cur_qm8") is False
    assert "[gdrive] upload skipped: GDRIVE_FOLDER_ID not set / rclone not found" in \
        capsys.readouterr().out


def test_disabled_under_pytest(env, monkeypatch):
    cfg = gd.config()
    assert cfg["configured"] is True and cfg["enabled"] is False
    assert gd._under_pytest() is True                # the real guard sees this run
    assert gd.schedule_match(EV, "2026cur_qm8") is False
    assert gd.schedule_video(EV, "2026cur_qm8", __file__) is False


def test_status_does_not_echo_folder_id(on):
    st = gd.status()
    assert "folder123" not in json.dumps(st)
    assert st["configured"] and st["in_flight"] == {} and st["videos_pending"] == 0


def test_schedule_needs_keys(on):
    assert gd.schedule_match(EV, None) is False
    assert gd.schedule_match(None, "2026cur_qm8") is False
    assert on.calls == []


# ----------------------------------------------------------------- match files + coalescing
def test_match_upload_stages_pretty_match_file_and_compact_export(on):
    assert gd.schedule_match(EV, "2026cur_qm8") is True
    _wait_idle()
    (a1, t1, c1), (a2, t2, c2) = on.calls
    assert a1[2] == "gdrive:2026cur/2026cur_matches/2026cur_qm8.json"
    assert c1 == export.pretty({"event_key": EV, "match": {"match_key": "2026cur_qm8",
                                                            "xs": [1, 2, 3]}})
    assert a2[2] == "gdrive:2026cur/2026cur_scouting_export.json"
    assert c2 == '{"event_key":"2026cur","paths":true,"matches":[]}'
    assert t1 == t2 == gd.JSON_TIMEOUT_S
    last = gd.status()["last"][EV]
    assert last["ok"] is True and last["matches"] == ["2026cur_qm8"]


def test_matches_queued_while_busy_coalesce_into_one_export(on):
    started, release = threading.Event(), threading.Event()

    def hook(args):
        if args[2].endswith("2026cur_qm1.json"):
            started.set()
            assert release.wait(5)
    on.hook = hook
    assert gd.schedule_match(EV, "2026cur_qm1")
    assert started.wait(5)
    assert gd.schedule_match(EV, "2026cur_qm2")
    assert gd.schedule_match(EV, "2026cur_qm3")
    assert gd.schedule_match(EV, "2026cur_qm2")      # duplicate: uploaded once
    st = gd.status()["in_flight"][EV]
    assert st == {"uploading": ["2026cur_qm1"], "pending": ["2026cur_qm2", "2026cur_qm3"]}
    release.set()
    _wait_idle()
    exp = "gdrive:2026cur/2026cur_scouting_export.json"
    m = "gdrive:2026cur/2026cur_matches/{}.json".format
    assert on.remotes() == [m("2026cur_qm1"), exp, m("2026cur_qm2"), m("2026cur_qm3"), exp]
    assert gd.status()["last"][EV]["matches"] == ["2026cur_qm2", "2026cur_qm3"]
    assert gd.status()["in_flight"] == {}


def test_one_failing_match_does_not_block_others_or_export(on, monkeypatch, capsys):
    on.fail_on = ("2026cur_qm2.json",)
    real = api._match_sync

    def match_sync(ev, mk, team):
        if mk == "2026cur_qm3":
            raise RuntimeError("no analysed match")
        return real(ev, mk, team)
    monkeypatch.setattr(api, "_match_sync", match_sync)
    started, release = threading.Event(), threading.Event()
    on.hook = lambda args: (started.set(), release.wait(5)) if "qm0" in args[2] else None
    gd.schedule_match(EV, "2026cur_qm0")
    assert started.wait(5)
    for mk in ("2026cur_qm1", "2026cur_qm2", "2026cur_qm3", "2026cur_qm4"):
        gd.schedule_match(EV, mk)
    release.set()
    _wait_idle()
    last = gd.status()["last"][EV]
    assert last["ok"] is False
    assert last["matches"] == ["2026cur_qm1", "2026cur_qm4"]
    assert set(last["errors"]) == {"2026cur_qm2", "2026cur_qm3"}
    assert last["event_export"]["remote"] == "2026cur/2026cur_scouting_export.json"
    assert on.remotes()[-1] == "gdrive:2026cur/2026cur_scouting_export.json"
    out = capsys.readouterr().out
    assert "[gdrive] 2026cur/2026cur_qm2: match file failed" in out
    assert "[gdrive] 2026cur/2026cur_qm3: match file failed" in out


def test_missing_match_reports_http_detail(on, monkeypatch):
    from fastapi import HTTPException

    def missing(ev, mk, team):
        raise HTTPException(404, f"no analysed match {mk!r} in {ev}")
    monkeypatch.setattr(api, "_match_sync", missing)
    with pytest.raises(gd.GDriveError, match="no analysed match"):
        gd.upload_match(EV, "2026cur_qm99")


# ----------------------------------------------------------------- videos
def test_video_after_render_uploads_when_rendered(on, tmp_path):
    vid = tmp_path / "hud.mp4"
    vid.write_bytes(b"\x00" * 1000)
    gd.expect_video_after_render("job1", EV, "2026cur_qm8")
    assert gd.status()["videos_awaiting_render"] == 1
    assert gd.on_render_done("job1", "rendered", str(vid)) is True
    _wait_idle()
    (args, timeout, content), = on.calls
    assert args == ["copyto", str(vid), "gdrive:2026cur/videos/2026cur_qm8.mp4"]
    assert timeout == gd.VIDEO_TIMEOUT_S
    assert gd.status()["last_video"][EV]["ok"] is True
    assert gd.status()["videos_awaiting_render"] == 0


def test_video_not_uploaded_when_render_failed_or_unexpected(on, tmp_path, capsys):
    vid = tmp_path / "hud.mp4"
    vid.write_bytes(b"x")
    gd.expect_video_after_render("job1", EV, "2026cur_qm8")
    assert gd.on_render_done("job1", "render_failed", str(vid)) is False
    assert gd.on_render_done("job1", "rendered", str(vid)) is False   # already popped
    assert gd.on_render_done("other", "rendered", str(vid)) is False  # never expected
    _wait_idle()
    assert on.calls == []
    assert "video not uploaded (render render_failed)" in capsys.readouterr().out


def test_missing_video_is_skipped(on, tmp_path, capsys):
    assert gd.schedule_video(EV, "2026cur_qm8", str(tmp_path / "nope.mp4")) is False
    assert gd.schedule_video(EV, "2026cur_qm8", None) is False
    assert on.calls == []
    assert "video skipped, no file" in capsys.readouterr().out


def test_videos_upload_one_at_a_time(on, tmp_path, monkeypatch):
    active, peak = [0], [0]
    lock = threading.Lock()

    def slow_run(args, timeout):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.02)
        with lock:
            active[0] -= 1
        on.calls.append((args, timeout, None))
        return ""
    monkeypatch.setattr(gd, "_run", slow_run)
    for i in range(3):
        v = tmp_path / f"v{i}.mp4"
        v.write_bytes(b"x")
        assert gd.schedule_video(EV, f"2026cur_qm{i}", str(v))
    _wait_idle()
    assert peak[0] == 1
    assert [a[2] for a, _, _ in on.calls] == [
        f"gdrive:2026cur/videos/2026cur_qm{i}.mp4" for i in range(3)]


def test_video_failure_is_recorded_not_raised(on, tmp_path):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    on.fail_on = ("videos/",)
    assert gd.schedule_video(EV, "2026cur_qm8", str(v))
    _wait_idle()
    last = gd.status()["last_video"][EV]
    assert last["ok"] is False and "quota" in last["error"]


# ----------------------------------------------------------------- knobs
def test_knobs_registered_off_in_uploads_section():
    from web.job_config import KNOBS, JobConfig
    by = {k.name: k for k in KNOBS}
    for name, env_var in (("gdrive_upload", "GDRIVE_UPLOAD"),
                          ("gdrive_upload_video", "GDRIVE_UPLOAD_VIDEO")):
        k = by[name]
        assert k.type is bool and k.default is False and k.section == "Uploads"
        assert k.env == env_var
    cfg = JobConfig({"gdrive_upload": True, "gdrive_upload_video": True})
    assert cfg.gdrive_upload is True and cfg.gdrive_upload_video is True


# ----------------------------------------------------------------- pretty() move
_NUM_LIST = re.compile(r"\[\s+([-\d.eE+,\s]+?)\s+\]")
_PAIR_LIST = re.compile(r"\[\s+(\[-?\d+, -?\d+\](?:,\s+\[-?\d+, -?\d+\])*)\s+\]")


def _old_pretty(obj):          # scripts/export_scouting.py before the move
    txt = json.dumps(obj, indent=2)
    txt = _NUM_LIST.sub(lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", txt)
    txt = _PAIR_LIST.sub(lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]", txt)
    return txt + "\n"


def test_pretty_moved_unchanged():
    sample = {"t": [0.0, 0.1, -2.5e-3, 12], "pairs": [[0, 5], [-1, 7]],
              "robots": [{"team": 3128, "x": [1.25, 2.5], "name": "a, b"}], "empty": [],
              "mixed": [1, "a"], "nested": {"k": [[1.5, 2.5], [3, 4]]}}
    assert export.pretty(sample) == _old_pretty(sample)
    assert '"t": [0.0, 0.1, -0.0025, 12]' in export.pretty(sample)
    script = _load_script("export_scouting")
    assert script.pretty is export.pretty
    assert script.pretty(sample) == _old_pretty(sample)


# ----------------------------------------------------------------- CLI
def _md(mk, n, rid):
    return SimpleNamespace(match_key=mk, comp_level="qm", set_number=1, match_number=n,
                           replay_id=rid, replay_dir=None)


@pytest.fixture
def cli(monkeypatch):
    mod = _load_script("upload_gdrive")
    monkeypatch.setattr(api, "_load", lambda ev: (
        [_md("2026cur_qm10", 10, "r10"), _md("2026cur_qm2", 2, "r2")],
        [{"match_key": "2026cur_qm5", "reason": "no readable meta.json"}]))
    return mod


def test_cli_dry_run_lists_uploads(env, cli, monkeypatch, capsys):
    def no_rclone(args, timeout):
        raise AssertionError("dry run must not call rclone")
    monkeypatch.setattr(gd, "_run", no_rclone)
    assert cli.main([EV, "--dry-run"]) == 0
    out = capsys.readouterr().out
    lines = [ln.split() for ln in out.splitlines()[1:]]
    assert lines == [["match", "2026cur/2026cur_matches/2026cur_qm2.json"],
                     ["match", "2026cur/2026cur_matches/2026cur_qm10.json"],
                     ["event", "export", "2026cur/2026cur_scouting_export.json"]]
    assert "would upload 3 file(s) to gdrive:" in out


def test_cli_dry_run_match_filter_and_video(env, cli, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_video_for", lambda md: (
        ("/x/hud.mp4", "") if md.match_key == "2026cur_qm10" else (None, "HUD render queued")))
    assert cli.main([EV, "--dry-run", "--video", "--no-event-export",
                     "--match", "2026cur_qm10", "--match", "2026cur_qm2"]) == 0
    out = capsys.readouterr().out
    assert "skip video 2026cur_qm2: HUD render queued" in out
    assert "video        2026cur/videos/2026cur_qm10.mp4" in out
    assert "scouting_export" not in out
    assert cli.main([EV, "--dry-run", "--match", "2026cur_qm5"]) == 1
    assert "no readable meta.json" in capsys.readouterr().err


def test_cli_uploads_and_exit_code(on, cli, capsys):
    assert cli.main([EV, "--match", "2026cur_qm2"]) == 0
    assert on.remotes() == ["gdrive:2026cur/2026cur_matches/2026cur_qm2.json",
                            "gdrive:2026cur/2026cur_scouting_export.json"]
    on.fail_on = ("scouting_export",)
    assert cli.main([EV, "--match", "2026cur_qm2"]) == 1
    assert "1 uploaded, 1 failed" in capsys.readouterr().out


def test_cli_check(on, cli, monkeypatch, capsys):
    monkeypatch.setattr(gd, "_run", lambda args, timeout: "2026cur/\nnotes.txt\n")
    assert cli.main(["--check"]) == 0
    assert "gdrive: (folder set" in capsys.readouterr().out
    monkeypatch.setenv("GDRIVE_FOLDER_ID", "")
    assert cli.main(["--check"]) == 1
    assert "GDRIVE_FOLDER_ID not set" in capsys.readouterr().err


# ----------------------------------------------------------------- app hooks
def _job(**kw):
    base = dict(job_id="job1", event_key=EV, match_key="2026cur_qm8",
                _gdrive_after_finalize=True, _replay_render_pending=False,
                _gdrive_video_path=None,
                config=SimpleNamespace(gdrive_upload=True, gdrive_upload_video=True))
    base.update(kw)
    return SimpleNamespace(**base)


def test_app_hook_routes_video_by_render_mode(on, monkeypatch, tmp_path):
    import web.app as app_mod
    seen = []
    monkeypatch.setattr(gd, "schedule_match", lambda ev, mk: seen.append(("match", ev, mk)) or True)
    monkeypatch.setattr(gd, "schedule_video", lambda ev, mk, p: seen.append(("video", mk, p)))
    monkeypatch.setattr(gd, "expect_video_after_render",
                        lambda rid, ev, mk: seen.append(("expect", rid, mk)))
    hook = app_mod.InferenceJob._schedule_gdrive_upload
    # post-render pending: the video is expected after the render
    job = _job(_replay_render_pending=True)
    hook(job)
    assert seen == [("match", EV, "2026cur_qm8"), ("expect", "job1", "2026cur_qm8")]
    assert job._gdrive_after_finalize is False
    hook(job)                                            # once per job
    assert len(seen) == 2
    # live-encoded video: uploaded straight away
    seen.clear()
    hook(_job(_gdrive_video_path="/r/hud.mp4"))
    assert seen[-1] == ("video", "2026cur_qm8", "/r/hud.mp4")
    # video knob off / upload knob never set
    seen.clear()
    hook(_job(config=SimpleNamespace(gdrive_upload=True, gdrive_upload_video=False)))
    hook(_job(_gdrive_after_finalize=False))
    assert seen == [("match", EV, "2026cur_qm8")]


def test_render_done_hands_video_to_uploader(on, monkeypatch):
    import web.app as app_mod
    seen = []
    monkeypatch.setattr(gd, "on_render_done", lambda *a: seen.append(a))
    app_mod._gdrive_render_done("job1", "rendered", "/r/hud.mp4")
    assert seen == [("job1", "rendered", "/r/hud.mp4")]
    monkeypatch.setattr(gd, "on_render_done", lambda *a: 1 / 0)
    app_mod._gdrive_render_done("job1", "rendered", "/r/hud.mp4")      # never raises


def test_status_endpoint(on):
    from fastapi.testclient import TestClient
    import web.app as app_mod
    r = TestClient(app_mod.app).get("/api/gdrive/status")
    assert r.status_code == 200
    body = r.json()
    assert body["configured"] is True and body["remote"] == "gdrive:"
    assert "folder123" not in r.text
