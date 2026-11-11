"""Render status (HUD post-render spec section 7): the replays columns and
their migration, run() with the lockfile, the global queue's lifecycle, the
startup sweep, POST /render (409 while in flight) and the endpoints."""
import asyncio
import json
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import web.app as app_mod
import web.database as database
from web import hud_render
from tests.test_hud_render import make_replay


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "t.db")
    asyncio.run(database.init_db())
    monkeypatch.setattr(app_mod, "_render_queue", None)
    monkeypatch.setattr(app_mod, "_render_queue_loop", None)
    monkeypatch.setattr(app_mod, "_render_inflight", set())
    return tmp_path / "t.db"


def _row(rid):
    return database.get_replay_sync(rid)


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


_HOLDER = ("import fcntl, os, sys, time\n"
           "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644)\n"
           "fcntl.flock(fd, fcntl.LOCK_EX)\n"
           "os.ftruncate(fd, 0); os.write(fd, str(os.getpid()).encode())\n"
           "print('held', flush=True)\n"
           "time.sleep(120)\n")


def hold_lock_in_subprocess(d):
    """Another process holding ``d/render.lock`` (flock) until killed."""
    p = subprocess.Popen([sys.executable, "-c", _HOLDER, str(d / "render.lock")],
                         stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held"
    return p


def kill(p):
    p.kill()
    p.wait()
    p.stdout.close()


# --- columns ---

def test_migration_adds_the_render_columns(tmp_path, monkeypatch):
    dbfile = tmp_path / "old.db"
    con = sqlite3.connect(dbfile)
    con.execute("CREATE TABLE replays (id TEXT PRIMARY KEY, created_at TEXT, label TEXT, "
                "video_path TEXT, meta_path TEXT, status TEXT)")
    con.execute("INSERT INTO replays (id, label, status) VALUES ('old', 'qm1', 'complete')")
    con.commit()
    con.close()
    monkeypatch.setattr(database, "DB_PATH", dbfile)
    asyncio.run(database.init_db())
    asyncio.run(database.init_db())                        # idempotent
    row = database.get_replay_sync("old")
    assert row["render_status"] is None and row["render_error"] is None
    asyncio.run(database.set_render_status("old", "render_failed", "boom"))
    assert (_row("old")["render_status"], _row("old")["render_error"]) == ("render_failed", "boom")


def test_knob_off_row_is_unchanged(db):
    row = dict(id="j", created_at="2026-11-10T00:00:00", label="qm1", event_key="2026cur",
               match_key="2026cur_qm1", comp_level=None, blue_score=1, red_score=2,
               blue_teams="[1]", red_teams="[2]", video_path="v/hud.mp4", meta_path="v/meta.json",
               source_video_path="s.mp4", thumb_path="v/thumb.jpg", fps=8.0, frame_count=5,
               duration_sec=0.6, width=640, height=480, codec="h264", status="complete",
               processing_sec=3.0, timings="{}")
    asyncio.run(database.insert_replay(row))
    got = _row("j")
    assert {k: got[k] for k in row} == row
    assert got["render_status"] is None and got["render_error"] is None


def test_status_values_and_columns_are_validated(db):
    with pytest.raises(ValueError):
        database.set_render_status_sync("x", "done")
    with pytest.raises(ValueError):
        database.set_render_status_sync("x", "rendered", None, label="nope")


# --- run() and the lockfile ---

def test_run_sets_rendering_then_rendered_with_the_video_columns(db, tmp_path, monkeypatch):
    rid, d, _ = make_replay(tmp_path)
    seen = []
    real = hud_render.render_replay
    monkeypatch.setattr(hud_render, "render_replay",
                        lambda r, p=None: (seen.append((_row(r)["render_status"],
                                                        (d / "render.lock").exists())),
                                           real(r, p))[1])
    code, stats = hud_render.run(rid)
    assert code == 0 and seen == [("rendering", True)]
    row = _row(rid)
    assert row["render_status"] == "rendered" and row["render_error"] is None
    assert row["codec"] == stats["codec"] and (row["width"], row["height"]) == (320, 460)
    assert not (d / "render.lock").exists()


def test_run_records_the_failure_reason(db, tmp_path):
    rid, d, _ = make_replay(tmp_path)
    from pathlib import Path
    Path(_row(rid)["source_video_path"]).unlink()
    code, out = hud_render.run(rid)
    assert code == 1 and out["error"].startswith("source video missing")
    assert _row(rid)["render_status"] == "render_failed"
    assert _row(rid)["render_error"].startswith("source video missing")


def test_a_live_lock_blocks_and_a_stale_one_is_taken_over(db, tmp_path):
    rid, d, _ = make_replay(tmp_path)
    holder = hold_lock_in_subprocess(d)
    try:
        code, out = hud_render.run(rid)
        assert code == 3 and out["error"] == f"locked by pid {holder.pid}"
        assert _row(rid)["render_status"] is None
    finally:
        kill(holder)                                   # the kernel drops its flock
    assert (d / "render.lock").exists()                # the file is left behind ...
    code, _ = hud_render.run(rid)                      # ... and is no obstacle
    assert code == 0 and _row(rid)["render_status"] == "rendered"
    assert not (d / "render.lock").exists()


def test_two_acquirers_one_wins(tmp_path):
    import os
    first = hud_render.acquire_lock(tmp_path)
    assert first is not None
    assert hud_render.acquire_lock(tmp_path) is None
    assert hud_render.lock_holder(tmp_path) == os.getpid()
    hud_render.release_lock(first)
    assert hud_render.lock_holder(tmp_path) is None
    again = hud_render.acquire_lock(tmp_path)
    assert again is not None
    hud_render.release_lock(again)


def test_a_lock_file_left_by_a_dead_process_is_acquirable(tmp_path):
    holder = hold_lock_in_subprocess(tmp_path)
    assert hud_render.lock_holder(tmp_path) == holder.pid
    assert hud_render.acquire_lock(tmp_path) is None
    kill(holder)
    assert hud_render.lock_holder(tmp_path) is None
    lock = hud_render.acquire_lock(tmp_path)
    assert lock is not None
    hud_render.release_lock(lock)


def test_liveness_is_the_flock_not_the_pid(tmp_path):
    import os
    (tmp_path / "render.lock").write_text(str(os.getpid()))   # a live pid, but no flock
    assert hud_render.lock_holder(tmp_path) is None
    lock = hud_render.acquire_lock(tmp_path)
    assert lock is not None
    hud_render.release_lock(lock)
    assert hud_render.lock_holder(tmp_path / "nowhere") is None
    assert not (tmp_path / "nowhere").exists()


def test_run_without_a_meta_path_fails_before_locking(db, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    asyncio.run(database.insert_replay({"id": "nometa", "created_at": "2026-11-10T00:00:00",
                                        "label": "x", "meta_path": None, "status": "complete"}))
    code, out = hud_render.run("nometa")
    assert code == 1 and out["error"] == "meta.json missing"
    assert _row("nometa")["render_status"] == "render_failed"
    assert not (tmp_path / "render.lock").exists()


def test_run_removes_render_leftovers_first(db, tmp_path):
    from pathlib import Path
    rid, d, _ = make_replay(tmp_path)
    (d / "hud.render.mp4").write_bytes(b"half a video")
    (d / "thumb.render.jpg").write_bytes(b"half a thumb")
    Path(_row(rid)["source_video_path"]).unlink()      # fails before encoding
    assert hud_render.run(rid)[0] == 1
    assert not (d / "hud.render.mp4").exists() and not (d / "thumb.render.jpg").exists()


# --- the queue ---

def test_queue_lifecycle_runs_the_subprocess(db, tmp_path):
    rid, d, _ = make_replay(tmp_path)

    async def go():
        assert await app_mod.enqueue_render(rid) is True
        assert _row(rid)["render_status"] == "queued"
        assert await app_mod.enqueue_render(rid) is False             # already in flight
        await app_mod._render_queue.join()

    asyncio.run(go())
    row = _row(rid)
    assert row["render_status"] == "rendered", row["render_error"]
    assert row["codec"] and (d / "hud.mp4").exists() and app_mod._render_inflight == set()


def test_worker_marks_a_crashed_renderer_failed(db, tmp_path, monkeypatch):
    rid, d, _ = make_replay(tmp_path)
    monkeypatch.setattr(app_mod, "_render_command",
                        lambda r: [sys.executable, "-c", "import sys; print('boom'); sys.exit(7)"])

    async def go():
        await app_mod.enqueue_render(rid)
        await app_mod._render_queue.join()

    asyncio.run(go())
    assert _row(rid)["render_status"] == "render_failed"
    assert _row(rid)["render_error"] == "renderer exited 7: boom"


def test_finalize_queues_only_when_pending(db, monkeypatch):
    calls = []

    async def fake(rid):
        calls.append(rid)
        return True

    monkeypatch.setattr(app_mod, "enqueue_render", fake)
    job = app_mod.InferenceJob.__new__(app_mod.InferenceJob)
    job.job_id = "j1"
    job._replay_render_pending = False
    asyncio.run(job._queue_hud_render())
    job._replay_render_pending = True
    asyncio.run(job._queue_hud_render())
    asyncio.run(job._queue_hud_render())
    assert calls == ["j1"] and job._replay_render_pending is False


# --- startup sweep ---

def test_startup_sweep(db, tmp_path, monkeypatch):
    q = _FakeQueue()
    monkeypatch.setattr(app_mod, "_render_queue_get", lambda: q)
    rows = {}
    for rid, status in (("q", "queued"), ("r", "rendering"), ("held", "rendering"),
                        ("done", "rendered")):
        rows[rid] = make_replay(tmp_path, rid=rid, n_meta=2)[1]
        database.set_render_status_sync(rid, status)
    holder = hold_lock_in_subprocess(rows["held"])
    try:
        assert asyncio.run(app_mod._sweep_stale_renders()) == {"requeued": 1, "failed": 1}
    finally:
        kill(holder)
    assert q.items == ["q"] and _row("q")["render_status"] == "queued"   # re-enqueued
    assert (_row("r")["render_status"], _row("r")["render_error"]) == ("render_failed",
                                                                       "server restarted")
    assert _row("held")["render_status"] == "rendering"
    assert _row("done")["render_status"] == "rendered"


def test_knob_on_replay_rows_are_inserted_queued(db, tmp_path, monkeypatch):
    job = app_mod.InferenceJob.__new__(app_mod.InferenceJob)
    job._replay_draw_list = False
    assert job._replay_row_render_fields() == {}                       # knob off: no change
    job._replay_draw_list = True
    assert job._replay_row_render_fields() == {"render_status": "queued"}
    # A server that dies between the insert and the queue leaves a queued
    # row, which the startup sweep re-enqueues.
    rid, d, _ = make_replay(tmp_path)
    row = database.get_replay_sync(rid)
    asyncio.run(database.insert_replay({**row, **job._replay_row_render_fields()}))
    assert _row(rid)["render_status"] == "queued"
    q = _FakeQueue()
    monkeypatch.setattr(app_mod, "_render_queue_get", lambda: q)
    assert asyncio.run(app_mod._sweep_stale_renders()) == {"requeued": 1, "failed": 0}
    assert q.items == [rid]


def test_a_failure_to_queue_marks_the_row_failed(db, tmp_path, monkeypatch):
    rid, d, _ = make_replay(tmp_path)
    database.set_render_status_sync(rid, "queued")

    async def boom(r):
        raise RuntimeError("no loop")
    monkeypatch.setattr(app_mod, "enqueue_render", boom)
    job = app_mod.InferenceJob.__new__(app_mod.InferenceJob)
    job.job_id, job._replay_render_pending = rid, True
    asyncio.run(job._queue_hud_render())
    assert _row(rid)["render_status"] == "render_failed"
    assert _row(rid)["render_error"] == "could not queue the render: no loop"


def test_render_timeout_scales_with_the_frame_count():
    assert app_mod._render_timeout_s({"frame_count": 100}) == 600
    assert app_mod._render_timeout_s({"frame_count": 4000}) == 2000
    assert app_mod._render_timeout_s({"frame_count": None}) == 1800
    assert app_mod._render_timeout_s(None) == 1800
    assert app_mod._render_timeout_s({"frame_count": 10**9}) == 3600       # clamped
    for junk in ("abc", 0, -5, "nan", float("inf"), [], {}):
        assert app_mod._render_timeout_s({"frame_count": junk}) == 1800, junk


def test_kill_render_never_signals_a_reaped_process(monkeypatch):
    import os
    sent = []
    monkeypatch.setattr(os, "killpg", lambda pg, sig: sent.append(("killpg", pg)))
    reaped = SimpleNamespace(pid=12345, returncode=0, kill=lambda: sent.append("kill"))
    app_mod._kill_render(reaped, 12345)
    assert sent == []


def test_kill_render_uses_the_captured_pgid_else_kill(monkeypatch):
    import os
    sent = []
    monkeypatch.setattr(os, "killpg", lambda pg, sig: sent.append(("killpg", pg)))
    live = SimpleNamespace(pid=12345, returncode=None, kill=lambda: sent.append("kill"))
    app_mod._kill_render(live, 777)
    assert sent == [("killpg", 777)]
    sent.clear()
    app_mod._kill_render(live, None)                   # no pgid captured: just the child
    assert sent == ["kill"]


def test_worker_kills_a_render_that_times_out(db, tmp_path, monkeypatch):
    rid, d, _ = make_replay(tmp_path)
    monkeypatch.setattr(app_mod, "_render_command",
                        lambda r: [sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(app_mod, "_render_timeout_s", lambda row: 0.5)

    async def go():
        await app_mod.enqueue_render(rid)
        await asyncio.wait_for(app_mod._render_queue.join(), 20)

    asyncio.run(go())
    assert _row(rid)["render_status"] == "render_failed"
    assert _row(rid)["render_error"] == "timeout after 0.5 s"
    assert app_mod._render_inflight == set()


# --- endpoints ---

class _FakeQueue:
    def __init__(self):
        self.items = []

    def put_nowait(self, x):
        self.items.append(x)


def test_render_endpoint_queues_and_409s_while_in_flight(db, tmp_path, monkeypatch):
    rid, d, _ = make_replay(tmp_path)
    q = _FakeQueue()
    monkeypatch.setattr(app_mod, "_render_queue_get", lambda: q)
    c = TestClient(app_mod.app)
    r = c.post(f"/api/replays/{rid}/render")
    assert r.status_code == 200 and r.json() == {"status": "queued", "id": rid}
    assert q.items == [rid] and _row(rid)["render_status"] == "queued"
    busy = c.post(f"/api/replays/{rid}/render")
    assert busy.status_code == 409
    assert busy.json()["error"] == "a render is running — re-render again when it finishes"
    assert c.post("/api/replays/nope/render").status_code == 404


def test_render_endpoint_409s_while_the_script_holds_the_lock(db, tmp_path, monkeypatch):
    import os
    rid, d, _ = make_replay(tmp_path)
    monkeypatch.setattr(app_mod, "_render_queue_get", lambda: _FakeQueue())
    database.set_render_status_sync(rid, "rendering")
    holder = hold_lock_in_subprocess(d)
    c = TestClient(app_mod.app)
    try:
        assert c.post(f"/api/replays/{rid}/render").status_code == 409
    finally:
        kill(holder)                                                    # script died
    assert c.post(f"/api/replays/{rid}/render").status_code == 200


def test_list_and_meta_endpoints_report_the_render(db, tmp_path):
    rid, d, _ = make_replay(tmp_path)
    hud_render.run(rid)
    c = TestClient(app_mod.app)
    item = next(x for x in c.get("/api/replays").json() if x["id"] == rid)
    assert item["render_status"] == "rendered" and item["render_error"] is None
    info = c.get(f"/api/replays/{rid}").json()["replay_info"]
    assert info["render_status"] == "rendered" and info["render_stats"]["frames"] == 12
    assert c.get(f"/api/replays/{rid}/video").status_code == 200
