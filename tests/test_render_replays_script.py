"""scripts/render_replays.py: selection, sequential renders that record their
own status, the shared lockfile, and the exit code."""
import asyncio
import os
import sqlite3

import pytest

import scripts.render_replays as rr
import web.database as database
from tests.test_hud_render import make_replay


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "t.db")
    asyncio.run(database.init_db())
    return tmp_path / "t.db"


def _set_created(db, rid, when):
    con = sqlite3.connect(db)
    con.execute("UPDATE replays SET created_at = ? WHERE id = ?", (when, rid))
    con.commit()
    con.close()


def _status(rid):
    return database.get_replay_sync(rid)["render_status"]


def test_list_selects_by_since_and_renders_nothing(db, tmp_path, capsys):
    for rid, when in (("a", "2026-11-09T20:00:00"), ("b", "2026-11-10T09:00:00"),
                      ("c", "2026-11-10T10:00:00")):
        make_replay(tmp_path, rid=rid, n_meta=2)
        _set_created(db, rid, when)
    assert rr.main(["--since", "2026-11-10", "--db", str(db), "--list"]) == 0
    out = capsys.readouterr().out
    assert "2 replay(s) selected" in out and "  b  " in out and "  a  " not in out
    assert not (tmp_path / "b" / "hud.mp4").exists() and _status("b") is None


def test_renders_in_order_and_records_status(db, tmp_path, capsys):
    make_replay(tmp_path, rid="a", n_meta=3)
    make_replay(tmp_path, rid="b", n_meta=3)
    assert rr.main(["--replays", "b", "a", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert out.index("  b ") < out.index("  a ") and "done: 2 rendered, 0 skipped, 0 failed" in out
    assert _status("a") == _status("b") == "rendered"


def test_a_replay_locked_by_the_server_is_skipped(db, tmp_path, capsys):
    _, d, _ = make_replay(tmp_path, rid="a", n_meta=2)
    from tests.test_hud_render_status import hold_lock_in_subprocess, kill
    holder = hold_lock_in_subprocess(d)
    try:
        assert rr.main(["--replays", "a", "--db", str(db)]) == 0
    finally:
        kill(holder)
    assert f"skipped (locked by pid {holder.pid}" in capsys.readouterr().out
    assert _status("a") is None and (d / "render.lock").exists()


def test_failures_and_unknown_ids_set_the_exit_code(db, tmp_path, capsys):
    _, d, _ = make_replay(tmp_path, rid="a", n_meta=2)
    (tmp_path / "a_src.mp4").unlink()
    assert rr.main(["--replays", "a", "zzz", "--db", str(db)]) == 1
    out = capsys.readouterr().out
    assert "zzz: not found" in out and "FAILED source video missing" in out
    assert _status("a") == "render_failed"
