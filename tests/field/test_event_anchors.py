"""Event anchor store (web/field/event_anchors.py): the sync path on a tmp
sqlite DB and a tmp tile root, and the async path through web/database.py."""
import asyncio
import sqlite3

import pytest

from web.field import event_anchors as ea


def _a(team="3128", kind="digits", p=0.9, frame_num=420, colour="blue", tile=b"\xff\xd8jpeg"):
    return ea.Anchor(team=team, kind=kind, p=p, frame_num=frame_num, alliance_color=colour,
                     tile=tile)


@pytest.fixture()
def store(tmp_path):
    return {"db_path": tmp_path / "t.db", "root": tmp_path / "tiles"}


def _files(store, ns="live", ev="2026cur"):
    return sorted((store["root"] / ns / ev).glob("*.jpg"))


def test_schedule_rank_orders_quals_then_playoffs():
    assert ea.schedule_rank("2026cur_qm8") == 8
    assert ea.schedule_rank("2026cur_qm118") == 118
    assert (ea.schedule_rank("2026cur_qm124") < ea.schedule_rank("2026cur_sf1m1")
            < ea.schedule_rank("2026cur_sf13m1") < ea.schedule_rank("2026cur_f1m2"))
    assert ea.schedule_rank("garbage") is None and ea.schedule_rank(None) is None


def test_round_trip_keeps_metadata_and_tile_bytes(store):
    n = ea.replace("eval-a", "2026cur", "2026cur_qm3",
                   [_a(), _a(team="6773", kind="elimination", p=0.86, frame_num=500, tile=b"T2")],
                   replay_id="r3", **store)
    assert n == 2
    got = ea.fetch("eval-a", "2026cur", ["3128", "6773", "9066"], "2026cur_qm8", **store)
    a = got["3128"][0]
    assert (a.team, a.kind, a.p, a.frame_num, a.alliance_color, a.tile) == (
        "3128", "digits", 0.9, 420, "blue", b"\xff\xd8jpeg")
    assert (a.match_key, a.match_number, a.replay_id) == ("2026cur_qm3", 3, "r3")
    assert got["6773"][0].kind == "elimination" and got["6773"][0].tile == b"T2"
    assert "9066" not in got
    assert (store["root"] / "eval-a" / "2026cur" / f"{a.id}.jpg").read_bytes() == a.tile


def test_fetch_excludes_the_asking_match_and_unasked_teams(store):
    ea.replace("live", "2026cur", "2026cur_qm8", [_a()], **store)
    assert ea.fetch("live", "2026cur", ["3128"], "2026cur_qm8", **store) == {}
    assert ea.fetch("live", "2026cur", ["6773"], "2026cur_qm11", **store) == {}
    assert ea.fetch("live", "2026cur", [], "2026cur_qm11", **store) == {}
    assert len(ea.fetch("live", "2026cur", [3128], "2026cur_qm11", **store)["3128"]) == 1


def test_rerun_replaces_the_match_rows_and_deletes_old_files(store):
    ea.replace("live", "2026cur", "2026cur_qm3", [_a(frame_num=1), _a(frame_num=2)], **store)
    old = _files(store)
    assert len(old) == 2
    ea.replace("live", "2026cur", "2026cur_qm3", [_a(frame_num=3)], **store)
    got = ea.fetch("live", "2026cur", ["3128"], "2026cur_qm99", **store)["3128"]
    assert [a.frame_num for a in got] == [3]
    assert len(_files(store)) == 1 and not any(p.exists() for p in old)


def test_namespaces_are_isolated(store):
    ea.replace("live", "2026cur", "2026cur_qm3", [_a(frame_num=1)], **store)
    ea.replace("eval-inc", "2026cur", "2026cur_qm3", [_a(frame_num=2)], **store)
    ea.replace("eval-inc", "2026cur", "2026cur_qm3", [], **store)      # re-run with nothing
    assert [a.frame_num for a in ea.fetch("live", "2026cur", ["3128"], "x", **store)["3128"]] == [1]
    assert ea.fetch("eval-inc", "2026cur", ["3128"], "x", **store) == {}


def test_max_match_number_cutoff_is_inclusive(store):
    for n in (3, 7, 12):
        ea.replace("live", "2026cur", f"2026cur_qm{n}", [_a(frame_num=n)], **store)
    got = ea.fetch("live", "2026cur", ["3128"], "2026cur_qm20", max_match_number=7, **store)
    assert sorted(a.match_number for a in got["3128"]) == [3, 7]
    assert ea.fetch("live", "2026cur", ["3128"], "2026cur_qm20", 2, **store) == {}


def test_a_missing_tile_file_drops_only_that_anchor(store):
    ea.replace("live", "2026cur", "2026cur_qm3", [_a(frame_num=1), _a(frame_num=2)], **store)
    min(_files(store), key=lambda p: int(p.stem)).unlink()
    got = ea.fetch("live", "2026cur", ["3128"], "x", **store)["3128"]
    assert [a.frame_num for a in got] == [2]


def test_a_failed_write_leaves_the_old_rows_and_no_new_files(store, monkeypatch):
    ea.replace("live", "2026cur", "2026cur_qm3", [_a(frame_num=1)], **store)
    real, calls = ea.write_tile, {"n": 0}

    def flaky(root, rel, data):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        real(root, rel, data)

    monkeypatch.setattr(ea, "write_tile", flaky)
    with pytest.raises(OSError):
        ea.replace("live", "2026cur", "2026cur_qm3", [_a(frame_num=5), _a(frame_num=6)], **store)
    got = ea.fetch("live", "2026cur", ["3128"], "x", **store)["3128"]
    assert [a.frame_num for a in got] == [1]
    assert len(_files(store)) == 1


def test_bad_names_and_kinds_are_refused_before_anything_is_written(store):
    for ns in ("../x", "", "a/b"):
        with pytest.raises(ValueError):
            ea.replace(ns, "2026cur", "2026cur_qm3", [_a()], **store)
    with pytest.raises(ValueError):
        ea.replace("live", "2026cur", "2026cur_qm3", [_a(kind="vibes")], **store)
    assert not store["root"].exists() and not store["db_path"].exists()


def test_fetch_without_a_db_or_a_table_is_empty(store):
    assert ea.fetch("live", "2026cur", ["3128"], "x", **store) == {}
    assert not store["db_path"].exists()                     # read-only: nothing created
    sqlite3.connect(str(store["db_path"])).close()
    assert ea.fetch("live", "2026cur", ["3128"], "x", **store) == {}


def test_clear_removes_one_namespace_of_one_event(store):
    ea.replace("eval-full", "2026cur", "2026cur_qm3", [_a()], **store)
    ea.replace("eval-full", "2026hop", "2026hop_qm3", [_a()], **store)
    assert ea.clear("eval-full", "2026cur", **store) == 1
    assert ea.fetch("eval-full", "2026cur", ["3128"], "x", **store) == {}
    assert ea.fetch("eval-full", "2026hop", ["3128"], "x", **store)
    assert _files(store, "eval-full", "2026cur") == []
