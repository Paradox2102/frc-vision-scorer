"""Per-slot label confidence: the label team's share of the slot's team score
+ the runner-up, from the whole-match resolve to the DB, API and offline
checker. Every DB test uses a temp database."""
from __future__ import annotations

import asyncio
import importlib.util
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import web.database as database
from web.field.identity_resolve import label_share, resolve_slot_labels

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Pure share computation
# --------------------------------------------------------------------------- #

def test_share_and_runner_up():
    out = label_share({"3128": 6.2, "6773": 2.1, "870": 1.7}, "3128")
    assert out == {"label_share": 0.62, "runner_up_team": "6773", "runner_up_share": 0.21}


def test_single_team_slot_is_100_percent():
    assert label_share({"3128": 3.0}, "3128") == {
        "label_share": 1.0, "runner_up_team": None, "runner_up_share": None}


def test_no_evidence_is_none():
    none = {"label_share": None, "runner_up_team": None, "runner_up_share": None}
    assert label_share(None, "3128") == none
    assert label_share({}, "3128") == none
    assert label_share({"3128": 0.0}, "3128") == none


def test_label_without_score_has_zero_share():
    out = label_share({"6773": 2.0}, "3128")
    assert out["label_share"] == 0.0 and out["runner_up_team"] == "6773"
    assert out["runner_up_share"] == 1.0


def test_unlabelled_slot_suggests_best_team():
    out = label_share({"6773": 3.0, "870": 1.0}, None)
    assert out == {"label_share": None, "runner_up_team": "6773", "runner_up_share": 0.75}


def test_negative_scores_are_clamped_and_ties_break_by_team():
    out = label_share({"1": 2.0, "3": 1.0, "2": 1.0, "9": -5.0}, "1")
    assert out == {"label_share": 0.5, "runner_up_team": "2", "runner_up_share": 0.25}


def test_resolve_slot_labels_returns_label_conf():
    ilog = [{"frame_num": f, "identity_id": 10 + s, "slot_id": s,
             "alliance": "blue" if s < 3 else "red"} for f in (8, 16) for s in range(6)]
    reads = [
        {"frame": 8, "robot_id": 10, "matched_team": "4786", "vote_weight": 1.0},
        {"frame": 16, "robot_id": 10, "matched_team": "4786", "vote_weight": 1.0},
        {"frame": 16, "robot_id": 10, "matched_team": "6800", "vote_weight": 0.5},
    ]
    meta = {s: {"team_number": None, "confirmed": False,
                "alliance": "blue" if s < 3 else "red"} for s in range(6)}
    res = resolve_slot_labels(reads, ilog, {}, {},
                              lambda s: "blue" if s < 3 else "red",
                              ["4786", "6800", "870"], ["1", "2", "3"], meta)
    assert res["slot_meta"][0]["team_number"] == "4786"
    assert res["label_conf"][0] == {"label_share": 0.8, "runner_up_team": "6800",
                                    "runner_up_share": 0.2}
    assert res["label_conf"][5]["label_share"] is None


# --------------------------------------------------------------------------- #
# Migration (temp DB) + persistence
# --------------------------------------------------------------------------- #

def _cols(path):
    con = sqlite3.connect(path)
    try:
        return {r[1] for r in con.execute("PRAGMA table_info(match_robot_identity)")}
    finally:
        con.close()


def test_migration_adds_columns_to_existing_table_idempotently(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE match_robot_identity (
        job_id TEXT NOT NULL, identity_id INTEGER NOT NULL, slot INTEGER,
        slot_label TEXT, alliance TEXT, team_number TEXT, confirmed INTEGER DEFAULT 0,
        source TEXT, confidence REAL, created_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (job_id, identity_id))""")
    con.execute("INSERT INTO match_robot_identity (job_id, identity_id, slot, slot_label, "
                "alliance, team_number, confirmed, source, confidence) "
                "VALUES ('j', 7, 0, 'B1', 'blue', '3128', 1, 'ocr', 0.8)")
    con.commit()
    con.close()
    monkeypatch.setattr(database, "DB_PATH", path)
    asyncio.run(database.init_db())
    asyncio.run(database.init_db())                 # idempotent
    assert {"label_share", "runner_up_team", "runner_up_share"} <= _cols(path)
    rows = asyncio.run(database.get_match_identities("j"))
    assert rows[0]["team_number"] == "3128" and rows[0]["confidence"] == 0.8
    assert rows[0]["label_share"] is None and rows[0]["runner_up_team"] is None


def test_fresh_db_has_columns_and_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "new.db")
    asyncio.run(database.init_db())
    row = {"job_id": "j", "identity_id": 3, "slot": 0, "slot_label": "B1",
           "alliance": "blue", "team_number": "3128", "confirmed": 1, "source": "ocr",
           "confidence": 0.7, "label_share": 0.62, "runner_up_team": "6773",
           "runner_up_share": 0.21}
    asyncio.run(database.upsert_match_robot_identities([row]))
    got = asyncio.run(database.get_match_identities("j"))[0]
    assert (got["label_share"], got["runner_up_team"], got["runner_up_share"]) == (
        0.62, "6773", 0.21)
    # The single-row writer carries them too.
    asyncio.run(database.upsert_match_robot_identity({**row, "runner_up_team": "870"}))
    assert asyncio.run(database.get_match_identities("j"))[0]["runner_up_team"] == "870"


# --------------------------------------------------------------------------- #
# API: identities list + correction
# --------------------------------------------------------------------------- #

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "t.db")
    asyncio.run(database.init_db())
    from web.app import app
    return TestClient(app)


def _seed(share=0.62, ru="6773", ru_share=0.21, team="3128"):
    asyncio.run(database.insert_replay({
        "id": "job-1", "created_at": "2026-11-05T00:00:00", "label": "Qual 8",
        "event_key": None, "match_key": None, "comp_level": None,
        "blue_score": 1, "red_score": 2,
        "blue_teams": f'["{team}","2","3"]', "red_teams": '["4","5","6"]',
        "video_path": "x/hud.mp4", "meta_path": "x/meta.json",
        "thumb_path": "x/thumb.jpg", "source_video_path": "x.mp4",
        "fps": 8.0, "frame_count": 5, "duration_sec": 0.6,
        "width": 160, "height": 120, "codec": "mp4v", "status": "complete",
    }))
    asyncio.run(database.upsert_match_robot_identity({
        "job_id": "job-1", "identity_id": 11, "slot": 0, "slot_label": "B1",
        "alliance": "blue", "team_number": team, "confirmed": 1, "source": "global_resolve",
        "confidence": 0.9, "label_share": share, "runner_up_team": ru,
        "runner_up_share": ru_share}))


def _slot0(c):
    r = c.get("/api/replays/job-1/identities")
    assert r.status_code == 200
    return [x for x in r.json()["robots"] if x["slot"] == 0][0]


def test_identities_endpoint_exposes_share_and_runner_up(client):
    _seed()
    s0 = _slot0(client)
    assert s0["label_share"] == 0.62 and s0["runner_up_team"] == "6773"
    assert s0["runner_up_share"] == 0.21


def test_correct_to_runner_up_swaps_label_and_runner_up(client):
    _seed()
    r = client.post("/api/replays/job-1/correct", json={"slot": 0, "team_number": "6773"})
    assert r.status_code == 200, r.text
    s0 = _slot0(client)
    assert s0["team_number"] == "6773"
    assert (s0["label_share"], s0["runner_up_team"], s0["runner_up_share"]) == (
        0.21, "3128", 0.62)


def test_correct_to_other_team_clears_share(client):
    _seed()
    client.post("/api/replays/job-1/correct", json={"slot": 0, "team_number": "999"})
    s0 = _slot0(client)
    assert s0["team_number"] == "999" and s0["label_share"] is None
    assert s0["runner_up_team"] == "3128" and s0["runner_up_share"] == 0.62


def test_resave_same_team_keeps_share(client):
    _seed()
    client.post("/api/replays/job-1/correct", json={"slot": 0, "team_number": "3128"})
    s0 = _slot0(client)
    assert s0["label_share"] == 0.62 and s0["runner_up_team"] == "6773"


# --------------------------------------------------------------------------- #
# Offline checker prints share + runner-up
# --------------------------------------------------------------------------- #

def test_offline_check_reports_share_and_runner_up():
    spec = importlib.util.spec_from_file_location(
        "identity_offline_check", ROOT / "scripts" / "identity_offline_check.py")
    ioc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ioc)
    frames = list(range(0, 40, 8))
    binds = [(10 + s, s, 0, None) for s in range(6)]
    truth = {"B1": "4786", "B2": "3181", "B3": "2992",
             "R1": "6800", "R2": "1477", "R3": "4256"}
    reads = [{"frame": 16, "robot_id": 10, "matched_team": "4786", "vote_weight": 1.0},
             {"frame": 24, "robot_id": 10, "matched_team": "3181", "vote_weight": 0.25}]
    res = ioc.run_check(binds, frames, reads, truth)
    b1 = res["slots"]["B1"]
    assert b1["label_share"] == 0.8 and b1["runner_up_team"] == "3181"
    text = ioc.format_report(res)
    assert "share" in text and "runner-up" in text
    assert "80%" in text and "3181 20%" in text
