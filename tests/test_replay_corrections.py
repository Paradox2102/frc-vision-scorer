"""Recorded slot corrections (HUD post-render spec section 2): POST /correct
appends {slot, old, new} to replay_slot_corrections, and DELETE removes them."""
import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import web.database as database
from web import hud_labels
from web.hud_draw import slot_label


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "t.db")
    asyncio.run(database.init_db())
    from web.app import app
    return TestClient(app), tmp_path


def _seed(tmp_path, rid="job-1", summary=None, frames=None, slot=0, team="111",
          identity_id=None):
    d = tmp_path / rid
    d.mkdir(exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"summary": summary or {}, "frames": frames or [],
                                             "events": []}))
    asyncio.run(database.insert_replay({
        "id": rid, "created_at": "2026-11-10T00:00:00", "label": "qm1",
        "event_key": None, "match_key": None, "blue_teams": '["111"]', "red_teams": "[]",
        "video_path": str(d / "hud.mp4"), "meta_path": str(d / "meta.json"),
        "thumb_path": str(d / "thumb.jpg"), "source_video_path": "x.mp4",
        "fps": 8.0, "frame_count": len(frames or []), "status": "complete"}))
    asyncio.run(database.upsert_match_robot_identity({
        "job_id": rid, "identity_id": identity_id if identity_id is not None else slot,
        "slot": slot, "slot_label": slot_label(slot) if slot is not None else None,
        "alliance": "blue", "team_number": team, "confirmed": 1, "source": "ocr"}))
    return d


def _rows(rid="job-1"):
    return [{k: r[k] for k in ("slot", "old_team", "new_team")}
            for r in asyncio.run(database.get_slot_corrections(rid))]


def test_old_comes_from_slot_labels_with_ambiguous_as_none(client):
    c, tmp = client
    _seed(tmp, summary={"slot_labels": [{"slot": 0, "slot_label": "B1",
                                         "team_number": "AMBIGUOUS"}]})
    r = c.post("/api/replays/job-1/correct", json={"slot": 0, "team_number": "999"})
    assert r.status_code == 200, r.text
    assert r.json()["render_correction"] == {"recorded": True, "slot": 0, "old": None,
                                             "new": "999"}
    assert _rows() == [{"slot": 0, "old_team": None, "new_team": "999"}]


def test_old_falls_back_to_the_identity_row_without_slot_labels(client):
    c, tmp = client
    _seed(tmp, summary={})
    c.post("/api/replays/job-1/correct", json={"slot": 0, "team_number": "999"})
    assert _rows() == [{"slot": 0, "old_team": "111", "new_team": "999"}]


def test_a_second_correction_records_the_first_ones_new_team_as_old(client):
    c, tmp = client
    _seed(tmp, summary={"slot_labels": [{"slot": 0, "team_number": "111"}]})
    c.post("/api/replays/job-1/correct", json={"slot": 0, "team_number": "222"})
    c.post("/api/replays/job-1/correct", json={"slot": 0, "team_number": "333"})
    assert _rows() == [{"slot": 0, "old_team": "111", "new_team": "222"},
                       {"slot": 0, "old_team": "222", "new_team": "333"}]


def test_a_correction_with_no_slot_is_not_recorded(client):
    c, tmp = client
    _seed(tmp, slot=None, identity_id=7)
    r = c.post("/api/replays/job-1/correct", json={"identity_id": 7, "team_number": "999"})
    assert r.status_code == 200, r.text
    assert r.json()["render_correction"]["recorded"] is False
    assert _rows() == []


def test_delete_cascades_to_the_correction_rows(client):
    c, tmp = client
    _seed(tmp)
    _seed(tmp, rid="job-2")
    c.post("/api/replays/job-1/correct", json={"slot": 0, "team_number": "999"})
    c.post("/api/replays/job-2/correct", json={"slot": 0, "team_number": "888"})
    assert c.delete("/api/replays/job-1").status_code == 200
    assert _rows("job-1") == [] and len(_rows("job-2")) == 1


def test_recorded_old_is_what_the_renderer_compares_frames_against(client):
    c, tmp = client
    frames = [{"i": i, "frame_num": 100 + i, "robots": [
        {"slot": "B1", "team_number": "111" if i < 3 else "444", "confirmed": False}]}
        for i in range(5)]
    _seed(tmp, summary={"slot_labels": [{"slot": 0, "team_number": "111"}]}, frames=frames,
          team="555")    # the identity row disagrees: slot_labels wins
    c.post("/api/replays/job-1/correct", json={"slot": 0, "team_number": "999"})
    rows = database.get_slot_corrections_sync("job-1")
    corr = hud_labels.effective_corrections(rows)
    assert corr == {0: ("111", "999")}
    assert hud_labels.relabel_counts(frames, corr, slot_label) == {"B1": 3}



def test_a_failed_correction_record_never_blocks_the_correction(client, monkeypatch):
    """A DB error while recording the correction must not stop the identity update."""
    from web import app as app_mod

    async def boom(*a, **k):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(app_mod, "_record_slot_correction", boom)
    c, tmp = client
    _seed(tmp, summary={})
    r = c.post("/api/replays/job-1/correct", json={"slot": 0, "team_number": "999"})
    assert r.status_code == 200, r.text
    assert r.json()["render_correction"] == {"recorded": False,
                                             "reason": "record failed: database is locked"}
    ident = asyncio.run(database.get_match_identities("job-1"))
    assert [row["team_number"] for row in ident] == ["999"]
