import asyncio
import json
import pytest

import web.database as database


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    asyncio.run(database.init_db())
    return database


def _sample_row(replay_id="job-1"):
    return dict(
        id=replay_id, created_at="2026-06-03T00:00:00", label="Qual 10",
        event_key=None, match_key=None, comp_level=None,
        blue_score=70, red_score=67,
        blue_teams=json.dumps([581, 9785, 573]), red_teams=json.dumps([1732, 4698, 3297]),
        video_path=f"web/replays/{replay_id}/hud.mp4",
        meta_path=f"web/replays/{replay_id}/meta.json",
        source_video_path="web/uploads/x.mp4",
        thumb_path=f"web/replays/{replay_id}/thumb.jpg",
        fps=8.0, frame_count=1583, duration_sec=197.9,
        width=1280, height=1020, codec="avc1", status="complete",
    )


def test_insert_list_get_delete_replay(db):
    asyncio.run(db.insert_replay(_sample_row("job-1")))
    asyncio.run(db.insert_replay(_sample_row("job-2")))

    rows = asyncio.run(db.list_replays())
    assert [r["id"] for r in rows] == ["job-2", "job-1"]  # newest first by created_at desc, id tiebreak

    one = asyncio.run(db.get_replay("job-1"))
    assert one["label"] == "Qual 10"
    assert one["blue_score"] == 70

    asyncio.run(db.delete_replay("job-1"))
    assert asyncio.run(db.get_replay("job-1")) is None
    assert [r["id"] for r in asyncio.run(db.list_replays())] == ["job-2"]


def test_insert_replay_is_idempotent_upsert(db):
    asyncio.run(db.insert_replay(_sample_row("job-1")))
    row = _sample_row("job-1")
    row["status"] = "partial"
    asyncio.run(db.insert_replay(row))  # re-insert same id
    got = asyncio.run(db.get_replay("job-1"))
    assert got["status"] == "partial"
    assert len(asyncio.run(db.list_replays())) == 1
