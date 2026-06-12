import asyncio
import numpy as np
import pytest

import web.database as database


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    asyncio.run(database.init_db())
    return database


def _emb_bytes(*vals):
    return np.array(vals, dtype=np.float32).tobytes()


def test_insert_and_fetch_gallery_exemplars_by_event(db):
    asyncio.run(db.insert_gallery_exemplar({
        "event_key": "2026wamon", "team": "1787", "alliance_color": "blue",
        "embedding": _emb_bytes(1, 0, 0), "legibility": 0.9,
    }))
    asyncio.run(db.insert_gallery_exemplar({
        "event_key": "2026wamon", "team": "254", "alliance_color": "red",
        "embedding": _emb_bytes(0, 1, 0), "legibility": 0.8,
    }))
    asyncio.run(db.insert_gallery_exemplar({
        "event_key": "2026other", "team": "9", "alliance_color": "red",
        "embedding": _emb_bytes(0, 0, 1), "legibility": 0.7,
    }))

    rows = asyncio.run(db.get_gallery_exemplars("2026wamon"))
    assert len(rows) == 2
    teams = sorted(r["team"] for r in rows)
    assert teams == ["1787", "254"]
    # embedding survives round-trip as bytes -> float32
    blue = next(r for r in rows if r["team"] == "1787")
    arr = np.frombuffer(blue["embedding"], dtype=np.float32)
    assert np.allclose(arr, [1, 0, 0])


def test_get_gallery_exemplars_empty_for_unknown_event(db):
    assert asyncio.run(db.get_gallery_exemplars("nope")) == []


def test_get_gallery_exemplars_filters_to_match_teams(db):
    for team in ("573", "9785", "581", "1732", "4698", "3297", "254"):
        asyncio.run(db.insert_gallery_exemplar({
            "event_key": "2026hop", "team": team, "alliance_color": "blue",
            "embedding": _emb_bytes(1, 0, 0), "legibility": 0.9,
        }))
    asyncio.run(db.insert_gallery_exemplar({
        "event_key": "2026other", "team": "573", "alliance_color": "blue",
        "embedding": _emb_bytes(1, 0, 0), "legibility": 0.9,
    }))

    roster = ["573", 9785, "581", "1732", "4698", "3297", "573"]
    rows = asyncio.run(db.get_gallery_exemplars("2026hop", teams=roster))

    assert {row["team"] for row in rows} == {
        "573", "9785", "581", "1732", "4698", "3297",
    }
    assert len(rows) == 6


def test_gallery_exemplar_records_source_job_id(db):
    # The source match (job_id) is persisted so the UI can offer that match's
    # roster as the relabel dropdown.
    asyncio.run(db.insert_gallery_exemplar({
        "event_key": "2026wamon", "team": "1787", "alliance_color": "blue",
        "embedding": _emb_bytes(1, 0, 0), "legibility": 0.9, "job_id": "job-abc",
    }))
    rows = asyncio.run(db.get_gallery_exemplars("2026wamon"))
    assert rows[0]["job_id"] == "job-abc"


def test_gallery_exemplar_job_id_optional(db):
    # Older deposits without a job_id still insert and read back as None.
    asyncio.run(db.insert_gallery_exemplar({
        "event_key": "2026wamon", "team": "254", "alliance_color": "red",
        "embedding": _emb_bytes(0, 1, 0), "legibility": 0.8,
    }))
    rows = asyncio.run(db.get_gallery_exemplars("2026wamon"))
    assert rows[0]["job_id"] is None


def test_previous_gallery_exemplars_excludes_current_future_and_unknown(db):
    event_id = asyncio.run(db.upsert_event(
        "2026wamon", "WAMon", 2026, "Somewhere", "2026-03-01", "2026-03-03"
    ))
    for number in (1, 2, 3):
        asyncio.run(db.upsert_match(
            event_id, f"2026wamon_qm{number}", "qm", number
        ))

    for source_match_key, team in (
        ("2026wamon_qm1", "111"),
        ("2026wamon_qm2", "222"),
        ("2026wamon_qm3", "333"),
        (None, "999"),
    ):
        asyncio.run(db.insert_gallery_exemplar({
            "event_key": "2026wamon",
            "source_match_key": source_match_key,
            "team": team,
            "alliance_color": "blue",
            "embedding": _emb_bytes(1, 0, 0),
            "legibility": 0.9,
        }))

    rows = asyncio.run(db.get_previous_gallery_exemplars(
        "2026wamon", "2026wamon_qm2"
    ))
    assert [row["team"] for row in rows] == ["111"]


def test_previous_gallery_exemplars_verified_only(db):
    # Unverified deposits are candidates awaiting review, not identity
    # evidence: verified_only must exclude them from the match-time gallery.
    event_id = asyncio.run(db.upsert_event(
        "2026wamon", "WAMon", 2026, "Somewhere", "2026-03-01", "2026-03-03"
    ))
    for number in (1, 2):
        asyncio.run(db.upsert_match(
            event_id, f"2026wamon_qm{number}", "qm", number
        ))

    for team, verified in (("111", 1), ("222", 0), ("333", 1)):
        asyncio.run(db.insert_gallery_exemplar({
            "event_key": "2026wamon",
            "source_match_key": "2026wamon_qm1",
            "team": team,
            "alliance_color": "blue",
            "embedding": _emb_bytes(1, 0, 0),
            "legibility": 0.9,
            "verified": verified,
        }))

    rows = asyncio.run(db.get_previous_gallery_exemplars(
        "2026wamon", "2026wamon_qm2", verified_only=True
    ))
    assert sorted(row["team"] for row in rows) == ["111", "333"]
    assert all(row["verified"] for row in rows)

    # Default (verified_only=False) still returns every prior deposit, which
    # is what the review UI and opt-out path rely on.
    all_rows = asyncio.run(db.get_previous_gallery_exemplars(
        "2026wamon", "2026wamon_qm2"
    ))
    assert sorted(row["team"] for row in all_rows) == ["111", "222", "333"]


def test_previous_gallery_exemplars_verified_only_composes_with_team_filter(db):
    event_id = asyncio.run(db.upsert_event(
        "2026wamon", "WAMon", 2026, "Somewhere", "2026-03-01", "2026-03-03"
    ))
    for number in (1, 2):
        asyncio.run(db.upsert_match(
            event_id, f"2026wamon_qm{number}", "qm", number
        ))
    for team, verified in (("111", 1), ("111", 0), ("999", 1)):
        asyncio.run(db.insert_gallery_exemplar({
            "event_key": "2026wamon",
            "source_match_key": "2026wamon_qm1",
            "team": team,
            "alliance_color": "blue",
            "embedding": _emb_bytes(1, 0, 0),
            "legibility": 0.9,
            "verified": verified,
        }))

    rows = asyncio.run(db.get_previous_gallery_exemplars(
        "2026wamon", "2026wamon_qm2", teams=["111"], verified_only=True
    ))
    assert len(rows) == 1
    assert rows[0]["team"] == "111"
    assert rows[0]["verified"]
