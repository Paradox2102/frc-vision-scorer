"""Server-side roster fallback: the alliance gate must work even when the
browser never pushes a roster over the websocket (the bug that let blue team
1684's number land on red robot 7421 in 2026cur_qm2)."""
import asyncio

import web.app as app_mod
import web.database as database


def _fake_match(robots):
    async def _get(match_key):
        return {"tba_match_key": match_key, "robots": robots}
    return _get


def test_load_db_roster_splits_by_alliance(monkeypatch):
    monkeypatch.setattr(database, "get_match_by_key", _fake_match([
        {"alliance": "blue", "position": 1, "team_number": "1684"},
        {"alliance": "blue", "position": 2, "team_number": "6615"},
        {"alliance": "red", "position": 1, "team_number": "7421"},
        {"alliance": "red", "position": 2, "team_number": "8575"},
    ]))
    match_teams, blue, red = asyncio.run(app_mod._load_db_roster("2026cur_qm2"))
    assert blue == ["1684", "6615"]
    assert red == ["7421", "8575"]
    assert match_teams == ["1684", "6615", "7421", "8575"]


def test_load_db_roster_skips_null_teams(monkeypatch):
    monkeypatch.setattr(database, "get_match_by_key", _fake_match([
        {"alliance": "blue", "position": 1, "team_number": "1684"},
        {"alliance": "blue", "position": 2, "team_number": None},
        {"alliance": "red", "position": 1, "team_number": None},
    ]))
    match_teams, blue, red = asyncio.run(app_mod._load_db_roster("k"))
    assert blue == ["1684"] and red == []


def test_load_db_roster_none_when_no_match_key():
    assert asyncio.run(app_mod._load_db_roster(None)) is None


def test_load_db_roster_none_when_match_absent(monkeypatch):
    async def _none(match_key):
        return None
    monkeypatch.setattr(database, "get_match_by_key", _none)
    assert asyncio.run(app_mod._load_db_roster("nope")) is None


def test_load_db_roster_none_when_roster_empty(monkeypatch):
    monkeypatch.setattr(database, "get_match_by_key", _fake_match([]))
    assert asyncio.run(app_mod._load_db_roster("k")) is None
