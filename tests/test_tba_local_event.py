import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))

import tba  # noqa: E402


def _fixture(tmp_path, monkeypatch, key="2026simtest", blue=("frc3128", "frc6773")):
    data = {
        "event": {"key": key, "name": "Sim", "year": 2026},
        "matches": [{
            "key": f"{key}_qm1", "comp_level": "qm", "set_number": 1, "match_number": 1,
            "alliances": {"red": {"team_keys": ["frc1", "frc2", "frc3"], "score": 10},
                          "blue": {"team_keys": list(blue), "score": 20}},
            "videos": [], "time": None, "score_breakdown": None}],
        "teams": [{"key": f"frc{n}", "team_number": n, "nickname": f"Team {n}"}
                  for n in (1, 2, 3, 3128, 6773)],
    }
    (tmp_path / f"{key}.json").write_text(json.dumps(data))
    monkeypatch.setattr(tba, "LOCAL_EVENTS_DIR", tmp_path)


def _no_network(monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("TBA must not be called for fixture events")
    monkeypatch.setattr(tba, "tba_get", boom)


def test_fixture_loaded_without_tba(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    _no_network(monkeypatch)
    ev = asyncio.run(tba.load_event_data("2026simtest"))
    assert ev["event_key"] == "2026simtest" and ev["teams"]["3128"]["name"] == "Team 3128"
    assert ev["matches"][0]["key"] == "2026simtest_qm1"


def test_two_team_alliance_and_null_breakdown(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    _no_network(monkeypatch)
    m = asyncio.run(tba.load_event_data("2026simtest"))["matches"][0]
    assert m["blue_teams"] == ["3128", "6773"] and len(m["red_teams"]) == 3
    assert m["blue_hub"] == {} and m["red_hub"] == {}
    assert m["time"] is None and m["video_key"] is None


def test_path_escape_rejected(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    (tmp_path.parent / "evil.json").write_text("{}")
    for bad in ("../evil", "..", "a/b", "", "x.json"):
        assert tba.load_local_event(bad) is None


def test_unknown_key_falls_through_to_tba(tmp_path, monkeypatch):
    monkeypatch.setattr(tba, "LOCAL_EVENTS_DIR", tmp_path)
    assert tba.load_local_event("2026nope") is None


def test_committed_fixtures_load():
    for key in ("2026mantasim", "2026mantatru"):
        ev = asyncio.run(tba.load_event_data(key))
        m = ev["matches"][0]
        assert m["key"] == f"{key}_qm8" and m["blue_teams"] == ["3128", "6773"]
        assert m["red_teams"] == ["6497", "2724", "2637"]
        assert (m["red_score"], m["blue_score"]) == (292, 330)
