"""/api/scouting router: synthetic matches + a fake metric, no app import, no real DB reads."""
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.scouting import api, loader, registry
from web.scouting.model import MatchData, Sample, TeamMatch


class FakeMetric(registry.Metric):
    key = "fake_metric"
    title = "Fake"
    version = 1
    years = None
    description = "per-team sample counts"
    default_params = {"scale": 1.0, "flag": False, "label": "x"}
    param_bounds = {"scale": (0.1, 10.0)}

    def compute(self, matches, params):
        teams = {}
        for md in matches:
            for t, tm in md.teams.items():
                row = teams.setdefault(str(t), {"n": 0})
                row["n"] += len(tm.samples) * params["scale"]
        return {"teams": teams, "echo": {"flag": params["flag"], "label": params["label"]}}


class Fake2025Only(registry.Metric):
    key = "fake_2025_only"
    title = "2025 only"
    years = (2025,)

    def compute(self, matches, params):
        return {"teams": {}}


def _match(key, red, blue, labelled):
    md = MatchData(event_key="2026tst", match_key=key, comp_level="qm",
                   match_number=int(key.split("qm")[1]), set_number=1, year=2026,
                   replay_id=f"r-{key}", replay_created_at="2026-11-12T00:00:00",
                   legacy_labels=False, alliances={"red": red, "blue": blue},
                   hub_auto={"red": 10, "blue": 5}, frame_period_s=0.2)
    for team, n in labelled.items():
        alliance, station = md.station_of(team)
        tm = TeamMatch(team=team, alliance=alliance, station=station)
        tm.samples = [Sample(frame_num=i, t_match=i * 0.2, period="auto", remaining=20 - i * 0.2,
                             x=1.0, y=1.0, ax=1.0, ay=1.0, dt=0.2) for i in range(n)]
        md.teams[team] = tm
    return md


MATCHES = [
    _match("2026tst_qm1", [1, 2, 3], [4, 5, 6], {1: 3, 4: 2}),
    _match("2026tst_qm2", [1, 7, 8], [9, 10, 11], {1: 5, 9: 1}),
]
SKIPPED = [{"match_key": "2026tst_qm3", "reason": "no match clock (summary.periods)"}]


@pytest.fixture
def client(monkeypatch, tmp_path):
    registry._ensure_loaded()
    monkeypatch.setitem(registry._REGISTRY, "fake_metric", FakeMetric())
    monkeypatch.setitem(registry._REGISTRY, "fake_2025_only", Fake2025Only())
    monkeypatch.setattr(loader, "load_event", lambda ek, db=None: (list(MATCHES), list(SKIPPED)))
    monkeypatch.setattr(loader, "source_fingerprint", lambda ek, db=None: ("fp", ek))
    registry._CACHE.clear()
    registry._LOAD_CACHE.clear()

    db = tmp_path / "s.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE replays (id TEXT, event_key TEXT, match_key TEXT, status TEXT, "
                 "created_at TEXT, meta_path TEXT)")
    conn.executemany("INSERT INTO replays VALUES (?,?,?,?,?,?)", [
        ("a", "2026tst", "2026tst_qm1", "complete", "t", None),
        ("b", "2026tst", "2026tst_qm1", "complete", "t", None),   # A/B rerun, same match
        ("c", "2026tst", "2026tst_qm2", "complete", "t", None),
        ("d", "2025old", "2025old_qm1", "complete", "t", None),
        ("e", "2026tst", "2026tst_qm4", "processing", "t", None),
    ])
    conn.commit()
    conn.close()
    monkeypatch.setattr(api, "DB_PATH", db)

    app = FastAPI()
    app.include_router(api.router)
    yield TestClient(app)
    registry._CACHE.clear()
    registry._LOAD_CACHE.clear()


def test_events_counts_distinct_complete_matches(client):
    r = client.get("/api/scouting/events")
    assert r.status_code == 200
    ev = {e["event_key"]: e for e in r.json()}
    assert ev["2026tst"]["n_matches"] == 2 and ev["2026tst"]["year"] == 2026
    assert ev["2025old"]["n_matches"] == 1
    keys26 = {m["key"] for m in ev["2026tst"]["metrics"]}
    keys25 = {m["key"] for m in ev["2025old"]["metrics"]}
    assert "fake_metric" in keys26 and "fake_2025_only" not in keys26
    assert "fake_2025_only" in keys25
    assert r.json()[0]["event_key"] == "2026tst"          # most matches first


def test_metrics_lists_applicable_metrics_and_sources(client):
    r = client.get("/api/scouting/2026tst/metrics")
    assert r.status_code == 200
    d = r.json()
    keys = {m["key"] for m in d["metrics"]}
    assert "fake_metric" in keys and "fake_2025_only" not in keys
    assert d["year"] == 2026
    assert [s["match_key"] for s in d["sources"]] == ["2026tst_qm1", "2026tst_qm2"]
    assert d["skipped"] == SKIPPED


def test_teams_from_lineups_and_labels(client):
    d = client.get("/api/scouting/2026tst/teams").json()
    by = {t["team_number"]: t for t in d["teams"]}
    assert [t["team_number"] for t in d["teams"]] == sorted(by)
    assert by[1]["n_matches"] == 2 and by[1]["n_labelled"] == 2
    assert by[2]["n_matches"] == 1 and by[2]["n_labelled"] == 0
    assert set(by) == set(range(1, 12))


def test_metric_envelope_and_team_filter(client):
    env = client.get("/api/scouting/2026tst/metric/fake_metric").json()
    assert env["metric"] == "fake_metric" and env["schema_version"] == 1
    assert set(env["teams"]) == {"1", "4", "9"}
    assert env["teams"]["1"] == {"n": 8.0, "team_number": 1}

    one = client.get("/api/scouting/2026tst/metric/fake_metric?team=1").json()
    assert set(one["teams"]) == {"1"} and one["echo"] == env["echo"]
    none = client.get("/api/scouting/2026tst/metric/fake_metric?team=999").json()
    assert none["teams"] == {}


def test_metric_param_overrides_from_query(client):
    env = client.get("/api/scouting/2026tst/metric/fake_metric"
                     "?scale=2&flag=true&label=hi&ignored=7").json()
    assert env["params"] == {"scale": 2.0, "flag": True, "label": "hi"}
    assert env["teams"]["1"]["n"] == 16.0 and env["echo"] == {"flag": True, "label": "hi"}
    assert client.get("/api/scouting/2026tst/metric/fake_metric?scale=abc").status_code == 400
    assert client.get("/api/scouting/2026tst/metric/fake_metric?flag=maybe").status_code == 400


def test_out_of_bounds_or_non_finite_params_are_400(client):
    base = "/api/scouting/2026tst/metric/fake_metric"
    for bad in ("0.001", "1e9", "nan", "inf", "-inf"):
        assert client.get(f"{base}?scale={bad}").status_code == 400, bad
    assert client.get(f"{base}?scale=10").status_code == 200


def test_metric_cache_is_bounded(client, monkeypatch):
    monkeypatch.setattr(registry, "_CACHE_MAX", 3)
    base = "/api/scouting/2026tst/metric/fake_metric"
    for i in range(8):
        assert client.get(f"{base}?scale={1 + i / 10}").status_code == 200
    assert len(registry._CACHE) <= 3


def test_unknown_metric_404_and_wrong_year_400(client):
    assert client.get("/api/scouting/2026tst/metric/nope").status_code == 404
    r = client.get("/api/scouting/2026tst/metric/fake_2025_only")
    assert r.status_code == 400 and "2026" in r.json()["detail"]


def test_export_returns_one_envelope_per_applicable_metric(client):
    d = client.get("/api/scouting/2026tst/export").json()
    assert d["event_key"] == "2026tst"
    metrics = [e["metric"] for e in d["envelopes"]]
    assert "fake_metric" in metrics and "fake_2025_only" not in metrics
    assert len(metrics) == len(set(metrics))
    for e in d["envelopes"]:
        assert e["schema"] == "frc-vision-scorer.scouting" and e["event_key"] == "2026tst"


def test_field_image_2026(client):
    meta = client.get("/api/scouting/field-image/2026/meta")
    assert meta.status_code == 200
    m = meta.json()
    assert "path" not in m and m["field_top_left_px"] and m["field_bottom_right_px"]
    assert m["url"] == "/api/scouting/field-image/2026"
    img = client.get("/api/scouting/field-image/2026")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert client.get("/api/scouting/field-image/1999/meta").status_code == 404
    assert client.get("/api/scouting/field-image/1999").status_code == 404


# ----------------------------------------------------------------- robot paths
def test_export_carries_every_lineup_robots_path(client):
    d = client.get("/api/scouting/2026tst/export").json()
    assert d["path_frame"]["perspective"] == "absolute"
    qm1 = next(m for m in d["matches"] if m["match_key"] == "2026tst_qm1")
    assert qm1["robots_version"] == 2 and qm1["auto_s"] > 0
    assert [(r["team_number"], r["alliance"], r["alliance_position"]) for r in qm1["robots"]] == [
        (1, "red", 1), (2, "red", 2), (3, "red", 3), (4, "blue", 1), (5, "blue", 2), (6, "blue", 3)]
    r1 = qm1["robots"][0]["path"]
    assert r1["n_points"] == 3 and len(r1["t"]) == len(r1["x"]) == len(r1["y"]) == len(r1["frame"]) == 3
    assert r1["segments"] == [[0, 2]]
    unseen = qm1["robots"][1]["path"]                 # in the lineup, never identified
    assert unseen["n_points"] == 0 and unseen["t"] == [] and unseen["segments"] == []


def test_export_paths_false_drops_robots(client):
    d = client.get("/api/scouting/2026tst/export?paths=false").json()
    assert "path_frame" not in d
    assert all("robots" not in m for m in d["matches"])


def test_match_endpoint_and_team_filter(client):
    d = client.get("/api/scouting/2026tst/match/2026tst_qm2").json()
    assert d["schema"] == "frc-vision-scorer.scouting-match"
    assert d["match"]["match_key"] == "2026tst_qm2" and len(d["match"]["robots"]) == 6
    one = client.get("/api/scouting/2026tst/match/2026tst_qm2?team=9").json()
    assert [r["team_number"] for r in one["match"]["robots"]] == [9]
    assert one["match"]["robots"][0]["path"]["n_points"] == 1
    assert client.get("/api/scouting/2026tst/match/2026tst_qm2?team=1234").status_code == 404
    miss = client.get("/api/scouting/2026tst/match/2026tst_qm3")
    assert miss.status_code == 404 and "no match clock" in miss.json()["detail"]
    assert client.get("/api/scouting/2026tst/match/2026tst_qm99").status_code == 404
