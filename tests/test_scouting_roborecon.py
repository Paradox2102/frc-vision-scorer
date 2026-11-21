"""RoboReconSink (HTTP push to RoboRecon) and auto-push after analysis.
No network: a fake session records requests and replays canned responses."""
import json
import threading

import pytest

from web.scouting import autopush, export
from web.scouting.export import RoboReconError, RoboReconSink, _batches

TOKEN = "rrvs_test_secret_value"


class FakeResponse:
    def __init__(self, status_code, body=None, text=""):
        self.status_code, self._body, self.text = status_code, body, text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    """Records calls; answers POSTs like RoboRecon (every envelope stored)
    unless a canned response is queued."""
    def __init__(self, responses=None, get_response=None, raise_exc=None):
        self.calls = []
        self.responses = list(responses or [])
        self.get_response = get_response
        self.raise_exc = raise_exc

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url, headers, None))
        if self.raise_exc:
            raise self.raise_exc
        return self.get_response or FakeResponse(
            200, {"success": True, "data": {"schema": "frc-vision-scorer.scouting",
                                            "supportedSchemaVersions": [1]}})

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append(("POST", url, headers, data))
        if self.raise_exc:
            raise self.raise_exc
        if self.responses:
            return self.responses.pop(0)
        envs = json.loads(data)["envelopes"]
        results = [{"index": i, "metric": e["metric"], "eventKey": e["event_key"],
                    "status": "stored", "teamsStored": len(e["teams"]),
                    "teamsSkippedStale": 0, "eventKnown": True} for i, e in enumerate(envs)]
        return FakeResponse(200, {"success": True, "data": {
            "stored": len(envs), "rejected": 0, "results": results}})


def _env(metric="auto_paths", teams=None):
    return {"schema": "frc-vision-scorer.scouting", "schema_version": 1, "metric": metric,
            "metric_version": 2, "event_key": "2026cur", "year": 2026,
            "generated_at": "2026-11-21T10:00:00+00:00",
            "teams": teams if teams is not None else {"3128": {"team_number": 3128, "matches": []}}}


def _sink(session, **kw):
    return RoboReconSink(base_url="https://rr.example/", token=TOKEN, session=session, **kw)


# ----------------------------------------------------------------- config / check
def test_reads_env_and_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("ROBORECON_URL", "https://staging.example.net/")
    monkeypatch.setenv("ROBORECON_TOKEN", TOKEN)
    s = RoboReconSink(session=FakeSession())
    assert s.url == "https://staging.example.net/api/integrations/vision-scouting"


def test_check_sends_bearer_and_reports_schema_support():
    fake = FakeSession()
    info = _sink(fake).check()
    assert info == {"ok": True, "url": "https://rr.example/api/integrations/vision-scouting",
                    "supported_schema_versions": [1]}
    method, _, headers, _ = fake.calls[0]
    assert method == "GET" and headers["Authorization"] == f"Bearer {TOKEN}"


@pytest.mark.parametrize("status,needle", [
    (401, "rejected the token"), (503, "no VISION_SCOUTING_API_KEY"),
    (404, "not found"), (500, "returned 500"),
])
def test_http_failures_raise_without_leaking_the_token(status, needle):
    fake = FakeSession(get_response=FakeResponse(status, {"error": "nope"}),
                       responses=[FakeResponse(status, {"error": "nope"})])
    for call in (lambda s: s.check(), lambda s: s.push([_env()])):
        with pytest.raises(RoboReconError, match=needle) as ei:
            call(_sink(fake))
        assert TOKEN not in str(ei.value)


def test_network_error_is_wrapped():
    fake = FakeSession(raise_exc=ConnectionError(f"boom Authorization: Bearer {TOKEN}"))
    with pytest.raises(RoboReconError, match="could not reach") as ei:
        _sink(fake).push([_env()])
    assert TOKEN not in str(ei.value)


# ----------------------------------------------------------------- push
def test_push_reports_per_envelope_results():
    fake = FakeSession()
    out = _sink(fake).push([_env("auto_paths"), _env("cycles")])
    assert out["ok"] and out["requests"] == 1 and out["stored"] == 2 and out["rejected"] == 0
    assert [r["metric"] for r in out["results"]] == ["auto_paths", "cycles"]
    body = json.loads(fake.calls[0][3])
    assert [e["metric"] for e in body["envelopes"]] == ["auto_paths", "cycles"]


def test_partial_rejection_is_returned_not_raised():
    fake = FakeSession(responses=[FakeResponse(207, {"success": False, "data": {
        "stored": 1, "rejected": 1, "results": [
            {"index": 0, "metric": "auto_paths", "status": "stored", "teamsStored": 1},
            {"index": 1, "metric": "cycles", "status": "rejected", "errors": ["bad"]}]}})])
    out = _sink(fake).push([_env("auto_paths"), _env("cycles")])
    assert not out["ok"] and out["stored"] == 1 and out["rejected"] == 1
    assert out["results"][1]["errors"] == ["bad"]


def test_malformed_body_400_marks_every_envelope_rejected():
    fake = FakeSession(responses=[FakeResponse(400, {"success": False, "error": "Body must be JSON"})])
    out = _sink(fake).push([_env(), _env("cycles")])
    assert out["rejected"] == 2 and out["results"][0]["errors"] == ["Body must be JSON"]


def test_batches_respect_envelope_count_limit():
    fake = FakeSession()
    sink = _sink(fake)
    sink.MAX_ENVELOPES = 2
    out = sink.push([_env(f"m{i}") for i in range(5)])
    assert out["requests"] == 3 and out["stored"] == 5
    assert [len(json.loads(c[3])["envelopes"]) for c in fake.calls] == [2, 2, 1]


def test_oversized_envelope_is_split_by_whole_team():
    # Each team's payload stays intact (RoboRecon replaces per team: a team
    # split across requests would lose matches); teams are spread over parts.
    teams = {str(t): {"team_number": t, "matches": [{"match_key": f"m{i}", "pts": "x" * 300}
                                                    for i in range(5)]}
             for t in range(1, 11)}
    env = _env(teams=teams)
    one_team = len(json.dumps({"1": teams["1"]}))
    batches = list(_batches([env], max_envelopes=100, max_bytes=one_team * 3 + 600))
    sent = [e for batch, _ in batches for e in batch]
    assert len(sent) > 1
    assert sorted(k for e in sent for k in e["teams"]) == sorted(teams)
    for e in sent:
        assert e["metric"] == "auto_paths" and e["event_key"] == "2026cur"
        for k, payload in e["teams"].items():
            assert payload == teams[k]                       # whole team, all matches
    for _, body in batches:
        assert len(body) <= one_team * 3 + 600


# ----------------------------------------------------------------- auto-push
@pytest.fixture
def autopush_on(monkeypatch):
    monkeypatch.setenv("ROBORECON_AUTO_PUSH", "1")
    monkeypatch.setenv("ROBORECON_URL", "https://rr.example")
    monkeypatch.setenv("ROBORECON_TOKEN", TOKEN)
    monkeypatch.delenv("ROBORECON_AUTO_PUSH_METRICS", raising=False)
    monkeypatch.setattr(autopush, "_under_pytest", lambda: False)   # lift the in-test guard
    autopush._state.clear()
    autopush._last.clear()
    yield
    autopush._state.clear()
    autopush._last.clear()


def test_autopush_off_unless_enabled_and_configured(monkeypatch, autopush_on):
    monkeypatch.setenv("ROBORECON_AUTO_PUSH", "0")
    assert autopush.schedule("2026cur") is False
    monkeypatch.setenv("ROBORECON_AUTO_PUSH", "1")
    monkeypatch.setenv("ROBORECON_TOKEN", "")
    assert autopush.config()["enabled"] is False
    assert autopush.schedule("2026cur") is False


def test_autopush_disabled_under_pytest(monkeypatch, autopush_on):
    monkeypatch.setattr(autopush, "_under_pytest", lambda: True)
    assert autopush.config()["enabled"] is False
    monkeypatch.undo()
    assert autopush._under_pytest() is True          # the real guard sees this run


def test_autopush_defaults_to_auto_paths_and_coalesces(monkeypatch, autopush_on):
    import web.scouting.api as api
    first_started, release = threading.Event(), threading.Event()
    calls = []

    def fake_push(event_key, metrics):
        calls.append((event_key, list(metrics)))
        if len(calls) == 1:
            first_started.set()
            assert release.wait(5)
        return {"ok": True, "stored": 1, "rejected": 0, "metrics": metrics, "results": []}

    monkeypatch.setattr(api, "push_event", fake_push)
    assert autopush.schedule("2026cur") is True
    assert first_started.wait(5)
    # three more matches finish while the first push is running
    assert all(autopush.schedule("2026cur") for _ in range(3))
    release.set()
    for t in threading.enumerate():
        if t.name == "roborecon-push-2026cur":
            t.join(5)
    assert calls == [("2026cur", ["auto_paths"]), ("2026cur", ["auto_paths"])]   # 1 + 1 coalesced
    assert autopush.status()["last"]["2026cur"]["ok"] is True
    assert autopush.status()["in_flight"] == {}


def test_autopush_failure_is_logged_not_raised(monkeypatch, autopush_on, capsys):
    import web.scouting.api as api

    def boom(event_key, metrics):
        raise RoboReconError("RoboRecon rejected the token (401)")

    monkeypatch.setattr(api, "push_event", boom)
    assert autopush.schedule("2026cur") is True
    for t in threading.enumerate():
        if t.name == "roborecon-push-2026cur":
            t.join(5)
    last = autopush.status()["last"]["2026cur"]
    assert last["ok"] is False and "401" in last["error"]
    assert "auto-push 2026cur failed" in capsys.readouterr().out
