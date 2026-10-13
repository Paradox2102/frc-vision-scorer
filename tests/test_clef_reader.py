"""team_reader="clef": Cloudflare Workers AI decision-model bumper reader.

All tests are OFFLINE. The only network function in web/clef_reader.py is
``http_post``; the autouse ``no_network`` fixture replaces it (and
urllib/socket) with tripwires and asserts at teardown that nothing tried to
reach the network. Tests that need responses install a scripted ``FakeCF``.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import socket
import types
import urllib.error
import urllib.request

import pytest
from PIL import Image

import web.app as app_module
from web import clef_reader
from web.clef_reader import ClefSession
from web.job_config import JobConfig

RED = ["2992", "4256", "1477"]
BLUE = ["4786", "6800", "870"]
ALL6 = RED + BLUE
ACCOUNT = "acct-0123456789abcdef"
TOKEN = "tok-SECRET-do-not-log"


# --------------------------------------------------------------------------- #
# Network guard + fakes
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Every network path is a tripwire; teardown asserts none fired.

    The reader catches exceptions per crop (a tripwire raising inside it would
    be swallowed into an error entry), so attempts are RECORDED and checked
    here rather than relying on the raise alone."""
    attempts = []

    def _tripwire(*a, **k):
        attempts.append(("network", a[:1]))
        raise AssertionError("network call attempted in an offline test")

    monkeypatch.setattr(urllib.request, "urlopen", _tripwire)
    monkeypatch.setattr(socket, "create_connection", _tripwire)
    monkeypatch.setattr(clef_reader, "http_post", _tripwire)
    # Fixed fake credentials: never pick up a real .env value.
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", TOKEN)
    for var in ("CLEF_MIN_TEAM_PROB", "CLEF_CONCURRENCY", "CLEF_TIMEOUT_S",
                "CLEF_RETRIES", "CLEF_BACKOFF_S"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(clef_reader, "_default_session", None)
    yield attempts
    assert not attempts, f"offline test reached the network: {attempts}"


def _answers(team_probs=None, surface="bumper", legible=0.9):
    team_probs = team_probs or {"4786": 0.97, "none": 0.02, "other": 0.01}
    top = max(team_probs, key=team_probs.get)
    return {
        "legible": {"type": "noul", "noul": legible},
        "surface": {"type": "choice", "choice": surface,
                    "probabilities": {surface: 0.9}, "confidence": 0.9},
        "team": {"type": "choice", "choice": top, "probabilities": team_probs,
                 "confidence": team_probs[top]},
    }


def _ok(answers, input_tokens=400, wrapped=True):
    result = {"model": "clef-flash", "answers": answers,
              "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
    body = {"result": result, "success": True, "errors": []} if wrapped else result
    return 200, json.dumps(body)


def _quota_429():
    return 429, json.dumps({
        "result": None, "success": False,
        "errors": [{"code": 4006, "message": (
            "you have used up your daily free allocation of 10,000 neurons, "
            "please upgrade to Cloudflare's Workers Paid plan")}],
    })


class FakeCF:
    """Scripted stand-in for clef_reader.http_post: pops one response per
    call (the last one repeats). A response is (status, text), an Exception
    to raise, or a callable(payload) -> (status, text)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, data, headers, timeout):
        payload = json.loads(data.decode("utf-8"))
        self.calls.append({"url": url, "payload": payload,
                           "headers": dict(headers), "timeout": timeout})
        resp = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(resp, Exception):
            raise resp
        if callable(resp):
            return resp(payload)
        return resp


def _install(monkeypatch, fake):
    monkeypatch.setattr(clef_reader, "http_post", fake)
    return fake


def _session(**kw):
    kw.setdefault("sleep_fn", lambda _s: None)
    kw.setdefault("log_fn", lambda _m: None)
    return ClefSession(**kw)


def _jpeg_b64(w=80, h=40, color=(200, 30, 30)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _team_criteria(payload):
    return payload["questions"]["team"]["criteria"]


# --------------------------------------------------------------------------- #
# Payload shape
# --------------------------------------------------------------------------- #

def test_payload_offers_full_roster_plus_escapes_with_alliance_labels():
    img, _w, _h = clef_reader.clef_image(base64.b64decode(_jpeg_b64()))
    p = clef_reader.build_read_payload("clef-flash", img, RED, BLUE, "blue")
    assert p["model"] == "clef-flash"
    crit = _team_criteria(p)
    assert set(crit) == set(ALL6) | {"none", "other"}
    assert crit["4786"] == "Team 4786 (blue alliance)"
    assert crit["2992"] == "Team 2992 (red alliance)"
    assert "not any of the listed teams" in crit["other"].lower()
    # Known bumper colour is stated, but all six teams are still offered.
    assert "BLUE bumper pixels" in p["state"]
    assert "red alliance 2992, 4256, 1477" in p["state"]
    assert "blue alliance 4786, 6800, 870" in p["state"]
    # Bench question set: legible (noul), surface + team (choice).
    q = p["questions"]
    assert q["legible"]["type"] == "noul"
    assert q["surface"]["type"] == "choice"
    assert set(q["surface"]["criteria"]) == {"bumper", "printed", "none"}
    assert q["team"]["type"] == "choice"
    assert len(p["images"]) == 1
    assert p["images"][0].startswith("data:image/jpeg;base64,")


def test_payload_omits_colour_hint_when_alliance_unknown():
    img, _w, _h = clef_reader.clef_image(base64.b64decode(_jpeg_b64()))
    p = clef_reader.build_read_payload("clef", img, RED, BLUE, "unknown")
    assert p["model"] == "clef"
    assert "bumper pixels" not in p["state"]
    assert set(_team_criteria(p)) == set(ALL6) | {"none", "other"}


def test_option_order_is_deterministic_per_seed():
    img, _w, _h = clef_reader.clef_image(base64.b64decode(_jpeg_b64()))
    a = clef_reader.build_read_payload("clef-flash", img, RED, BLUE, "red", seed="crop-a")
    b = clef_reader.build_read_payload("clef-flash", img, RED, BLUE, "red", seed="crop-a")
    assert list(_team_criteria(a)) == list(_team_criteria(b))


def test_image_small_crop_upscaled_to_256_min_side():
    out, w, h = clef_reader.clef_image(base64.b64decode(_jpeg_b64(80, 40)))
    assert min(w, h) >= 256
    with Image.open(io.BytesIO(out)) as im:
        assert im.size == (w, h)
    assert clef_reader.clef_image_tokens(w, h) <= 1024


def test_image_large_crop_shrunk_under_token_cap():
    out, w, h = clef_reader.clef_image(base64.b64decode(_jpeg_b64(3000, 2000)))
    assert clef_reader.clef_image_tokens(w, h) <= 1024
    assert round(w / 32) * round(h / 32) + 3 <= 1024
    assert len(out) <= clef_reader.CLEF_MAX_IMAGE_BYTES


def test_image_in_range_jpeg_passes_through_unchanged():
    raw = base64.b64decode(_jpeg_b64(512, 384))
    out, w, h = clef_reader.clef_image(raw)
    assert out == raw and (w, h) == (512, 384)


def test_classify_payload_shape():
    img, _w, _h = clef_reader.clef_image(base64.b64decode(_jpeg_b64(300, 300)))
    p = clef_reader.build_classify_payload("clef-flash", img)
    q = p["questions"]["object"]
    assert q["type"] == "choice"
    assert set(q["criteria"]) == {"robot", "equipment", "person", "unclear"}
    assert p["model"] == "clef-flash" and len(p["images"]) == 1


# --------------------------------------------------------------------------- #
# Response mapping
# --------------------------------------------------------------------------- #

_ALLIANCE = {**{t: "red" for t in RED}, **{t: "blue" for t in BLUE}}


def test_map_confident_team():
    res = clef_reader.map_read_answers(
        _answers({"4786": 0.92, "6800": 0.05, "none": 0.03}),
        _ALLIANCE, "blue", min_team_prob=0.5)
    assert res["team_number"] == "4786"
    assert res["alliance"] == "blue"
    assert res["confidence"] == pytest.approx(0.92)
    assert res["digit_prob"] == pytest.approx(0.92)
    assert res["raw_read"] == "4786"
    assert res["surface"] == "bumper"
    assert res["error"] is None
    assert res["clef_probs"] == {"4786": 0.92, "6800": 0.05, "none": 0.03}
    json.dumps(res)  # diagnostics-safe


def test_map_alliance_comes_from_roster_not_hint():
    res = clef_reader.map_read_answers(
        _answers({"2992": 0.9, "none": 0.1}), _ALLIANCE, "blue", min_team_prob=0.5)
    assert res["team_number"] == "2992" and res["alliance"] == "red"


def test_map_team_below_threshold_is_no_team():
    res = clef_reader.map_read_answers(
        _answers({"4786": 0.45, "none": 0.40, "other": 0.15}),
        _ALLIANCE, "blue", min_team_prob=0.5)
    assert res["team_number"] is None
    assert res["confidence"] == 0.0
    assert res["raw_read"] == "4786"          # the chosen option id
    assert res["alliance"] == "blue"          # expected alliance when no team
    assert res["digit_prob"] == pytest.approx(0.45)


@pytest.mark.parametrize("escape", ["none", "other"])
def test_map_escape_winner(escape):
    probs = {escape: 0.8, "4786": 0.15, "870": 0.05}
    res = clef_reader.map_read_answers(
        _answers(probs, surface="none"), _ALLIANCE, "unknown", min_team_prob=0.5)
    assert res["team_number"] is None
    assert res["raw_read"] == escape
    assert res["alliance"] == "unknown"
    assert res["surface"] == "unknown"       # "none" normalises to unknown


def test_map_printed_surface_survives_for_the_surface_gate():
    res = clef_reader.map_read_answers(
        _answers({"1477": 0.95, "none": 0.05}, surface="printed"),
        _ALLIANCE, "red", min_team_prob=0.5)
    assert res["team_number"] == "1477" and res["surface"] == "printed"


def test_min_team_prob_env_override(monkeypatch):
    monkeypatch.setenv("CLEF_MIN_TEAM_PROB", "0.7")
    fake = _install(monkeypatch, FakeCF(_ok(_answers({"4786": 0.65, "none": 0.35}))))
    out = _session().read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    assert len(fake.calls) == 1
    assert out[0]["team_number"] is None and out[0]["raw_read"] == "4786"


def test_a_0_6_read_gets_half_vote_weight():
    """confidence = the option probability, so the app's 0.85 full-weight bar
    applies to it directly: 0.6 -> half weight, 0.9 -> full."""
    lo = clef_reader.map_read_answers(_answers({"4786": 0.6, "none": 0.4}),
                                      _ALLIANCE, "blue", min_team_prob=0.5)
    hi = clef_reader.map_read_answers(_answers({"4786": 0.9, "none": 0.1}),
                                      _ALLIANCE, "blue", min_team_prob=0.5)
    weight = lambda c: 1.0 if c >= 0.85 else 0.5  # app.py's vote rule
    assert weight(lo["confidence"]) == 0.5
    assert weight(hi["confidence"]) == 1.0


@pytest.mark.parametrize("wrapped", [True, False])
def test_wrapped_and_unwrapped_responses(monkeypatch, wrapped):
    fake = _install(monkeypatch, FakeCF(_ok(_answers(), input_tokens=321, wrapped=wrapped)))
    s = _session()
    out = s.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    assert out[0]["team_number"] == "4786"
    assert s.stats()["input_tokens"] == 321
    assert len(fake.calls) == 1


def test_success_false_body_is_an_error_not_a_crash(monkeypatch):
    _install(monkeypatch, FakeCF((200, json.dumps(
        {"result": None, "success": False, "errors": [{"code": 5000, "message": "bad"}]}))))
    out = _session().read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    assert out[0]["team_number"] is None and out[0]["error"]


def test_request_targets_model_route_with_bearer_auth(monkeypatch):
    fake = _install(monkeypatch, FakeCF(_ok(_answers())))
    _session(model_key="clef-omni").read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    call = fake.calls[0]
    assert call["url"] == (f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}"
                           "/ai/run/@cf/cloudflare/clef-omni")
    assert call["payload"]["model"] == "clef-omni"
    assert call["headers"]["Authorization"] == f"Bearer {TOKEN}"


# --------------------------------------------------------------------------- #
# Quota / auth latch, retries
# --------------------------------------------------------------------------- #

def test_quota_exhausted_latches_and_stops_all_http(monkeypatch):
    fake = _install(monkeypatch, FakeCF(_quota_429(), _ok(_answers())))
    logs = []
    s = _session(log_fn=logs.append)
    first = s.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    assert first[0]["team_number"] is None and "quota" in first[0]["error"].lower()
    assert len(fake.calls) == 1, "quota exhaustion must not be retried"
    assert not s.available and s.disabled_kind == "quota"
    latch_lines = [m for m in logs if "quota exhausted" in m.lower()]
    assert len(latch_lines) == 1
    assert "reads disabled for this job" in latch_lines[0]

    # Every later read short-circuits with NO HTTP.
    later = s.read_bumpers([_jpeg_b64()] * 4, RED, BLUE, ["blue"] * 4)
    assert len(fake.calls) == 1
    assert all(r["team_number"] is None and r["error"] for r in later)
    assert s.classify_crops([_jpeg_b64(300, 300)])[0]["object"] == "unclear"
    assert len(fake.calls) == 1
    st = s.stats()
    assert st["quota_errors"] == 1 and st["skipped"] == 5
    assert len([m for m in logs if "quota exhausted" in m.lower()]) == 1


def test_quota_detected_by_message_without_code(monkeypatch):
    _install(monkeypatch, FakeCF((429, "You have used up your daily free allocation")))
    s = _session()
    s.read_bumpers([_jpeg_b64()], RED, BLUE, ["red"])
    assert s.disabled_kind == "quota"


def test_transient_5xx_is_retried_then_succeeds(monkeypatch):
    sleeps = []
    fake = _install(monkeypatch, FakeCF((503, "upstream busy"), _ok(_answers())))
    s = _session(sleep_fn=sleeps.append)
    out = s.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    assert out[0]["team_number"] == "4786"
    assert len(fake.calls) == 2 and len(sleeps) == 1
    assert s.available and s.stats()["retries"] == 1


def test_plain_429_rate_limit_is_transient(monkeypatch):
    fake = _install(monkeypatch, FakeCF((429, json.dumps(
        {"success": False, "errors": [{"code": 3040, "message": "rate limited"}]})),
        _ok(_answers())))
    s = _session()
    assert s.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])[0]["team_number"] == "4786"
    assert len(fake.calls) == 2 and s.available


def test_network_error_is_retried(monkeypatch):
    fake = _install(monkeypatch, FakeCF(
        urllib.error.URLError(ConnectionResetError("reset")), _ok(_answers())))
    out = _session().read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    assert out[0]["team_number"] == "4786" and len(fake.calls) == 2


def test_transient_gives_up_without_latching(monkeypatch):
    fake = _install(monkeypatch, FakeCF((500, "boom")))
    s = _session(retries=2)
    out = s.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    assert out[0]["error"] and len(fake.calls) == 3
    assert s.available, "transient failures must not disable the reader"
    assert s.stats()["errors"] == 1


def test_bad_request_is_not_retried(monkeypatch):
    fake = _install(monkeypatch, FakeCF((400, "invalid questions")))
    s = _session()
    out = s.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    assert out[0]["error"] and len(fake.calls) == 1 and s.available


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_latches(monkeypatch, status):
    fake = _install(monkeypatch, FakeCF((status, json.dumps(
        {"success": False, "errors": [{"code": 10000, "message": "Authentication error"}]}))))
    logs = []
    s = _session(log_fn=logs.append)
    s.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    s.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    assert len(fake.calls) == 1
    assert s.disabled_kind == "auth" and not s.available
    assert len([m for m in logs if "reads disabled for this job" in m]) == 1


def test_missing_credentials_disable_without_http(monkeypatch, no_network):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID")
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")
    logs = []
    s = _session(log_fn=logs.append)
    assert not s.available and s.disabled_kind == "credentials"
    out = s.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    assert out[0]["error"] and out[0]["team_number"] is None
    assert len(logs) == 1  # one clear line, at construction


def test_credentials_never_appear_in_logs_or_errors(monkeypatch, capsys):
    # The error body echoes the account id (as some Cloudflare errors do).
    _install(monkeypatch, FakeCF((400, f"no such route for account {ACCOUNT} token {TOKEN}")))
    logs = []
    s = ClefSession(sleep_fn=lambda _s: None, log_fn=logs.append)
    out = s.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    _install(monkeypatch, FakeCF((401, f"bad token {TOKEN}")))
    out += s.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    text = "\n".join(logs) + capsys.readouterr().out + json.dumps(out) + s.summary_line()
    assert TOKEN not in text and ACCOUNT not in text


def test_concurrency_env_and_parallel_reads(monkeypatch):
    monkeypatch.setenv("CLEF_CONCURRENCY", "3")
    fake = _install(monkeypatch, FakeCF(_ok(_answers())))
    s = _session()
    assert s.concurrency == 3
    out = s.read_bumpers([_jpeg_b64()] * 7, RED, BLUE, ["blue"] * 7)
    assert len(out) == 7 and len(fake.calls) == 7
    assert all(r["team_number"] == "4786" for r in out)


# --------------------------------------------------------------------------- #
# Cost / latency counters
# --------------------------------------------------------------------------- #

def test_cost_and_latency_counters(monkeypatch):
    _install(monkeypatch, FakeCF(_ok(_answers(), input_tokens=400),
                                 _ok(_answers(), input_tokens=600)))
    s = _session()
    s.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    s.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    st = s.stats()
    assert st["model"] == "clef-flash"
    assert st["requests"] == 2 and st["ok"] == 2
    assert st["input_tokens"] == 1000
    assert st["cost_usd"] == pytest.approx(1000 * 0.038 / 1e6)
    assert st["errors"] == 0 and st["quota_errors"] == 0
    assert st["latency_p50_s"] is not None and st["latency_p50_s"] >= 0.0
    line = s.summary_line()
    assert line.startswith("[clef]")
    assert "2 reads" in line and "0 errors" in line and "$" in line
    json.dumps(st)


def test_cost_uses_per_model_price(monkeypatch):
    _install(monkeypatch, FakeCF(_ok(_answers(), input_tokens=1_000_000)))
    s = _session(model_key="clef")
    s.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    assert s.stats()["cost_usd"] == pytest.approx(0.24)


def test_quota_error_counted_in_summary(monkeypatch):
    _install(monkeypatch, FakeCF(_quota_429()))
    s = _session()
    s.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    assert s.stats()["quota_errors"] == 1
    assert "quota" in s.summary_line().lower()


# --------------------------------------------------------------------------- #
# Classification (equipment adjudication)
# --------------------------------------------------------------------------- #

def test_classify_maps_choice(monkeypatch):
    fake = _install(monkeypatch, FakeCF(_ok({"object": {
        "type": "choice", "choice": "equipment",
        "probabilities": {"robot": 0.1, "equipment": 0.85, "person": 0.03, "unclear": 0.02},
        "confidence": 0.85}})))
    out = _session().classify_crops([_jpeg_b64(300, 300)])
    assert out[0]["object"] == "equipment" and out[0]["error"] is None
    assert len(fake.calls) == 1
    assert set(fake.calls[0]["payload"]["questions"]) == {"object"}


def test_classify_error_is_unclear(monkeypatch):
    _install(monkeypatch, FakeCF((400, "nope")))
    out = _session().classify_crops([_jpeg_b64(300, 300)])
    assert out[0]["object"] == "unclear" and out[0]["error"]


def test_adjudication_routes_to_clef_when_reader_is_clef(monkeypatch):
    called = {"clef": 0}

    def _boom(*_a, **_k):
        raise AssertionError("vLLM classify must not run for a clef job")

    def _fake_clef(crops, session=None):
        called["clef"] += 1
        return [{"object": "robot", "error": None} for _ in crops]

    monkeypatch.setattr(app_module, "_vllm_classify_crops", _boom)
    monkeypatch.setattr(app_module, "_clef_classify_crops", _fake_clef)
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    job.team_reader = "clef"
    job._clef_session = None
    job.config = JobConfig({"team_reader": "clef"})
    ident = types.SimpleNamespace(equipment_streak=2)
    job.identity_registry = types.SimpleNamespace(identities={5: ident})
    job.diagnostic = types.SimpleNamespace(record_adjudication=lambda **k: None)
    job._current_frame = 10
    job._slot_label = lambda s: f"S{s}"
    asyncio.run(job._run_adjudication([(5, 1, _jpeg_b64(300, 300))]))
    assert called["clef"] == 1 and ident.equipment_streak == 0


# --------------------------------------------------------------------------- #
# app.py dispatch + full roster threading
# --------------------------------------------------------------------------- #

def _forbid_local_readers(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("clef reads must never reach the vLLM / in-process Qwen readers")
    monkeypatch.setattr(app_module, "_vllm_read_bumpers", _boom)
    monkeypatch.setattr(app_module, "_qwen_vl_read_bumpers", _boom)
    monkeypatch.setattr(app_module, "_get_qwen_vl", _boom)


def test_vision_read_bumpers_dispatches_to_clef(monkeypatch):
    _forbid_local_readers(monkeypatch)
    fake = _install(monkeypatch, FakeCF(_ok(_answers({"6800": 0.9, "none": 0.1}))))
    s = _session()
    out = app_module._vision_read_bumpers(
        [_jpeg_b64()], ALL6, BLUE, RED, expected_alliances=["blue"],
        provider="clef", clef_session=s)
    assert out[0]["team_number"] == "6800" and out[0]["alliance"] == "blue"
    assert len(fake.calls) == 1


def test_clef_quota_never_falls_back_to_local_model(monkeypatch):
    _forbid_local_readers(monkeypatch)
    _install(monkeypatch, FakeCF(_quota_429()))
    s = _session()
    for _ in range(3):
        out = app_module._vision_read_bumpers(
            [_jpeg_b64()], ALL6, BLUE, RED, expected_alliances=["blue"],
            provider="clef", clef_session=s)
        assert out[0]["team_number"] is None and out[0]["error"]


def test_clef_internal_exception_returns_error_entries(monkeypatch):
    _forbid_local_readers(monkeypatch)

    def _explode(*_a, **_k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(app_module, "_clef_read_bumpers", _explode)
    out = app_module._vision_read_bumpers(
        [_jpeg_b64(), _jpeg_b64()], ALL6, BLUE, RED, provider="clef")
    assert len(out) == 2 and all(r["error"] and r["team_number"] is None for r in out)


def test_full_roster_offers_confirmed_team_missing_from_remaining(monkeypatch):
    """Callers may pass roster-minus-confirmed lists; the clef path must still
    offer all six teams (+ escapes) when full_roster is given."""
    fake = _install(monkeypatch, FakeCF(_ok(_answers({"4786": 0.95, "none": 0.05}))))
    remaining_blue = ["6800", "870"]          # 4786 already confirmed elsewhere
    out = app_module._clef_read_bumpers(
        [_jpeg_b64()], ["2992", "4256", "1477", "6800", "870"], remaining_blue, RED,
        expected_alliances=["blue"], full_roster={"red": RED, "blue": BLUE},
        session=_session())
    assert set(_team_criteria(fake.calls[0]["payload"])) == set(ALL6) | {"none", "other"}
    assert out[0]["team_number"] == "4786"


def _make_fused_job(team_reader, confirmed_team="4786"):
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    job.config = JobConfig({"team_reader": team_reader}
                           if team_reader != "local" else {})
    job._camera_centers_xy = {}
    job.match_teams = ALL6
    job.match_blue_teams = BLUE
    job.match_red_teams = RED
    job.robot_tracks = {9: types.SimpleNamespace(
        team_number=confirmed_team, team_number_confirmed=True)}
    job.vlm_status = {}
    job._vlm_task = None
    job._claude_vision_calls = 0
    job._current_frame = 100
    job.video_fps = 30.0
    job.field_tracker = types.SimpleNamespace(last_det_to_slot={0: 1})
    ident = types.SimpleNamespace(active_slot=1, alliance="blue", last_vlm_time=0.0,
                                  vlm_fail_count=0, backoff_until=-999.0)
    job.identity_registry = types.SimpleNamespace(identities={7: ident})
    job._occ = {}
    job.confidence_config = types.SimpleNamespace(backoff_base_sec=2.0)
    job.team_reader = team_reader
    job._clef_session = None
    job._get_or_create_slot_track = lambda slot_id: None
    job.co_occlusion_tracker = types.SimpleNamespace(get_active_events=lambda: [])
    job.apply_gallery_matches = lambda crops: None
    return job


def _fused_obs():
    return [types.SimpleNamespace(
        evidence=[("main", _jpeg_b64(), 1.0, (4.0, 3.0))],
        detection=types.SimpleNamespace(pos=(4.0, 3.0)))]


def test_fused_pass_threads_full_roster_for_clef(monkeypatch):
    seen = []

    def _rec(crops, teams, blue, red, expected_alliances=None, provider="local", **kw):
        seen.append({"provider": provider, **kw})
        return [{"team_number": None, "confidence": 0.0, "raw_read": "none"}]

    monkeypatch.setattr(app_module, "_vision_read_bumpers", _rec)
    job = _make_fused_job("clef")
    job._clef_session = _session()
    asyncio.run(job.read_fused_bumper_numbers(_fused_obs(), {7: 1.0}))
    assert seen[0]["provider"] == "clef"
    assert seen[0]["full_roster"] == {"red": RED, "blue": BLUE}
    assert seen[0]["clef_session"] is job._clef_session


def test_fused_pass_non_clef_call_is_unchanged(monkeypatch):
    """vLLM/local/deepseek calls get no new kwargs (byte-identical call)."""
    seen = []

    def _rec(crops, teams, blue, red, expected_alliances=None, provider="local", **kw):
        seen.append(kw)
        return [{"team_number": None, "confidence": 0.0}]

    monkeypatch.setattr(app_module, "_vision_read_bumpers", _rec)
    asyncio.run(_make_fused_job("vllm").read_fused_bumper_numbers(_fused_obs(), {7: 1.0}))
    assert seen == [{}]


def test_fused_pass_end_to_end_offers_confirmed_team(monkeypatch):
    fake = _install(monkeypatch, FakeCF(_ok(_answers({"4786": 0.95, "none": 0.05}))))
    job = _make_fused_job("clef", confirmed_team="4786")
    job._clef_session = _session()
    asyncio.run(job.read_fused_bumper_numbers(_fused_obs(), {7: 1.0}))
    assert len(fake.calls) == 1
    assert set(_team_criteria(fake.calls[0]["payload"])) == set(ALL6) | {"none", "other"}
    assert job.vlm_status["provider"] == "clef"


def test_apply_bumper_read_passes_full_roster_for_clef(monkeypatch):
    seen = []

    def _rec(crops, teams, blue, red, expected_alliances=None, provider="local", **kw):
        seen.append({"blue": blue, **kw})
        return [{"team_number": "4786", "confidence": 0.9}]

    monkeypatch.setattr(app_module, "_vision_read_bumpers", _rec)
    job = _make_fused_job("clef")
    robot = types.SimpleNamespace(alliance="blue", team_ocr_votes=None)
    job._get_slot_track_if_exists = lambda slot: robot
    job.robot_tracks = {3: robot}
    app_module.InferenceJob._apply_bumper_read(
        job, 3, _jpeg_b64(), ["6800", "870"], ["6800", "870"], [], source_weight=1.0)
    assert seen[0]["blue"] == ["6800", "870"]          # caller's remaining list
    assert seen[0]["full_roster"] == {"red": RED, "blue": BLUE}


def test_identify_pass_threads_full_roster_for_clef(monkeypatch):
    import numpy as np
    seen = []

    def _rec(crops, teams, blue, red, expected_alliances=None, provider="local", **kw):
        seen.append({"provider": provider, **kw})
        return [{"team_number": None, "confidence": 0.0, "raw_read": "none"}]

    monkeypatch.setattr(app_module, "_vision_read_bumpers", _rec)
    job = _make_fused_job("clef", confirmed_team="4786")
    job._clef_session = _session()
    robot = types.SimpleNamespace(robot_id=1, alliance="blue", team_number=None,
                                  team_number_confirmed=False, team_ocr_votes=None)
    job.robot_tracks = {
        1: robot,
        9: types.SimpleNamespace(team_number="4786", team_number_confirmed=True),
    }
    job._get_or_create_slot_track = lambda cid: robot
    job._bumper_crop_b64 = lambda *a, **k: (_jpeg_b64(), 0.9)
    job._slot_label = lambda s: f"S{s}"
    job.diagnostic = types.SimpleNamespace(record_vlm_read=lambda **k: None)
    app_module.InferenceJob._vision_identify_robots_impl(
        job, np.zeros((10, 10, 3), np.uint8), [(1, (0, 0, 5, 5))])
    assert seen and seen[0]["provider"] == "clef"
    assert seen[0]["full_roster"] == {"red": RED, "blue": BLUE}


# --------------------------------------------------------------------------- #
# Job wiring: config, vLLM lifecycle, warnings, usage summary, diagnostics
# --------------------------------------------------------------------------- #

def test_job_config_accepts_clef_and_clef_model():
    cfg = JobConfig({"team_reader": "clef"})
    assert cfg.team_reader == "clef" and cfg.clef_model == "clef-flash"
    for m in ("clef-flash", "clef", "clef-omni"):
        assert JobConfig({"team_reader": "clef", "clef_model": m}).clef_model == m
    with pytest.raises(ValueError):
        JobConfig({"clef_model": "clef-ultra"})
    # Default reader unchanged.
    assert "clef_model" not in JobConfig().non_defaults()


def test_legacy_team_reader_arg_accepts_clef():
    # Mirrors InferenceJob.__init__'s legacy (config=None) branch.
    assert "clef" in app_module._TEAM_READERS


def test_vllm_not_started_for_clef_jobs():
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    job.config = JobConfig({"team_reader": "clef", "vllm_phase_exclusive": True})
    job.team_reader = "clef"
    assert job._needs_vllm_server() is False
    job.team_reader = "vllm"
    assert job._needs_vllm_server() is True


def test_ensure_stopped_is_a_noop_when_no_server():
    from web.vllm_lifecycle import ensure_stopped

    async def _boom(*_a, **_k):
        raise AssertionError("must not touch a server that is not running")

    def _http_boom(*_a, **_k):
        raise AssertionError("must not touch a server that is not running")

    assert asyncio.run(ensure_stopped(check_fn=lambda: False, run_script=_boom,
                                      http_fn=_http_boom)) is True


class _WS:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)


def test_unavailable_clef_warns_ui_once(monkeypatch):
    _install(monkeypatch, FakeCF(_quota_429()))
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    job.websocket = _WS()
    job._websocket_send_lock = asyncio.Lock()
    job.vlm_status = {}
    job._clef_session = _session()
    job._clef_session.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])

    async def run():
        await job._maybe_warn_clef_unavailable()
        await job._maybe_warn_clef_unavailable()

    asyncio.run(run())
    warns = [m for m in job.websocket.messages if m.get("type") == "degradation_warning"]
    assert len(warns) == 1 and warns[0]["component"] == "clef"
    assert "quota" in warns[0]["message"].lower()
    assert job.degradation_warnings[0]["count"] == 1


def test_finalize_clef_usage_writes_timings_once(monkeypatch, capsys):
    _install(monkeypatch, FakeCF(_ok(_answers(), input_tokens=500)))
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    job.timings = {}
    gauges = {}
    job.identity_diagnostics = types.SimpleNamespace(
        gauge=lambda k, v: gauges.__setitem__(k, v))
    job._clef_session = _session()
    job._clef_session.read_bumpers([_jpeg_b64()] * 2, RED, BLUE, ["blue"] * 2)
    job._finalize_clef_usage()
    job._finalize_clef_usage()  # idempotent
    t = job.timings
    assert t["clef.requests"] == 2 and t["clef.input_tokens"] == 1000
    assert t["clef.quota_errors"] == 0 and t["clef.errors"] == 0
    assert t["clef.cost_cents"] == pytest.approx(1000 * 0.038 / 1e6 * 100)
    assert "clef.latency_p50_s" in t
    assert all(isinstance(v, (int, float)) for v in t.values())  # survives round()
    assert gauges["clef_usage"]["input_tokens"] == 1000
    printed = [l for l in capsys.readouterr().out.splitlines() if "[clef]" in l]
    assert len(printed) == 1 and "2 reads" in printed[0]


def test_finalize_clef_usage_noop_for_other_readers():
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    job.timings = {}
    job._finalize_clef_usage()
    assert job.timings == {}


def test_diagnostic_row_carries_clef_probs_only_when_present(tmp_path):
    from web.diagnostic import DiagnosticRecorder
    path = tmp_path / "d.jsonl"
    rec = DiagnosticRecorder(path)
    common = dict(frame=1, robot_id=2, crop_b64="x", legibility=0.0, raw_read="4786",
                  matched_team="4786", alliance_read="blue", confidence=0.9,
                  vote_weight=1.0, surface="bumper")
    rec.record_vlm_read(**common)
    rec.record_vlm_read(**common, clef_probs={"4786": 0.9, "none": 0.1})
    rec.close()
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert "clef_probs" not in rows[0]
    assert rows[1]["clef_probs"] == {"4786": 0.9, "none": 0.1}
