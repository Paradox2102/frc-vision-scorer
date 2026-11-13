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


def test_vllm_ready_check_runs_without_phase_exclusive():
    # Resident reader (big card): the job still verifies/starts its vlm_model,
    # but the phase-exclusive "reclaims the card" meaning stays off.
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    job.config = JobConfig({"team_reader": "vllm", "vllm_phase_exclusive": False})
    job.team_reader = "vllm"
    assert job._needs_vllm_server() is False
    assert job._needs_vllm_ready() is True
    for reader in ("clef", "jev", "local"):
        job.team_reader = reader
        assert job._needs_vllm_ready() is False


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


# =========================================================================== #
# clef_multiframe: multi-image requests (newest crop + buffered earlier crops)
# =========================================================================== #

def _mf_answers(team_probs=None, same=0.93, **kw):
    a = _answers(team_probs, **kw)
    if same is not None:
        a["same_robot"] = {"type": "noul", "noul": same}
    return a


def _crops(n):
    return [_jpeg_b64(80 + 8 * i, 40, (200, 30 + 20 * i, 30)) for i in range(n)]


def _prep(b64):
    return clef_reader.clef_image(base64.b64decode(b64))


def test_multi_payload_images_state_and_same_robot_question():
    imgs = [_prep(c)[0] for c in _crops(3)]
    p = clef_reader.build_read_payload_multi("clef-flash", imgs, RED, BLUE, "blue",
                                             seed="newest", span_s=1.37)
    assert p["model"] == "clef-flash"
    assert len(p["images"]) == 3
    assert all(u.startswith("data:image/jpeg;base64,") for u in p["images"])
    assert p["images"][0] == clef_reader._data_url(imgs[0])           # newest first
    assert p["state"].startswith(
        "3 images: the same robot's bumper at different moments over ~1.4 s, newest first.")
    assert "red alliance 2992, 4256, 1477" in p["state"]
    assert "BLUE bumper pixels" in p["state"]
    q = p["questions"]
    assert q["same_robot"]["type"] == "noul"
    assert q["same_robot"]["instructions"] == "Do all images show the same robot?"
    assert set(q["same_robot"]["criteria"]) == {"true", "false"}
    # The other questions (and the option order for the same seed) match a
    # single-crop read of the newest crop.
    single = clef_reader.build_read_payload("clef-flash", imgs[0], RED, BLUE, "blue",
                                            seed="newest")
    assert {k: v for k, v in q.items() if k != "same_robot"} == single["questions"]
    assert list(_team_criteria(p)) == list(_team_criteria(single))


def test_multi_payload_rejects_more_than_api_max_images():
    imgs = [_prep(c)[0] for c in _crops(5)]
    with pytest.raises(ValueError):
        clef_reader.build_read_payload_multi("clef-flash", imgs, RED, BLUE)


def test_list_item_is_one_request_with_n_images(monkeypatch):
    fake = _install(monkeypatch, FakeCF(_ok(_mf_answers({"4786": 0.97, "none": 0.03}))))
    s = _session()
    crops = _crops(3)
    frames = [{"frame": 180, "cam": "main"}, {"frame": 150, "cam": "corner_left"},
              {"frame": 90, "cam": "main"}]
    out = s.read_bumpers([crops], RED, BLUE, ["blue"],
                         item_meta=[{"span_s": 1.5, "frames": frames}])
    assert len(fake.calls) == 1
    p = fake.calls[0]["payload"]
    assert len(p["images"]) == 3
    assert p["images"] == [clef_reader._data_url(_prep(c)[0]) for c in crops]
    assert p["state"].startswith("3 images: the same robot's bumper at different moments "
                                 "over ~1.5 s, newest first.")
    assert "same_robot" in p["questions"]
    r = out[0]
    assert r["team_number"] == "4786" and r["error"] is None
    assert r["clef_same_robot"] == pytest.approx(0.93)
    assert r["clef_frames"] == frames
    # Result shape: the single-read keys plus the two multi-frame keys.
    single_keys = set(clef_reader.map_read_answers(
        _answers(), _ALLIANCE, "blue", 0.5))
    assert set(r) == single_keys | {"clef_same_robot", "clef_frames"}
    st = s.stats()
    assert st["mf_reads"] == 1 and st["mf_images"] == 3 and st["ok"] == 1
    assert st["inconsistent"] == 0


def test_list_item_capped_at_four_images(monkeypatch):
    fake = _install(monkeypatch, FakeCF(_ok(_mf_answers())))
    s = _session()
    s.read_bumpers([_crops(6)], RED, BLUE, ["blue"])
    assert len(fake.calls[0]["payload"]["images"]) == clef_reader.CLEF_MAX_IMAGES == 4
    assert s.stats()["mf_images"] == 4


def test_single_element_list_reads_exactly_like_its_string(monkeypatch):
    c = _jpeg_b64()
    fake = _install(monkeypatch, FakeCF(_ok(_answers())))
    a = _session().read_bumpers([c], RED, BLUE, ["blue"])
    b = _session().read_bumpers([[c]], RED, BLUE, ["blue"])
    assert fake.calls[0]["payload"] == fake.calls[1]["payload"]
    assert a == b and "clef_same_robot" not in b[0]


def test_string_items_unchanged_next_to_list_items(monkeypatch):
    c1, rest = _jpeg_b64(), _crops(2)
    fake = _install(monkeypatch, FakeCF(lambda p: _ok(
        _mf_answers() if len(p["images"]) > 1 else _answers())))
    s = _session(concurrency=1)
    s.read_bumpers([c1], RED, BLUE, ["blue"])
    base = fake.calls[0]["payload"]
    out = s.read_bumpers([c1, rest], RED, BLUE, ["blue", "blue"],
                         item_meta=[None, {"span_s": 0.5, "frames": None}])
    single = [c["payload"] for c in fake.calls[1:] if len(c["payload"]["images"]) == 1]
    multi = [c["payload"] for c in fake.calls[1:] if len(c["payload"]["images"]) == 2]
    assert single == [base] and len(multi) == 1
    assert "clef_same_robot" not in out[0] and "clef_same_robot" in out[1]
    assert "clef_frames" not in out[1]          # no frames in its meta


def test_multi_token_estimate_sums_every_image(monkeypatch):
    result = {"model": "clef-flash", "answers": _mf_answers()}   # no usage block
    fake = _install(monkeypatch, FakeCF((200, json.dumps({"result": result, "success": True}))))
    s = _session()
    crops = _crops(3)
    s.read_bumpers([crops], RED, BLUE, ["blue"], item_meta=[{"span_s": 1.0}])
    p = fake.calls[0]["payload"]
    whs = [_prep(c)[1:] for c in crops]
    text = len(json.dumps(p["state"])) + len(json.dumps(p["questions"]))
    expected = sum(clef_reader.clef_image_tokens(w, h) for w, h in whs) + text // 4
    st = s.stats()
    assert st["input_tokens"] == expected and st["estimated_token_responses"] == 1
    assert clef_reader.estimate_tokens(p, whs) == expected
    # The single-image form is unchanged.
    assert clef_reader.estimate_tokens(p, whs[0]) == (
        clef_reader.clef_image_tokens(*whs[0]) + text // 4)


def test_same_robot_gate_marks_read_inconsistent(monkeypatch):
    _install(monkeypatch, FakeCF(_ok(_mf_answers({"4786": 0.97, "none": 0.03}, same=0.2))))
    s = _session()
    s.multiframe = True
    r = s.read_bumpers([_crops(4)], RED, BLUE, ["blue"])[0]
    assert r["team_number"] is None and r["confidence"] == 0.0
    assert r["error"] == clef_reader.INCONSISTENT_ERROR == "tracklet_inconsistent"
    assert r["clef_same_robot"] == pytest.approx(0.2)
    assert r["raw_read"] == "4786"              # what it read is kept for diagnostics
    assert r["alliance"] == "blue"
    st = s.stats()
    assert st["inconsistent"] == 1 and st["mf_reads"] == 1 and st["errors"] == 0
    line = s.summary_line()
    assert "1 multi-frame (4 images), 1 inconsistent" in line


def test_same_robot_threshold_is_configurable(monkeypatch):
    _install(monkeypatch, FakeCF(_ok(_mf_answers(same=0.2))))
    r = _session(same_robot_min=0.1).read_bumpers([_crops(2)], RED, BLUE, ["blue"])[0]
    assert r["team_number"] == "4786" and r["error"] is None


def test_missing_same_robot_answer_does_not_gate(monkeypatch):
    _install(monkeypatch, FakeCF(_ok(_mf_answers(same=None))))
    r = _session().read_bumpers([_crops(2)], RED, BLUE, ["blue"])[0]
    assert r["team_number"] == "4786" and r["clef_same_robot"] is None


def test_multi_read_when_latched_makes_no_http(monkeypatch, no_network):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")
    s = _session()
    frames = [{"frame": 2, "cam": "main"}, {"frame": 1, "cam": "main"}]
    r = s.read_bumpers([_crops(2)], RED, BLUE, ["blue"], item_meta=[{"frames": frames}])[0]
    assert r["team_number"] is None and r["error"] and r["clef_frames"] == frames
    assert s.stats()["skipped"] == 1 and s.stats()["mf_reads"] == 0


def test_summary_line_unchanged_without_multiframe(monkeypatch):
    _install(monkeypatch, FakeCF(_ok(_answers())))
    s = _session()
    s.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    assert "multi-frame" not in s.summary_line()


def test_job_config_multiframe_knobs():
    cfg = JobConfig({"team_reader": "clef", "clef_multiframe": True})
    assert cfg.clef_multiframe is True
    assert cfg.clef_mf_frames == 4 and cfg.clef_mf_window_s == 3.0
    assert cfg.clef_mf_same_robot_min == 0.5
    d = JobConfig()
    assert d.clef_multiframe is False
    for k in ("clef_multiframe", "clef_mf_frames", "clef_mf_window_s", "clef_mf_same_robot_min"):
        assert k not in d.non_defaults()
    for bad in ({"clef_mf_frames": 5}, {"clef_mf_frames": 1},
                {"clef_mf_same_robot_min": 1.5}, {"clef_mf_window_s": 0.0}):
        with pytest.raises(ValueError):
            JobConfig(bad)


def test_job_clef_session_takes_multiframe_knobs():
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    job.team_reader = "clef"
    job.config = JobConfig({"team_reader": "clef", "clef_multiframe": True,
                            "clef_mf_same_robot_min": 0.7})
    s = job._get_clef_session()
    assert s.multiframe is True and s.same_robot_min == 0.7
    job2 = app_module.InferenceJob.__new__(app_module.InferenceJob)
    job2.team_reader = "clef"
    job2.config = JobConfig({"team_reader": "clef"})
    assert job2._get_clef_session().multiframe is False


# --------------------------------------------------------------------------- #
# Read-pass wiring (InferenceJob._read_fused_bumper_numbers_impl)
# --------------------------------------------------------------------------- #

import numpy as np                                        # noqa: E402
from unittest.mock import MagicMock                       # noqa: E402
from web.diagnostic import DiagnosticRecorder             # noqa: E402
from web.field.identity import IdentityRegistry, RobotIdentity   # noqa: E402
from web.field.identity_diagnostics import DiagnosticEvents      # noqa: E402
from web.field.identity_confidence import ConfidenceConfig       # noqa: E402

_BOX = (200, 100, 340, 220)


def _view(f):
    """A distinct synthetic 'view frame' per source frame: red band + white
    blocks (some legibility), varying with f so crops differ."""
    rng = np.random.default_rng(f)
    img = rng.integers(0, 60, (400, 600, 3), dtype=np.uint8)
    img[160:220, 200:340] = (30, 30, 200)                  # BGR red band
    img[175:205, 220 + (f % 7) * 4:240 + (f % 7) * 4] = 255
    return img


def _mf_read_job(tmp_path, multiframe=True):
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    cfg = {"team_reader": "clef"}
    if multiframe:
        cfg["clef_multiframe"] = True
    job.config = JobConfig(cfg)
    job.team_reader = "clef"
    job._camera_centers_xy = {}
    job.match_teams = ALL6
    job.match_blue_teams = BLUE
    job.match_red_teams = RED
    job.vlm_status = {}
    job._vlm_task = None
    job._claude_vision_calls = 0
    job._current_frame = 0
    job._evidence_frame = 0
    job.video_fps = 60.0
    job.field_tracker = types.SimpleNamespace(last_det_to_slot={0: 1}, binding_events=[],
                                              _id_to_slot={})
    reg = IdentityRegistry()
    reg.identities[7] = RobotIdentity(identity_id=7, alliance="blue", active_slot=1)
    reg._next_id = 8
    job.identity_registry = reg
    job.robot_tracks = {}
    job._last_bindings = {}
    job._slot_source = {}
    job._slot_crops = {}
    job.scoring_events = []
    job.user_seed_zones_raw = []
    job._occ = {}
    job._occ_by_cam = {}
    job.confidence_config = ConfidenceConfig()
    job.co_occlusion_tracker = types.SimpleNamespace(get_active_events=lambda: [])
    job.apply_gallery_matches = lambda crops: None
    job.identity_diagnostics = MagicMock()
    tmp_path.mkdir(parents=True, exist_ok=True)
    job.diagnostic = DiagnosticRecorder(tmp_path / "diag.jsonl", ws_send=None)
    job._clef_session = _session()
    return job


def _mf_fused(job, f):
    frame = _view(f)

    def _enc(_f=frame):
        return job._bumper_crop_b64(_f, _BOX, enforce_floor=False)[0]
    lazy = app_module._LazyCrop(
        _enc, src=(frame, _BOX, False) if job._clef_mf_enabled() else None)
    return [types.SimpleNamespace(
        evidence=[("main", lazy, 1.0, (4.0, 3.0))], track_id=5,
        detection=types.SimpleNamespace(pos=(4.0, 3.0)),
        relink_evidence=types.SimpleNamespace(type="continued"))], frame


def _run_frames(job, frames):
    fused = frame = None
    for f in frames:
        fused, frame = _mf_fused(job, f)
        job._clef_mf_push(fused, f)
    return fused, frame


def _read(job, fused, crop_frame):
    asyncio.run(job.read_fused_bumper_numbers(
        fused, {7: 1.0}, det_to_slot={0: 1}, crop_frame=crop_frame,
        slot_identity={1: 7}))


def _diag_rows(job, tmp_path):
    job.diagnostic.flush()
    return [json.loads(l) for l in (tmp_path / "diag.jsonl").read_text().splitlines()
            if json.loads(l)["type"] == "vlm_read"]


def test_knob_off_clef_payload_is_byte_identical(monkeypatch, tmp_path):
    """Knob off (and knob on with nothing usable buffered) sends today's
    single-crop request byte-for-byte, through today's call signature."""
    fake = _install(monkeypatch, FakeCF(_ok(_answers())))
    seen = []
    real = app_module._vision_read_bumpers

    def _spy(*a, **k):
        seen.append(sorted(k))
        return real(*a, **k)
    monkeypatch.setattr(app_module, "_vision_read_bumpers", _spy)

    off = _mf_read_job(tmp_path, multiframe=False)
    fused, frame = _mf_fused(off, 180)
    assert fused[0].evidence[0][1].src is None          # no frame pinned when off
    _read(off, fused, 180)
    newest = off._bumper_crop_b64(frame, _BOX, enforce_floor=False)[0]
    img, _w, _h = _prep(newest)
    expected = clef_reader.build_read_payload(
        "clef-flash", img, RED, BLUE, "blue",
        options=clef_reader.roster_options(RED, BLUE), seed=newest)
    assert fake.calls[0]["payload"] == expected
    assert seen[0] == ["clef_session", "expected_alliances", "full_roster", "provider"]

    # Knob on, but only the newest frame buffered -> single-crop fallback.
    on = _mf_read_job(tmp_path / "x", multiframe=True)
    fused_on, _ = _run_frames(on, [180])
    _read(on, fused_on, 180)
    assert fake.calls[1]["payload"] == expected
    assert seen[1] == seen[0]
    # Knob on, buffer never built (no push at all) -> same.
    on2 = _mf_read_job(tmp_path / "y", multiframe=True)
    fused2, _ = _mf_fused(on2, 180)
    _read(on2, fused2, 180)
    assert fake.calls[2]["payload"] == expected
    assert on._mf_counts["single"] == 1 and on._mf_counts["multi"] == 0


def test_multiframe_read_sends_newest_plus_buffered_frames(monkeypatch, tmp_path):
    fake = _install(monkeypatch, FakeCF(_ok(_mf_answers({"4786": 0.97, "none": 0.03}))))
    job = _mf_read_job(tmp_path)
    fused, frame = _run_frames(job, range(0, 181, 6))
    newest = job._bumper_crop_b64(frame, _BOX, enforce_floor=False)[0]
    _read(job, fused, 180)

    assert len(fake.calls) == 1
    p = fake.calls[0]["payload"]
    assert len(p["images"]) == 4
    assert p["images"][0] == clef_reader._data_url(_prep(newest)[0])
    assert p["state"].startswith("4 images: the same robot's bumper at different moments over ~")
    assert "newest first" in p["state"] and "same_robot" in p["questions"]
    single = clef_reader.build_read_payload("clef-flash", _prep(newest)[0], RED, BLUE,
                                            "blue", seed=newest)
    assert list(_team_criteria(p)) == list(_team_criteria(single))

    # Every single-crop consumer got the NEWEST crop; the row is stamped with
    # the newest crop's frame and records which frames/cams were sent.
    rows = _diag_rows(job, tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["frame"] == 180 and row["robot_id"] == 7 and row["crop_b64"] == newest
    assert row["matched_team"] == "4786"
    assert row["clef_same_robot"] == pytest.approx(0.93)
    fr = row["clef_frames"]
    assert len(fr) == 4 and fr[0] == {"frame": 180, "cam": "main"}
    assert [f["frame"] for f in fr] == sorted((f["frame"] for f in fr), reverse=True)
    assert job.last_ocr_crops[0]["crop"] == newest
    slot_crops = next(iter(job._slot_crops.values()))
    assert slot_crops[0]["crop_b64"] == newest

    # ONE vote for the whole multi-image read.
    robot = next(iter(job.robot_tracks.values()))
    assert robot.team_ocr_votes == {"4786": 1.0}
    st = job._clef_session.stats()
    assert st["mf_reads"] == 1 and st["mf_images"] == 4 and st["requests"] == 1

    # The frames used are marked sent: the next read uses fresh ones.
    used = {f["frame"] for f in fr}
    assert job._crop_buffer.sent_frames(7) >= used
    fused2, _ = _run_frames(job, [186])
    job.identity_registry.identities[7].backoff_until = -999.0
    _read(job, fused2, 186)
    fr2 = _diag_rows(job, tmp_path)[-1]["clef_frames"]
    assert fr2[0]["frame"] == 186 and not ({f["frame"] for f in fr2} & used)


def test_tracklet_inconsistent_read_casts_no_vote_and_raises_alarm(monkeypatch, tmp_path):
    _install(monkeypatch, FakeCF(_ok(_mf_answers({"4786": 0.97, "none": 0.03}, same=0.1))))
    job = _mf_read_job(tmp_path)
    fused, _ = _run_frames(job, range(0, 181, 6))
    _read(job, fused, 180)
    robot = next(iter(job.robot_tracks.values()))
    assert not robot.team_ocr_votes
    ident = job.identity_registry.identities[7]
    assert ident.vlm_fail_count == 1                       # a failed read -> backoff
    calls = [c for c in job.identity_diagnostics.record.call_args_list
             if c.args and c.args[0] == DiagnosticEvents.VLM_TRACKLET_INCONSISTENT]
    assert len(calls) == 1
    kw = calls[0].kwargs
    assert kw["identity_id"] == 7 and kw["p_same_robot"] == pytest.approx(0.1)
    assert len(kw["frames"]) == 4 and kw["frames"][0]["frame"] == 180
    job.identity_diagnostics.increment.assert_any_call("vlm.tracklet_inconsistent")
    assert job._crop_buffer.entries(7) == []               # buffer flushed
    assert job._crop_buffer.stats()["flushes"]["inconsistent"] == 1
    row = _diag_rows(job, tmp_path)[0]
    assert row["matched_team"] is None and row["clef_same_robot"] == pytest.approx(0.1)
    assert job._clef_session.stats()["inconsistent"] == 1


def test_read_falls_back_to_single_crop_when_identity_moved_slots(monkeypatch, tmp_path):
    """The background pass may start after a re-bind: if the buffer does not
    hold this identity on the crop's slot at the crop frame, read single."""
    fake = _install(monkeypatch, FakeCF(_ok(_answers())))
    job = _mf_read_job(tmp_path)
    fused, _ = _run_frames(job, range(0, 181, 6))
    job.identity_registry.identities[7].active_slot = 2      # re-bound before the pass ran
    asyncio.run(job.read_fused_bumper_numbers(
        fused, {7: 1.0}, det_to_slot={0: 2}, crop_frame=180, slot_identity={2: 7}))
    assert len(fake.calls[0]["payload"]["images"]) == 1
    assert "clef_frames" not in _diag_rows(job, tmp_path)[0]


def test_contradiction_and_roster_clear_flush_the_buffer(tmp_path):
    job = _mf_read_job(tmp_path)
    _run_frames(job, range(0, 31, 6))
    assert job._crop_buffer.entries(7)
    job._mf_flush(7, "contradiction")
    assert job._crop_buffer.entries(7) == []
    assert job._crop_buffer.stats()["flushes"] == {"contradiction": 1}
    # The real call sites route through _mf_flush.
    import inspect
    src = inspect.getsource(app_module.InferenceJob)
    for reason in ("contradiction", "roster_clear", "co_occlusion", "inconsistent"):
        assert f'"{reason}")' in src


def test_co_occlusion_resolution_flushes_both_identities(monkeypatch, tmp_path):
    job = _mf_read_job(tmp_path)
    reg = job.identity_registry
    reg.identities[8] = RobotIdentity(identity_id=8, alliance="blue", active_slot=2,
                                      team_number="6800", is_ambiguous=True)
    reg.identities[7].team_number = "4786"
    reg.identities[7].is_ambiguous = True
    for f in range(0, 31, 6):
        fa, _ = _mf_fused(job, f)
        fb, _ = _mf_fused(job, f + 1000)
        fb[0].track_id = 6
        job.field_tracker.last_det_to_slot = {0: 1, 1: 2}
        job._clef_mf_push(fa + fb, f)
    assert job._crop_buffer.entries(7) and job._crop_buffer.entries(8)
    event = types.SimpleNamespace(event_id=1, status="pending_vlm", identity_ids={7, 8},
                                  alliance="blue")
    resolved = []
    job.co_occlusion_tracker = types.SimpleNamespace(
        get_active_events=lambda: [event],
        resolve_event=lambda eid, mapping: resolved.append(mapping))
    job.deferred_attributions = []
    # Reads come back swapped: slot 1 shows 6800, slot 2 shows 4786.
    monkeypatch.setattr(app_module, "_vision_read_bumpers", lambda crops, *a, **k: [
        {"team_number": "6800", "confidence": 0.95, "raw_read": "6800"},
        {"team_number": "4786", "confidence": 0.95, "raw_read": "4786"}][:len(crops)])
    fa, _ = _mf_fused(job, 36)
    fb, _ = _mf_fused(job, 1036)
    fb[0].track_id = 6
    asyncio.run(job.read_fused_bumper_numbers(
        fa + fb, {7: 1.0, 8: 1.0}, det_to_slot={0: 1, 1: 2}, crop_frame=36,
        slot_identity={1: 7, 2: 8}))
    assert resolved == [{8: 1, 7: 2}]
    assert job._crop_buffer.entries(7) == [] and job._crop_buffer.entries(8) == []
    assert job._crop_buffer.stats()["flushes"]["co_occlusion"] == 2


def test_encode_plan_drops_failed_older_crop_and_collapses():
    good = np.full((30, 80, 3), 120, np.uint8)
    empty = np.zeros((0, 0, 3), np.uint8)
    items = ["NEWEST", ["NEWEST2", good, empty]]
    metas = [None, {"span_s": 9.9, "frames": [{"frame": 100, "cam": "main"},
                                              {"frame": 70, "cam": "main"},
                                              {"frame": 40, "cam": "main"}]}]
    out, m = app_module._clef_mf_encode_plan(items, metas, 60.0)
    assert out[0] == "NEWEST" and m[0] is None
    assert out[1][0] == "NEWEST2" and len(out[1]) == 2
    assert out[1][1] == app_module._encode_band_crop_b64(good)
    assert m[1]["frames"] == metas[1]["frames"][:2] and m[1]["span_s"] == 0.5
    out2, m2 = app_module._clef_mf_encode_plan([["ONLY", empty]],
                                               [{"frames": [{"frame": 1, "cam": "main"},
                                                            {"frame": 0, "cam": "main"}]}], 60.0)
    assert out2 == ["ONLY"] and m2 == [None]


def test_buffered_crop_encodes_like_the_lazy_newest_crop():
    frame = _view(42)
    band, leg = app_module._band_crop_for_buffer((frame, _BOX, False))
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    b64, score = job._bumper_crop_b64(frame, _BOX, enforce_floor=False)
    assert app_module._encode_band_crop_b64(band) == b64
    assert leg == pytest.approx(score)


def test_finalize_reports_multiframe_counters(monkeypatch, capsys):
    _install(monkeypatch, FakeCF(_ok(_mf_answers(same=0.1), input_tokens=900)))
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    job.team_reader = "clef"
    job.config = JobConfig({"team_reader": "clef", "clef_multiframe": True})
    job.timings = {}
    gauges = {}
    job.identity_diagnostics = types.SimpleNamespace(
        gauge=lambda k, v: gauges.__setitem__(k, v))
    job._clef_session = _session()
    job._clef_session.multiframe = True
    job._clef_session.read_bumpers([_crops(3)], RED, BLUE, ["blue"])
    job._mf_counts = {"multi": 1, "single": 2}
    job._crop_buffer = app_module.CropBuffer(fps=60.0)
    job._finalize_clef_usage()
    t = job.timings
    assert t["clef.mf_reads"] == 1 and t["clef.mf_images"] == 3
    assert t["clef.inconsistent"] == 1 and t["clef.mf_single_fallbacks"] == 2
    assert all(isinstance(v, (int, float)) for v in t.values())
    assert gauges["clef_mf_buffer"]["multi"] == 1
    printed = [l for l in capsys.readouterr().out.splitlines() if "[clef]" in l]
    assert "1 multi-frame (3 images), 1 inconsistent" in printed[0]


def test_finalize_omits_multiframe_timings_when_knob_off(monkeypatch):
    _install(monkeypatch, FakeCF(_ok(_answers())))
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    job.team_reader = "clef"
    job.config = JobConfig({"team_reader": "clef"})
    job.timings = {}
    job._clef_session = _session()
    job._clef_session.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    job._finalize_clef_usage()
    assert not any(k.startswith("clef.mf_") or k == "clef.inconsistent" for k in job.timings)


def test_diagnostic_row_carries_multiframe_fields_only_when_present(tmp_path):
    from web.diagnostic import DiagnosticRecorder
    path = tmp_path / "d.jsonl"
    rec = DiagnosticRecorder(path)
    common = dict(frame=180, robot_id=2, crop_b64="x", legibility=0.0, raw_read="4786",
                  matched_team="4786", alliance_read="blue", confidence=0.9,
                  vote_weight=1.0, surface="bumper")
    rec.record_vlm_read(**common)
    rec.record_vlm_read(**common, clef_probs={"4786": 0.9},
                        clef_frames=[{"frame": 180, "cam": "main"},
                                     {"frame": 150, "cam": "corner_left"}],
                        clef_same_robot=0.88)
    rec.close()
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert "clef_frames" not in rows[0] and "clef_same_robot" not in rows[0]
    assert rows[1]["clef_frames"] == [{"frame": 180, "cam": "main"},
                                      {"frame": 150, "cam": "corner_left"}]
    assert rows[1]["clef_same_robot"] == 0.88
    assert rows[1]["frame"] == 180 and rows[1]["crop_b64"] == "x"
