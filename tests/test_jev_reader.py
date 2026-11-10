"""team_reader="jev": TypeSafe Jev decision-model bumper reader.

Offline: every network path is a tripwire; responses are scripted through
clef_reader.http_post (the one network call JevSession shares with Clef).
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import socket
import types
import urllib.request
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

import web.app as app_module
from web import clef_reader, jev_reader
from web.diagnostic import DiagnosticRecorder
from web.field.identity import IdentityRegistry, RobotIdentity
from web.field.identity_confidence import (
    ConfidenceConfig, is_tentative_read, read_can_lock, read_vote_base)
from web.field.identity_resolve import apply_evidence_floor
from web.jev_reader import JevSession
from web.job_config import JobConfig

RED = ["2992", "4256", "1477"]
BLUE = ["4786", "6800", "870"]
ALL6 = RED + BLUE
TOKEN = "jev-SECRET-do-not-log"
ESC = ("not_robot", "unreadable", "printed", "other")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    attempts = []

    def _tripwire(*a, **k):
        attempts.append(("network", a[:1]))
        raise AssertionError("network call attempted in an offline test")

    monkeypatch.setattr(urllib.request, "urlopen", _tripwire)
    monkeypatch.setattr(socket, "create_connection", _tripwire)
    monkeypatch.setattr(clef_reader, "http_post", _tripwire)
    monkeypatch.setenv("JEV_TYPESAFE_API_TOKEN", TOKEN)
    for var in ("CLEF_MIN_TEAM_PROB", "CLEF_CONCURRENCY", "CLEF_RETRIES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(app_module, "_jev_default", None)
    yield attempts
    assert not attempts, f"offline test reached the network: {attempts}"


def _answers(team_probs):
    top = max(team_probs, key=team_probs.get)
    return {"team": {"type": "choice", "choice": top, "probabilities": team_probs,
                     "confidence": team_probs[top]}}


def _ok(answers, input_tokens=1500):
    return 200, json.dumps({"model": "jev-1.14.0", "answers": answers,
                            "usage": {"input_tokens": input_tokens, "output_tokens": 100}})


class FakeJev:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, data, headers, timeout):
        payload = json.loads(data.decode("utf-8"))
        self.calls.append({"url": url, "payload": payload, "headers": dict(headers)})
        resp = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(resp, Exception):
            raise resp
        return resp


def _install(monkeypatch, fake):
    monkeypatch.setattr(clef_reader, "http_post", fake)
    return fake


def _session(**kw):
    kw.setdefault("sleep_fn", lambda _s: None)
    kw.setdefault("log_fn", lambda _m: None)
    return JevSession(**kw)


def _jpeg(w=80, h=40, color=(200, 30, 30), fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format=fmt, quality=90)
    return buf.getvalue()


def _jpeg_b64(w=80, h=40) -> str:
    return base64.b64encode(_jpeg(w, h)).decode()


def _roster_alliance():
    return {**{t: "red" for t in RED}, **{t: "blue" for t in BLUE}}


# --------------------------------------------------------------------------- #
# Image prep bounds
# --------------------------------------------------------------------------- #

def _size(b: bytes):
    return Image.open(io.BytesIO(b)).size


@pytest.mark.parametrize("w,h", [(80, 40), (30, 30), (2000, 100), (4000, 3000),
                                 (300, 290), (1600, 1600), (10, 1200)])
def test_jev_image_lands_inside_api_bounds(w, h):
    img, ow, oh, _pad = jev_reader.jev_image(_jpeg(w, h))
    assert _size(img) == (ow, oh)
    assert jev_reader.JEV_MIN_SIDE <= ow <= jev_reader.JEV_MAX_SIDE
    assert jev_reader.JEV_MIN_SIDE <= oh <= jev_reader.JEV_MAX_SIDE
    assert jev_reader.JEV_MIN_PIXELS <= ow * oh <= jev_reader.JEV_MAX_PIXELS


def test_jev_image_letterboxes_extreme_aspect():
    img, w, h, padded = jev_reader.jev_image(_jpeg(3000, 100))
    assert padded and (w, h) == (1568, 280)
    # Pad is neutral gray (top-left corner), image content in the middle.
    px = Image.open(io.BytesIO(img)).convert("RGB").getpixel((5, 2))
    assert all(abs(c - 128) < 12 for c in px)


def test_jev_image_in_range_jpeg_passes_through():
    raw = _jpeg(400, 300)
    img, w, h, padded = jev_reader.jev_image(raw)
    assert img == raw and (w, h) == (400, 300) and not padded


def test_jev_image_reencodes_png():
    img, w, h, _ = jev_reader.jev_image(_jpeg(400, 300, fmt="PNG"))
    assert Image.open(io.BytesIO(img)).format == "JPEG" and (w, h) == (400, 300)


# --------------------------------------------------------------------------- #
# Payload shape
# --------------------------------------------------------------------------- #

def test_payload_shape_ref_model_header_and_escapes(monkeypatch):
    fake = _install(monkeypatch, FakeJev(_ok(_answers({"4786": 0.9, "unreadable": 0.1}))))
    _session().read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    call = fake.calls[0]
    assert call["url"] == "https://api.typesafe.ai/v1/systemone"
    h = call["headers"]
    assert h["Authorization"] == f"Bearer {TOKEN}"
    assert h["TypeSafe-Danger-Zone-Images"] == "true"
    p = call["payload"]
    assert p["model"] == "jev-1.14.0"                       # pinned
    ref = jev_reader.JEV_ASSET_REF
    assert p["state"]["crop"] == ref                        # asset ref is referenced
    asset = p["assets"][ref]
    assert asset["type"] == "image" and asset["format"] == "jpg"
    assert not asset["data_base64"].startswith("data:")
    w, h_ = _size(base64.b64decode(asset["data_base64"]))
    assert min(w, h_) >= 280
    assert set(p["questions"]) == {"team"}                  # no subject question
    crit = p["questions"]["team"]["criteria"]
    # JEV_ALLIANCE_OPTIONS is off: even with a blue colour hint, all 6 roster
    # teams + the four escapes, under the neutral roster line.
    assert set(crit) == set(RED) | set(BLUE) | set(ESC)
    assert crit["4786"] == "Team 4786 (blue alliance)"
    assert "Teams in this match:" in p["state"]["context"]
    assert "Teams on the blue alliance" not in p["state"]["context"]
    assert "`crop`" in p["questions"]["team"]["instructions"]


def test_payload_unknown_alliance_offers_full_roster(monkeypatch):
    fake = _install(monkeypatch, FakeJev(_ok(_answers({"4786": 0.9, "unreadable": 0.1}))))
    _session().read_bumpers([_jpeg_b64()], RED, BLUE, ["unknown"], extra_teams=["9999"])
    p = fake.calls[0]["payload"]
    assert set(p["questions"]["team"]["criteria"]) == set(ALL6) | {"9999"} | set(ESC)
    assert "Teams in this match: 2992, 4256, 1477, 4786, 6800, 870, 9999." in p["state"]["context"]


def test_payload_option_order_is_deterministic_per_crop():
    opts = jev_reader.jev_options(RED, BLUE, (), "red")
    a = jev_reader.build_jev_payload("jev-1.14.0", b"x", opts, "red", seed="crop-a")
    b = jev_reader.build_jev_payload("jev-1.14.0", b"x", opts, "red", seed="crop-a")
    assert list(a["questions"]["team"]["criteria"]) == list(b["questions"]["team"]["criteria"])


def test_unknown_jev_model_rejected():
    with pytest.raises(ValueError):
        JevSession("jev-9")


# --------------------------------------------------------------------------- #
# Answer mapping
# --------------------------------------------------------------------------- #

def _map(probs, expected="blue", tentative_min_p=0.17):
    return jev_reader.map_jev_answers(_answers(probs), _roster_alliance(), expected,
                                      0.5, tentative_min_p)


def test_map_confident_team_is_a_normal_read():
    r = _map({"4786": 0.81, "6800": 0.09, "870": 0.0, "unreadable": 0.1,
              "not_robot": 0.0, "printed": 0.0, "other": 0.0})
    assert r["team_number"] == "4786" and r["confidence"] == 0.81
    assert r["tentative"] is False and r["alliance"] == "blue"
    assert r["surface"] == "bumper" and r["raw_read"] == "4786"
    assert r["runner_up"] == "6800" and r["runner_up_p"] == 0.09
    assert r["jev_probs"]["unreadable"] == 0.1


def test_map_unreadable_top_with_team_above_floor_is_tentative():
    r = _map({"6800": 0.24, "4786": 0.05, "870": 0.01, "unreadable": 0.68,
              "not_robot": 0.0, "printed": 0.0, "other": 0.02})
    assert r["team_number"] == "6800" and r["tentative"] is True
    assert r["confidence"] == 0.24
    assert r["runner_up"] == "4786" and r["runner_up_p"] == 0.05
    assert r["raw_read"] == "unreadable"


def test_map_unreadable_top_below_floor_is_no_team():
    r = _map({"6800": 0.16, "4786": 0.05, "unreadable": 0.79})
    assert r["team_number"] is None and r["tentative"] is False
    assert r["confidence"] == 0.0 and "runner_up" not in r


def test_map_tentative_disabled_by_zero_floor():
    r = _map({"6800": 0.4, "unreadable": 0.6}, tentative_min_p=0.0)
    assert r["team_number"] is None


@pytest.mark.parametrize("escape", ["not_robot", "printed", "other"])
def test_map_other_escape_tops_name_no_team(escape):
    probs = {"6800": 0.3, "4786": 0.05, "unreadable": 0.05, escape: 0.6}
    r = _map(probs)
    assert r["team_number"] is None and r["tentative"] is False
    if escape == "printed":
        assert r["surface"] == "printed"
    if escape == "not_robot":
        assert r["surface"] == "unknown"


def test_map_zero_probability_runner_up_is_omitted():
    r = _map({"4786": 0.95, "6800": 0.0, "870": 0.0, "unreadable": 0.05})
    assert r["team_number"] == "4786" and "runner_up" not in r


def test_map_alliance_comes_from_roster():
    r = _map({"2992": 0.9, "unreadable": 0.1}, expected="blue")
    assert r["alliance"] == "red"


# --------------------------------------------------------------------------- #
# Vote weight / lock guards (pure helpers used at every vote site)
# --------------------------------------------------------------------------- #

def test_tentative_vote_weight_is_p_times_source_weight():
    res = {"team_number": "6800", "confidence": 0.24, "tentative": True}
    assert read_vote_base(res) == pytest.approx(0.24)
    assert read_vote_base(res) * 1.5 == pytest.approx(0.36)
    # Strictly below a normal sub-0.85 read's 0.5, even for a high p.
    assert read_vote_base({"confidence": 0.7, "tentative": True}) < 0.5
    # Non-tentative unchanged.
    assert read_vote_base({"confidence": 0.6}) == 0.5
    assert read_vote_base({"confidence": 0.9}) == 1.0


def test_tentative_reads_never_lock():
    assert is_tentative_read({"tentative": True})
    assert not read_can_lock({"confidence": 0.99, "tentative": True})
    assert read_can_lock({"confidence": 0.9})
    assert not read_can_lock({"confidence": 0.6})


# --------------------------------------------------------------------------- #
# Transport behaviour (shared with Clef)
# --------------------------------------------------------------------------- #

def test_missing_token_disables_without_http(monkeypatch, no_network):
    monkeypatch.delenv("JEV_TYPESAFE_API_TOKEN")
    logs = []
    s = _session(log_fn=logs.append)
    out = s.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    assert not s.available and out[0]["team_number"] is None
    assert "jev unavailable" in out[0]["error"]
    assert logs and "JEV_TYPESAFE_API_TOKEN" in logs[0] and logs[0].startswith("[jev]")


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_latches(monkeypatch, status):
    fake = _install(monkeypatch, FakeJev((status, '{"error": "bad token"}')))
    logs = []
    s = _session(log_fn=logs.append)
    s.read_bumpers([_jpeg_b64()] * 3, RED, BLUE, ["blue"] * 3)
    assert not s.available and s.disabled_kind == "auth"
    assert len(fake.calls) <= 3 and "TypeSafe" in logs[0]


def test_429_is_retried(monkeypatch):
    fake = _install(monkeypatch, FakeJev((429, "slow down"),
                                         _ok(_answers({"4786": 0.9, "unreadable": 0.1}))))
    out = _session().read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    assert out[0]["team_number"] == "4786" and len(fake.calls) == 2


def test_token_never_in_errors_or_logs(monkeypatch, capsys):
    _install(monkeypatch, FakeJev((400, f"bad request echo {TOKEN}")))
    out = _session().read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    assert TOKEN not in out[0]["error"]
    assert TOKEN not in capsys.readouterr().out


def test_accounting_and_summary(monkeypatch):
    _install(monkeypatch, FakeJev(_ok(_answers({"4786": 0.9, "unreadable": 0.1}), 2000)))
    s = _session()
    s.read_bumpers([_jpeg_b64()] * 2, RED, BLUE, ["blue"] * 2)
    st = s.stats()
    assert st["requests"] == 2 and st["input_tokens"] == 4000
    assert st["cost_usd"] == pytest.approx(4000 * 0.042 / 1e6)
    assert s.summary_line().startswith("[jev] jev-1.14.0: 2 reads")


def test_multiframe_list_item_reads_newest_crop_only(monkeypatch):
    fake = _install(monkeypatch, FakeJev(_ok(_answers({"4786": 0.9, "unreadable": 0.1}))))
    out = _session().read_bumpers([[_jpeg_b64(), _jpeg_b64(90, 40)]], RED, BLUE, ["blue"])
    assert len(fake.calls) == 1 and len(fake.calls[0]["payload"]["assets"]) == 1
    assert out[0]["team_number"] == "4786"


def test_classify_payload_and_mapping(monkeypatch):
    fake = _install(monkeypatch, FakeJev(_ok({"object": {
        "type": "choice", "choice": "equipment",
        "probabilities": {"robot": 0.1, "equipment": 0.85, "person": 0.0, "unclear": 0.05}}})))
    out = _session().classify_crops([_jpeg_b64()])
    assert out[0]["object"] == "equipment"
    p = fake.calls[0]["payload"]
    assert p["state"]["crop"] == jev_reader.JEV_ASSET_REF and "object" in p["questions"]


# --------------------------------------------------------------------------- #
# App wiring
# --------------------------------------------------------------------------- #

def _forbid_local_readers(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("jev reads must never reach the vLLM / in-process Qwen readers")
    monkeypatch.setattr(app_module, "_vllm_read_bumpers", _boom)
    monkeypatch.setattr(app_module, "_qwen_vl_read_bumpers", _boom)
    monkeypatch.setattr(app_module, "_get_qwen_vl", _boom)


def test_vision_read_bumpers_dispatches_to_jev(monkeypatch):
    _forbid_local_readers(monkeypatch)
    fake = _install(monkeypatch, FakeJev(_ok(_answers({"6800": 0.9, "unreadable": 0.1}))))
    out = app_module._vision_read_bumpers(
        [_jpeg_b64()], ALL6, BLUE, RED, expected_alliances=["blue"],
        provider="jev", clef_session=_session())
    assert out[0]["team_number"] == "6800" and len(fake.calls) == 1


def test_jev_without_session_uses_jev_default_not_clef(monkeypatch):
    _forbid_local_readers(monkeypatch)
    fake = _install(monkeypatch, FakeJev(_ok(_answers({"6800": 0.9, "unreadable": 0.1}))))
    app_module._vision_read_bumpers([_jpeg_b64()], ALL6, BLUE, RED,
                                    expected_alliances=["blue"], provider="jev")
    assert fake.calls[0]["url"] == jev_reader.JEV_URL


def test_jev_auth_failure_never_falls_back_to_local_model(monkeypatch):
    _forbid_local_readers(monkeypatch)
    _install(monkeypatch, FakeJev((401, "no")))
    s = _session()
    for _ in range(3):
        out = app_module._vision_read_bumpers(
            [_jpeg_b64()], ALL6, BLUE, RED, expected_alliances=["blue"],
            provider="jev", clef_session=s)
        assert out[0]["team_number"] is None and out[0]["error"]


def test_jev_internal_exception_returns_error_entries(monkeypatch):
    _forbid_local_readers(monkeypatch)

    def _explode(*_a, **_k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(app_module, "_clef_read_bumpers", _explode)
    out = app_module._vision_read_bumpers([_jpeg_b64()] * 2, ALL6, BLUE, RED, provider="jev")
    assert len(out) == 2 and all(r["error"].startswith("jev:") for r in out)


def test_knob_defaults_and_choices():
    cfg = JobConfig({"team_reader": "jev"})
    assert cfg.team_reader == "jev"
    assert cfg.jev_model == "jev-1.14.0"
    assert cfg.jev_tentative_min_p == pytest.approx(0.17)
    assert "jev_model" not in JobConfig().non_defaults()
    assert "jev_tentative_min_p" not in JobConfig().non_defaults()
    with pytest.raises(ValueError):
        JobConfig({"jev_model": "jev-0.1"})
    assert "jev" in app_module._TEAM_READERS


def test_job_builds_jev_session_from_knobs():
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    job.config = JobConfig({"team_reader": "jev", "jev_tentative_min_p": 0.3})
    job.team_reader = "jev"
    s = job._get_clef_session()
    assert isinstance(s, JevSession) and s.tentative_min_p == 0.3
    assert s.model_key == "jev-1.14.0"
    assert job._clef_read_kwargs("jev")["clef_session"] is s
    assert job._needs_vllm_server() is False
    assert job._clef_mf_enabled() is False


class _WS:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)


def test_unavailable_jev_warns_ui_as_jev(monkeypatch):
    _install(monkeypatch, FakeJev((401, "no")))
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    job.websocket = _WS()
    job._websocket_send_lock = asyncio.Lock()
    job.vlm_status = {}
    job._clef_session = _session()
    job._clef_session.read_bumpers([_jpeg_b64()], RED, BLUE, ["blue"])
    asyncio.run(job._maybe_warn_clef_unavailable())
    warns = [m for m in job.websocket.messages if m.get("type") == "degradation_warning"]
    assert len(warns) == 1 and warns[0]["component"] == "jev"
    assert "Jev bumper reader disabled" in warns[0]["message"]


def test_finalize_writes_jev_timings(monkeypatch, capsys):
    _install(monkeypatch, FakeJev(_ok(_answers({"4786": 0.9, "unreadable": 0.1}), 1000)))
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    job.timings = {}
    gauges = {}
    job.identity_diagnostics = types.SimpleNamespace(gauge=lambda k, v: gauges.__setitem__(k, v))
    job._clef_session = _session()
    job._clef_session.read_bumpers([_jpeg_b64()] * 2, RED, BLUE, ["blue"] * 2)
    job._finalize_clef_usage()
    t = job.timings
    assert t["jev.requests"] == 2 and t["jev.input_tokens"] == 2000
    assert t["jev.cost_cents"] == pytest.approx(2000 * 0.042 / 1e6 * 100)
    assert t["jev.errors"] == 0 and "jev.latency_p50_s" in t
    assert not any(k.startswith("clef.") for k in t)
    assert gauges["jev_usage"]["input_tokens"] == 2000
    assert any(l.strip().startswith("[jev]") for l in capsys.readouterr().out.splitlines())


def test_census_message_names_jev(capsys):
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    job.config = JobConfig({"team_reader": "jev", "read_census": True})
    job.team_reader = "jev"
    assert job._read_census_enabled() is True
    out = capsys.readouterr().out
    assert "on Jev" in out and "WARNING" not in out


# --------------------------------------------------------------------------- #
# Diagnostic row + evidence floor
# --------------------------------------------------------------------------- #

def test_diagnostic_row_carries_tentative_and_runner_up(tmp_path):
    path = tmp_path / "d.jsonl"
    rec = DiagnosticRecorder(path)
    common = dict(frame=1, robot_id=2, crop_b64="x", legibility=0.0, raw_read="unreadable",
                  matched_team="6800", alliance_read="blue", confidence=0.24,
                  vote_weight=0.24, surface="bumper")
    rec.record_vlm_read(**common)
    rec.record_vlm_read(**common, jev_probs={"6800": 0.24, "unreadable": 0.7},
                        tentative=True, runner_up="4786", runner_up_p=0.04)
    rec.close()
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    for k in ("jev_probs", "tentative", "runner_up", "runner_up_p"):
        assert k not in rows[0]
    assert rows[1]["tentative"] is True and rows[1]["runner_up"] == "4786"
    assert rows[1]["runner_up_p"] == 0.04 and rows[1]["jev_probs"]["unreadable"] == 0.7


def test_read_diag_extras_only_sets_present_fields():
    assert app_module._read_diag_extras({"team_number": "1"}) == {}
    assert app_module._read_diag_extras({"tentative": False, "jev_probs": {"1": 1.0}}) == {
        "jev_probs": {"1": 1.0}}
    ex = app_module._read_diag_extras({"tentative": True, "runner_up": "2", "runner_up_p": 0.1,
                                       "clef_same_robot": 0.9})
    assert ex == {"tentative": True, "runner_up": "2", "runner_up_p": 0.1,
                  "clef_same_robot": 0.9}


def test_evidence_floor_treats_jev_rows_like_clef_rows():
    rows = [
        {"matched_team": "6800", "confidence": 0.24, "tentative": True,
         "jev_probs": {"6800": 0.24, "unreadable": 0.7}},
        {"matched_team": "4786", "confidence": 0.9,
         "jev_probs": {"4786": 0.9, "unreadable": 0.1}},
        {"matched_team": "870", "confidence": 0.3},           # other reader: untouched
        {"matched_team": None, "jev_probs": {"unreadable": 1.0}},
    ]
    kept, dropped = apply_evidence_floor(rows, min_conf=0.65)
    assert dropped == 1 and [r["matched_team"] for r in kept] == ["4786", "870", None]
    # Margin is against the "unreadable" escape for jev rows.
    kept, dropped = apply_evidence_floor(rows[1:2], min_margin=0.85)
    assert dropped == 1


# --------------------------------------------------------------------------- #
# Live read path: tentative vote weight, no lock, row fields
# --------------------------------------------------------------------------- #

_BOX = (200, 100, 340, 220)


def _view(f):
    rng = np.random.default_rng(f)
    img = rng.integers(0, 60, (400, 600, 3), dtype=np.uint8)
    img[160:220, 200:340] = (30, 30, 200)
    img[175:205, 220:240] = 255
    return img


def _read_job(tmp_path):
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    job.config = JobConfig({"team_reader": "jev"})
    job.team_reader = "jev"
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
    job.gallery = None
    job.identity_diagnostics = MagicMock()
    job.diagnostic = DiagnosticRecorder(tmp_path / "diag.jsonl", ws_send=None)
    job._clef_session = _session()
    return job


def _read_once(job, f):
    frame = _view(f)

    def _enc(_f=frame):
        return job._bumper_crop_b64(_f, _BOX, enforce_floor=False)[0]
    lazy = app_module._LazyCrop(_enc, src=None)
    fused = [types.SimpleNamespace(
        evidence=[("main", lazy, 1.0, (4.0, 3.0))], track_id=5,
        detection=types.SimpleNamespace(pos=(4.0, 3.0)),
        relink_evidence=types.SimpleNamespace(type="continued"))]
    asyncio.run(job.read_fused_bumper_numbers(
        fused, {7: 1.0}, det_to_slot={0: 1}, crop_frame=f, slot_identity={1: 7}))


def _rows(job, tmp_path):
    job.diagnostic.flush()
    return [json.loads(l) for l in (tmp_path / "diag.jsonl").read_text().splitlines()
            if json.loads(l)["type"] == "vlm_read"]


def test_tentative_reads_vote_weakly_and_never_lock(monkeypatch, tmp_path):
    tentative = {"6800": 0.3, "4786": 0.02, "870": 0.0, "unreadable": 0.66,
                 "not_robot": 0.0, "printed": 0.0, "other": 0.02}
    _install(monkeypatch, FakeJev(_ok(_answers(tentative))))
    job = _read_job(tmp_path)
    contradictions = []
    monkeypatch.setattr(job, "_handle_confirmed_vlm_contradiction",
                        lambda *a, **k: contradictions.append(a))
    for f in (100, 200, 300, 400):        # > the 3-consecutive lock run
        _read_once(job, f)
    robot = job._get_slot_track_if_exists(1)
    assert robot is not None
    assert robot.team_ocr_votes["6800"] == pytest.approx(4 * 0.3)
    assert robot.team_number_confirmed is False and robot.team_number != "6800"
    assert contradictions == []
    ident = job.identity_registry.identities[7]
    assert ident.vlm_fail_count == 4 and not ident.has_vlm_read
    rows = _rows(job, tmp_path)
    assert len(rows) == 4
    for r in rows:
        assert r["tentative"] is True and r["matched_team"] == "6800"
        assert r["vote_weight"] == pytest.approx(0.3)
        assert r["runner_up"] == "4786" and r["jev_probs"]["unreadable"] == 0.66


def test_confident_jev_reads_still_lock(monkeypatch, tmp_path):
    _install(monkeypatch, FakeJev(_ok(_answers({"6800": 0.95, "unreadable": 0.05}))))
    job = _read_job(tmp_path)
    for f in (100, 200, 300):
        _read_once(job, f)
    robot = job._get_slot_track_if_exists(1)
    assert robot.team_number == "6800" and robot.team_number_confirmed
    rows = _rows(job, tmp_path)
    assert all("tentative" not in r and r["vote_weight"] == 1.0 for r in rows)


# --------------------------------------------------------------------------- #
# Window-label requests (web/field/window_labels.py)
# --------------------------------------------------------------------------- #

def _who(probs):
    top = max(probs, key=probs.get)
    return {"who": {"type": "choice", "choice": top, "probabilities": probs,
                    "confidence": probs[top]}}


@pytest.mark.parametrize("kind,name,ref", [("wide", "crop", "crop_image"),
                                           ("zoom", "crop", "crop_image"),
                                           ("grid", "crops", "crops_image")])
def test_window_payload_shape(kind, name, ref):
    payload, sizes = jev_reader.build_jev_window_payload(
        "jev-1.14.0", _jpeg(480, 480), kind, RED, BLUE)
    assert payload["model"] == "jev-1.14.0"
    assert payload["state"][name] == ref and set(payload["assets"]) == {ref}
    assert sizes == [(480, 480)]
    crit = payload["questions"]["team"]["criteria"]
    assert list(crit)[:6] == RED + BLUE
    assert crit[RED[0]] == f"Team {RED[0]} (red bumper)"
    assert crit[BLUE[0]] == f"Team {BLUE[0]} (blue bumper)"
    assert set(crit) - set(RED + BLUE) == {"not_robot", "unreadable", "printed", "other"}
    assert crit["not_robot"].startswith("No robot at the marked spot")
    ctx = payload["state"]["context"]
    assert f"red alliance {', '.join(RED)}; blue alliance {', '.join(BLUE)}" in ctx
    assert ("tick marks" in ctx) == (kind != "zoom")
    assert ("nearest the centre" in ctx) == (kind == "zoom")


def test_window_payload_upscales_small_images_to_jevs_minimum():
    _payload, sizes = jev_reader.build_jev_window_payload(
        "jev-1.14.0", _jpeg(240, 240), "wide", RED, BLUE)
    assert sizes == [(280, 280)]


def test_lineup_payload_has_two_referenced_assets():
    payload, sizes = jev_reader.build_jev_lineup_payload(
        "jev-1.14.0", _jpeg(1536, 1440), _jpeg(960, 960), ["A", "B", "C"])
    assert payload["state"]["lineup"] == "lineup_image"
    assert payload["state"]["crops"] == "crops_image"
    assert set(payload["assets"]) == {"lineup_image", "crops_image"}
    assert sizes == [(1536, 1440), (960, 960)]
    crit = payload["questions"]["who"]["criteria"]
    assert list(crit) == ["A", "B", "C", "none", "unsure"]
    ctx = payload["state"]["context"]
    assert "shows 3 FRC robots" in ctx and "labelled A to C" in ctx
    assert "Yellow balls" in ctx


def test_estimate_tokens_accepts_one_size_or_a_list():
    payload = {"state": {"context": "x" * 190}, "questions": {}}
    one = jev_reader.jev_estimate_tokens(payload, (415, 100))
    two = jev_reader.jev_estimate_tokens(payload, [(415, 100), (415, 200)])
    assert two - one == 200


def test_estimate_tokens_one_size_numpy_size_and_one_element_list_agree():
    payload = {"state": {"context": "x" * 190}, "questions": {"q": "y" * 40}}
    w, h = 830, 300
    plain = jev_reader.jev_estimate_tokens(payload, (w, h))
    numpy_pair = jev_reader.jev_estimate_tokens(payload, (np.int64(w), np.int64(h)))
    listed = jev_reader.jev_estimate_tokens(payload, [(w, h)])
    text = len(json.dumps(payload["state"])) + len(json.dumps(payload["questions"]))
    assert plain == numpy_pair == listed == int(w * h / 415 + text / 1.9)


def test_jev_asset_is_public_and_lineup_payload_can_share_one_card_asset():
    card, crops = _jpeg(1536, 1440), _jpeg(960, 960)
    asset = jev_reader.jev_asset(card)
    assert asset[1] == (1536, 1440) and asset[0]["type"] == "image"
    p1, s1 = jev_reader.build_jev_lineup_payload("jev-1.14.0", card, crops, ["A", "B"],
                                                 lineup_asset=asset)
    p2, _s2 = jev_reader.build_jev_lineup_payload("jev-1.14.0", card, crops, ["A", "B"],
                                                  lineup_asset=asset)
    assert p1["assets"]["lineup_image"] is asset[0]
    assert p1["assets"]["lineup_image"] is p2["assets"]["lineup_image"]
    plain, plain_sizes = jev_reader.build_jev_lineup_payload("jev-1.14.0", card, crops,
                                                             ["A", "B"])
    assert p1 == plain and s1 == plain_sizes


def test_ask_returns_probabilities_in_order_and_counts_usage(monkeypatch):
    fake = _install(monkeypatch, FakeJev(_ok(_who({"A": 0.9, "unsure": 0.1}), 4000)))
    s = _session(concurrency=1)
    items = [jev_reader.build_jev_lineup_payload("jev-1.14.0", _jpeg(400, 300), _jpeg(300, 300), ["A"])
             for _ in range(3)]
    out = s.ask(items, "who")
    assert [o["probs"]["A"] for o in out] == [0.9, 0.9, 0.9]
    assert len(fake.calls) == 3
    assert fake.calls[0]["headers"]["TypeSafe-Danger-Zone-Images"] == "true"
    st = s.stats()
    assert st["requests"] == 3 and st["input_tokens"] == 12000 and st["errors"] == 0


def test_ask_reports_errors_without_raising_and_latches_on_auth(monkeypatch):
    _install(monkeypatch, FakeJev((400, f"bad payload, token {TOKEN}")))
    s = _session(concurrency=1)
    item = jev_reader.build_jev_window_payload("jev-1.14.0", _jpeg(480, 480), "wide", RED, BLUE)
    out = s.ask([item, item], "team")
    assert all("error" in o for o in out) and s.stats()["errors"] == 2 and s.available
    assert TOKEN not in json.dumps(out)

    _install(monkeypatch, FakeJev((401, f"bad token {TOKEN}")))
    s2 = _session(concurrency=1)
    out2 = s2.ask([item, item], "team")
    assert all("error" in o for o in out2) and not s2.available
    assert TOKEN not in json.dumps(out2)


def test_ask_turns_a_malformed_item_into_an_error_entry(monkeypatch):
    _install(monkeypatch, FakeJev(_ok(_answers({"2992": 0.9, "unreadable": 0.1}))))
    s = _session(concurrency=1)
    item = jev_reader.build_jev_window_payload("jev-1.14.0", _jpeg(480, 480), "wide", RED, BLUE)
    out = s.ask([item, "not-a-pair", None], "team")
    assert len(out) == 3 and out[0]["probs"]["2992"] == 0.9
    assert "error" in out[1] and "error" in out[2]


def test_ask_missing_question_is_an_error(monkeypatch):
    _install(monkeypatch, FakeJev(_ok(_who({"A": 1.0}))))
    s = _session(concurrency=1)
    item = jev_reader.build_jev_window_payload("jev-1.14.0", _jpeg(480, 480), "wide", RED, BLUE)
    assert "error" in s.ask([item], "team")[0]


def test_lineup_payload_colour_sentence_is_optional_and_off_by_default():
    card, crops = _jpeg(1536, 1440), _jpeg(960, 960)
    plain, _ = jev_reader.build_jev_lineup_payload("jev-1.14.0", card, crops, ["A", "B"])
    off, _ = jev_reader.build_jev_lineup_payload("jev-1.14.0", card, crops, ["A", "B"],
                                                 colour_note=False)
    on, _ = jev_reader.build_jev_lineup_payload("jev-1.14.0", card, crops, ["A", "B"],
                                                colour_note=True)
    assert plain == off and jev_reader.LINEUP_COLOUR_NOTE not in plain["state"]["context"]
    ctx = on["state"]["context"]
    assert jev_reader.LINEUP_COLOUR_NOTE in ctx
    assert ctx.index(jev_reader.LINEUP_COLOUR_NOTE) < ctx.index("Yellow balls")
    assert on["questions"] == plain["questions"] and on["assets"] == plain["assets"]
