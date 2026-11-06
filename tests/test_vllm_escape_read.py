"""Per-model bumper-read prompt variant in _vllm_read_bumpers.

The default model (qwen3-vl-4b) must keep the incumbent prompt + schema
byte-for-byte; Qwen3.5-4B (vlm_models read_prompt="escape") gets the escape
prompt, which decides the crop's subject before any digits so junk crops
(ball piles, field structures) can't be given a copied roster number.
"""
import base64
import io
import json
import urllib.request

import pytest
from PIL import Image

import web.app as app
from web import vlm_models
from web.diagnostic import DiagnosticRecorder

RED = ["254", "1678", "4414"]
BLUE = ["118", "971", "2056"]
MATCH = RED + BLUE

# The incumbent prompt for a red expected alliance — copied verbatim so any
# drift in the default model's payload fails here.
INCUMBENT_RED_PROMPT = (
    "This image shows ONE FRC robot from a side or elevated camera angle. "
    "Around the base of the robot is a bumper: a solid RED or BLUE band with a "
    "WHITE team number printed on it. FRC team numbers are 1 to 5 digits "
    "(for example 294, 2485, 4322).\n\n"
    "Computer vision sees RED bumper pixels, so partial or uncertain reads "
    "should be matched only against these red alliance teams: 254, 1678, 4414.\n"
    "Carefully read the WHITE digits on the bumper, left to right. Report the "
    "exact number you SEE. Do not use robot paint or mechanisms for alliance color; "
    "only the bumper band matters. If the digits are clear, report them even if the "
    "color looks surprising.\n\n"
    'Reply with ONLY compact JSON and nothing else: '
    '{"team": "<the digits, or null if you cannot read them>", '
    '"color": "red" or "blue", '
    '"surface": "bumper" if the digits are on a robot\'s bumper band, '
    'or "printed" if they are printed on a crate, bin, sign, or other '
    'field structure}.'
)


class _Probe:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _crop_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (200, 10, 10)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def reader(monkeypatch):
    """Run _vllm_read_bumpers against a canned reply; returns (run, payloads)."""
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Probe())
    payloads = []
    reply = {"content": None}

    def fake_post(payload):
        payloads.append(payload)
        return {"choices": [{"message": {"content": reply["content"]}}]}

    monkeypatch.setattr(app, "_post_vlm_chat", fake_post)

    def run(content: str, expected: str = "red") -> dict:
        reply["content"] = content
        return app._vllm_read_bumpers([_crop_b64()], MATCH, BLUE, RED, [expected])[0]

    yield run, payloads
    app._apply_vlm_model(vlm_models.DEFAULT_KEY)


def _prompt(payload: dict) -> str:
    return payload["messages"][0]["content"][1]["text"]


def test_default_model_payload_is_incumbent(reader):
    run, payloads = reader
    app._apply_vlm_model(vlm_models.DEFAULT_KEY)
    res = run('{"team": "254", "color": "red", "surface": "bumper"}')
    p = payloads[-1]
    assert _prompt(p) == INCUMBENT_RED_PROMPT
    rf = p["response_format"]["json_schema"]
    assert rf["name"] == "bumper_read"
    assert rf["schema"] is app._BUMPER_READ_SCHEMA
    assert "chat_template_kwargs" not in p
    assert p["model"] == app.VLM_MODEL_NAME
    # Incumbent result keys unchanged (no "subject").
    assert set(res) == {"team_number", "alliance", "confidence", "raw_read",
                        "digit_prob", "surface"}
    assert res["team_number"] == "254"


def test_escape_payload_for_qwen35(reader):
    run, payloads = reader
    app._apply_vlm_model("qwen3.5-4b")
    run('{"subject": "bumper", "team": "254", "color": "red"}', expected="red")
    p = payloads[-1]
    rf = p["response_format"]["json_schema"]
    assert rf["name"] == "bumper_read_escape"
    schema = rf["schema"]
    assert schema is app._BUMPER_READ_ESCAPE_SCHEMA
    # Guided decoding emits properties in order: subject must come first.
    assert list(schema["properties"]) == ["subject", "team", "color"]
    assert schema["properties"]["subject"]["enum"] == [
        "bumper", "no_digits", "printed", "not_robot"]
    assert schema["properties"]["color"]["enum"] == ["red", "blue", "none"]
    assert schema["required"] == ["subject", "team", "color"]
    prompt = _prompt(p)
    assert "Teams on the red alliance in this match: 254, 1678, 4414.\n" in prompt
    assert prompt.startswith("This image is an automatic crop from a robot detector")
    assert "Computer vision sees" not in prompt
    # Qwen3.5 extra_body (thinking off) merged top-level; served HF id.
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert p["model"] == "Qwen/Qwen3.5-4B"


def test_escape_hint_lines():
    assert (app._escape_read_hint_line("blue", RED, BLUE, MATCH)
            == "Teams on the blue alliance in this match: 118, 971, 2056.")
    assert (app._escape_read_hint_line("unknown", RED, BLUE, MATCH)
            == "Teams in this match: 254, 1678, 4414, 118, 971, 2056.")


def test_escape_unknown_alliance_uses_match_teams(reader):
    run, payloads = reader
    app._apply_vlm_model("qwen3.5-4b")
    run('{"subject": "not_robot", "team": null, "color": "none"}', expected="unknown")
    assert ("Teams in this match: 254, 1678, 4414, 118, 971, 2056.\n"
            in _prompt(payloads[-1]))


def test_escape_bumper_keeps_roster_matched_team(reader):
    run, _ = reader
    app._apply_vlm_model("qwen3.5-4b")
    res = run('{"subject": "bumper", "team": "1678", "color": "red"}')
    assert res["team_number"] == "1678"
    assert res["alliance"] == "red"
    assert res["surface"] == "bumper"
    assert res["subject"] == "bumper"
    assert res["raw_read"] == "1678"
    assert res["confidence"] > 0


@pytest.mark.parametrize("subject,surface", [
    ("no_digits", "unknown"), ("printed", "printed"), ("not_robot", "unknown")])
def test_escape_non_bumper_subject_forces_no_team(reader, subject, surface):
    run, _ = reader
    app._apply_vlm_model("qwen3.5-4b")
    # Even if the model emits a roster number, a non-bumper subject drops it.
    res = run(json.dumps({"subject": subject, "team": "254", "color": "red"}))
    assert res["team_number"] is None
    assert res["raw_read"] == ""
    assert res["confidence"] == 0.0
    assert res["surface"] == surface
    assert res["subject"] == subject


def test_escape_unparseable_reply(reader):
    run, _ = reader
    app._apply_vlm_model("qwen3.5-4b")
    res = run("not json at all")
    assert res["team_number"] is None
    assert res["subject"] is None
    assert res["surface"] == "unknown"


def test_parse_escape_read_fenced_json():
    team, color, surface, subject = app._parse_escape_read(
        '```json\n{"subject": "bumper", "team": 4414, "color": "red"}\n```')
    assert (team, color, surface, subject) == ("4414", "red", "bumper", "bumper")


def test_apply_vlm_model_sets_and_resets_read_prompt():
    try:
        app._apply_vlm_model("qwen3.5-4b")
        assert app._active_vlm_read_prompt == "escape"
        app._apply_vlm_model(vlm_models.DEFAULT_KEY)
        assert app._active_vlm_read_prompt == "incumbent"
        app._apply_vlm_model("qwen3.5-4b")
        app._apply_vlm_model(None)  # no knob -> default
        assert app._active_vlm_read_prompt == "incumbent"
        app._apply_vlm_model("qwen3-vl-8b")
        assert app._active_vlm_read_prompt == "incumbent"
    finally:
        app._apply_vlm_model(vlm_models.DEFAULT_KEY)


def test_vlm_read_row_records_subject_only_when_present(tmp_path):
    out = tmp_path / "diag.jsonl"
    rec = DiagnosticRecorder(out, ws_send=None)
    common = dict(robot_id=3, crop_b64="abc", legibility=0.0, raw_read="",
                  matched_team=None, alliance_read="unknown", confidence=0.0,
                  vote_weight=0.5)
    rec.record_vlm_read(frame=10, subject="not_robot", **common)
    rec.record_vlm_read(frame=11, subject=None, **common)
    rec.close()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["subject"] == "not_robot"
    assert "subject" not in rows[1]
