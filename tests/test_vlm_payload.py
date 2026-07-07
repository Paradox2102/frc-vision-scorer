"""Payload construction for vLLM chat reads (structured output plumbing)."""
import io
import json
import urllib.error
import urllib.request

from web.app import (
    _vlm_chat_payload,
    _BUMPER_READ_SCHEMA,
    _OBJECT_CLASS_SCHEMA,
    _post_vlm_chat,
)


def test_basic_payload_shape():
    p = _vlm_chat_payload("data:image/jpeg;base64,AAA", "read it", max_tokens=64)
    assert p["temperature"] == 0
    assert p["max_tokens"] == 64
    content = p["messages"][0]["content"]
    assert content[0] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}}
    assert content[1] == {"type": "text", "text": "read it"}
    assert "response_format" not in p  # no schema given


def test_logprobs_flag():
    p = _vlm_chat_payload("d", "p", max_tokens=8, logprobs=True)
    assert p["logprobs"] is True


def test_guided_schema_attached():
    p = _vlm_chat_payload("d", "p", max_tokens=8, schema_name="bumper_read",
                          schema=_BUMPER_READ_SCHEMA, guided=True)
    rf = p["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "bumper_read"
    assert rf["json_schema"]["schema"] is _BUMPER_READ_SCHEMA


def test_guided_false_omits_schema():
    p = _vlm_chat_payload("d", "p", max_tokens=8, schema_name="x",
                          schema=_BUMPER_READ_SCHEMA, guided=False)
    assert "response_format" not in p


def test_bumper_schema_matches_reply_contract():
    props = _BUMPER_READ_SCHEMA["properties"]
    assert set(props) == {"team", "color", "surface"}
    assert props["color"]["enum"] == ["red", "blue"]
    assert props["surface"]["enum"] == ["bumper", "printed", "unclear"]


def test_object_schema_matches_reply_contract():
    props = _OBJECT_CLASS_SCHEMA["properties"]
    assert props["object"]["enum"] == ["robot", "equipment", "person", "unclear"]


class _FakeResponse:
    """Minimal context-manager stand-in for the urlopen() response."""

    def __init__(self, body: dict):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_post_vlm_chat_retries_on_400_and_strips_schema(monkeypatch):
    payload = {
        "model": "qwen",
        "temperature": 0,
        "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
    }
    fake_body = {"choices": [{"message": {"content": "4174"}}]}
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b""))
        return _FakeResponse(fake_body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = _post_vlm_chat(payload)

    assert result == fake_body
    assert len(calls) == 2

    first_data = json.loads(calls[0].data.decode("utf-8"))
    second_data = json.loads(calls[1].data.decode("utf-8"))
    assert "response_format" not in second_data
    expected_first_minus_schema = {k: v for k, v in first_data.items() if k != "response_format"}
    assert second_data == expected_first_minus_schema


def test_post_vlm_chat_non_400_propagates(monkeypatch):
    payload = {"model": "qwen", "response_format": {"type": "json_schema"}}
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        raise urllib.error.HTTPError(req.full_url, 500, "server error", {}, io.BytesIO(b""))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    try:
        _post_vlm_chat(payload)
        assert False, "expected HTTPError to propagate"
    except urllib.error.HTTPError as e:
        assert e.code == 500

    assert len(calls) == 1


def test_post_vlm_chat_400_without_response_format_propagates(monkeypatch):
    payload = {"model": "qwen"}  # no response_format key at all
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b""))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    try:
        _post_vlm_chat(payload)
        assert False, "expected HTTPError to propagate"
    except urllib.error.HTTPError as e:
        assert e.code == 400

    assert len(calls) == 1


def test_post_vlm_chat_retry_failure_propagates(monkeypatch):
    payload = {"model": "qwen", "response_format": {"type": "json_schema"}}
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b""))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    try:
        _post_vlm_chat(payload)
        assert False, "expected HTTPError to propagate"
    except urllib.error.HTTPError as e:
        assert e.code == 400

    assert len(calls) == 2
