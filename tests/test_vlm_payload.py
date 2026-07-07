"""Payload construction for vLLM chat reads (structured output plumbing)."""
from web.app import _vlm_chat_payload, _BUMPER_READ_SCHEMA, _OBJECT_CLASS_SCHEMA


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
