"""When the vLLM team reader is unreachable, bumper reads must degrade to the
in-process Qwen reader — not return a batch of dead "connection refused" entries.

Regression for the production symptom where every bumper read showed
``<urlopen error [Errno 111] Connection refused>`` at 0% confidence: the vLLM
server never came up (phase-exclusive lifecycle), and `_vllm_read_bumpers`
swallowed the per-crop connection errors instead of signalling the caller to
fall back. The fallback in `_vision_read_bumpers` only fires on an exception,
so it never triggered and the whole match's identity pipeline was starved.
"""
import urllib.error
import urllib.request

import web.app as app_module
from web.app import _vision_read_bumpers


def _refuse(*_a, **_k):
    raise urllib.error.URLError(
        ConnectionRefusedError(111, "Connection refused")
    )


def test_vllm_unreachable_falls_back_to_local_reader(monkeypatch):
    # Every HTTP call to the vLLM server is refused (server is down).
    monkeypatch.setattr(urllib.request, "urlopen", _refuse)

    sentinel = [{"team_number": "4174", "alliance": "blue",
                 "confidence": 0.9, "raw_read": "4174"}]
    called = {"qwen": False}

    def fake_qwen(crops, *a, **k):
        called["qwen"] = True
        return sentinel

    monkeypatch.setattr(app_module, "_qwen_vl_read_bumpers", fake_qwen)

    out = _vision_read_bumpers(
        ["fakecropb64"], ["4174"], ["4174"], [],
        expected_alliances=["blue"], provider="vllm",
    )

    assert called["qwen"], "should fall back to in-process Qwen when vLLM is down"
    assert out == sentinel
    # Crucially, no read should carry the swallowed connection-refused error.
    assert all("Connection refused" not in str(r.get("error", "")) for r in out)
