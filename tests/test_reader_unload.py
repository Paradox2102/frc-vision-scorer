"""The in-process bumper reader must be evictable before SAM3 tracking.

team_reader=local lazy-loads Qwen3-VL into a module global that holds ~8GB
of VRAM for the life of the server. SAM3's offload heuristic then sees
<16GB free and pushes tracker state to the CPU, slowing propagation ~30%.
Phase exclusivity already takes the external vLLM server down before
tracking; the in-process reader needs the same treatment. The reader
reloads lazily on the next read pass, so eviction is quality-neutral.
"""
import web.app as app_module
from web.app import _unload_qwen_vl


def test_unload_clears_cached_model(monkeypatch):
    monkeypatch.setattr(app_module, "_qwen_vl_model", object())
    monkeypatch.setattr(app_module, "_qwen_vl_processor", object())

    assert _unload_qwen_vl() is True

    assert app_module._qwen_vl_model is None
    assert app_module._qwen_vl_processor is None


def test_unload_noop_when_not_loaded(monkeypatch):
    monkeypatch.setattr(app_module, "_qwen_vl_model", None)
    monkeypatch.setattr(app_module, "_qwen_vl_processor", None)

    assert _unload_qwen_vl() is False
