"""Registry of selectable bumper-reader VLM models (all vLLM-served, HF format)."""
import pytest

from web import vlm_models


def test_default_entry_is_the_incumbent():
    # The default must stay byte-for-byte today's behavior: Qwen3-VL-4B at 0.45.
    assert vlm_models.DEFAULT_KEY == "qwen3-vl-4b"
    m = vlm_models.get("qwen3-vl-4b")
    assert m.hf_repo == "Qwen/Qwen3-VL-4B-Instruct"
    assert m.gpu_util == 0.45


def test_model_keys_ordered_default_first():
    keys = vlm_models.model_keys()
    assert keys[0] == vlm_models.DEFAULT_KEY
    for k in ("glm-4.6v-flash", "qwen3-vl-8b", "lfm2.5-vl"):
        assert k in keys
    # The abliterated Huihui 24B was dropped (no vision-intact non-GGUF quant).
    assert "huihui-mistral-24b" not in keys


def test_glm_carries_reasoning_and_thinking_flags():
    # GLM needs --reasoning-parser glm45 to serve, and enable_thinking=false so
    # 64-token reads don't come back empty.
    assert vlm_models.extra_serve_args("glm-4.6v-flash") == ["--reasoning-parser", "glm45"]
    assert vlm_models.extra_body("glm-4.6v-flash") == {
        "chat_template_kwargs": {"enable_thinking": False}}


def test_default_model_has_no_extra_serve_args_or_body():
    # The incumbent must stay byte-identical: no extra flags, no extra payload.
    assert vlm_models.extra_serve_args("qwen3-vl-4b") == []
    assert vlm_models.extra_body("qwen3-vl-4b") == {}


def test_get_unknown_raises():
    with pytest.raises(ValueError):
        vlm_models.get("nope-not-a-model")


def test_payload_model_is_hf_repo():
    assert vlm_models.payload_model("qwen3-vl-4b") == "Qwen/Qwen3-VL-4B-Instruct"
    # Unknown key falls back to the provided default.
    assert vlm_models.payload_model("nope", default="X") == "X"


def test_gpu_util_accessor():
    assert vlm_models.gpu_util("qwen3-vl-4b") == 0.45
    assert vlm_models.gpu_util("glm-4.6v-flash") == 0.70
    assert vlm_models.gpu_util("qwen3-vl-8b") == 0.65


def test_final_repo_ids():
    assert vlm_models.get("glm-4.6v-flash").hf_repo == "Mitchins/GLM-4.6V-Flash-FP8-Block128"
    assert vlm_models.get("qwen3-vl-8b").hf_repo == "Qwen/Qwen3-VL-8B-Instruct-FP8"
    assert vlm_models.get("lfm2.5-vl").hf_repo == "LiquidAI/LFM2.5-VL-1.6B"


def test_served_matches_substring_and_none():
    assert vlm_models.served_matches("qwen3-vl-4b", "Qwen/Qwen3-VL-4B-Instruct") is True
    # vLLM may append a suffix/revision — substring still matches.
    assert vlm_models.served_matches("qwen3-vl-4b", "Qwen/Qwen3-VL-4B-Instruct@main") is True
    assert vlm_models.served_matches("qwen3-vl-4b", "some/OtherModel") is False
    assert vlm_models.served_matches("qwen3-vl-4b", None) is False


def test_status_ready_when_cached(monkeypatch):
    monkeypatch.setattr(vlm_models, "_hf_cached", lambda repo: True)
    assert vlm_models.status("qwen3-vl-4b") == "ready"


def test_status_needs_download_when_absent(monkeypatch):
    monkeypatch.setattr(vlm_models, "_hf_cached", lambda repo: False)
    assert vlm_models.status("qwen3-vl-4b") == "needs_download"


def test_describe_rows_shape(monkeypatch):
    monkeypatch.setattr(vlm_models, "_hf_cached", lambda repo: False)
    rows = vlm_models.describe()
    assert rows and rows[0]["key"] == vlm_models.DEFAULT_KEY
    for r in rows:
        assert set(r) == {"key", "label", "hf_repo", "status", "notes", "approx_gb"}
        assert r["status"] in ("ready", "needs_download")
