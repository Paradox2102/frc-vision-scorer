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


def test_qwen35_4b_entry_fields():
    m = vlm_models.get("qwen3.5-4b")
    assert m.label == "Qwen3.5-4B (escape prompt)"
    assert m.hf_repo == "Qwen/Qwen3.5-4B"
    assert m.gpu_util == 0.5
    assert m.approx_gb == 9.3
    # start_vllm.sh passes --dtype float16 first; argparse last-wins -> bf16.
    assert vlm_models.extra_serve_args("qwen3.5-4b") == ["--dtype", "bfloat16"]
    # Qwen3.5 thinks by default; 64-token reads need thinking off.
    assert vlm_models.extra_body("qwen3.5-4b") == {
        "chat_template_kwargs": {"enable_thinking": False}}
    assert m.read_prompt == "escape"


def test_qwen35_4b_appended_after_existing_entries():
    keys = vlm_models.model_keys()
    assert keys[0] == vlm_models.DEFAULT_KEY
    assert keys.index("qwen3.5-4b") > keys.index("lfm2.5-vl")
    # Big-card readers come after every 24 GB-capable entry.
    assert keys[-2:] == ("qwen3-vl-30b-a3b", "qwen3.6-35b-a3b")


def test_read_prompt_helper():
    # Only Qwen3.5-4B uses the escape prompt (it makes Qwen3-VL over-conservative).
    assert vlm_models.read_prompt("qwen3.5-4b") == "escape"
    for k in ("qwen3-vl-4b", "glm-4.6v-flash", "qwen3-vl-8b", "lfm2.5-vl",
              "qwen3-vl-30b-a3b"):
        assert vlm_models.read_prompt(k) == "incumbent"
    assert vlm_models.read_prompt("qwen3.6-35b-a3b") == "escape"
    # Unknown keys never change the read prompt.
    assert vlm_models.read_prompt("nope-not-a-model") == "incumbent"


def test_qwen35_served_matches_is_specific():
    assert vlm_models.served_matches("qwen3.5-4b", "Qwen/Qwen3.5-4B") is True
    assert vlm_models.served_matches("qwen3.5-4b", "Qwen/Qwen3-VL-4B-Instruct") is False
    assert vlm_models.served_matches("qwen3-vl-4b", "Qwen/Qwen3.5-4B") is False


def test_vlm_model_knob_choices_include_qwen35():
    from web import job_config
    knob = next(k for k in job_config.KNOBS if k.name == "vlm_model")
    assert "qwen3.5-4b" in knob.choices


def test_big_card_entries():
    vl = vlm_models.get("qwen3-vl-30b-a3b")
    assert vl.hf_repo == "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    assert vl.vram_gb == 42.0
    assert vlm_models.extra_serve_args("qwen3-vl-30b-a3b") == ["--dtype", "bfloat16"]
    assert vlm_models.extra_body("qwen3-vl-30b-a3b") == {}
    q = vlm_models.get("qwen3.6-35b-a3b")
    assert q.hf_repo == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert q.vram_gb == 48.0
    assert vlm_models.extra_body("qwen3.6-35b-a3b") == {
        "chat_template_kwargs": {"enable_thinking": False}}
    assert vlm_models.served_matches("qwen3.6-35b-a3b", "Qwen/Qwen3.6-35B-A3B-FP8")
    assert not vlm_models.served_matches("qwen3.6-35b-a3b", "Qwen/Qwen3.5-4B")


def test_serve_util_keeps_registry_fraction_on_small_or_unknown_cards(monkeypatch):
    monkeypatch.delenv("VLM_SERVER_GPU_GB", raising=False)
    for key in vlm_models.model_keys():
        m = vlm_models.get(key)
        assert vlm_models.serve_util(key, 23.6) == m.gpu_util  # 3090
        assert vlm_models.serve_util(key, None) == m.gpu_util


def test_serve_util_is_absolute_on_big_cards(monkeypatch):
    monkeypatch.delenv("VLM_SERVER_GPU_GB", raising=False)
    card = 95.6  # RTX PRO 6000: 97887 MiB
    # No vram_gb: the 3090-proven budget, gpu_util x 24.
    assert vlm_models.serve_util("qwen3-vl-8b", card) == round(0.65 * 24 / card, 3)
    assert vlm_models.serve_util("qwen3.5-4b", card) == round(0.5 * 24 / card, 3)
    # vram_gb wins when set.
    assert vlm_models.serve_util("qwen3.6-35b-a3b", card) == round(48.0 / card, 3)
    assert vlm_models.serve_util("qwen3-vl-30b-a3b", card) == round(42.0 / card, 3)


def test_serve_util_env_override(monkeypatch):
    monkeypatch.setenv("VLM_SERVER_GPU_GB", "40")
    assert vlm_models.serve_util("qwen3-vl-8b", 95.6) == round(40 / 95.6, 3)
    # The override applies on small cards too, clamped to 0.92.
    assert vlm_models.serve_util("qwen3-vl-8b", 23.6) == 0.92
    monkeypatch.setenv("VLM_SERVER_GPU_GB", "junk")
    assert vlm_models.serve_util("qwen3-vl-8b", 23.6) == 0.65


def test_lifecycle_start_exports_serve_util(monkeypatch):
    import asyncio
    from web import vllm_lifecycle

    seen = {}

    class _Proc:
        async def wait(self):
            return 0

    async def _fake_exec(*args, env=None, **_kw):
        seen["args"], seen["env"] = args, env
        return _Proc()

    monkeypatch.delenv("VLM_SERVER_GPU_GB", raising=False)
    monkeypatch.setattr(vllm_lifecycle, "_card_total_gib", lambda: 95.6)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    prev = vllm_lifecycle.active_model()
    try:
        vllm_lifecycle.set_active_model("qwen3.6-35b-a3b")
        asyncio.run(vllm_lifecycle._default_run_script("start"))
    finally:
        vllm_lifecycle.set_active_model(prev)
    assert seen["env"]["VLM_MODEL_NAME"] == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert seen["env"]["VLM_SERVER_GPU_UTIL"] == str(round(48.0 / 95.6, 3))
    assert seen["args"][-2:] == ("--dtype", "bfloat16")
