"""Registry of selectable bumper-reader VLM models (all served by vLLM).

Every model is served by the SAME vLLM OpenAI-compatible server
(``VLM_SERVER_URL``, port 8006, web/start_vllm.sh) in native HF format
(safetensors, possibly a quantized repo), so the read path in web/app.py and the
phase-exclusive sleep/wake lifecycle are untouched — switching models is just a
different ``VLM_MODEL_NAME`` (and per-model ``VLM_SERVER_GPU_UTIL``) at launch.

vLLM pulls weights from the HF hub automatically on first serve, so "download if
missing" is native — no second serving stack, no GGUF, no symlinks. ``status``
reports ``ready`` (repo already in the HF cache) or ``needs_download`` (vLLM will
fetch it on first serve) so the UI can annotate the picker.

Nothing here imports web.app (or torch); huggingface_hub is imported lazily
inside ``status`` only. It is safe to import from web/job_config.py at module
load for the ``vlm_model`` knob's choices, and from web/vllm_lifecycle.py for
model-aware start/verify.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

#: The default model — byte-for-byte today's behavior (vLLM serving 4B fp16 at
#: gpu-util 0.45). Keep this entry's hf_repo/gpu_util in lockstep with
#: web/start_vllm.sh's defaults.
DEFAULT_KEY = "qwen3-vl-4b"


@dataclass(frozen=True)
class VlmModel:
    key: str
    label: str              # human name for the UI picker
    hf_repo: str            # HF hub id vLLM serves; also the served /v1/models id
    gpu_util: float         # per-model VLM_SERVER_GPU_UTIL (fraction of 24GB)
    notes: str = ""         # UI hint / provenance
    approx_gb: Optional[float] = None  # rough download size for the "needs download" hint
    # Extra `vllm serve` CLI args appended by web/start_vllm.sh at launch (e.g.
    # GLM's reasoning parser). Empty for models that need no special flags.
    extra_serve_args: tuple = ()
    # Extra keys merged TOP-LEVEL into the chat/completions payload by
    # _vlm_chat_payload when this model is active (e.g. disabling GLM's thinking
    # so 64-token reads don't come back empty). Empty for others.
    extra_body: dict = field(default_factory=dict)
    # Which bumper-read prompt/schema web/app.py's _vllm_read_bumpers sends for
    # this model: "incumbent" (the original prompt + _BUMPER_READ_SCHEMA) or
    # "escape" (subject decided first: bumper/no_digits/printed/not_robot, so
    # junk crops have an explicit way out instead of a copied roster number).
    # The escape prompt makes Qwen3-VL models over-conservative, so it is a
    # per-model choice, not a global switch.
    read_prompt: str = "incumbent"


# Order here is the UI order (default first). Every model is served by the same
# vLLM server in HF format; the phase-exclusive lifecycle gives the reader the
# whole card during the analysis phase, so the FP8 quants can run at high
# gpu_util. Repo ids + gpu_util verified on HF 2026-07-22.
_REGISTRY: dict[str, VlmModel] = {
    "qwen3-vl-4b": VlmModel(
        key="qwen3-vl-4b",
        label="Qwen3-VL-4B (default)",
        hf_repo="Qwen/Qwen3-VL-4B-Instruct",
        gpu_util=0.45,
        notes="Incumbent — fp16, fast, adequate for bumper digits.",
        approx_gb=8.0,
    ),
    "glm-4.6v-flash": VlmModel(
        key="glm-4.6v-flash",
        label="GLM-4.6V-Flash (FP8)",
        hf_repo="Mitchins/GLM-4.6V-Flash-FP8-Block128",
        # 0.70×24 = 16.8GB: 12.4GB FP8 weights + ~4.4GB KV. Not higher: the
        # analysis loop runs YOLO/DINOv2/person-det (~3GB) in the app process
        # on the same card while the reader serves (the same coexistence that
        # pins the incumbent at 0.45).
        gpu_util=0.70,
        notes="FP8 block-128 (12.4 GB, vision tower BF16); bf16 original doesn't "
              "fit the 3090. Thinking disabled for bumper reads.",
        approx_gb=12.4,
        # GLM-4.6V needs its reasoning parser to serve, and it THINKS by default:
        # at the 64-token bumper-read budget the reasoning channel eats the whole
        # allowance and the reply content comes back EMPTY (verified live). The
        # extra_body disables thinking so the 64-token guided read returns JSON.
        extra_serve_args=("--reasoning-parser", "glm45"),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "qwen3-vl-8b": VlmModel(
        key="qwen3-vl-8b",
        label="Qwen3-VL-8B (FP8)",
        hf_repo="Qwen/Qwen3-VL-8B-Instruct-FP8",
        # 0.65×24 = 15.6GB: 10.6GB FP8 weights + ~5GB KV, leaving room for the
        # app process's analysis-loop models (see the GLM entry's note).
        gpu_util=0.65,
        notes="Official FP8 (10.6 GB); runs W8A16 via Marlin on the 3090. "
              "Larger, slower, finer reads than 4B.",
        approx_gb=10.6,
    ),
    "lfm2.5-vl": VlmModel(
        key="lfm2.5-vl",
        label="LFM2.5-VL 1.6B",
        hf_repo="LiquidAI/LFM2.5-VL-1.6B",
        gpu_util=0.45,
        notes="bf16 (3.2 GB), tiny/fast. Needs vLLM >=0.23 (we run 0.25.1, "
              "upgraded 2026-07-22); verified serving on this stack.",
        approx_gb=3.2,
    ),
    "qwen3.5-4b": VlmModel(
        key="qwen3.5-4b",
        label="Qwen3.5-4B (escape prompt)",
        hf_repo="Qwen/Qwen3.5-4B",
        # 0.5x24 = 12GB: 9.3GB bf16 weights + ~2.7GB KV/encoder. Not higher: the
        # analysis loop runs YOLO/DINOv2/person-det (~3GB) in the app process
        # on the same card while the reader serves (see the GLM entry's note).
        gpu_util=0.5,
        notes="bf16 (9.3 GB), served with the escape read prompt. 2026-11-05 "
              "200-crop bench: 70/90 correct, 0 wrong, 2% FP at 0.9 s/crop "
              "(incumbent prompt: 87% FP — copies roster numbers onto ball "
              "piles/field structures; Qwen3-VL-8B: 69/0/1% at 2.1 s).",
        approx_gb=9.3,
        # start_vllm.sh passes --dtype float16 then appends "$@"; argparse is
        # last-wins (verified live), so this runs Qwen3.5 (hybrid Gated-DeltaNet)
        # in its native bf16.
        extra_serve_args=("--dtype", "bfloat16"),
        # Qwen3.5 thinks by default; at the 64-token read budget the reasoning
        # eats the allowance and the reply comes back empty (as with GLM).
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        read_prompt="escape",
    ),
}


def model_keys() -> tuple:
    """Registry keys in UI order (default first) — the vlm_model knob choices."""
    return tuple(_REGISTRY.keys())


def get(key: str) -> VlmModel:
    model = _REGISTRY.get(key)
    if model is None:
        raise ValueError(
            f"unknown VLM model: {key!r}; valid: {list(_REGISTRY)}")
    return model


def payload_model(key: str, default: Optional[str] = None) -> str:
    """The value for the chat payload's ``model`` field — the served HF id.

    vLLM matches this against its served model id, which is the launched HF repo.
    """
    try:
        return get(key).hf_repo
    except ValueError:
        return default if default is not None else key


def gpu_util(key: str) -> float:
    return get(key).gpu_util


def extra_serve_args(key: str) -> list:
    """Extra `vllm serve` CLI args for this model (empty list if none)."""
    return list(get(key).extra_serve_args)


def extra_body(key: str) -> dict:
    """Extra chat/completions payload keys for this model (empty dict if none)."""
    return dict(get(key).extra_body)


def read_prompt(key: str) -> str:
    """The bumper-read prompt variant for this model ("incumbent" | "escape").

    Unknown keys get "incumbent" so a bad key never changes the read prompt."""
    try:
        return get(key).read_prompt
    except ValueError:
        return "incumbent"


def served_matches(key: str, served_id: Optional[str]) -> bool:
    """True if ``served_id`` (from GET /v1/models) is the model for ``key``.

    vLLM reports the HF repo it was launched with, so a match means the served
    id equals (or contains) the registry's hf_repo. Substring keeps it robust to
    a trailing revision/suffix vLLM might append.
    """
    if not served_id:
        return False
    return get(key).hf_repo in served_id


def _hf_cached(repo_id: str) -> bool:
    """True if ``repo_id`` has a snapshot in the local HF hub cache."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        return any(r.repo_id == repo_id for r in info.repos)
    except Exception:
        return False


def status(key: str) -> str:
    """``ready`` if the repo is in the HF cache, else ``needs_download``.

    vLLM fetches an uncached repo automatically on first serve, so
    ``needs_download`` is selectable — it just means the first run pays the
    download. There is no "downloading" race state (vLLM owns the fetch).
    """
    model = get(key)
    return "ready" if _hf_cached(model.hf_repo) else "needs_download"


def describe() -> list[dict]:
    """Registry rows for the /api/vlm_models endpoint / UI picker."""
    return [
        {
            "key": m.key,
            "label": m.label,
            "hf_repo": m.hf_repo,
            "status": status(m.key),
            "notes": m.notes,
            "approx_gb": m.approx_gb,
        }
        for m in _REGISTRY.values()
    ]
