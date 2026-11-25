"""Cloudflare Clef decision models (clef-flash 9B; also Cloudflare/clef 27B) behind the
Jev-wire shim, in-process via the vendor's ``joint_schema_model.py``.

Repo:     --model (shim arg), default ``Cloudflare/clef-flash`` (9B, Qwen3.5-9B backbone).
          ``Cloudflare/clef`` (27B) ships the byte-identical joint_schema_model.py and the
          same processor/architecture (Qwen3_5ForConditionalGeneration), so
          ``SHIM_ARGS="--model Cloudflare/clef"`` should work unchanged (needs ~55 GB bf16).
Licence:  Apache-2.0 (both repos).

Readout:  the model's native joint schema head -- one logit per allowed option of every
          question in a single forward pass (no generation, no thinking). We call the
          vendor's ``encode_record`` -> ``collate_records`` -> ``ClefModel.forward`` and
          softmax each question's logits, exactly as the vendor ``systemone()`` does, but
          without its 4-decimal rounding (keeps small probabilities for calibration).

Mapping:  the Jev request maps 1:1 onto the vendor record:
            state      -> record["state"] as-is (dict rendered by the vendor as sorted
                          compact JSON; the ``crop`` field keeps its asset-ref string)
            assets     -> record["images"] (PIL, request order). The vendor format has no
                          asset refs: images are always placed as vision tokens at the
                          start of the STATE block, before the state JSON.
            questions  -> record["questions"] unchanged (type/instructions/criteria all
                          encoded; the vendor sorts choice options by option id).
          Image size: vendor processor defaults (min 65,536 px, 16 px patches, 2x2 merge);
          the bench's jev_image prep (short side >= 280) already exceeds the minimum.
          Optional (off by default): ``--max-pixels N`` (shim extra arg via SHIM_ARGS) ->
          media_kwargs size={"longest_edge": N, ...} to cap image tokens (transformers 5
          ignores a plain ``max_pixels`` kwarg).

Weights:  bf16 on cuda, default HF cache (snapshot_download); no quantisation.

Venv (fresh Ubuntu, NVIDIA driver supporting CUDA >= 12.8; cu128 wheels):
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv venvs/clef --python 3.12
    uv pip install --python venvs/clef/bin/python \\
        "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" \\
        --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python venvs/clef/bin/python \\
        "transformers==5.10.2" pillow safetensors huggingface_hub accelerate
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


DEFAULT_REPO = "Cloudflare/clef-flash"


def _import_vendor(snapshot: Path):
    spec = importlib.util.spec_from_file_location(
        "joint_schema_model", snapshot / "joint_schema_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["joint_schema_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def _extra_int(extra: list, flag: str):
    for i, tok in enumerate(extra or []):
        if tok == flag and i + 1 < len(extra):
            return int(extra[i + 1])
        if tok.startswith(flag + "="):
            return int(tok.split("=", 1)[1])
    return None


class Backend:
    def __init__(self, args):
        import torch
        from huggingface_hub import snapshot_download

        self.torch = torch
        repo = args.model or DEFAULT_REPO
        path = Path(repo) if Path(repo).is_dir() else Path(snapshot_download(repo))
        self.jsm = _import_vendor(path)
        self.model, self.processor = self.jsm.load_release_model(path, device=args.device)
        self.device = torch.device(args.device)
        self.name = repo.rstrip("/").split("/")[-1]
        max_pixels = _extra_int(getattr(args, "extra", None), "--max-pixels")
        self.media_kwargs = ({"size": {"longest_edge": max_pixels,
                                       "shortest_edge": min(65536, max_pixels)}}
                             if max_pixels else None)
        if args.device.startswith("cuda"):
            print(f"[clef] weights on GPU: {torch.cuda.memory_allocated() / 2**30:.2f} GiB",
                  flush=True)

    def answer(self, state, images, questions):
        torch = self.torch
        record = {"state": state, "questions": questions}
        if images:
            record["images"] = list(images.values())
        if self.media_kwargs:
            record["media_kwargs"] = self.media_kwargs
        tok = self.processor.tokenizer
        enc = self.jsm.encode_record(tok, record, processor=self.processor)
        batch = self.jsm.collate_records([enc], tok.pad_token_id, self.device)
        with torch.inference_mode():
            logits = self.model(batch)[0]
        answers = {}
        for q, ql in zip(enc.questions, logits):
            probs = ql.float().softmax(-1).tolist()
            answers[q.question_id] = {"probabilities": dict(zip(q.option_ids, probs))}
        return {"answers": answers, "input_tokens": len(enc.input_ids)}
