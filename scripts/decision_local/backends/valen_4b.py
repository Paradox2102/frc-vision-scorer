"""Valen-Decision-1.5-4B backend for the Jev-wire shim.

Model:    --model repo (default Valen-Team/Valen-Decision-1.5-4B; same file serves the 9B via
          SHIM_ARGS="--model Valen-Team/Valen-Decision-1.5-9B") (Qwen3.5 base + decision head), Apache-2.0,
          trust_remote_code (vendor files read; HF-download-only, no network/subprocess).
Readout:  the model's native ``model.predict({"state": ..., "questions": {...}})`` -- the
          "SystemOne" internal request format, shared-state execution (one forward pass,
          block-bidirectional attention), calibrated per-candidate softmax over the decision
          head's candidate logits (temperature 1.0). No generation, no thinking.
Mapping:  the bench's request is already in the vendor's native schema:
            - questions: passed verbatim (``type: choice``, ``instructions``, ``criteria``
              {option_id: description}); the vendor prompt is ``Question i / Task: choice /
              Question: <instructions> / Candidates: <id>: <description>``.
            - state: one user message = ["crop:" text, the image, the state's ``context``
              text]. The image is written to a temp file (vendor API takes file paths) and
              processed by the vendor's Qwen3.5 processor at native resolution (no resize).
          Nothing is dropped. Option order is kept as sent.
Venv:     (SCRATCH = any dir; fresh Ubuntu, driver for CUDA >= 12.8 incl. 13.x; needs uv)
            uv venv $SCRATCH/venvs/valen-4b --python 3.12
            uv pip install --python $SCRATCH/venvs/valen-4b/bin/python \
                "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128
            uv pip install --python $SCRATCH/venvs/valen-4b/bin/python \
                "transformers==5.23.0" pillow av
Run:      GPU_LOCK=$SCRATCH/gpu.lock scripts/decision_local/run_bench.sh \
              $SCRATCH/venvs/valen-4b/bin/python scripts/decision_local/backends/valen_4b.py \
              valen-4b 8103 --options alliance --score
"""
import tempfile
from pathlib import Path

import torch
from transformers import AutoModel

MODEL_ID = "Valen-Team/Valen-Decision-1.5-4B"


def _allow_missing_flash_attn():
    """block_attention.py imports flash_attn only inside the flash-attention code path, which
    sdpa never takes; transformers' static import check would still demand the package.
    Skip just that one requirement (no vendor file edited, no flash_attn build needed)."""
    from transformers import dynamic_module_utils as dmu
    orig = dmu.check_imports

    def check_imports(filename):
        try:
            return orig(filename)
        except ImportError as e:
            if "flash_attn" in str(e) and "pip install flash_attn" in str(e) and str(e).count("pip install") == 1:
                return dmu.get_relative_imports(filename)
            raise
    dmu.check_imports = check_imports


class Backend:
    name = "valen-decision"

    def __init__(self, args):
        _allow_missing_flash_attn()
        repo = args.model or MODEL_ID
        self.name = repo.rstrip("/").split("/")[-1].lower()
        self.model = AutoModel.from_pretrained(
            repo, trust_remote_code=True, dtype="auto", attn_implementation="sdpa",
        ).to(args.device).eval()
        # Card: full-precision matmuls for the heads.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        self.tmp = tempfile.TemporaryDirectory(prefix="valen-bench-")

    def answer(self, state, images, questions):
        context = state.get("context", "") if isinstance(state, dict) else str(state)
        content = []
        for i, (ref, img) in enumerate(images.items()):
            path = Path(self.tmp.name) / f"{i}.png"
            img.save(path)
            label = next((k for k, v in state.items() if v == ref), ref) if isinstance(state, dict) else ref
            content.append({"type": "text", "text": f"{label}:"})
            content.append({"type": "image_url", "image_url": {"url": str(path)}})
        content.append({"type": "text", "text": context})
        request = {"state": {"messages": [{"role": "user", "content": content}]},
                   "questions": {qid: {"type": "choice", "instructions": q["instructions"],
                                       "criteria": dict(q["criteria"])}
                                 for qid, q in questions.items()}}
        res = self.model.predict(request, media_root=self.tmp.name)
        return {"answers": {qid: {"probabilities": a["probabilities"]}
                            for qid, a in res["answers"].items()},
                "input_tokens": res["usage"]["input_tokens"]}
