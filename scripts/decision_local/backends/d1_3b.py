"""Backend: LiquidAI/d1-3B (LFM2.5-VL-3B post-trained as a one-pass "System One" decision model).

Repo:    LiquidAI/d1-3B   (trust_remote_code; vendor code = modeling_d1/runner/api/prompt/hybrid/lfm2_vl)
Licence: LFM Open License v1.0 (HF "other"). Apache-2.0-style grant (use, modify, redistribute,
         sublicense; patent grant) BUT commercial use is only licensed while the user/entity's
         annual revenue is below US$10M; above that, commercial use is not licensed (non-profits
         501(c)(3) exempt for non-commercial/research). Redistribution requires shipping the licence
         and marking modified files.
Readout: the model's native ``model.system_one(state, questions, images=[...])``: one forward pass,
         zero generated tokens; each choice question's probabilities are a softmax over the
         single-token option-letter logits (A, B, C ...) at the answer slot. No generation/parsing.

Request mapping:
  * state (dict) is rendered by the vendor as indented JSON under "QUESTION:". The image goes in as
    the native ``images=[PIL]`` argument (placed at the head of the user turn by the chat template).
    The only edit: the state's ``crop`` asset ref (e.g. "crop") is replaced by the words
    "the attached image", because the vendor API has no named assets; the context text, which
    refers to `crop`, is otherwise untouched.
  * questions pass through verbatim ({"type": "choice", "instructions", "criteria"}): the vendor
    prints the instructions, then lists criteria descriptions as lettered options ("A <description>").
    Option ids (bumper / 1678 / not_robot ...) are NOT shown to the model, only their descriptions
    (the vendor's default option_style "desc"); the letters are mapped back to ids by position.
  * Both questions are answered in a single request, state+image read once (the vendor's tree path).
  * Lost: nothing material. Images are capped at 1 Mpx by the vendor and tiled by the processor.

Venv recipe (SCRATCH=<scratchpad>):
  uv venv $SCRATCH/venvs/d1-3b --python 3.12
  uv pip install --python $SCRATCH/venvs/d1-3b/bin/python "transformers>=5.14" torch torchvision pillow
  (default PyPI torch wheels are CUDA 12.x builds and run on the 3090)
Run: GPU_LOCK=$SCRATCH/gpu.lock scripts/decision_local/run_bench.sh \\
       $SCRATCH/venvs/d1-3b/bin/python scripts/decision_local/backends/d1_3b.py d1-3b 8105 --score
"""
from __future__ import annotations

import copy
import sys

import torch


class Backend:
    name = "d1-3b"

    def __init__(self, args):
        from transformers import AutoModel

        repo = args.model or "LiquidAI/d1-3B"
        torch.cuda.reset_peak_memory_stats()
        self.model = AutoModel.from_pretrained(
            repo, trust_remote_code=True, dtype=torch.bfloat16).to(args.device).eval()
        self.device = args.device
        # warm the engine (tokeniser + processor) so first-request latency is honest
        print(f"[d1_3b] loaded {repo}; peak VRAM after load "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", file=sys.stderr, flush=True)
        self._n = 0

    def answer(self, state, images, questions):
        pics = list(images.values())
        st = copy.deepcopy(state)
        if isinstance(st, dict) and images:
            for k, v in st.items():
                if isinstance(v, str) and v in images:
                    st[k] = "the attached image"
        res = self.model.system_one(st, questions, images=pics)
        self._n += 1
        if self._n % 25 == 1:
            print(f"[d1_3b] req {self._n}: peak VRAM "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", file=sys.stderr, flush=True)
        return {"answers": {qid: {"probabilities": a["probabilities"]}
                            for qid, a in res["answers"].items()},
                "input_tokens": res["usage"]["input_tokens"]}
