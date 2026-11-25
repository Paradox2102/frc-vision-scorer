"""Intern-Decision-4B backend for the Jev-wire shim.

Repo:     internlm/Intern-Decision-4B (Qwen3.5-4B based; Apache-2.0, derived from Qwen3.5
          whose licence ships as LICENSE-QWEN)
Readout:  native DecisionEngine.predict() from the repo's own inference.py. One causal
          forward over the chat-templated prompt (system + state + schema + images) plus a
          masked assistant JSON skeleton; logits before each <decision> placeholder are
          softmaxed over the single-token option symbols A, B, C... and temperature-
          calibrated (T=1.992, the checkpoint default). No generate(), no thinking.

Request mapping:
  - questions -> passed through unchanged (instructions + per-option descriptions are
    rendered by the vendor compiler as "A = option_id: description").
  - state     -> vendor JSON-dumps it into the user message. The bench's state is
    {"crop": "<asset ref>", "context": "..."}; the asset ref is replaced with
    "Picture 1 (the attached image)" because the vendor takes images as a separate
    ordered list (rendered with vision ids), not by ref. A string state is passed as is.
  - images    -> decoded PIL images are written losslessly as PNG to a temp dir (the vendor
    API only accepts file paths) in request order, then deleted.

Venv recipe (SCRATCH = the session scratchpad):
  uv venv $SCRATCH/venvs/intern-decision-4b --python 3.12
  uv pip install --python $SCRATCH/venvs/intern-decision-4b/bin/python \\
      torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python $SCRATCH/venvs/intern-decision-4b/bin/python \\
      transformers==5.14.1 "Pillow>=10" huggingface_hub
  # weights land in the default HF cache (first use: snapshot_download below)

The vendor inference.py is loaded from the HF snapshot directory (it loads the checkpoint
with local_files_only=True from that same directory).
"""
import copy
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ID = "internlm/Intern-Decision-4B"


class Backend:
    name = "intern-decision-4b"

    def __init__(self, args):
        from huggingface_hub import snapshot_download
        ckpt = snapshot_download(args.model or REPO_ID)
        spec = importlib.util.spec_from_file_location("intern_decision_inference",
                                                      str(Path(ckpt) / "inference.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["intern_decision_inference"] = mod
        spec.loader.exec_module(mod)
        self.engine = mod.DecisionEngine(ckpt, device=args.device)
        scratch = os.environ.get("TMPDIR") or tempfile.gettempdir()
        self.tmp = tempfile.mkdtemp(prefix="intern_decision_", dir=scratch)
        import atexit
        atexit.register(shutil.rmtree, self.tmp, True)

    def answer(self, state, images, questions):
        paths, refs = [], []
        for i, (ref, img) in enumerate(images.items()):
            p = os.path.join(self.tmp, f"img{i}.png")
            img.save(p)
            paths.append(p)
            refs.append(ref)
        st = copy.deepcopy(state)
        if isinstance(st, dict):
            for k, v in list(st.items()):
                if isinstance(v, str) and v in refs:
                    st[k] = f"Picture {refs.index(v) + 1} (the attached image)"
        request = {"state": st, "questions": questions}
        if paths:
            request["images"] = paths
        res = self.engine.predict(request)
        return {"answers": {q: {"probabilities": a["probabilities"]}
                            for q, a in res["answers"].items()},
                "input_tokens": res["usage"].get("input_tokens")}
