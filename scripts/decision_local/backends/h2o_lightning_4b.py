"""H2O-Lightning-4B (h2oai/h2o-lightning-4b) as a bumper-bench decision backend.

Repo:     h2oai/h2o-lightning-4b, pinned revision 659fc412de8486dbc2477b0d1cb4121e97e72471
          (a fine-tune of Qwen/Qwen3.5-4B; Qwen3_5ForConditionalGeneration, bf16 weights, 9.3 GB).
Licence:  Apache-2.0 (base Qwen3.5-4B: Apache-2.0).

Readout (the vendor's native one, run in-process with transformers instead of through vLLM):
  The vendor ships a stdlib "shim" (h2o_lightning_shim.py) that sits in front of vLLM 0.30. For each question it
  renders the trained prompt (Qwen3.5 chat template, thinking off, the decision-engine system prompt, the "plain"
  layout `record: ... / question: ... / options: A) name: description ...`, compact JSON for a dict state, images
  as image parts ahead of the user text, assistant turn pre-filled with "Answer:"), asks vLLM for ONE output token
  with `logprob_token_ids` = the option labels' tokens, and turns those into probabilities with softmax(lp / T)
  (T = serve_config.json's image temperature, 1.0, for any request that carries an image). There is no
  transformers/pooled path in the vendor package: pooled_readout.py is for a vLLM pooling-runner engine.
  This backend imports the vendor shim module itself (sha256-pinned) and uses its OWN functions for everything
  except the forward pass: Contract (serve_config.json), bind_assets + extract_images (the `assets` ref -> image
  binding, `[image 1]` placeholder), render_state, Contract.image_prompt / question_prompt (the exact prompt text),
  the label-token derivation of Shim.check_backend, and format_answer (temperature, choice/noul/score shapes).
  The forward pass is transformers' Qwen3_5 model: the final-norm hidden state at the last prompt position times the
  lm_head rows of the asked labels' tokens, in fp32 (the card's `head_dtype: float32`); the softmax restricted to
  those tokens equals the shim's softmax over vLLM's full-vocab logprobs (the normaliser cancels).

Attention mode: two-way (the vendor's recommended default, `serve_config.json two_way.default`): bidirectional
  attention on the 8 full-attention layers, Gated DeltaNet layers causal -- implemented here by passing no mask and
  `is_causal=False` on those layers' SDPA. `--attention-mode one-way` (or env H2O_LIGHTNING_MODE=one-way) reads
  causally like stock vLLM. At start-up the backend reads the vendor's three TWO_WAY_PROBES and refuses to start if
  the two-way answers are not within serve_config.json's probe_tolerance (0.05) of the release's recorded answers,
  or if two-way and one-way do not differ by >= 0.05 (the vendor's own start-up guard).

Request mapping (bench -> native): state {"crop": "crop_image", "context": "..."} + assets {"crop_image": jpg} is
  rebuilt into the vendor wire (`assets` with the decoded crop re-encoded as lossless PNG) and bound by the vendor's
  own assets.bind, so the record becomes {"crop":"[image 1]","context":"..."} with the crop as image part 1. Question
  instructions and every criteria description go into the prompt verbatim (nothing dropped). Each question is one
  forward pass; the crop's vision features are computed once per request and reused. Images are resized by the
  model's Qwen2VL image processor with longest_edge = serve_config.json's max_pixels (1,638,400) and the shipped
  shortest_edge (65,536), as vLLM would. Long-record fitting (BM25 selector) is not implemented: bench prompts are
  ~1k tokens, far under the vendor's 40,960 context; an over-long prompt raises instead.

Venv recipe (SCRATCH = the agent scratchpad):
  uv venv $SCRATCH/venvs/h2o-lightning-4b --python 3.12
  uv pip install --python $SCRATCH/venvs/h2o-lightning-4b/bin/python torch==2.13.0 torchvision==0.28.0 \
      --index-url https://download.pytorch.org/whl/cu129
  uv pip install --python $SCRATCH/venvs/h2o-lightning-4b/bin/python "transformers==5.20.2" pillow numpy \
      huggingface_hub accelerate
  (No flash-linear-attention / causal-conv1d: transformers' torch fallbacks run the Gated DeltaNet layers.
  cuDNN is disabled in-process; see __init__.)
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path

REPO_ID = "h2oai/h2o-lightning-4b"
REVISION = "659fc412de8486dbc2477b0d1cb4121e97e72471"
# sha256 of the vendor files this backend executes (reviewed copies); the shim itself pins its siblings
VENDOR_SHA256 = {
    "h2o_lightning_shim.py": "287434ee8e1df2b6b2fe52b86f503c33cc536e14ad822ee36855f81645572870",
    "assets.py": "e30ff2efd7628ade3cdc548c936751db9c433e6ad3a143f4d122b0baddcdc94c",
    "box.py": "e26ad9d8ea1ac501730d2d770c2aeebda642962bdb1d591c86f0cfb843274301",
    "selector_v2.py": "bd518a2c3bea9167803b6c21e480b37386fa212cf35101093b84e75c5307b7e3",
    "pooled_readout.py": "d59e62ccb3b5af742021e8064a5987ee27291744ea043ad94fbc8baa4eee1199",
    "serve_config.json": "f4fa2806a0c4d332c7dc271244c292a29ab94bd844bf5e68c3dcf29690e5828e",
}
FILES = ["model.safetensors", "config.json", "generation_config.json", "preprocessor_config.json",
         "chat_template.jinja", "tokenizer/*", "tokenizer_contract.json", *VENDOR_SHA256]
MAX_LEN = 40960                      # serve.sh's --max-model-len
IMAGE_MIN_PIXELS = 65536             # preprocessor_config.json size.shortest_edge (vLLM keeps it)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_vendor_shim(snap: Path):
    for name, want in VENDOR_SHA256.items():
        got = _sha256(snap / name)
        if got != want:
            raise RuntimeError(f"{name} sha256 {got[:16]} is not the reviewed {want[:16]}")
    spec = importlib.util.spec_from_file_location("h2o_lightning_shim", snap / "h2o_lightning_shim.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h2o_lightning_shim"] = mod
    spec.loader.exec_module(mod)
    return mod


def _mode_from(extra) -> str:
    mode = os.environ.get("H2O_LIGHTNING_MODE", "two-way")
    extra = list(extra or [])
    for i, a in enumerate(extra):
        if a == "--attention-mode" and i + 1 < len(extra):
            mode = extra[i + 1]
        elif a.startswith("--attention-mode="):
            mode = a.split("=", 1)[1]
    return mode


class Backend:
    def __init__(self, args):
        os.environ.setdefault("LIGHTNING_IMAGE_URLS", "off")   # the shim never fetches links (none are sent anyway)
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

        self.torch = torch
        # cuDNN off: on this host cuDNN dlopens a sub-library from the system's /lib/x86_64-linux-gnu cuDNN 9 that
        # does not match the wheel's 9.20 (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH in the vision tower's Conv3d
        # patch embed). That conv is a stride=kernel patch projection; torch's native kernel computes the same.
        torch.backends.cudnn.enabled = False
        snap = Path(args.model) if args.model and Path(args.model).is_dir() else Path(
            snapshot_download(REPO_ID, revision=REVISION, allow_patterns=FILES))
        S = self.S = _load_vendor_shim(snap)
        cfg = json.loads((snap / "serve_config.json").read_text())
        cfg["_config_dir"] = str(snap)
        self.c = S.Contract(cfg)
        self.mode = S._mode_of(_mode_from(getattr(args, "extra", None)), "bidirectional")
        self.name = f"{self.c.model}@{REVISION[:8]} ({'two-way' if self.mode == 'bidirectional' else 'one-way'}, transformers)"

        # the training tokenizer (Qwen2Tokenizer from tokenizer/, as serve.sh passes --tokenizer); check its contract
        self.tok = AutoTokenizer.from_pretrained(str(snap / "tokenizer"))
        contract = snap / "tokenizer_contract.json"
        if _sha256(contract) != S.TOKENIZER_CONTRACT_SHA256:
            raise RuntimeError("tokenizer_contract.json is not the shipped file")
        probes = json.loads(contract.read_text())["probes"]
        bad = [p["id"] for p in probes if self._encode(p["text"]) != p["token_ids"]]
        if bad:
            raise RuntimeError(f"tokenizer splits {len(bad)}/{len(probes)} contract probes differently: {bad[:6]}")
        self._label_ids()
        self._check_chat()

        self.improc = AutoImageProcessor.from_pretrained(str(snap))
        self.image_token = "<|image_pad|>"
        self.dev = getattr(args, "device", None) or "cuda"
        t0 = time.perf_counter()
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(snap), dtype=torch.bfloat16, attn_implementation="sdpa").to(self.dev).eval()
        self.image_token_id = int(self.model.config.image_token_id)
        if self.tok.convert_tokens_to_ids(self.image_token) != self.image_token_id:
            raise RuntimeError("tokenizer and config disagree on the image token id")
        self.full_attn = [layer.self_attn for layer in self.model.model.language_model.layers
                          if hasattr(layer, "self_attn")]            # the full-attention layers (GDN: linear_attn)
        W = self.model.lm_head.weight
        ids = [self.c.label_ids[l] for l in self.c.labels]
        self.label_rows = {i: k for k, i in enumerate(ids)}
        self.W = W[torch.tensor(ids, device=W.device)].float()          # [n_labels, hidden] fp32 head rows
        print(f"[h2o-lightning] weights on {self.dev} in {time.perf_counter() - t0:.1f}s; "
              f"{len(self.full_attn)} full-attention layers; mode {self.mode}", flush=True)
        self._probe_guard()
        torch.cuda.reset_peak_memory_stats()
        self.n_req = 0

    # ------------------------------------------------------------------ tokenizer / prompt checks
    def _encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _label_ids(self):
        """Shim.check_backend: every label is ONE token after the answer prefill (" <label>")."""
        S, c = self.S, self.c
        n = len(c.labels)
        probe = {"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": f"d{i}" for i in range(n)}}
        text = c.question_prompt(S.render_state({"x": 1}, c.state_compact), probe)[0]
        base = self._encode(text)
        ids, bad = {}, []
        for label in c.labels:
            full = self._encode(text + " " + label)
            if len(full) != len(base) + 1 or full[:len(base)] != base:
                bad.append(label)
            else:
                ids[label] = full[-1]
        if bad or len(set(ids.values())) != len(ids):
            raise RuntimeError(f"labels not one distinct token at the answer slot: {bad[:8]}")
        c.label_ids = ids

    def _check_chat(self):
        """Shim.check_chat: the chat template renders the image-path messages to the text-path prompt."""
        S, c = self.S, self.c
        probe = {"type": "choice", "instructions": "Which?", "criteria": {"a": "first", "b": "second"}}
        user, *_ = c.question_parts(S.render_state({"x": 1}, c.state_compact), probe)
        want = self._encode(S.render_chat(c.system, user, c.prefill, c.chat))
        msgs = S.chat_messages(c.system, user, c.prefill, [])
        got = self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                           continue_final_message=True, enable_thinking=False)
        if hasattr(got, "keys"):
            got = got["input_ids"]
        if list(got) != want:
            raise RuntimeError("chat template does not render the trained text prompt")

    # ------------------------------------------------------------------ forward
    def _logits(self, text: str, image=None, mode=None):
        """Label logits (fp32) at the last position of `text`; `image` = (feats, grid_thw, tokens per image)."""
        torch = self.torch
        mode = mode or self.mode
        if image is not None:
            _feats, _grid, per_image = image
            parts = text.split(self.image_token)
            if len(parts) - 1 != len(per_image):
                raise RuntimeError("image slots in the prompt do not match the images")
            text = parts[0] + "".join(self.image_token * n + p for n, p in zip(per_image, parts[1:]))
        ids = self._encode(text)
        if len(ids) + self.S.CONTEXT_SLACK > MAX_LEN:
            raise RuntimeError(f"prompt of {len(ids)} tokens is over the {MAX_LEN}-token context")
        input_ids = torch.tensor([ids], device=self.dev)
        m = self.model.model
        kw = {}
        if image is not None:
            feats, grid, _ = image
            mm_types = (input_ids == self.image_token_id).int()
            pos, _ = m.get_rope_index(input_ids, mm_token_type_ids=mm_types, image_grid_thw=grid)
            kw = {"mm_encoder_outputs": {"image": feats}, "image_grid_thw": grid, "position_ids": pos,
                  "mm_token_type_ids": mm_types}
        causal = mode != "bidirectional"
        for a in self.full_attn:
            a.is_causal = causal
        try:
            out = m(input_ids=input_ids, attention_mask={"full_attention": None, "linear_attention": None},
                    use_cache=False, **kw)
        finally:
            for a in self.full_attn:
                a.is_causal = True
        h = out.last_hidden_state[0, -1].float()
        return (self.W @ h).tolist(), ids

    def _image_feats(self, pil_images):
        torch = self.torch
        enc = self.improc(images=pil_images, return_tensors="pt",
                          size={"shortest_edge": IMAGE_MIN_PIXELS,
                                "longest_edge": int(self.c.max_pixels or 16777216)})
        pv = enc["pixel_values"].to(self.dev)
        grid = enc["image_grid_thw"].to(self.dev)
        merge = self.model.model.visual.spatial_merge_size
        per_image = (grid.prod(-1) // merge ** 2).tolist()
        feats = self.model.model.get_image_features(pv, grid, return_dict=True)
        return feats, grid, per_image

    def _read(self, prompt_text, labels, image=None, mode=None):
        """The readout as the shim's `one`/`one_chat`: per-label scores in label order (softmax-invariant logits)."""
        logits, ids = self._logits(prompt_text, image, mode)
        return [logits[self.label_rows[self.c.label_ids[l]]] for l in labels], ids

    # ------------------------------------------------------------------ start-up guard (vendor TWO_WAY_PROBES)
    def _probe_guard(self):
        S, c = self.S, self.c
        ref = c.two_way.probe_reference if c.two_way else None
        worst_ref, worst_sep = 0.0, 0.0
        with self.torch.inference_mode():
            for k, body in enumerate(S.TWO_WAY_PROBES):
                st = S.render_state(body["state"], c.state_compact)
                for qid, q in body["questions"].items():
                    text, names, _d, labels = c.question_prompt(st, q, qid)
                    p2 = S.probabilities(self._read(text, labels, None, "bidirectional")[0], 1.0)
                    p1 = S.probabilities(self._read(text, labels, None, "causal")[0], 1.0)
                    worst_sep = max(worst_sep, max(abs(a - b) for a, b in zip(p1, p2)))
                    if ref is not None:
                        d = max(abs(p - ref[k][qid][nm]) for p, nm in zip(p2, names))
                        worst_ref = max(worst_ref, d)
                        print(f"[h2o-lightning] probe {k}/{qid}: two-way {dict(zip(names, [round(x, 4) for x in p2]))}"
                              f" ref {ref[k][qid]} |d|max {d:.4f}; one-way {[round(x, 4) for x in p1]}", flush=True)
        print(f"[h2o-lightning] probe guard: max |two-way - release ref| {worst_ref:.4f} "
              f"(tol {c.two_way.probe_tolerance}), max |two-way - one-way| {worst_sep:.4f}", flush=True)
        if ref is not None and worst_ref > c.two_way.probe_tolerance:
            raise RuntimeError(f"two-way probes off the release reference by {worst_ref:.4f}")
        if worst_sep < S.TWO_WAY_MIN_SEPARATION:
            raise RuntimeError(f"two-way and one-way readouts agree to {worst_sep:.4f}: bidirectional not applied")

    # ------------------------------------------------------------------ the contract
    def answer(self, state, images, questions):
        S, c = self.S, self.c
        torch = self.torch
        assets = {}
        for ref, img in images.items():
            buf = io.BytesIO()
            img.save(buf, format="PNG")                  # lossless re-encode of the decoded crop
            assets[ref] = {"type": "image", "format": "png", "data_base64": base64.b64encode(buf.getvalue()).decode()}
        body = {"state": state, "questions": questions}
        if assets:
            body["assets"] = assets
        body = S.bind_assets(body)
        st, uris = S.extract_images(body, c.max_images)
        qs = body["questions"]
        temps, image_t, floor = self._temperatures()
        answers, ntoks = {}, []
        with torch.inference_mode():
            image = None
            if uris:
                from PIL import Image
                pils = [Image.open(io.BytesIO(base64.b64decode(u.split(",", 1)[1]))).convert("RGB") for u in uris]
                image = self._image_feats(pils)
            full_txt = S.render_state(st, c.state_compact)
            for qid, q in qs.items():
                if image is not None:
                    _msgs, text, names, descs, labels = c.image_prompt(full_txt, q, uris, qid)
                else:
                    text, names, descs, labels = c.question_prompt(full_txt, q, qid)
                lp, ids = self._read(text, labels, image)
                ntoks.append(ids)
                typ = q.get("type", "choice")
                t = image_t if image is not None else temps[typ]
                a = S.format_answer(typ, names, descs, lp, t, floor, S.noul_threshold(q))
                if typ == "noul":
                    keys = list(questions[qid].get("criteria") or {"true": 0, "false": 0})
                    probs = {keys[0]: a["noul"], keys[1]: 1.0 - a["noul"]} if len(keys) >= 2 else \
                        {"true": a["noul"], "false": 1.0 - a["noul"]}
                elif typ == "score":
                    crit = questions[qid].get("criteria")
                    keys = list(crit) if isinstance(crit, dict) else [str(i) for i in range(len(names))]
                    probs = {k: a["probabilities"][str(i)] for i, k in enumerate(keys)}
                else:
                    probs = dict(a["probabilities"])
                answers[qid] = {"probabilities": probs}
        self.n_req += 1
        if self.n_req in (1, 5) or self.n_req % 50 == 0:
            print(f"[h2o-lightning] req {self.n_req}: peak VRAM allocated "
                  f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB, "
                  f"reserved {torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GiB", flush=True)
        # usage as the vendor counts it: the shared head (system, record, image) once plus each question's tail
        shared = S.common_prefix_len(ntoks)
        return {"answers": answers, "input_tokens": sum(len(t) for t in ntoks) - shared * (len(ntoks) - 1)}

    def _temperatures(self):
        if self.mode == "bidirectional" and self.c.two_way is not None:
            tw = self.c.two_way
            return tw.temperature_by_type, tw.image_temperature, tw.noul_floor
        return self.c.temperature_by_type, self.c.image_temperature, self.c.noul_floor
