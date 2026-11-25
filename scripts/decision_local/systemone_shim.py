#!/usr/bin/env python3
"""Minimal Jev-wire (`POST /v1/systemone`) server around a local decision model.

Lets the bumper-crop bench (`scripts/bench_bumper_readers.py run --arms local`)
score open-weight decision models with the jev arm's exact payload. Stdlib HTTP
only, so it runs inside whatever venv a model's vendor code needs:

    <model-venv>/bin/python scripts/decision_local/systemone_shim.py \\
        --backend scripts/decision_local/backends/clef_flash.py --port 8100

Wire (the subset the bench sends; mirrors TypeSafe Jev 1.14):

    request  {"model": str, "state": dict|str,
              "assets": {ref: {"type": "image", "format": "jpg", "data_base64": str}},
              "questions": {qid: {"type": "choice", "instructions": str,
                                  "criteria": {option_id: description}}}}
    response {"model": str,
              "answers": {qid: {"type": "choice", "choice": option_id,
                                "probabilities": {option_id: p}, "confidence": p}},
              "usage": {"input_tokens": int|None}}

A backend module defines ``class Backend`` with:

    def __init__(self, args: argparse.Namespace)   # load weights (args.device, args.model)
    name: str                                      # reported as response["model"]
    def answer(self, state, images, questions) -> dict
        # state: the request's state (dict or str), asset refs left in place
        # images: {ref: PIL.Image.Image (RGB)} in request order
        # questions: the request's questions dict (ordered)
        # returns {"answers": {qid: {"probabilities": {opt: p}, ...}},
        #          "input_tokens": int|None}
        # probabilities must cover every criteria key and sum to ~1.

Requests are served one at a time (one model, one GPU).
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def load_backend(path: str):
    spec = importlib.util.spec_from_file_location("decision_backend", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["decision_backend"] = mod
    spec.loader.exec_module(mod)
    return mod.Backend


def normalise(answers: dict, questions: dict) -> dict:
    """Fill choice/confidence, renormalise, and check every option is present."""
    out = {}
    for qid, q in questions.items():
        a = answers.get(qid)
        if a is None:
            raise ValueError(f"backend returned no answer for question {qid!r}")
        probs = {str(k): float(v) for k, v in a["probabilities"].items()}
        missing = set(q.get("criteria", {})) - set(probs)
        if missing:
            raise ValueError(f"{qid}: backend probabilities miss options {sorted(missing)}")
        tot = sum(probs.values())
        if tot <= 0:
            raise ValueError(f"{qid}: probabilities sum to {tot}")
        probs = {k: v / tot for k, v in probs.items()}
        top = max(probs, key=probs.get)
        out[qid] = {"type": q.get("type", "choice"), "choice": top,
                    "probabilities": probs, "confidence": probs[top],
                    **{k: v for k, v in a.items() if k not in ("probabilities",)}}
        if q.get("type") == "noul" and "true" in probs:
            out[qid]["noul"] = probs["true"]  # Workers AI / Jev noul answer field
    return out


def decode_data_url(s) -> bytes:
    """Workers AI image item: a data URL string or {"content_type", "base64"}."""
    if isinstance(s, dict):
        return base64.b64decode(s["base64"])
    return base64.b64decode(s.split(",", 1)[1] if s.startswith("data:") else s)


def make_handler(backend, lock: threading.Lock):
    from PIL import Image

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # quiet; the bench records latency
            pass

        def _send(self, code: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") in ("/healthz", "/health"):
                self._send(200, {"ok": True, "model": backend.name})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            # /v1/systemone: Jev wire (named `assets`, referenced from the state).
            # /v1/clef: Workers AI Clef wire (`images` list of data URLs, placed before
            # the state; response wrapped in {"success", "result"}).
            route = self.path.rstrip("/")
            if route not in ("/v1/systemone", "/v1/clef"):
                self._send(404, {"error": "not found"})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                images = {}
                if route == "/v1/clef":
                    for i, item in enumerate(req.get("images") or []):
                        images[f"image_{i}"] = Image.open(
                            io.BytesIO(decode_data_url(item))).convert("RGB")
                for ref, asset in (req.get("assets") or {}).items():
                    if asset.get("type") != "image":
                        raise ValueError(f"asset {ref!r}: only images are supported")
                    raw = base64.b64decode(asset["data_base64"])
                    images[ref] = Image.open(io.BytesIO(raw)).convert("RGB")
                questions = req["questions"]
                t0 = time.perf_counter()
                with lock:
                    res = backend.answer(req.get("state"), images, questions)
                dt = time.perf_counter() - t0
                body = {
                    "model": backend.name,
                    "answers": normalise(res["answers"], questions),
                    "usage": {"input_tokens": res.get("input_tokens"), "output_tokens": 0},
                    "timing": {"model_s": round(dt, 4)},
                }
                if route == "/v1/clef":
                    body = {"success": True, "result": body}
                self._send(200, body)
            except Exception as e:  # noqa: BLE001 - report to the bench, keep serving
                traceback.print_exc()
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, help="path to a backend module")
    ap.add_argument("--model", default=None, help="HF repo id or local dir (backend default)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    args, extra = ap.parse_known_args()
    args.extra = extra
    Backend = load_backend(str(Path(args.backend).resolve()))
    t0 = time.perf_counter()
    backend = Backend(args)
    print(f"[shim] loaded {backend.name} in {time.perf_counter() - t0:.1f}s; "
          f"serving http://{args.host}:{args.port}/v1/systemone", flush=True)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(backend, threading.Lock()))
    srv.serve_forever()


if __name__ == "__main__":
    main()
