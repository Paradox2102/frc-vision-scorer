#!/bin/bash
# Pod-side bootstrap for the FRC vision scorer (runs ON the RunPod pod).
#
#   bash /workspace/frc/deploy/runpod/bootstrap.sh [--no-start] [--no-prefetch]
#
# Idempotent and safe to re-run (runpod_cli.py `up` runs it after every push,
# and after a pod stop/start the container disk is fresh again):
#   1. apt: ffmpeg, tesseract-ocr, libGL/glib (container disk; reinstalled after a stop)
#   2. uv + deno binaries on the volume (/workspace/tools)
#   3. /workspace/frc/.venv (Python 3.12) from requirements-runpod*.txt — skipped
#      when the requirement files' hash matches the last successful install
#   4. prefetch SAM3 (gated, needs HF_TOKEN in .env) / DINOv3 / EasyOCR weights
#      into /workspace/cache
#   5. import smoke test (CUDA usable on this GPU arch, `import web.app`)
#   6. profile.env from the GPU's VRAM tier (kept if it already exists)
#   7. (re)start the server and wait for HTTP 200 on localhost:8004
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pod_env.sh
source "$HERE/pod_env.sh"

START=1
PREFETCH=1
for arg in "$@"; do
  case "$arg" in
    --no-start) START=0 ;;
    --no-prefetch) PREFETCH=0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

UV_VERSION="0.10.0"
DENO_VERSION="2.8.1"

mkdir -p "$FRC_LOGS" "$FRC_TOOLS/bin" "$FRC_CACHE" "$FRC_TUNE" "$HF_HOME" "$EASYOCR_MODULE_PATH" \
         "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR"
cd "$FRC_APP"
step() { echo; echo "=== [bootstrap] $* ($(date +%H:%M:%S))"; }

# ---------------------------------------------------------------- 1. apt
step "system packages"
need_apt=0
for bin in ffmpeg tesseract unzip curl git; do
  command -v "$bin" >/dev/null 2>&1 || need_apt=1
done
if [ ! -e /usr/lib/x86_64-linux-gnu/libGL.so.1 ]; then need_apt=1; fi
if [ "$need_apt" = 1 ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    ffmpeg tesseract-ocr tesseract-ocr-eng libgl1 libglib2.0-0 unzip curl git ca-certificates \
    >/dev/null
fi
echo "ffmpeg: $(ffmpeg -version 2>/dev/null | head -1 | cut -c1-40)"
echo "tesseract: $(tesseract --version 2>&1 | head -1)"

# ---------------------------------------------------------------- 2. uv + deno
step "uv $UV_VERSION + deno $DENO_VERSION"
if ! "$FRC_TOOLS/bin/uv" --version 2>/dev/null | grep -q "$UV_VERSION"; then
  tmp="$(mktemp -d)"
  curl -fsSL "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" \
    | tar -xz -C "$tmp"
  install -m 755 "$tmp"/uv-x86_64-unknown-linux-gnu/uv "$tmp"/uv-x86_64-unknown-linux-gnu/uvx "$FRC_TOOLS/bin/"
  rm -rf "$tmp"
fi
uv --version
if ! "$FRC_TOOLS/deno/bin/deno" --version 2>/dev/null | grep -q "deno $DENO_VERSION"; then
  mkdir -p "$FRC_TOOLS/deno/bin"
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/deno.zip" \
    "https://github.com/denoland/deno/releases/download/v$DENO_VERSION/deno-x86_64-unknown-linux-gnu.zip"
  unzip -q -o "$tmp/deno.zip" -d "$FRC_TOOLS/deno/bin"
  chmod +x "$FRC_TOOLS/deno/bin/deno"
  rm -rf "$tmp"
fi
# web/app.py also prepends ~/.deno/bin when it exists; keep that path valid too.
mkdir -p "$HOME/.deno"
ln -sfn "$FRC_TOOLS/deno/bin" "$HOME/.deno/bin"
deno --version | head -1

# ---------------------------------------------------------------- 3. venv
step "python venv"
REQ="$HERE/requirements-runpod.txt"
REQ_GIT="$HERE/requirements-runpod-git.txt"
REQ_OVR="$HERE/requirements-runpod-overrides.txt"
want_hash="$(cat "$REQ" "$REQ_GIT" "$REQ_OVR" | sha256sum | cut -c1-16)"
have_hash="$(cat "$FRC_APP/.venv/.frc-req-hash" 2>/dev/null || true)"
if [ ! -x "$FRC_APP/.venv/bin/python" ]; then
  uv venv --managed-python --python 3.12 "$FRC_APP/.venv"   # interpreter on the volume
  have_hash=""
fi
if [ "$want_hash" != "$have_hash" ]; then
  export VIRTUAL_ENV="$FRC_APP/.venv"
  uv pip install --python "$FRC_APP/.venv/bin/python" --index-strategy unsafe-best-match \
    --override "$REQ_OVR" -r "$REQ"
  # cv2 must be the headless build (no libGL dependency, same as local .venv).
  uv pip uninstall --python "$FRC_APP/.venv/bin/python" opencv-python || true
  uv pip install --python "$FRC_APP/.venv/bin/python" --reinstall --no-deps \
    "$(grep -E '^opencv-python-headless==' "$REQ")"
  SAM2_BUILD_CUDA=0 uv pip install --python "$FRC_APP/.venv/bin/python" \
    --no-deps --no-build-isolation -r "$REQ_GIT"
  echo "$want_hash" > "$FRC_APP/.venv/.frc-req-hash"
else
  echo "venv up to date ($want_hash)"
fi
PY="$FRC_APP/.venv/bin/python"
"$PY" -c "import torch, cv2, numpy; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| cv2', cv2.__version__, '| numpy', numpy.__version__)"

# ---------------------------------------------------------------- 4. prefetch
if [ "$PREFETCH" = 1 ]; then
  step "prefetch model weights into $FRC_CACHE"
  "$PY" - <<'PYEOF' || echo "[bootstrap] WARNING: prefetch had failures (see above); first job will retry the downloads"
import os, sys, traceback
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(os.environ["FRC_APP"]) / ".env")
ok = True

def attempt(label, fn):
    global ok
    try:
        fn()
        print(f"  ok   {label}")
    except Exception as e:  # keep going; report all
        ok = False
        print(f"  FAIL {label}: {type(e).__name__}: {str(e)[:300]}")

def sam3():
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN missing from .env (facebook/sam3 is gated)")
    from sam3.model_builder import download_ckpt_from_hf
    p = download_ckpt_from_hf("sam3")
    print(f"       {p} ({Path(p).stat().st_size / 1e9:.2f} GB)")

def dinov3():
    import timm
    timm.create_model("timm/vit_small_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0)

def easyocr_models():
    import easyocr
    easyocr.Reader(["en"], gpu=False, verbose=False)

attempt("SAM3 checkpoint (facebook/sam3, gated)", sam3)
attempt("DINOv3 ViT-S (S4 backbone fallback / dinov3 embedder)", dinov3)
attempt("EasyOCR en", easyocr_models)
if not ok:
    print("  hint: SAM3 403/401 = HF_TOKEN lacks access to facebook/sam3 -> request access on huggingface.co "
          "with that account, then `runpod_cli.py secrets` + `runpod_cli.py ssh -- bash "
          "/workspace/frc/deploy/runpod/bootstrap.sh`")
    sys.exit(1)
PYEOF
fi

# ---------------------------------------------------------------- 5. smoke test
step "import smoke test"
"$PY" - <<'PYEOF'
import sys, time
t0 = time.time()
import torch
assert torch.cuda.is_available(), "CUDA not available to torch (driver/CUDA mismatch?)"
cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
archs = torch.cuda.get_arch_list()
name = torch.cuda.get_device_name(0)
print(f"GPU {name} {arch}; torch built for {archs}")
if arch not in archs and f"compute_{cap[0]}{cap[1]}" not in archs:
    sys.exit(f"torch {torch.__version__} has no kernels for {arch}")
x = torch.randn(512, 512, device="cuda")
assert torch.isfinite((x @ x).sum()).item()
import sam3, sam2, ultralytics, timm, transformers, easyocr, pupil_apriltags, yt_dlp  # noqa: F401
import web.app  # noqa: F401  (full server import: every module + weight path constants)
print(f"import web.app OK in {time.time() - t0:.1f}s; MODEL_PATH={web.app.MODEL_PATH} "
      f"exists={web.app.MODEL_PATH.exists()} BUMPER_MODEL_PATH exists={web.app.BUMPER_MODEL_PATH.exists()}")
PYEOF

# ---------------------------------------------------------------- 6. profile
step "knob profile"
python3 "$HERE/profiles.py" --write-profile "$FRC_APP/profile.env"
grep -v '^#' "$FRC_APP/profile.env"

# ---------------------------------------------------------------- 7. server
if [ "$START" = 1 ]; then
  step "server"
  bash "$HERE/serve.sh" restart
fi
step "done"
