#!/bin/bash
# Pod-side: the dedicated vLLM venv (/workspace/frc/.venv-vllm) + reader weights.
# Called by bootstrap.sh as a SOFT step (failure only breaks `tune` arm Q, the
# resident Qwen3.5-4B reader). vLLM pins its own torch build (0.25.1 -> torch
# 2.11 cu13), so it must never share the app's .venv; web/start_vllm.sh expects
# .venv-vllm/bin/{python,vllm} and its bundled nvidia/cu13 libs. ~9 GB on the
# volume; skipped when already installed at the pinned version.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pod_env.sh
source "$HERE/pod_env.sh"

VLLM_VERSION="${FRC_VLLM_VERSION:-0.25.1}"   # the version start_vllm.sh is tuned for
VENV="$FRC_APP/.venv-vllm"
APP_PY="$FRC_APP/.venv/bin/python"

[ -x "$VENV/bin/python" ] || uv venv --managed-python --python 3.12 "$VENV"
# Always run (a no-op when satisfied): it also repairs a venv whose vLLM
# dependencies drifted. The lock holds the Nov 12 known-good set; a
# non-default FRC_VLLM_VERSION installs unconstrained.
LOCK="$HERE/requirements-vllm-lock.txt"
if [ "$VLLM_VERSION" = "0.25.1" ] && [ -f "$LOCK" ]; then
  uv pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION" -c "$LOCK"
else
  uv pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION"
fi
"$VENV/bin/python" -c 'import vllm, transformers; print("vllm", vllm.__version__, "transformers", transformers.__version__)'
[ -x "$VENV/bin/vllm" ] || { echo "missing $VENV/bin/vllm" >&2; exit 1; }

# Reader weights for every vlm_model key a tune arm uses (public repos).
cd "$FRC_APP"
"$APP_PY" - <<'PYEOF'
import sys
sys.path[:0] = [".", "deploy/runpod"]
from huggingface_hub import snapshot_download
import profiles
from web import vlm_models

keys = sorted({a.knobs["vlm_model"] for a in profiles.ARMS if "vlm_model" in a.knobs})
for k in keys:
    repo = vlm_models.get(k).hf_repo
    print(f"  prefetch {k}: {repo} -> {snapshot_download(repo)}")
PYEOF
