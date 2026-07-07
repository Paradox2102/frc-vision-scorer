#!/bin/bash
# Start the bumper-reading VLM as a vLLM OpenAI-compatible server
# (team_reader="vllm" / VLM_SERVER_URL in app.py).
#
# Runs from the dedicated .venv-vllm: vLLM pins its own torch build and must
# never share a venv with the app (.venv).

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

MODEL="${VLM_MODEL_NAME:-Qwen/Qwen3-VL-4B-Instruct}"
PORT="${VLM_SERVER_PORT:-8006}"
# 0.45 of the 3090's 24GB ≈ 11GB: ~8GB fp16 weights plus KV-cache/encoder
# headroom, leaving the rest for SAM2/SAM3 + YOLO in the app process.
# Don't trim below 0.45: at 0.42 the engine refuses to start (KV cache for
# one max-len request doesn't fit, and available-KV drifts run to run).
# SAM3 coexistence headroom comes from SAM3_CHUNK_FRAMES instead.
UTIL="${VLM_SERVER_GPU_UTIL:-0.45}"

echo "Starting vLLM team reader: $MODEL on http://localhost:$PORT"
echo "Logs: /tmp/frc-vllm.log"

# The system nvcc (CUDA 12.0) is too old to JIT flashinfer's sampling
# kernels; vLLM's native torch sampler is fine for 64-token reads.
export VLLM_USE_FLASHINFER_SAMPLER=0

# Mount the dev sleep routes (/sleep, /wake_up, /is_sleeping) that
# web/vllm_lifecycle.py uses to offload the GPU pool between phases instead
# of killing the process — this is what kills the ~60s per-job cold start.
export VLLM_SERVER_DEV_MODE=1

# min/max_pixels mirror the in-process AutoProcessor settings so both
# providers feed the model identical resolutions.
#
# Launch the console script THROUGH the venv's python rather than executing
# it directly: console-script shebangs bake in the venv's absolute path at
# creation time, so they break the moment the project dir is moved (e.g.
# /home/nick/frc-vision-scorer -> /home/nick/AI/frc-vision-scorer) with
# "cannot execute: required file not found". bin/python is an absolute
# symlink to the system interpreter and survives the move, so invoking the
# script as its argument is relocation-proof.
"$PROJECT_ROOT/.venv-vllm/bin/python" "$PROJECT_ROOT/.venv-vllm/bin/vllm" serve "$MODEL" \
  --port "$PORT" \
  --enable-sleep-mode \
  --dtype float16 \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-kwargs '{"min_pixels": 200704, "max_pixels": 1003520}' \
  > /tmp/frc-vllm.log 2>&1 &
echo $! > /tmp/frc-vllm.pid

echo "Started with PID: $(cat /tmp/frc-vllm.pid)"
echo "To stop: web/stop_vllm.sh"
