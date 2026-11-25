#!/usr/bin/env bash
# Serve one local decision model behind the Jev-wire shim and run the bumper-crop
# bench's `local` arm against it, holding a GPU lock so only one model is loaded
# at a time.
#
#   scripts/decision_local/run_bench.sh <venv-python> <backend.py> <label> <port> \
#       [bench run args...]          # e.g. --limit 5   |   --options missing-true --score
#
# Env: GPU_LOCK (default /tmp/frc-decision-bench-gpu.lock), SHIM_ARGS (extra shim args,
#      e.g. "--model /path/to/weights"), LOAD_TIMEOUT_S (default 1200),
#      BENCH_PY (python for the bench itself, default the repo .venv),
#      BENCH_ARM (local = Jev-wire payload, default; local-clef = Clef Workers AI payload).
set -euo pipefail
PY=$1; BACKEND=$2; LABEL=$3; PORT=$4; shift 4
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LOCK=${GPU_LOCK:-/tmp/frc-decision-bench-gpu.lock}
LOG_DIR="$REPO/data/eval/bumper_crop_bench/results/local_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/shim-$LABEL-$(date +%Y%m%d-%H%M%S).log"

exec 9>"$LOCK"
echo "[run_bench] $LABEL: waiting for GPU lock $LOCK" >&2
flock 9
echo "[run_bench] $LABEL: lock held; starting shim on :$PORT (log $LOG)" >&2

# shellcheck disable=SC2086
"$PY" "$REPO/scripts/decision_local/systemone_shim.py" --backend "$BACKEND" \
    --port "$PORT" ${SHIM_ARGS:-} >"$LOG" 2>&1 &
SHIM=$!
trap 'kill $SHIM 2>/dev/null || true; wait $SHIM 2>/dev/null || true' EXIT

deadline=$(( $(date +%s) + ${LOAD_TIMEOUT_S:-1200} ))
until curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null; do
    if ! kill -0 $SHIM 2>/dev/null; then
        echo "[run_bench] shim died during load:" >&2; tail -40 "$LOG" >&2; exit 1
    fi
    if (( $(date +%s) > deadline )); then
        echo "[run_bench] shim load timed out" >&2; tail -40 "$LOG" >&2; exit 1
    fi
    sleep 3
done
grep '^\[shim\]' "$LOG" >&2 || true
nvidia-smi --query-gpu=memory.used --format=csv,noheader >&2 || true

cd "$REPO"
# BENCH_PY: the bench's local arm needs only pillow + requests, so a rented box can
# run it in the model venv (no repo .venv there).
"${BENCH_PY:-.venv/bin/python}" scripts/bench_bumper_readers.py run --arms "${BENCH_ARM:-local}" \
    --local-url "http://127.0.0.1:$PORT/v1/systemone" --local-label "$LABEL" \
    --concurrency 2 "$@"
