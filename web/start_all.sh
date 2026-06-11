#!/bin/bash
# Start the full stack: vLLM team reader + main app, wired together.
#
#   web/start_all.sh   — brings up both servers
#   web/stop_all.sh    — stops both
#
# Order matters: the vLLM server preallocates its GPU pool (~11 GB at util
# 0.45), and the app must see TEAM_READER=vllm so it never loads its own
# in-process copy of Qwen (two copies of the 4B fp16 weights do not fit).

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

"$PROJECT_ROOT/web/start_vllm.sh"

echo "Waiting for vLLM to come up (model load ~90s)..."
for i in $(seq 1 36); do
    if curl -s http://localhost:"${VLM_SERVER_PORT:-8006}"/v1/models 2>/dev/null | grep -q .; then
        echo "vLLM ready."
        break
    fi
    if ! kill -0 "$(cat /tmp/frc-vllm.pid 2>/dev/null)" 2>/dev/null; then
        echo "vLLM process died — check /tmp/frc-vllm.log. Starting app with local reader."
        break
    fi
    sleep 5
done

# Route bumper reads to the vLLM server, and trim SAM3 chunk windows so its
# video residency coexists with vLLM's pool (400-frame default assumes the
# whole card; 350 leaves margin — identity was flat across 150-800 in the
# chunk-size sweep).
export TEAM_READER=vllm
export SAM3_CHUNK_FRAMES="${SAM3_CHUNK_FRAMES:-350}"

"$PROJECT_ROOT/web/start.sh"
