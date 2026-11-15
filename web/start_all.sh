#!/bin/bash
# Start the app with the vLLM reader configured but NOT running.
#
#   web/start_all.sh   — starts the app (vLLM comes up on demand)
#   web/stop_all.sh    — stops the app and any running vLLM
#
# The vLLM server is fully job-managed (vllm_phase_exclusive knob, default
# on): a job stops any running instance before its tracking precompute
# (vLLM's ~11GB pool + SAM3's ~11GB peak OOMs the 24GB card), starts it for
# the analysis phase's bumper reads (~90s model load), and stops it again
# when the job finishes. Idle time between matches therefore costs zero
# VRAM. Co-residency chunk trimming (350/250/150) is gone — every size
# OOM'd the main view; SAM3's peak is model + per-object state, not frame
# residency.
#
# TEAM_READER only sets the DEFAULT reader; the UI radio per-job override
# wins either way. Default jev (Nov 6 2026 live-event config: hosted Jev
# image reader, needs JEV_TYPESAFE_API_TOKEN in .env, no vLLM/GPU). Export
# TEAM_READER=vllm before running this to get the local Qwen reader back;
# the app never preloads in-process Qwen when the reader is vllm.

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

export TEAM_READER="${TEAM_READER:-jev}"

# Dark re-anchor / dark eviction run on their job_config defaults again
# (June 11 2026 OOM containment lifted Nov 14 2026: the cause was the
# per-job VRAM leak fixed Nov 13; live A/B peak 12.7 GB on a 24 GB card).

"$PROJECT_ROOT/web/start.sh"
