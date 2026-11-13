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

# 2026-06-11 whole-card OOM containment: full-card SAM3 runs OOM'd at ~21GB
# even with the dark-reanchor attempt cap, while the previous day's runs
# (same matches, no dark-reanchor / no aggressive eviction / no
# expandable_segments) were clean. Until the instrumented chunk-VRAM logs
# attribute the growth, run yesterday's known-good behavior profile. The
# features stay in the tree and re-enable per job via config_json for
# A/B-ing the leak. (A caller's legacy SAM2_* name is still honoured.)
export SAM3_DARK_REANCHOR_ENABLED="${SAM3_DARK_REANCHOR_ENABLED:-${SAM2_DARK_REANCHOR_ENABLED:-0}}"
export SAM3_EVICT_DARK_FRAMES="${SAM3_EVICT_DARK_FRAMES:-${SAM2_EVICT_DARK_FRAMES:-99999}}"
export SAM3_EVICT_MIN_ABSENCES="${SAM3_EVICT_MIN_ABSENCES:-${SAM2_EVICT_MIN_ABSENCES:-99}}"

"$PROJECT_ROOT/web/start.sh"
