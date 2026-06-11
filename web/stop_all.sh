#!/bin/bash
# Stop the full stack: main app + vLLM team reader.
cd "$(dirname "$0")"
./stop.sh
./stop_vllm.sh
