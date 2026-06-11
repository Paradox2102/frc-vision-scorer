#!/bin/bash
# Stop the vLLM team-reader server

if [ -f /tmp/frc-vllm.pid ]; then
    PID=$(cat /tmp/frc-vllm.pid)
    echo "Stopping vLLM team reader (PID: $PID)"
    kill $PID 2>/dev/null
    rm /tmp/frc-vllm.pid
    echo "Stopped"
else
    echo "No PID file found"
fi
