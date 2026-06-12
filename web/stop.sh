#!/bin/bash
# Stop FRC Vision Scorer web application

if [ -f /tmp/frc-vision-dev.pid ]; then
    PID=$(cat /tmp/frc-vision-dev.pid)
    echo "Stopping FRC Vision Scorer (PID: $PID)"
    kill $PID 2>/dev/null
    # A mid-job server can ignore SIGTERM for a long time while still holding
    # ~17GB of VRAM, which OOMs the next server's SAM3 init. Wait for actual
    # death, then escalate.
    for _ in $(seq 1 20); do
        kill -0 $PID 2>/dev/null || break
        sleep 0.5
    done
    if kill -0 $PID 2>/dev/null; then
        echo "PID $PID ignored SIGTERM after 10s; sending SIGKILL"
        kill -9 $PID 2>/dev/null
        sleep 1
    fi
    rm /tmp/frc-vision-dev.pid
    echo "Stopped"
else
    echo "No PID file found"
fi
