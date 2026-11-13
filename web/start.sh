#!/bin/bash
# Start FRC Vision Scorer web application

# Run from the PROJECT ROOT: SAM checkpoints and model paths resolve relative
# to it; starting from web/ silently degraded mask tracking to Kalman tracking.
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

echo "Starting FRC Vision Scorer on http://localhost:8004"
echo "Logs: /tmp/frc-vision-dev.log"

# expandable_segments was added 2026-06-11 for an OOM that turned out to be
# real exhaustion, not fragmentation — and it is one of the two variables
# that changed before the whole-card 21GB OOMs began. Off until the
# instrumented chunk-VRAM logs clear it (set PYTORCH_ALLOC_CONF to opt in).

"$PROJECT_ROOT/.venv/bin/python" web/app.py > /tmp/frc-vision-dev.log 2>&1 &
echo $! > /tmp/frc-vision-dev.pid

echo "Started with PID: $(cat /tmp/frc-vision-dev.pid)"
echo "To stop: kill \$(cat /tmp/frc-vision-dev.pid)"
