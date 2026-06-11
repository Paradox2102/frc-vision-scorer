#!/bin/bash
# Start FRC Vision Scorer web application

# Run from the PROJECT ROOT: SAM2 checkpoints and model paths resolve relative
# to it; starting from web/ silently degraded SAM2 to Kalman tracking.
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

echo "Starting FRC Vision Scorer on http://localhost:8004"
echo "Logs: /tmp/frc-vision-dev.log"

"$PROJECT_ROOT/.venv/bin/python" web/app.py > /tmp/frc-vision-dev.log 2>&1 &
echo $! > /tmp/frc-vision-dev.pid

echo "Started with PID: $(cat /tmp/frc-vision-dev.pid)"
echo "To stop: kill \$(cat /tmp/frc-vision-dev.pid)"
