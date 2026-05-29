#!/bin/bash
# Start FRC Vision Scorer web application

cd "$(dirname "$0")"
PROJECT_ROOT="$(dirname "$(pwd)")"

echo "Starting FRC Vision Scorer on http://localhost:8004"
echo "Logs: /tmp/frc-vision-dev.log"

"$PROJECT_ROOT/.venv/bin/python" app.py > /tmp/frc-vision-dev.log 2>&1 &
echo $! > /tmp/frc-vision-dev.pid

echo "Started with PID: $(cat /tmp/frc-vision-dev.pid)"
echo "To stop: kill \$(cat /tmp/frc-vision-dev.pid)"
