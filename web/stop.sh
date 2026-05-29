#!/bin/bash
# Stop FRC Vision Scorer web application

if [ -f /tmp/frc-vision-dev.pid ]; then
    PID=$(cat /tmp/frc-vision-dev.pid)
    echo "Stopping FRC Vision Scorer (PID: $PID)"
    kill $PID 2>/dev/null
    rm /tmp/frc-vision-dev.pid
    echo "Stopped"
else
    echo "No PID file found"
fi
