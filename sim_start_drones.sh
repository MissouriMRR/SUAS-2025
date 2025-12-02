#!/bin/bash

SESSION_NAME="MultipleRotors"
NCOPTERS="${1:-1}"
OUT_PORT="${2:-14550}"
OUT_HOST="${3:-127.0.0.1}"


tmux kill-session -t "$SESSION_NAME"

# Start a new detached tmux session with a placeholder window
tmux new-session -d -s "$SESSION_NAME" -n "init"

echo "Starting $NCOPTERS drones..."

for ((i = 0; i < $NCOPTERS; i++)); do
    WINDOW_NAME="Drone_$i"

    tmux new-window -d -t "$SESSION_NAME" -n "$WINDOW_NAME"
    tmux send-keys -t "$SESSION_NAME:$WINDOW_NAME" "/ardupilot/Tools/autotest/sim_vehicle.py -v ArduCopter -f airsim-copter --instance $i" Enter
    sleep 1
done

# Kill the initial placeholder window
tmux kill-window -t "$SESSION_NAME:init"

# Attach to the session
tmux attach-session -t "$SESSION_NAME"