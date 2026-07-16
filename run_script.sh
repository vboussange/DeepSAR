#!/usr/bin/env bash

set -eu

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 SCRIPT [PREFIX]" >&2
    exit 2
fi

file="$1"
prefix="${2:-}"

if [ ! -f "$file" ]; then
    echo "Script not found: $file" >&2
    exit 2
fi

namesim_with_ext=$(basename "$file")
namesim="${namesim_with_ext%.*}"

if [ -n "$prefix" ]; then
    namesim="${prefix}_${namesim}"
fi

mkdir -p stdout

echo "Launching script for $namesim"

setsid nohup uv run python "$file" > "stdout/${namesim}.out" 2>&1 &

pid=$!

# Give the process a moment to appear in ps
sleep 0.2

pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
sid=$(ps -o sid= -p "$pid" | tr -d ' ')

echo "$pid" > "stdout/${namesim}_pid.txt"
echo "$pgid" > "stdout/${namesim}_pgid.txt"
echo "$sid" > "stdout/${namesim}_sid.txt"

echo "PID:  $pid"
echo "PGID: $pgid"
echo "SID:  $sid"
echo "Logs: stdout/${namesim}.out"
