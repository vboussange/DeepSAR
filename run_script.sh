#!/bin/bash
# https://stackoverflow.com/questions/17385794/how-to-get-the-process-id-to-kill-a-nohup-process

file=$1
prefix=$2  # Optional prefix argument

namesim_with_ext=$(basename ${file})
namesim="${namesim_with_ext%.*}"

# Apply prefix if provided
if [ -n "$prefix" ]; then
    namesim="${prefix}_${namesim}"
fi

# Ensure required directories exist
mkdir -p stdout

echo "Launching script for $namesim"
chmod +x $file
# source /home/boussang/miniforge3/bin/activate /home/boussang/SAR_modelling/.env-torch

nohup python $file > "stdout/${namesim}.out" 2>&1 &
echo $! > "stdout/${namesim}_save_pid.txt"
