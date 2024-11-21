#!/bin/bash
# https://stackoverflow.com/questions/17385794/how-to-get-the-process-id-to-kill-a-nohup-process
file="python/scripts/eva_processing/compile_eva_chelsa_EUNIS.py"
namesim_with_ext=$(basename ${file})
namesim="${namesim_with_ext%.*}"

echo "lauching script for $namesim"
chmod +x $file
source /home/boussang/miniforge3/bin/activate /home/boussang/SAR_modelling/python/.env
nohup python $file > "stdout/${namesim}.out" 2>&1

file="python/scripts/eva_processing/compile_eva_chelsa_EUNIS-sara.py"
namesim_with_ext=$(basename ${file})
namesim="${namesim_with_ext%.*}"

echo "lauching script for $namesim"
chmod +x $file
source /home/boussang/miniforge3/bin/activate /home/boussang/SAR_modelling/python/.env
nohup python $file > "stdout/${namesim}.out" 2>&1
