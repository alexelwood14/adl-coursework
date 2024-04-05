#!/usr/bin/env bash

#PBS -q ampereq
#PBS -l select=1:ncpus=1:ngpus=1:mem=100G
#PBS -l walltime=00:30:00

module use /software/x86/tools/nvidia/hpc_sdk/modulefiles
module load nvhpc/22.9
source adl-coursework/venv/bin/activate
python3 adl-coursework/src/CNN/runner.py --dataset-root="/lustre/projects/bristol/$USER/MagnaTagATune" --epochs=2 --val-frequency=2
