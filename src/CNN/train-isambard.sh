#!/usr/bin/env bash

#PBS -q ampereq
#PBS -l select=1:ncpus=1:ngpus=1:mem=100G
#PBS -l walltime=00:30:00

module load nvhpc/2023
export CUDA_LAUNCH_BLOCKING=1
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/2023/cuda/12.1/
eval "$(/lustre/home/br-aelwood/anaconda3/bin/conda shell.bash hook)"
conda init
conda activate pytorch-build
python3 adl-coursework/src/CNN/runner.py --dataset-root="/lustre/projects/bristol/$USER/MagnaTagATune" --epochs=2 --val-frequency=2
