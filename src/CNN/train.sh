#!/usr/bin/env bash

#SBATCH --job-name=adl-coursework
#SBATCH --partition teach_gpu
#SBATCH --nodes=1
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --gres=gpu:1
#SBATCH --account=coms030144
#SBATCH --time=0:30:00
#SBATCH --mem=8GB

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"

python runner.py --dataset-root="/mnt/storage/scratch/$USER/MagnaTagATune" --epochs=20 --val-frequency=5
