#!/bin/bash

#SBATCH --job-name=setup_env
#SBATCH --time=01:00:00
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu

export HF_HOME="$SCRATCH/models_cache"

mace_dir="${HOME}/MACE-fork"
source "${SCRATCH}/miniconda3/etc/profile.d/conda.sh"
conda activate mace
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py configs/object/erase_ship.yaml
