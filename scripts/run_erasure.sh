#!/bin/bash
#SBATCH --job-name=model_erasure
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=student

nvidia-smi -L

mace_dir="${HOME}/MACE"
conda init
source ~/.bashrc && conda activate mace
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1 python training.py configs/object/erase_ship.yaml
