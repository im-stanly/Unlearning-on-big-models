
#!/bin/bash

#SBATCH --job-name=run_inference
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
# erasure
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 6 --prompt 'a photo of the ship' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/erasure'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 6 --prompt 'a photo of the ship' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/erasure'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 6 --prompt 'a photo of the ship' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/erasure'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 6 --prompt 'a photo of the ship' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/erasure'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 6 --prompt 'a photo of the ship' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/erasure'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 6 --prompt 'a photo of the ship' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/erasure'
# generality
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 6 --prompt 'a photo of the boat' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/generality'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 6 --prompt 'a photo of the boat' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/generality'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 6 --prompt 'a photo of the vessel' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/generality'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 6 --prompt 'a photo of the vessel' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/generality'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 6 --prompt 'a photo of the watercraft' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/generality'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 6 --prompt 'a photo of the watercraft' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/generality'
# specificity
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 4 --prompt 'a photo of the airplane' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/specificity'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 4 --prompt 'a photo of the automobile' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/specificity'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 4 --prompt 'a photo of the bird' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/specificity'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 4 --prompt 'a photo of the cat' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/specificity'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 4 --prompt 'a photo of the deer' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/specificity'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 4 --prompt 'a photo of the dog' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/specificity'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 4 --prompt 'a photo of the frog' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/specificity'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 4 --prompt 'a photo of the horse' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/specificity'
cd "${mace_dir}" && CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --num_images 4 --prompt 'a photo of the truck' --model_path "$SCRATCH/saved_model/LoRA_fusion_model" --save_path './assets/createdUnlearningImages/specificity'
