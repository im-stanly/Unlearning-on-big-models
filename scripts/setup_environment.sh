#!/bin/bash

mace_url="https://github.com/Shilin-LU/MACE.git"
git clone "${mace_url}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Installing conda..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  chmod +x ./Miniconda3-latest-Linux-x86-64.sh
  echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
  rm ./Miniconda3-latest-Linux-x64-64.sh
else
  echo "Conda already installed"
fi

conda tos accept
conda create -n mace python=3.10
conda init
conda activate mace
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda==11.7 -c pytorch -c nvidia
pip install diffusers==0.22.0 transformers==4.46.2 huggingface_hub==0.25.2
pip install accelerate openai omegaconf opencv-python numpy==1.26.4

mace_dir="${HOME}/MACE"
data_cache_dir="${mace_dir}/cache"

if [ ! -d "${data_cache_dir}" ]; then
  mkdir -p "${data_cache_dir}"
fi