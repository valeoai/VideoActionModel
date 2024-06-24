#!/bin/bash

set -eu

module purge
module load cray-python

env_name=wm_world_model_env_2.4.0dev

# Install  virtualenv
python3 -m pip install --user --upgrade pip
pip3 install --user virtualenv --no-cache-dir

ENVS_ROOT_DIR="/lus/work/CT10/cin4181/SHARED/shared_python_envs"

# Create environnement
python3 -m virtualenv ${ENVS_ROOT_DIR}/${env_name}
chmod +x ${ENVS_ROOT_DIR}/${env_name}/bin/activate

# Activate environment and update pip
source ${ENVS_ROOT_DIR}/${env_name}/bin/activate

# load all necessary module for wm_metrics
module load craype-accel-amd-gfx90a craype-x86-trento # Compiler ?
module load PrgEnv-cray # devkit ?
module load amd-mixed # AMD hardware

# install all necessary python packages
python3 -m pip install --upgrade pip --no-cache-dir
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm5.7 --no-cache-dir
pip3 install einops omegaconf click scipy --no-cache-dir
pip3 install lightning==2.2.0 --no-cache-dir
pip3 install torchmetrics==0.11.4 --no-cache-dir
pip3 install hydra-core>=1.1.0 --no-cache-dir
pip3 install hydra-colorlog>=1.1.0 --no-cache-dir
pip3 install fairscale opencv-python-headless matplotlib pyquaternion wandb python-dotenv rich torch-summary timm lpips tensorboard deepspeed mup --no-cache-dir