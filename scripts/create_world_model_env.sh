#!/bin/bash

set -eu

module purge
module load cray-python

env_name=wm_world_model_env

# Install  virtualenv
python3 -m pip install --user --upgrade pip
pip3 install --user virtualenv --no-cache-dir

ENVS_ROOT_DIR="/lus/work/CT10/cin4181/SHARED/code/env"

# Create environnement
python3 -m virtualenv ${ENVS_ROOT_DIR}/${env_name}
chmod +x ${ENVS_ROOT_DIR}/${env_name}/bin/activate

# Activate environment and update pip
source ${ENVS_ROOT_DIR}/${env_name}/bin/activate

# this script load all necessary module for wm_metrics
source ./activate_world_model_env.sh

# install all necessary python packages
python3 -m pip install --upgrade pip --no-cache-dir
pip3 install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/rocm5.6 --no-cache-dir
pip3 install einops omegaconf click scipy --no-cache-dir
pip3 install lightning==2.2.0 --no-cache-dir
pip3 install torchmetrics==0.11.4 --no-cache-dir
pip3 install hydra-core>=1.1.0 --no-cache-dir
pip3 install hydra-colorlog>=1.1.0 --no-cache-dir
pip3 install fairscale opencv-python-headless pyquaternion wandb python-dotenv rich torch-summary timm tensorboard --no-cache-dir