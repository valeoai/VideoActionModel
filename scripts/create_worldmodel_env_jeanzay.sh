#!/bin/bash

set -eu

module purge
module load cuda/12.4.1

env_name=worldmodel

ENVS_ROOT_DIR=${ycy_CCFRWORK}/python_envs

# Create environnement
python3 -m virtualenv ${ENVS_ROOT_DIR}/${env_name}
chmod +x ${ENVS_ROOT_DIR}/${env_name}/bin/activate

# Activate environment
source ${ENVS_ROOT_DIR}/${env_name}/bin/activate

# Upgrade pip within the virtual environment
pip install --upgrade pip --no-cache-dir

# install all necessary python packages
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install einops omegaconf click scipy --no-cache-dir
pip install lightning==2.4.0 hydra-core>=1.1.0 hydra-colorlog>=1.1.0 --no-cache-dir
pip install torchmetrics==1.4.1 --no-cache-dir
pip install fairscale opencv-python-headless matplotlib pyquaternion wandb python-dotenv rich torch-summary tensorboard deepspeed mup --no-cache-dir

echo "Environment setup complete. To activate, use:"
echo "source ${ENVS_ROOT_DIR}/${env_name}/bin/activate"