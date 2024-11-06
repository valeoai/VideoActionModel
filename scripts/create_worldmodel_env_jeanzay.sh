#!/bin/bash

set -eu

# doc: http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-python-env-eng.html

python_module=pytorch-gpu/py3/2.4.0
env_name=worldmodel
ENVS_ROOT_DIR=${ycy_CCFRWORK}/python_envs
export PYTHONUSERBASE=${ENVS_ROOT_DIR}/${env_name}
mkdir -p "${PYTHONUSERBASE}"

module purge
module load arch/h100
module load "${python_module}"

pip install --upgrade lightning --user --no-cache-dir
pip install hydra-core>=1.1.0 hydra-colorlog>=1.1.0 mup deepspeed --user --no-cache-dir

echo "Environment setup complete. To activate, use:"
echo "module purge; module load ${python_module}"
echo "export PYTHONUSERBASE=${ENVS_ROOT_DIR}/${env_name}"
