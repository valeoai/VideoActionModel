#!/bin/bash

set -eu

# doc: http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-python-env-eng.html

python_module=pytorch-gpu/py3/2.4.0
env_name=video_action_model
ENVS_ROOT_DIR=${WORK}/python_envs
export PYTHONUSERBASE=${ENVS_ROOT_DIR}/${env_name}
mkdir -p "${PYTHONUSERBASE}"

module purge
module load arch/h100
module load "${python_module}"

pip install -e .

echo "Environment setup complete. To activate, use:"
echo "module purge; module load ${python_module}"
echo "export PYTHONUSERBASE=${ENVS_ROOT_DIR}/${env_name}"
