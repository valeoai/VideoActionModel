#!/bin/bash
#SBATCH --job-name=hq_worker
#SBATCH -A ycy@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-gc
#SBATCH --time=10:00:00
#SBATCH --output=hq_worker-%j.out
#SBATCH --error=hq_worker-%j.out

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/world_model
export MPICH_GPU_SUPPORT_ENABLED=1
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
# Important change when using deepspeed (which now uses triton)
# By default the cache dir will be $HOME/.triton
# We point it to $SCRATCH because the inodes quota is very limited on JeanZay
export TRITON_CACHE_DIR=$SCRATCH/.triton
export TMPDIR=$JOBSCRATCH

# Echo des commandes lancees
set -x

if [ -z "$1" ]; then
    echo "Usage: $0 <CPUS_PER_WORKER>"
    exit 1
fi
CPUS_PER_WORKER=$1

# Execution du code
srun hq worker start \
--cpus $CPUS_PER_WORKER \
--resource "gpus/nvidia=sum(1)" \
--no-detect-resources \
--idle-timeout 3000sec \
--manager slurm \
--on-server-lost finish-running
