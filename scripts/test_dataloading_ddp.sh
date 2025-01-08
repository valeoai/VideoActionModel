#!/bin/bash
#SBATCH --job-name=debugging
#SBATCH -A ycy@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=00:10:00
#SBATCH --output=debugging/%j.out
#SBATCH --error=debugging/%j.out

mkdir -p debugging

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/world_model

SCRIPT_DIR=$(dirname $(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}'))

srun python $SCRIPT_DIR/test_dataloading_ddp.py


# srun -A ycy@h100 -C h100 --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:4 --hint=nomultithread --qos=qos_gpu_h100-dev --time=00:45:00 bash
# torchrun --standalone --nproc-per-node=4 scripts/test_dataloading_ddp.py
