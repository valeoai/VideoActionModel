#!/bin/bash
#SBATCH --job-name=hq_worker
#SBATCH -A ycy@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-gc
#SBATCH --time=12:00:00
#SBATCH --output=hq_worker-%j.out
#SBATCH --error=hq_worker-%j.out

module purge
module load arch/h100
module load ffmpeg

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
--no-detect-resources \
--idle-timeout 3000sec \
--manager slurm \
--on-server-lost finish-running
