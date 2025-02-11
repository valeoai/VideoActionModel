#!/bin/bash
#SBATCH --job-name=hq_worker
#SBATCH -A ycy@h100
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00
#SBATCH --output=hq_worker-%j.out
#SBATCH --error=hq_worker-%j.out

module purge

# Echo des commandes lancees
set -x

# Execution du code
srun hq worker start \
--cpus 1 \
--no-detect-resources \
--idle-timeout 3000sec \
--manager slurm \
--on-server-lost finish-running
