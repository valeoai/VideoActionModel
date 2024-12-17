#!/bin/bash
#SBATCH --job-name=hq_server
#SBATCH -A fzh@v100
#SBATCH -C v100
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --output=hq_server-%j.out
#SBATCH --error=hq_server-%j.out

# Configuration of modules
module purge

# Echo des commandes lancees
set -x

# Execution du code
srun hq server start --idle-timeout=3000sec
