#!/bin/bash
#SBATCH --job-name=hq_server
#SBATCH -A fzh@v100
#SBATCH --partition=archive
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=hq_server-%j.out
#SBATCH --error=hq_server-%j.out

# Configuration of modules
module purge

# Echo des commandes lancees
set -x

# Execution du code
srun hq server start --idle-timeout=3000sec
