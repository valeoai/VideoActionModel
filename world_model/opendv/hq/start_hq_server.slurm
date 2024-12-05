#!/bin/bash
#SBATCH --job-name=hq_server         # nom du job
#SBATCH -A ycyh100                      # pour cibler les noeuds H100
#SBATCH -C h100                      # pour cibler les noeuds H100
# Ici, reservation de 3x24=72 CPU (pour 3 taches) et de 3 GPU (1 GPU par tache) sur un seul noeud :
#SBATCH --nodes=1                    # nombre de noeud
#SBATCH --gres=gpu:0                 # nombre de GPU par noeud (max 4 pour les noeuds H100)
# Sachant qu'ici on ne reserve qu'un seul GPU par tache (soit 1/4 des GPUs), l'ideal est de reserver 1/4 des CPU du noeud pour chaque tache:
#SBATCH --cpus-per-task=24           # nombre de CPU par tache (1/4 des CPUs ici)
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d’execution maximum demande (HH:MM:SS)
#SBATCH --output=hq-server-%j.out    # nom du fichier de sortie
#SBATCH --error=hq-server-%j.out     # nom du fichier d'erreur (ici commun avec la sortie)

cd $WORK/NextTokenPredictor

# Configuration of modules
module purge
module load ffmpeg

# Echo des commandes lancees
set -x

# Execution du code
srun hq worker start --cpus 24  --no-detect-resources --idle-timeout 300sec --manager slurm --on-server-lost finish-running
