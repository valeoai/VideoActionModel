#!/bin/bash
#SBATCH --job-name=hq_jobs         # nom du job
#SBATCH -A ycyh100                      # pour cibler les noeuds H100
#SBATCH -C h100                      # pour cibler les noeuds H100
# Ici, reservation de 3x24=72 CPU (pour 3 taches) et de 3 GPU (1 GPU par tache) sur un seul noeud :
#SBATCH --nodes=1                    # nombre de noeud
#SBATCH --gres=gpu:0                 # nombre de GPU par noeud (max 4 pour les noeuds H100)
# Sachant qu'ici on ne reserve qu'un seul GPU par tache (soit 1/4 des GPUs), l'ideal est de reserver 1/4 des CPU du noeud pour chaque tache:
#SBATCH --cpus-per-task=8           # nombre de CPU par tache (1/4 des CPUs ici)
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d’execution maximum demande (HH:MM:SS)
#SBATCH --output=hq-server-%j.out    # nom du fichier de sortie
#SBATCH --error=hq-server-%j.out     # nom du fichier d'erreur (ici commun avec la sortie)

cd $WORK/NextTokenPredictor

module purge
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/world_model
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
# Important change when using deepspeed (which now uses triton)
# By default the cache dir will be $HOME/.triton
# We point it to $SCRATCH because the inodes quota is very limited on JeanZay
export TRITON_CACHE_DIR=$SCRATCH/.triton
export TMPDIR=$JOBSCRATCH

# Echo des commandes lancees
set -x

# Execution du code
srun python world_model/opendv/create_opendv_tokens.py \
--video_list world_model/opendv/opendv_video.json \
--metadata ~/iveco/datasets_iveco_raw/OpenDV_Youtube/videos_metadata.csv \
--outdir ~/data/OpenDV_Youtube/tokens \
--tmpdir ~/data/OpenDV_Youtube/tmp \
--tokenizer_jit_path ~/iveco/scratch_iveco/world_model_JZGC4/jit_models/VQ_ds16_16384_llamagen.jit \
--num_writer_threads 5 \
--frames_queue_size 10000 \
--writer_queue_size 10000 \
--batch_size 64 \
--target_frame_rate 5 \
--target_width 512 \
--target_height 288
