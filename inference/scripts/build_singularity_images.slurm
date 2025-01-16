#!/bin/bash
#SBATCH --job-name=singularity
#SBATCH -A ycy@h100  # at this time singularity does not seem to be working on h100
#SBATCH -C h100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --nodes=1  # important to do this on one node
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=slurm_jobs_logs/stdout/%x_%A_%a.out
#SBATCH --error=slurm_jobs_logs/stdout/%x_%A_%a.out
#SBATCH --array=0-2%1  # 3 tasks that will run one after the other, this ensures no cache error in singularity

mkdir -p slurm_jobs_logs/stdout

export DOCKER_JZ_FOLDER=$ycy_ALL_CCFRSCRATCH/neuroncap_docker_file

TARGET_DOCKER_IMAGE[0]=$DOCKER_JZ_FOLDER/neurad.docker.tar.gz
TARGET_SINGULARITY_IMAGE[0]=$DOCKER_JZ_FOLDER/neurad.sif

TARGET_DOCKER_IMAGE[1]=$DOCKER_JZ_FOLDER/ncap.docker.tar.gz
TARGET_SINGULARITY_IMAGE[1]=$DOCKER_JZ_FOLDER/ncap.sif

TARGET_DOCKER_IMAGE[2]=$DOCKER_JZ_FOLDER/ncap_vai0rbis.docker.tar.gz
TARGET_SINGULARITY_IMAGE[2]=$DOCKER_JZ_FOLDER/ncap_vai0rbis.sif

# very important as singularity uses /tmp a lot
export TMPDIR=$JOBSCRATCH
export SINGULARITY_CACHEDIR=$JOBSCRATCH

module purge
module load singularity

singularity version

singularity build \
  ${TARGET_SINGULARITY_IMAGE[$SLURM_ARRAY_TASK_ID]} \
  docker-archive://${TARGET_DOCKER_IMAGE[$SLURM_ARRAY_TASK_ID]} &&

# On JZ we can not use singularity as we like:
# https://calcul.math.cnrs.fr/attachments/evt/2021-01-anf-ust4hpc-2021/support06.pdf
# (slide 12)
# So we need to copy the images to an authorized location
# here this copy to the folder
# $SINGULARITY_ALLOWED_DIR
# that is only accessible when loading the singularity module
idrcontmgr cp ${TARGET_SINGULARITY_IMAGE[$SLURM_ARRAY_TASK_ID]}

# srun -A ycy@h100 -C h100 --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=20 --gres=gpu:0 --hint=nomultithread --qos=qos_gpu_h100-dev --time=02:00:00 bash
# export TMPDIR=$JOBSCRATCH
# export SINGULARITY_CACHEDIR=$JOBSCRATCH
# singularity build \
# $ycy_ALL_CCFRSCRATCH/neuroncap_docker_file/ncap_vai0rbis.sif \
# docker-archive://$ycy_ALL_CCFRSCRATCH/neuroncap_docker_file/ncap_vai0rbis.docker.tar.gz
# idrcontmgr cp $ycy_ALL_CCFRSCRATCH/neuroncap_docker_file/ncap_vai0rbis.sif
