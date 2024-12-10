#!/bin/bash
#SBATCH --job-name=hq_jobs
#SBATCH -A fzh@v100
#SBATCH -C v100
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-dev
#SBATCH --time=01:00:00
#SBATCH --output=hq_jobs-%j.out
#SBATCH --error=hq_jobs-%j.out

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
--metadata $fzh_ALL_CCFRSCRATCH/OpenDV/videos_metadata.csv \
--outdir $fzh_ALL_CCFRSCRATCH/OpenDV/tokens \
--tmpdir $fzh_ALL_CCFRSCRATCH/OpenDV/tmp \
--tokenizer_jit_path $WORK/VQ_ds16_16384_llamagen.jit \
--num_writer_threads 5 \
--frames_queue_size 10000 \
--writer_queue_size 10000 \
--batch_size 64 \
--target_frame_rate 5 \
--target_width 512 \
--target_height 288
