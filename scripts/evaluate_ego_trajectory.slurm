#!/bin/bash
#SBATCH --job-name=nxt_eval
#SBATCH -A ycy@h100
#SBATCH -C h100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-gc
#SBATCH --time=10:10:00
#SBATCH --output=slurm_jobs_logs/stdout/ego_eval/%A_%a.out
#SBATCH --error=slurm_jobs_logs/stdout/ego_eval/%A_%a.out
#SBATCH --array=0-5

mkdir -p slurm_jobs_logs/stdout/ego_eval
mkdir -p slurm_jobs_logs/files/ego_eval


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

ckpt[0]=./tmp/vam_ckpt/action_expert_38k_attndim768_actdim192.ckpt
batch_size[0]=64
outdir[0]=action_expert_38k_attndim768_actdim192_5samples

ckpt[1]=./tmp/vam_ckpt/action_expert_77k_attndim768_actdim192.ckpt
batch_size[1]=64
outdir[1]=action_expert_77k_attndim768_actdim192_5samples

ckpt[2]=./tmp/vam_ckpt/action_expert_116k_attndim768_actdim192.ckpt
batch_size[2]=64
outdir[2]=action_expert_116k_attndim768_actdim192_5samples

ckpt[3]=./tmp/vam_ckpt/action_expert_38k_attndim1024_actdim256.ckpt
batch_size[3]=64
outdir[3]=action_expert_38k_attndim1024_actdim256_5samples

ckpt[4]=./tmp/vam_ckpt/action_expert_77k_attndim1024_actdim256.ckpt
batch_size[4]=64
outdir[4]=action_expert_77k_attndim1024_actdim256_5samples

ckpt[5]=./tmp/vam_ckpt/action_expert_139k_attndim2048_actdim512.ckpt
batch_size[5]=32
outdir[5]=action_expert_139k_attndim2048_actdim512_5samples


srun python scripts/evaluate_ego_trajectory.py \
--vam_checkpoint_path ${ckpt[${SLURM_ARRAY_TASK_ID}]} \
--outdir slurm_jobs_logs/files/ego_eval/${outdir[${SLURM_ARRAY_TASK_ID}]} \
--num_workers 24 \
--batch_size ${batch_size[${SLURM_ARRAY_TASK_ID}]} \
--store_trajectories True \
--num_sampled_trajectories 5
