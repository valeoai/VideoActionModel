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
#SBATCH --time=10:00:00
#SBATCH --output=slurm_jobs_logs/stdout/nxt_eval/%A_%a.out
#SBATCH --error=slurm_jobs_logs/stdout/nxt_eval/%A_%a.out
#SBATCH --array=1-5

mkdir -p slurm_jobs_logs/stdout/nxt_eval
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

ckpt[0]=$ycy_ALL_CCFRSCRATCH/output_data/vaiorbis/Vaiorbis_pretrained0000038823_DDP_Nodes6_BSperGPU16_totalBS384_attdim768_actdim192_0121_0053_1737417196/checkpoints/end_of_epoch_epoch=000_step=0000007251.ckpt
batch_size[0]=64
outdir[0]=action_expert_38k_attndim768_actdim192

ckpt[1]=$ycy_ALL_CCFRSCRATCH/output_data/vaiorbis/Vaiorbis_pretrained0000077646_DDP_Nodes6_BSperGPU16_totalBS384_attdim768_actdim192_0121_0112_1737418368/checkpoints/end_of_epoch_epoch=000_step=0000007251.ckpt
batch_size[1]=64
outdir[1]=action_expert_77k_attndim768_actdim192

ckpt[2]=$ycy_ALL_CCFRSCRATCH/output_data/vaiorbis/Vaiorbis_pretrained0000116469_DDP_Nodes6_BSperGPU16_totalBS384_attdim768_actdim192_0121_0112_1737418377/checkpoints/end_of_epoch_epoch=000_step=0000007251.ckpt
batch_size[2]=64
outdir[2]=action_expert_116k_attndim768_actdim192

ckpt[3]=$ycy_ALL_CCFRSCRATCH/output_data/vaiorbis/Vaiorbis_pretrained0000038823_DDP_Nodes6_BSperGPU16_totalBS384_attdim1024_actdim256_0121_0112_1737418373/checkpoints/end_of_epoch_epoch=000_step=0000007251.ckpt
batch_size[3]=64
outdir[3]=action_expert_38k_attndim1024_actdim256

ckpt[4]=$ycy_ALL_CCFRSCRATCH/output_data/vaiorbis/Vaiorbis_pretrained0000077646_DDP_Nodes6_BSperGPU16_totalBS384_attdim1024_actdim256_0121_0052_1737417153/checkpoints/end_of_epoch_epoch=000_step=0000007251.ckpt
batch_size[4]=64
outdir[4]=action_expert_77k_attndim1024_actdim256

ckpt[5]=$ycy_ALL_CCFRSCRATCH/output_data/vaiorbis/Vaiorbis_pretrained0000139763_DDP_Nodes12_BSperGPU8_totalBS384_attdim2048_actdim512_0121_2057_1737489441/checkpoints/end_of_epoch_epoch=000_step=0000007251.ckpt
batch_size[5]=32
outdir[5]=action_expert_139k_attndim2048_actdim512


srun python scripts/nxt_evaluation.py \
--vai0rbis_checkpoint_path ${ckpt[${SLURM_ARRAY_TASK_ID}]} \
--outdir slurm_jobs_logs/files/ego_eval/${outdir[${SLURM_ARRAY_TASK_ID}]} \
--num_workers 24 \
--batch_size ${batch_size[${SLURM_ARRAY_TASK_ID}]} \
