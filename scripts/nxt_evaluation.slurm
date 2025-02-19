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
#SBATCH --time=05:00:00
#SBATCH --output=slurm_jobs_logs/stdout/nxt_eval/%A_%a.out
#SBATCH --error=slurm_jobs_logs/stdout/nxt_eval/%A_%a.out
#SBATCH --array=1-7

mkdir -p slurm_jobs_logs/stdout/nxt_eval
mkdir -p slurm_jobs_logs/files/nxt_eval


module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/video_action_model
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

ckpt[1]=$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/wd_sweep/GPT2_OpenDV_Llamagen_768_Nodes16_BSperGPU6_totalBS384_weight_decay1e-07_0116_1243_1737027800/checkpoints/quarters_epoch=000_step=0000038823.ckpt
batch_size[1]=128
outfile[1]=./slurm_jobs_logs/files/nxt_eval/attndim768_38k.json

ckpt[2]=$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/wd_sweep/GPT2_OpenDV_Llamagen_768_Nodes16_BSperGPU6_totalBS384_weight_decay1e-07_0116_1243_1737027800/checkpoints/quarters_epoch=000_step=0000116469.ckpt
batch_size[2]=128
outfile[2]=./slurm_jobs_logs/files/nxt_eval/attndim768_116k.json

ckpt[3]=$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/wd_sweep/GPT2_OpenDV_Llamagen_1024_Nodes32_BSperGPU3_totalBS384_weight_decay1e-07_0118_0114_1737159286/checkpoints/quarters_epoch=000_step=0000038823.ckpt
batch_size[3]=128
outfile[3]=./slurm_jobs_logs/files/nxt_eval/attndim1024_38k.json

ckpt[4]=$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/wd_sweep/GPT2_OpenDV_Llamagen_1024_Nodes32_BSperGPU3_totalBS384_weight_decay1e-07_0118_0114_1737159286/checkpoints/quarters_epoch=000_step=0000116469.ckpt
batch_size[4]=128
outfile[4]=./slurm_jobs_logs/files/nxt_eval/attndim1024_116k.json

ckpt[5]=$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/wd_sweep/GPT2_OpenDV_Llamagen_2048_Nodes48_BSperGPU2_totalBS384_weight_decay1e-07_0118_0125_1737159936/checkpoints/before_drop_epoch=000_step=0000139763.ckpt
batch_size[5]=96
outfile[5]=./slurm_jobs_logs/files/nxt_eval/attndim2048_139k.json

ckpt[6]=$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/wd_sweep/GPT2_OpenDV_Llamagen_768_Nodes16_BSperGPU6_totalBS384_weight_decay1e-07_0116_1243_1737027800/checkpoints/quarters_epoch=000_step=0000077646.ckpt
batch_size[6]=128
outfile[6]=./slurm_jobs_logs/files/nxt_eval/attndim768_77k.json

ckpt[7]=$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/wd_sweep/GPT2_OpenDV_Llamagen_1024_Nodes32_BSperGPU3_totalBS384_weight_decay1e-07_0118_0114_1737159286/checkpoints/quarters_epoch=000_step=0000077646.ckpt
batch_size[7]=128
outfile[7]=./slurm_jobs_logs/files/nxt_eval/attndim1024_77k.json


srun python scripts/nxt_evaluation.py \
--gpt_checkpoint_path ${ckpt[${SLURM_ARRAY_TASK_ID}]} \
--outfile ${outfile[${SLURM_ARRAY_TASK_ID}]} \
--num_workers 24 \
--batch_size ${batch_size[${SLURM_ARRAY_TASK_ID}]} \
