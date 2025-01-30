#!/bin/bash
#SBATCH --job-name=nxt_eval
#SBATCH -A ycy@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-gc
#SBATCH --time=10:00:00
#SBATCH --output=slurm_jobs_logs/stdout/hbird_eval/%A_%a.out
#SBATCH --error=slurm_jobs_logs/stdout/hbird_eval/%A_%a.out
#SBATCH --array=0-5

mkdir -p slurm_jobs_logs/stdout/hbird_eval
mkdir -p slurm_jobs_logs/files/hbird_eval


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


# [PRETRAIN]
# 768, 90% data:
ckpt[0]="$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/wd_sweep/GPT2_OpenDV_Llamagen_768_Nodes16_BSperGPU6_totalBS384_weight_decay1e-07_0116_1243_1737027800/checkpoints/before_drop_epoch=000_step=0000139763.ckpt"
outfile[0]="./slurm_jobs_logs/files/hbird_eval/test_pretrain_768.json"
# 1024, 90% data:
ckpt[1]="$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/wd_sweep/GPT2_OpenDV_Llamagen_1024_Nodes32_BSperGPU3_totalBS384_weight_decay1e-07_0118_0114_1737159286/checkpoints/before_drop_epoch=000_step=0000139763.ckpt"
outfile[1]="./slurm_jobs_logs/files/hbird_eval/test_pretrain_1024.json"
# 2048, 90% data:
ckpt[2]="$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/wd_sweep/GPT2_OpenDV_Llamagen_2048_Nodes48_BSperGPU2_totalBS384_weight_decay1e-07_0118_0125_1737159936/checkpoints/before_drop_epoch=000_step=0000139763.ckpt"
outfile[2]="./slurm_jobs_logs/files/hbird_eval/test_pretrain_2048.json"

# [FINETUNE]
# 768, 90% data:
ckpt[3]="$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/finetuned/Finetuned_0000139763_mixOpendvNuplanNuscenes_GPT2_Llamagen_768_Nodes16_BSperGPU6_totalBS384_weight_decay1e-07_0122_2017_1737573432/checkpoints/end_of_epoch_epoch=001_step=0000155294.ckpt"
outfile[3]="./slurm_jobs_logs/files/hbird_eval/test_finetune_768.json"
# 1024, 90% data:
ckpt[4]="$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/finetuned/Finetuned_0000139763_mixOpendvNuplanNuscenes_GPT2_Llamagen_1024_Nodes32_BSperGPU3_totalBS384_weight_decay1e-07_0122_2024_1737573850/checkpoints/end_of_epoch_epoch=001_step=0000155294.ckpt"
outfile[4]="./slurm_jobs_logs/files/hbird_eval/test_finetune_1024.json"
# 2048, 90% data:
ckpt[5]="$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/finetuned/Finetuned_0000139763_mixOpendvNuplanNuscenes_GPT2_Llamagen_2048_Nodes48_BSperGPU2_totalBS384_weight_decay1e-07_0121_0119_1737418786/checkpoints/end_of_epoch_epoch=001_step=0000155294.ckpt"
outfile[5]="./slurm_jobs_logs/files/hbird_eval/test_finetune_2048.json"

srun python scripts/hbird_evaluation.py \
--gpt_checkpoint_path ${ckpt[$SLURM_ARRAY_TASK_ID]} \
--tokenizer_jit_path $ycy_ALL_CCFRWORK/llamagen_jit_models/VQ_ds16_16384_llamagen_encoder.jit \
--outfile ${outfile[$SLURM_ARRAY_TASK_ID]} \
--num_workers 16 \
--batch_size 16
