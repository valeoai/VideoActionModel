#!/bin/bash
#SBATCH --job-name=quality_eval
#SBATCH -A cya@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=12:00:00
#SBATCH --output=slurm_jobs_logs/stdout/quality_eval/%A_%a.out
#SBATCH --error=slurm_jobs_logs/stdout/quality_eval/%A_%a.out
#SBATCH --array=4-6

mkdir -p slurm_jobs_logs/stdout/quality_eval
mkdir -p slurm_jobs_logs/files/quality_eval


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

# [FINETUNE]
# 768, 90% data:
ckpt[0]="$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/finetuned/Finetuned_0000139763_mixOpendvNuplanNuscenes_GPT2_Llamagen_768_Nodes16_BSperGPU6_totalBS384_weight_decay1e-07_0122_2017_1737573432/checkpoints/end_of_epoch_epoch=001_step=0000155294.ckpt"
outfile[0]="./slurm_jobs_logs/files/quality_eval/test_layer12_finetune_768.json"
tokenizer_only[0]=False
# 1024, 90% data:
ckpt[1]="$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/finetuned/Finetuned_0000139763_mixOpendvNuplanNuscenes_GPT2_Llamagen_1024_Nodes32_BSperGPU3_totalBS384_weight_decay1e-07_0122_2024_1737573850/checkpoints/end_of_epoch_epoch=001_step=0000155294.ckpt"
outfile[1]="./slurm_jobs_logs/files/quality_eval/test_layer12_finetune_1024.json"
tokenizer_only[1]=False
# 2048, 90% data:
ckpt[2]="$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/finetuned/Finetuned_0000139763_mixOpendvNuplanNuscenes_GPT2_Llamagen_2048_Nodes48_BSperGPU2_totalBS384_weight_decay1e-07_0121_0119_1737418786/checkpoints/end_of_epoch_epoch=001_step=0000155294.ckpt"
outfile[2]="./slurm_jobs_logs/files/quality_eval/test_layer12_finetune_2048.json"
tokenizer_only[2]=False

# tokenizer only
ckpt[3]="none"
outfile[3]="./slurm_jobs_logs/files/quality_eval/llamagen.json"
tokenizer_only[3]=True

# [PRETRAIN]
# 768, 90% data:
ckpt[4]="$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/wd_sweep/GPT2_OpenDV_Llamagen_768_Nodes16_BSperGPU6_totalBS384_weight_decay1e-07_0116_1243_1737027800/checkpoints/vam_before_drop_epoch=000_step=0000139763_fused.pt"
outfile[4]="./slurm_jobs_logs/files/quality_eval/test_layer12_pretrain_768.json"
tokenizer_only[4]=False
# 1024, 90% data:
ckpt[5]="$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/wd_sweep/GPT2_OpenDV_Llamagen_1024_Nodes32_BSperGPU3_totalBS384_weight_decay1e-07_0118_0114_1737159286/checkpoints/vam_before_drop_epoch=000_step=0000139763_fused.pt"
outfile[5]="./slurm_jobs_logs/files/quality_eval/test_layer12_pretrain_1024.json"
tokenizer_only[5]=False
# 2048, 90% data:
ckpt[6]="$ycy_ALL_CCFRSCRATCH/vavim_2048_pretrain_fused.ckpt"
outfile[6]="./slurm_jobs_logs/files/quality_eval/test_layer12_pretrain_2048.json"
tokenizer_only[6]=False


srun python scripts/quality_evaluation.py \
--outfile ${outfile[$SLURM_ARRAY_TASK_ID]} \
--gpt_checkpoint_path ${ckpt[$SLURM_ARRAY_TASK_ID]} \
--dtype bf16 \
--tokenizer_only ${tokenizer_only[$SLURM_ARRAY_TASK_ID]} \
--per_proc_batch_size 32
