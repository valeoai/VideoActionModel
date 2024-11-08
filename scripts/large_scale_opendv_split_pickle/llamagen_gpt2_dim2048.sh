#!/bin/bash

# Define common variables
OUTPUT_DIR='/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/large_scale'
RUN_NAME='muP_GPT2_split_opendv_dim2048'
TRAIN_SCRIPT=/lustre/fswork/projects/rech/ycy/uyv67bd/NextTokenPredictor/world_model/train.py
DATA_DIR=/lustre/fsn1/projects/rech/ycy/commun/OpenDV_tokenized/frames512/VQ_ds16_16384_llamagen

STD=0.0843
LR=0.0092

# Function to create SBATCH script for a stage
create_sbatch_script() {
    local stage=$1
    local ckpt_path=$2
    local script_name="job_stage_${stage}.sh"

    cat << EOF > $script_name
#!/bin/bash

#SBATCH --job-name=muP_GPT2_split_opendv_dim2048_stage_${stage}
#SBATCH -C h100
#SBATCH -A ycy@h100
#SBATCH --nodes=48
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=$WORK/slurm_jobs_logs/stdout/%x_%j.out
#SBATCH --error=$WORK/slurm_jobs_logs/stdout/%x_%j.out

# Load modules and set environment variables
module purge
module load arch/h100 
module load pytorch-gpu/py3/2.4.0

export PYTHONUSERBASE=$WORK/python_envs/worldmodel
export MPICH_GPU_SUPPORT_ENABLED=1
export TRITON_CACHE_DIR=$SCRATCH/.triton
export HYDRA_FULL_ERROR=1
# export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

# Echo launched commands
set -x

srun python $TRAIN_SCRIPT \\
    experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction_stage_${stage} \\
    model.network.init_std=$STD \\
    optimizer.lr=$LR \\
    model.network.embedding_dim=2048 \\
    model.network.nb_heads=16 \\
    data.batch_size=2 \\
    data.num_workers=6 \\
    data.data_root_dir=$DATA_DIR \\
    paths.output_dir=$OUTPUT_DIR \\
    optimizer.weight_decay=0.1 \\
    ++trainer.devices=4 \\
    ++trainer.num_nodes=48 \\
    name=${RUN_NAME}_part_${stage} \\
    $([ -n "$ckpt_path" ] && echo "++ckpt_path=$ckpt_path")
EOF

    chmod +x $script_name
    echo $script_name
}

# Create SBATCH scripts for each stage
script1=$(create_sbatch_script 1)
script2=$(create_sbatch_script 2 "${OUTPUT_DIR}/${RUN_NAME}_part_1/checkpoints/last.ckpt")
script3=$(create_sbatch_script 3 "${OUTPUT_DIR}/${RUN_NAME}_part_2/checkpoints/last.ckpt")

# Submit jobs with dependencies
job1_id=$(sbatch --parsable $script1)
job2_id=$(sbatch --parsable --dependency=afterok:$job1_id $script2)
job3_id=$(sbatch --parsable --dependency=afterok:$job2_id $script3)

echo "Submitted jobs: $job1_id, $job2_id, $job3_id"