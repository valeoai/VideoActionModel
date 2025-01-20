
module purge
module load singularity
# we have to load singularity here to have access to the
# $SINGULARITY_ALLOWED_DIR environment variable

#################################################################
# Edit the following paths to match your setup
export BASE_DIR=$WORK
export NUSCENES_PATH=$ycy_ALL_CCFRSCRATCH/nuscenes_v2
# Model related stuff
export MODEL_NAME='NextTokenPredictor'
export MODEL_FOLDER=$BASE_DIR/$MODEL_NAME
export MODEL_IMAGE='ncap_vai0rbis.sif'
export MODEL_CONTAINER=$SINGULARITY_ALLOWED_DIR/$MODEL_IMAGE
## Tokenizer path
export IMAGE_TOKENIZER_PATH=$fzh_ALL_CCFRSCRATCH/neuroncap_worldmodel_ckpt/jit_models/VQ_ds16_16384_llamagen.jit
## Vai0rbis path
export VAI0RBIS_CKPT_PATH=$ycy_ALL_CCFRSCRATCH/test_fused_checkpoint/gpt_width768_action_dim192_fused.pt
# Rendering related stuff
export RENDERING_FOLDER=$BASE_DIR/'neurad-studio'
export RENDERING_CHECKPOITNS_PATH='checkpoints'
export RENDERING_CONTAINER=$SINGULARITY_ALLOWED_DIR/'neurad_70.sif'  # Changed compared to original code from neuro-ncap
# NCAP related stuff
export NCAP_FOLDER=$BASE_DIR/'neuro-ncap'
export NCAP_CONTAINER=$SINGULARITY_ALLOWED_DIR/'ncap.sif'
export LOG_DIR='logs/vai0rbis_ncap_w768'  # this path is inside the container & binded to $NCAP_FOLDER

# Evaluation default values, set to lower for debugging
export RUNS=50

#################################################################

# SLURM related stuff
export TIME_NOW=$(date +"%Y-%m-%d_%H-%M-%S")
export ACCOUNT='ycy'  # fzh
export GPU_TYPE='h100'  # v100
export QOS='qos_gpu_h100-gc'  # qos_gpu-t3
export WALL_TIME='02:00:00'
export NUM_CPUS=16  # 10

if [ ! -d $NCAP_FOLDER ]; then
    echo "NCAP folder not found"
    exit 1
fi

# assert all the other folders are present
if [ ! -d $MODEL_FOLDER ]; then
    echo "Model folder not found"
    exit 1
fi
if [ ! -d $RENDERING_FOLDER ]; then
    echo "Rendering folder not found"
    exit 1
fi

# assert all singularity files exist
if [ ! -f $MODEL_CONTAINER ]; then
    echo "Model container file not found"
    exit 1
fi
if [ ! -f $RENDERING_CONTAINER ]; then
    echo "Rendering container file not found"
    exit 1
fi
if [ ! -f $NCAP_CONTAINER ]; then
    echo "NCAP container file not found"
    exit 1
fi

mkdir -p $NCAP_FOLDER/$LOG_DIR/$TIME_NOW

# echo the absolute path of this file
JOB_FOLDER=$(dirname $(realpath $0))
export SINGULARITY_JOB_FILE=$JOB_FOLDER/_singularity_job.sh

for SCENARIO in "stationary" "frontal" "side"; do

    if [ $SCENARIO == "stationary" ]; then
        num_scenarios=10
    elif [ $SCENARIO == "frontal" ]; then
        num_scenarios=5
    elif [ $SCENARIO == "side" ]; then
        num_scenarios=5
    fi

    target_file=$JOB_FOLDER/_dispatch_scenario_${ACCOUNT}_${GPU_TYPE}_${SCENARIO}.slurm
    stdout_file=$BASE_DIR/slurm_jobs_logs/ncap/$SCENARIO
    echo "Submitting the following job: $target_file"
    echo "stdout file: $stdout_file"

    sed \
    -e 's|{{SCENARIO}}|'"$SCENARIO"'|g' \
    -e 's|{{NUM_SCENARIOS}}|'"$num_scenarios"'|g' \
    -e 's|{{ACCOUNT}}|'"$ACCOUNT"'|g' \
    -e 's|{{GPU_TYPE}}|'"$GPU_TYPE"'|g' \
    -e 's|{{NUM_CPUS}}|'"$NUM_CPUS"'|g' \
    -e 's|{{QOS}}|'"$QOS"'|g' \
    -e 's|{{WALL_TIME}}|'"$WALL_TIME"'|g' \
    -e 's|{{STDOUT_FILE}}|'"$stdout_file"'|g' \
    -e 's|{{SINGULARITY_JOB_FILE}}|'"$SINGULARITY_JOB_FILE"'|g' \
    < $JOB_FOLDER/_dispatch_scenario.slurm > \
     $target_file

    sbatch $target_file $SCENARIO --runs $RUNS

    # Move the target file to the directory of the stdout file for debugging purposes
    mkdir -p $(dirname $stdout_file)
    mv --backup=numbered $target_file $(dirname $stdout_file)/$(basename $target_file)

done
