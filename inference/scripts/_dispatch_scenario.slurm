#!/bin/bash
#SBATCH --job-name={{SCENARIO}}
#SBATCH -A {{ACCOUNT}}@{{GPU_TYPE}}
#SBATCH -C {{GPU_TYPE}}
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={{NUM_CPUS}}
#SBATCH --hint=nomultithread
#SBATCH --time={{WALL_TIME}}
#SBATCH --qos={{QOS}}
#SBATCH --output={{STDOUT_FILE}}_%A_%a.out
#SBATCH --error={{STDOUT_FILE}}_%A_%a.out
#SBATCH --array=1-{{NUM_SCENARIOS}}

mkdir -p $(dirname {{STDOUT_FILE}})

module purge
module load singularity
export TMPDIR=$JOBSCRATCH

scenario=${1:?"No scenario given"}

export SCENARIO=$scenario

SINGULARITY_START_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
srun bash {{SINGULARITY_JOB_FILE}} ncap_slurm_array_$scenario $scenario --scenario-category=$scenario ${@:2}
SINGULARITY_END_TIME=$(date +"%Y-%m-%d_%H-%M-%S")

echo "Singularity job started at $SINGULARITY_START_TIME and ended at $SINGULARITY_END_TIME"
