#!/bin/bash
#SBATCH --account=cin4181
#SBATCH --constraint=MI250
#SBATCH --job-name=multinode_batch_size_finder          # Job name
#SBATCH --output=multinode_batch_size_finder-%j.out     # Output file name
#SBATCH --error=multinode_batch_size_finder-%j.out      # Error file name
#SBATCH --nodes=2                                       # Number of nodes
#SBATCH --ntasks-per-node=8                             # Number of tasks per node
#SBATCH --gpus-per-node=8                               # Number of GPUs per node
#SBATCH --cpus-per-task=8                               # Distribute 64 cores evenly across 8 tasks(GPUs) = 8 CPUs/GPU
#SBATCH --exclusive                                     # Exclusive use of nodes
#SBATCH --hint=nomultithread                            # /!\ Caution, 'multithread' in Slurm vocabulary refers to hyperthreading.
#SBATCH --threads-per-core=1                            # 1 process per physical core, no hyperthreading, per default
#SBATCH --time=00:30:00                                 # Time limit hrs:min:sec

# Load required modules and source your Python environment
REPO_DIR="/lus/home/CT10/cin4181/fbartoccioni/NextTokenPredictor"

module purge

source ${REPO_DIR}/scripts/activate_world_model_env.sh

export MPICH_GPU_SUPPORT_ENABLED=1
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_PROCID="$SLURM_PROCID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_PORT="$MASTER_PORT
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export MIOPEN_USER_DB_PATH=/tmp/miopen_${MASTER_ADDR}_${SLURM_PROCID}
echo "MIOPEN_USER_DB_PATH="$MIOPEN_USER_DB_PATH
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${MASTER_ADDR}_${SLURM_PROCID}_cache/
echo "MIOPEN_CUSTOM_CACHE_DIR"=$MIOPEN_CUSTOM_CACHE_DIR
export HYDRA_FULL_ERROR=1

echo "SLURM_GPUS_ON_NODE="$SLURM_GPUS_ON_NODE
echo "SLURM_CPUS_ON_NODE="$SLURM_CPUS_ON_NODE

# echo of launched commands
set -x


# Path to your wrapper script and PyTorch script
WRAPPER_SCRIPT=${REPO_DIR}"/scripts/find_max_batch_size/wrapper_script.py"
PYTORCH_SCRIPT=${REPO_DIR}"/scripts/find_max_batch_size/pytorch_lightning_script.py"

# Iterate over batch sizes
for BATCH_SIZE in {1..10}
do
    echo "Testing batch size: $BATCH_SIZE"
    # Use srun for each batch size check, and launch it in parallel across nodes
    srun --ntasks=16 --gpus-per-task=${SLURM_GPUS_ON_NODE} \
         --cpus-per-task=$((${SLURM_CPUS_ON_NODE} / ${SLURM_GPUS_ON_NODE})) \
         python $WRAPPER_SCRIPT $PYTORCH_SCRIPT $BATCH_SIZE
done