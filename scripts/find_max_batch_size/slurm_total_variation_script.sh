#!/bin/bash
#SBATCH --account=cin4181
#SBATCH --constraint=MI250
#SBATCH --job-name=multinode_batch_size_finder          # Job name
#SBATCH --output=multinode_batch_size_finder-%j.out     # Output file name
#SBATCH --error=multinode_batch_size_finder-%j.out      # Error file name
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8 
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread                            # /!\ Caution, 'multithread' in Slurm vocabulary refers to hyperthreading.
#SBATCH --threads-per-core=1                            # 1 process per physical core, no hyperthreading, per default
#SBATCH --exclusive


# Load required modules and source your Python environment
REPO_DIR="/lus/home/CT10/cin4181/fbartoccioni/NextTokenPredictor"

module purge

source ${REPO_DIR}/scripts/activate_world_model_env.sh

WRAPPER_SCRIPT=${REPO_DIR}"/scripts/find_max_batch_size/wrapper_script.py"

# Loop over different total batch sizes
for TOTAL_BATCH_SIZE in 64 128 256; do
    # Loop over different model sizes
    for LAYER_DIM in 768; do
        # Loop over different numbers of nodes in use (1 to 8)
        for USE_NODES in 1 2; do
            # Calculate the total number of GPUs based on the number of nodes used
            NUM_GPUS=$(($USE_NODES * 8))

            # Calculate per-GPU batch size
            if (( NUM_GPUS != 0 )); then
                PER_GPU_BATCH_SIZE=$(($TOTAL_BATCH_SIZE / $NUM_GPUS))
            else
                PER_GPU_BATCH_SIZE=0  # Prevent division by zero
            fi

            NB_HEADS=$(($LAYER_DIM / 64))

            PYTORCH_SCRIPT="$REPO_DIR/world_model/train.py experiment=GPT2_vqgan_imagenet_f16_1024 trainer=deepspeed2 ++trainer.limit_train_batches=5 ++trainer.limit_val_batches=2 ++trainer.max_epochs=2 paths.quantized_nuscenes_root_dir=/scratch/sshfs3/ML-AI/world_model_project/data_preprocessed/nuscenes_tokenized/VQGAN_ImageNet_f16_1024/  model.network.nb_layers=12  data.train_dataset_params.sequence_length=16 data.dataloader_params.multiprocessing_context='fork' model.network.embedding_dim=$LAYER_DIM model.network.nb_heads=$NB_HEADS data.dataloader_params.batch_size=$PER_GPU_BATCH_SIZE  ++trainer.devices=$NUM_GPUS ++trainer.num_nodes=$USE_NODES  name=GPT2_atnorth_nuscenes_test_`date '+%m%d_%H%M_%s'`"

            # Run the job with srun adjusting the number of nodes dynamically
            srun --nodes=$USE_NODES --ntasks-per-node=$NUM_GPUS \
            python $WRAPPER_SCRIPT $PYTORCH_SCRIPT
        done
    done
done

echo "Completed batch size, model size, and node usage testing."
