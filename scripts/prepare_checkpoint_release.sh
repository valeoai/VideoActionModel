#!/bin/bash

: '
srun -A cya@h100 -C h100 --pty \
--nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:0 --hint=nomultithread \
--qos=qos_gpu_h100-dev --time=00:45:00 \
bash scripts/prepare_checkpoint_release.sh
'

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/video_action_model

# Data folder containing pickles and jsons
DATA_FOLDER="${ycy_ALL_CCFRSCRATCH}/release_weights_v1/datafiles"
# Source folder containing all checkpoints
SRC_FOLDER="${ycy_ALL_CCFRSCRATCH}/release_weights_v1"
file_count=$(find "$SRC_FOLDER" -type f -name "*.pt" | wc -l)
echo "Found $file_count files in $SRC_FOLDER"

# Destination folder for all copied videos
DEST_FOLDER="${ycy_ALL_CCFRSCRATCH}/release_weights_tar"
# Create destination folder if it doesn't exist
mkdir -p "$DEST_FOLDER"

python scripts/handle_checkpoints.py \
--mode create \
--checkpoint_dir $SRC_FOLDER \
--outdir $DEST_FOLDER \
--num_threads 16 \
--maxsize 2G

# Upload the weights with the GitHub CLI
# https://cli.github.com/manual/gh_release_uploads
find $DEST_FOLDER -type f -name "*.tar.gz*" | while read -r filepath; do
    # Copy the file to the destination with the new name
    echo "[ ] Uploading: $filepath"
    gh release upload v1.0.0 $filepath --clobber
    echo "[x] Uploaded: $filepath"
done

find $DATA_FOLDER -type f -name "*.tar.gz" | while read -r filepath; do
    # Copy the file to the destination with the new name
    echo "[ ] Uploading: $filepath"
    gh release upload v1.0.0 $filepath --clobber
    echo "[x] Uploaded: $filepath"
done
