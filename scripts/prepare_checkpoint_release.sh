#!/bin/bash

: '
srun -A ycy@h100 --pty \
--cpus-per-task=3 --hint=nomultithread \
--partition=prepost --time=02:00:00 \
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
--maxsize 1900MB

# Upload the weights with the GitHub CLI
# https://cli.github.com/manual/gh_release_uploads
find $DEST_FOLDER -type f -name "*.tar.gz*" | while read -r filepath; do
    # Copy the file to the destination with the new name
    echo "[ ] Uploading: $filepath"
    gh release upload v1.0.0 $filepath --clobber --repo valeoai/VideoActionModel
    echo "[x] Uploaded: $filepath"
done

find $DATA_FOLDER -type f -name "*.tar.gz" | while read -r filepath; do
    # Copy the file to the destination with the new name
    echo "[ ] Uploading: $filepath"
    gh release upload v1.0.0 $filepath --clobber --repo valeoai/VideoActionModel
    echo "[x] Uploaded: $filepath"
done

# Upload the JIT models for LlamaGen
gh release upload v1.0.0 $ycy_ALL_CCFRWORK/llamagen_jit_models/VQ_ds16_16384_llamagen_encoder.jit#llamagen_encoder.jit --clobber --repo valeoai/VideoActionModel
gh release upload v1.0.0 $ycy_ALL_CCFRWORK/llamagen_jit_models/VQ_ds16_16384_llamagen_decoder.jit#llamagen_decoder.jit --clobber --repo valeoai/VideoActionModel
