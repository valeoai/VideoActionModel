#################################################################
# Edit the following paths to match your setup
BASE_DIR='/mnt/data/shared/eramzi'
CKPT_BASE_DIR='/mnt/iveco'
# Model related stuff
MODEL_NAME='NextTokenPredictor'
MODEL_FOLDER=$BASE_DIR/$MODEL_NAME
MODEL_CHECKPOINT_PATH=''
MODEL_CFG_PATH='configs/inference_ncap.yaml'
MODEL_IMAGE='ncap_wm:latest'

NUSCENES_PATH='/media/data/datasets/nuscenes'

TORCH_HOME='/.cache'
MODEL_DOCKER_NAME='ncap_model_debug'

# Tokenizer paths
IMAGE_TOKENIZER_PATH=$CKPT_BASE_DIR/scratch_iveco/world_model_JZGC4/jit_models/VQ_ds16_16384_llamagen.jit
TRAJECTORY_TOKENIZER_PATH=$CKPT_BASE_DIR/scratch_iveco/world_model_JZGC4/jit_models/trajectory_decoder_RFSQ_2levels_8x5x5x5.jit
# WM paths
WM_BASE_LOGDIR=$CKPT_BASE_DIR/scratch_iveco/world_model_JZGC4/model_logs_and_checkpoints/Finetune_opendv_dim2048_part3_imitation_learning_nuscenes_2epoch_dropLR
WM_CKPT_PATH=$WM_BASE_LOGDIR/"checkpoints/step=0000000056_fused.pt"
WM_CONFIG_PATH=$WM_BASE_LOGDIR/"tensorboard/version_0/hparams.yaml"



docker run --name $MODEL_DOCKER_NAME --rm --gpus '"device=0"' \
    -v $MODEL_FOLDER:/model \
    -v $BASE_DIR/.cache:$TORCH_HOME \
    -v $IMAGE_TOKENIZER_PATH:/model/weights/tokenizers/image_tokenizer.jit \
    -v $TRAJECTORY_TOKENIZER_PATH:/model/weights/tokenizers/trajectory_tokenizer.jit \
    -v $WM_CKPT_PATH:/model/weights/world_model/checkpoint.pt \
    -v $WM_CONFIG_PATH:/model/weights/world_model/config.yaml \
    -v $NUSCENES_PATH:/model/data/nuscenes \
    -e TORCH_HOME=$TORCH_HOME \
    -w /model \
    --network host \
    -e PYTHONPATH=. \
    $MODEL_IMAGE \
    python -u inference/runner.py # \
    # --port $model_port \
    # --config_path $MODEL_CFG_PATH \
    # --checkpoint_path $MODEL_CHECKPOINT_PATH
