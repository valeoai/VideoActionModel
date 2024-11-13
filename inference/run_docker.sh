#################################################################
# Edit the following paths to match your setup
BASE_DIR='/mnt/data/shared/eramzi'
# Model related stuff
MODEL_NAME='NextTokenPredictor'
MODEL_FOLDER=$BASE_DIR/$MODEL_NAME
MODEL_CHECKPOINT_PATH=''
MODEL_CFG_PATH='configs/inference.yaml'
MODEL_IMAGE='ncap_wm:latest'

NUSCENES_PATH='/media/data/datasets/nuscenes'

TORCH_HOME='/.cache'
MODEL_DOCKER_NAME='ncap_model_debug'

CKPT_DIR=/mnt/data/shared/eramzi/llamagen_distilled/weights


docker run --name $MODEL_DOCKER_NAME --rm --gpus '"device=7"' \
    -v $MODEL_FOLDER:/model \
    -v $BASE_DIR/.cache:$TORCH_HOME \
    -v $CKPT_DIR:/model/weights \
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
