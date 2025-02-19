array_file=${1:?"No array file given"}
output_name=${2:?"No output name given (for logging)"}


# loop over all remaining args and check if "--spoof-renderer or --spoof_renderer" is in  ${@:4}
# if it is, set RENDERER_ARGS="--spoof-renderer" and remove it from the list of args
SHOULD_START_RENDERER=true
for arg in ${@:3}; do
  if [[ $arg == "--spoof-renderer" || $arg == "--spoof_renderer" ]]; then
    SHOULD_START_RENDERER=false
  fi
done

# loop over all remaining args and check if "--spoof-renderer or --spoof_renderer" is in  ${@:3}
# if it is, set RENDERER_ARGS="--spoof-renderer" and remove it from the list of args
SHOULD_START_MODEL=true
for arg in ${@:3}; do
  if [[ $arg == "--spoof-model" || $arg == "--spoof_model" ]]; then
    SHOULD_START_MODEL=false
  fi
done

# find two free ports
find_free_port() {
  python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

renderer_port=$(find_free_port)
model_port=$(find_free_port)
echo "Renderer port: $renderer_port"
echo "Model port: $model_port"
echo "Logging to: $NCAP_FOLDER/$LOG_DIR/$TIME_NOW/$output_name"

# figure out the array sequence (and abort if it is empty)
id_to_seq=$NCAP_FOLDER/scripts/arrays/${array_file}.txt
seq=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $id_to_seq)
[[ -z $seq ]] && echo "undefined sequence" && exit 0

echo "Sequence: $seq"

# TODO: how do we set the nuscenes path correctly here?
if [ $SHOULD_START_RENDERER == true ]; then
  echo "Running NeuRAD service in background with ${RENDERING_CONTAINER}..."
  singularity exec --nv \
    --bind $RENDERING_FOLDER:/neurad_studio \
    --bind $NUSCENES_PATH:/neurad_studio/data/nuscenes \
    --bind $HOME/.cache:/.cache \
    --pwd /neurad_studio \
    --env PYTHONPATH=. \
    --env TORCH_HOME=/.cache/torch \
    --env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64 \
    $RENDERING_CONTAINER \
    python -u nerfstudio/scripts/closed_loop/main.py \
    --port $renderer_port \
    --load-config /neurad_studio/$RENDERING_CHECKPOITNS_PATH/$seq/config.yml \
    --adjust_pose \
    &
fi

if [ $SHOULD_START_MODEL == true ]; then
  echo "Running $MODEL_NAME service in background with ${MODEL_CONTAINER}..."
  singularity exec --nv \
    --bind $MODEL_FOLDER:/model \
    --bind $IMAGE_TOKENIZER_PATH:/model/weights/image_tokenizer.jit \
    --bind $VAM_CKPT_PATH:/model/weights/vam.pt \
    --pwd /model \
    --env PYTHONPATH=. \
    --env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64 \
    $MODEL_CONTAINER \
    python -u inference/server.py \
    --port $model_port \
    --config_path /model/weights/image_tokenizer.jit \
    --checkpoint_path /model/weights/vam.pt \
    &
fi

echo "Running NeuroNCAP in foreground with ${NCAP_CONTAINER}..."
singularity exec --nv \
  --bind $NCAP_FOLDER:/neuro_ncap \
  --bind $NUSCENES_PATH:/neuro_ncap/data/nuscenes \
  --bind $LOG_DIR:/logs \
  --pwd /neuro_ncap \
  --env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64 \
  $NCAP_CONTAINER \
  python -u main.py \
  --engine.renderer.port $renderer_port \
  --engine.model.port $model_port \
  --engine.dataset.data_root /neuro_ncap/data/nuscenes \
  --engine.dataset.version v1.0-trainval \
  --engine.dataset.sequence $seq \
  --engine.logger.log-dir /logs/$output_name-$seq \
  --scenario-category $SCENARIO \
  --runs $RUNS \
