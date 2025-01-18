# Request interactive session with 1 GPU
# srun -A fzh@v100 -C v100 --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 --hint=nomultithread --qos=qos_gpu-t3 --time=00:30:00 bash

# srun -A ycy@h100 -C h100 --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 --hint=nomultithread --qos=qos_gpu_h100-dev --time=00:10:00 bash run_single_job.sh

module purge
module load singularity
export TMPDIR=$JOBSCRATCH
export SINGULARITY_CACHEDIR=$JOBSCRATCH

singularity version

export seq=0101
export scenario=stationary
export runs=50
export BASE_DIR=$WORK
export RENDERING_FOLDER=$BASE_DIR/neurad-studio
export RENDERING_CHECKPOITNS_PATH=/neurad_studio/checkpoints
export RENDERING_CONTAINER=$SINGULARITY_ALLOWED_DIR/neurad_70.sif
export MODEL_CONTAINER=$SINGULARITY_ALLOWED_DIR/ncap_vai0rbis.sif
export NCAP_FOLDER=$BASE_DIR/neuro-ncap
export NCAP_CONTAINER=$SINGULARITY_ALLOWED_DIR/ncap.sif

echo "Sequence: $seq"
echo "Scenario: $scenario"

## Tokenizer paths
export IMAGE_TOKENIZER_PATH=$fzh_ALL_CCFRSCRATCH/neuroncap_worldmodel_ckpt/jit_models/VQ_ds16_16384_llamagen.jit
## Vai0rbis paths
export VAI0RBIS_BASE_LOGDIR=$fzh_ALL_CCFRSCRATCH/neuroncap_worldmodel_ckpt/Finetune_opendv_dim2048_part3_imitation_learning_nuscenes_2epoch_dropLR
export VAI0RBIS_CKPT_PATH=$VAI0RBIS_BASE_LOGDIR/"checkpoints/step=0000000056_fused.pt"
export VAI0RBIS_CONFIG_PATH=$VAI0RBIS_BASE_LOGDIR/"checkpoints/hparams.yaml"

export renderer_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export model_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

singularity exec --nv \
    --bind $BASE_DIR/neurad-studio:/neurad_studio \
    --bind $ycy_ALL_CCFRSCRATCH/nuscenes_v2:/neurad_studio/data/nuscenes \
    --bind $HOME/.cache:/.cache \
    --pwd /neurad_studio \
    --env PYTHONPATH=. \
	--env TORCH_HOME=/.cache/torch \
	--env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64 \
    $RENDERING_CONTAINER \
    python -u nerfstudio/scripts/closed_loop/main.py \
    --port $renderer_port \
    --load-config $RENDERING_CHECKPOITNS_PATH/$seq/config.yml \
    --adjust_pose \
	&

singularity exec --nv \
    --bind $WORK/NextTokenPredictor:/model \
    --bind $IMAGE_TOKENIZER_PATH:/model/weights/tokenizers/image_tokenizer.jit \
    --pwd /model \
    --env PYTHONPATH=. \
	--env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64 \
    $MODEL_CONTAINER \
	python -u inference/server.py \
    --port $model_port \
    --config_path /model/configs/inference_ncap.yaml \
    --checkpoint_path null \
	&

singularity exec --nv \
  --bind $NCAP_FOLDER:/neuro_ncap \
  --bind $ycy_ALL_CCFRSCRATCH/nuscenes_v2:/neuro_ncap/data/nuscenes \
  --bind $ycy_CCFRSCRATCH/logs/debug_neuroncap:/neuro_ncap/logdir \
  --pwd /neuro_ncap \
  --env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64 \
  $NCAP_CONTAINER \
  python -u main.py \
  --engine.renderer.port $renderer_port \
  --engine.model.port $model_port \
  --engine.dataset.data_root /neuro_ncap/data/nuscenes \
  --engine.dataset.version v1.0-trainval \
  --engine.dataset.sequence $seq \
  --engine.logger.log-dir /neuro_ncap/logdir/$seq \
  --scenario-category $scenario \
  --runs $runs \
