
## Install

See an example Docker file at: https://github.com/valeoai/VisualQuantization/blob/master/fbartocc_Dockerfile_scene_token

Change the relevant paths in configs/paths


## DATA

For now two "quantized nuscenes" are available on the valeo's cluster. Both are accesible under `/master_datasets_preprocessed/nuscenes_quantized`
- VQGAN_ImageNet_f16_1024   available in `/datasets_local` on `[urus,viper]`
- VQGAN_OpenImage_f8_16384  available in `/datasets_local` on `[urus,mustang]`

## Run

Simple command to launch a debug run on valeo's cluster
```
cluster jobs add --name=worldmodel --restricted-to-machines=urus --cpu-memory=80 --gpus=1 -f Dockerfile_scene_token "export HYDRA_FULL_ERROR=1 && cd /home/fbartocc/workspace/scene_tokenization/NextTokenPrediction && pip install -e . && python ./world_model/train.py paths=valeo_debug experiment=GPT2_vqgan_imagenet_f16_1024 debug=gpu_limit_batches"
```

Use:
```
python ./world_model/train.py --help
```
to see the different configuraitions group currently available.

Example:
```
== Configuration groups ==
Compose your configuration from those groups (group=option)

callbacks: callbacks_debug, callbacks_training, early_stopping, learning_rate_monitor, model_checkpoint, model_summary, none, rich_progress_bar, tqdm_progress_bar
data: tokenized_sequence_nuscenes
data/transform: crop_and_resize
debug: default, fast_dev_run, gpu_limit_batches, gpu_overfit, overfit, profiler
experiment: GPT2_vqgan_imagenet_f16_1024
logger: csv, many_loggers, none, tensorboard, wandb
model: next_token_predictor
model/action_quantizer: random_speed_and_curvature
model/network: gpt2
model/sequence_adapter: gpt_adapter
optimizer: adam, adamW
paths: adastra, adastra_debug, default, valeo, valeo_debug
scheduler: none, step_lr
trainer: cpu, ddp, ddp_sim, default, gpu
```

To change a network, after adding the code and the config file under `configs/model/network` just add:
```
model/network=your_network
```
to use it

