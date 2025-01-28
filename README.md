# Video Action Model

## Install

To use the world model, you need to install the following dependencies:

```bash
git clone https://www.github.com/valeoai/NextTokenPredictor
cd NextTokenPredictor
pip install -e .
# to install torch at the same time: pip install -e ".[torch]"
# to code: pip install -e ".[dev]"
```

## DATA

Follow the instructions in the [opendv](vam/datalib/README.md) folder.

## Training

### Pre-training

```bash
python jeanzay_slurm_job_submit.py \
-n GPT2_OpenDV_Llamagen_768_Nodes16_BSperGPU6_totalBS384_weight_decay1e-07 \
--gpus_per_node 4 --nodes 16 \
-wt 20:00:00 \
-p 'experiment=video_pretraining_GPT2_llamagen_ds16_16384_opendv data.batch_size=6 paths.output_dir=$ycy_ALL_CCFRSCRATCH/VAM_test model.network.embedding_dim=768 callbacks=callbacks_opendv_training model.optimizer_conf.weight_decay=1e-07 model.network.init_std=0.0289 model.optimizer_conf.lr=0.0041 data.num_workers=6 ++trainer.max_epochs=1 ++trainer.val_check_interval=0.25'
```

### Fine-tuning

```bash
python jeanzay_slurm_job_submit.py \
-n Finetuned_0000038823_mixOpendvNuplanNuscenes_GPT2_Llamagen_768_Nodes16_BSperGPU6_totalBS384_weight_decay1e-07 \
--gpus_per_node 4 --nodes 16 \
-wt 02:30:00 \
-p 'experiment=finetune_mix_complet data.batch_size=6 paths.output_dir=$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/finetuned  model.network.embedding_dim=768 model.optimizer_conf.weight_decay=1e-07 model.optimizer_conf.lr=0.0041  data.num_workers=6 ckpt_path="$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/wd_sweep/GPT2_OpenDV_Llamagen_1024_Nodes32_BSperGPU3_totalBS384_weight_decay1e-07_0118_0114_1737159286/checkpoints/quarters_epoch\=000_step\=0000038823.ckpt"'
```

### Action learning

```bash
python jeanzay_slurm_job_submit.py \
-n VAM_pretrained0000038823_DDP_Nodes6_BSperGPU16_totalBS384_attdim768_actdim192 \
--gpus_per_node 4 \
--nodes 6 \
-wt 10:00:00 \
-p 'experiment=action_learning data.batch_size=16 model.vam_conf.gpt_config.embedding_dim=768  model.vam_conf.action_config.embedding_dim=192 model.vam_conf.action_config.init_std=0.0086 model.optimizer_conf.lr=0.0194  model.optimizer_conf.weight_decay=1e-07 paths.output_dir=$ycy_ALL_CCFRSCRATCH/output_data/VAM ++trainer.max_epochs=1 data.num_workers=6 model.vam_conf.gpt_checkpoint_path="$ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/finetuned/Finetuned_0000038823_mixOpendvNuplanNuscenes_GPT2_Llamagen_768_Nodes16_BSperGPU6_totalBS384_weight_decay1e-07_0119_1726_1737303976/checkpoints/end_of_epoch_epoch\=001_step\=0000054354_fused.pt" trainer.strategy=ddp +model.grad_logging=100'
```

## Inference

### Video generation

```python
import torch

from vam.video_pretraining import load_pretrained_gpt
from vam.utils import expand_path, plot_images
from vam.datalib import OpenDVTokensDataset, torch_image_to_plot

# Load the pretrained model and the tokenizer decoder.
gpt = load_pretrained_gpt(expand_path("XXX"))
image_detokenizer = torch.jit.load(expand_path("XXXX")).to("cuda")

# Load the dataset.
dts = OpenDVTokensDataset(
    data_root_dir="XXXX",
    video_list=["5pAf38x5z9Q"],  # This is one of the validation video from OpenDV
    sequence_length=8,
    subsampling_factor=5,
)

# Upper bound quality with ground truth tokens.
visual_tokens = dts[100]["visual_tokens"].to("cuda", non_blocking=True)
gt_images = image_detokenizer(dts[100]["visual_tokens"].to("cuda", non_blocking=True))
gt_images = torch_image_to_plot(gt_images)
plot_images(gt_images, 2, 4)

# Generate 4 frames in the future from the first 6 frames.
# Note: we can use bloat16 on A100 or H100 GPUs.
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    generated_frames = gpt.forward_inference(
        number_of_future_frames=4,
        burnin_visual_tokens=visual_tokens.unsqueeze(0)[:, :6],
    )

pred_images = image_detokenizer(generated_frames.squeeze(0))
pred_images = torch_image_to_plot(pred_images)
plot_images(pred_images, 2, 4)
```

### Action generation

```python
import pickle

import torch
from einops import rearrange, repeat

from vam.evaluation import min_ade
from vam.action_expert import load_inference_VAM
from vam.datalib import EgoTrajectoryDataset
from vam.utils import expand_path

vam = load_inference_VAM(expand_path("XXX"), "cuda")

with open("XXX", "rb") as f:
    nuscenes_pickle_data = pickle.load(f)

dataset = EgoTrajectoryDataset(
    pickle_data=nuscenes_pickle_data,
    tokens_rootdir=expand_path("XXX"),
)

num_sampling = 5
sample = dataset[0]
visual_tokens = sample["visual_tokens"].to("cuda", non_blocking=True)
visual_tokens = repeat(visual_tokens, "t h w -> b t h w", b=num_sampling)
commands = sample["high_level_command"].to("cuda", non_blocking=True)[-1:]
commands = repeat(commands, "t -> b t", b=num_sampling)

with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    trajectory = vam(visual_tokens, commands, torch.bfloat16)

ground_truth = sample["positions"].to("cuda", non_blocking=True)[-1:]

ground_truth = sample["positions"].to("cuda", non_blocking=True)[-1:]
trajectory = rearrange(trajectory, "s 1 t a -> 1 s t a")
loss = min_ade(trajectory, ground_truth)
print(loss)
```

### Neuro-NCAP

Please follow instruction on: [Neuro-NCAP](inference/README.md).

![teaser](.github/ressources/frontal_0103_run_45.gif)
