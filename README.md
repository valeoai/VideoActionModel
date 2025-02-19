<div align="center">

# VaViM and VaVAM: Autonomous Driving through Video Generative Modeling

[![Paper](http://img.shields.io/badge/paper-arxiv.0000.0000-B31B1B.svg)](https://arxiv.org/)
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://valeoai.github.io/vavim-vavam/)
[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.4.0-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/valeoai/VideoActionModel#license)

</div>

This is the code for VaViM (video generative model) and VaVAM (video-action model) as detailed in [our tech report](https://arxiv.org/).

We explores the potential of large-scale generative video models to enhance
autonomous driving capabilities, introducing an open-source autoregressive video
model (VaViM) and a companion video-action model (VaVAM).
VaViM is a simple autoregressive model that predicts frames using
spatio-temporal token sequences, while VaVAM leverages the learned representations to generate driving trajectories through imitation learning. Together, they offer a complete perception-to-action pipeline.

This is a research project released to the scientific community to foster advances in end-to-end autonomous driving. **It is not intended for deployment nor commercial use.** Please refer to the Licenses below.

## Installation

Before using VaViM or VaVAM you must install its dependencies. The easiest way to do it is to install this repo as a package with `pip`:

```bash
git clone https://www.github.com/valeoai/VideoActionModel
cd VideoActionModel
# Installs all runtime dependencies, except for PyTorch:
pip install -e .
# Installs all runtime dependencies, including PyTorch 2.4.0 (recommended version):
# pip install -e ".[torch]"
# Installs all runtime and development dependencies:
# pip install -e ".[dev]"
```

Installation in cluster / cloud environments may require adaptation to their software infrastructure. While we cannot offer guidance for each individual situation, we present, as an example, the [setup we employed in a SLURM environment](scripts/create_worldmodel_env_jeanzay.sh).

We also have  [Dockerfile](docker/Dockerfile) used to run the NeuroNCAP benchmark, which can provide additional hints on how to setup an environment.

## Repository structure

```text
VideoActionModel
|––inference    => NeuroNCAP evaluation
     \--scripts => Scripts to run the NeuroNCAP evaluation
|––mup_shapes   => mup parameterization
|––notebooks
     |--qualitative*.ipynb => Qualitative examples for VaViM an VaVAM
     \--scaling_laws.ipynb => Scaling laws of VaViM
|––scripts  => Utilities: fusing deepspeed checkpoints, evaluations...
\––vam      => Main source code
     |--action_expert     => Action expert
     |--datalib           => Data pre-processing and loading
     |--evaluation        => Evaluation utils
     \--video_pretraining => GPT-style model
```

## Data

To obtain and prepare the training and evaluation datasets, please follow the instructions in the [datalib folder](vam/datalib/README.md).

## Training

VaViM and VaVAM are large-scale models that require at least a few dozen latest-model GPUs to train. Please refer to the technical report for more details.

The instructions and scripts provided below were specifically tailored for the French Jean Zay cluster. Adapting them to other SLURM clusters could require some adjustments, completely different environments (e.g., cloud) may require extensive adaptation.

### Pre-training

To pre-train VaViM, you can use:

```bash
python jeanzay_slurm_job_submit.py \
-n VaViM_768_pretraining_OpenDV \
--gpus_per_node 4 --nodes 16 \
-wt 20:00:00 \
-p 'experiment=video_pretraining_GPT2_llamagen_ds16_16384_opendv data.batch_size=6 paths.output_dir=XXXXX model.network.embedding_dim=768 callbacks=callbacks_opendv_training model.optimizer_conf.weight_decay=1e-07 model.network.init_std=0.0289 model.optimizer_conf.lr=0.0041 data.num_workers=6 ++trainer.max_epochs=1 ++trainer.val_check_interval=0.25'
```

### Fine-tuning

First, generate the fine-tuning data:

```bash
python scripts/regain_index_from_train.py \
--ckpt /path/to/ckpt.pt \
--outdir $ycy_ALL_CCFRWORK \
--name checkpoint_pretraining
```

Then, launch the fine-tuning job:

```bash
python jeanzay_slurm_job_submit.py \
-n VaViM_768_finetuning_data_mix \
--gpus_per_node 4 --nodes 16 \
-wt 02:30:00 \
-p 'experiment=finetune_mix_complet data.batch_size=6 paths.output_dir=XXXXX  model.network.embedding_dim=768 model.optimizer_conf.weight_decay=1e-07 model.optimizer_conf.lr=0.0041 data.num_workers=6 ckpt_path="XXXXX"'
```

### Action learning

Finally, to train VaVAM with imitation learning:

```bash
python jeanzay_slurm_job_submit.py \
-n VaVAM_768_action_learning_nuPlan_nuScenes \
--gpus_per_node 4 \
--nodes 6 \
-wt 10:00:00 \
-p 'experiment=action_learning data.batch_size=16 model.vam_conf.gpt_config.embedding_dim=768 model.vam_conf.action_config.embedding_dim=192 model.vam_conf.action_config.init_std=0.0086 model.optimizer_conf.lr=0.0194 model.optimizer_conf.weight_decay=1e-07 paths.output_dir=XXXXX ++trainer.max_epochs=1 data.num_workers=6 model.vam_conf.gpt_checkpoint_path="XXXXX" trainer.strategy=ddp +model.grad_logging=100'
```

Outside an SLURM environement, you can launch the `vam/train.py` script, and add the following to your command line:

```bash
name=XXX \  # Name of the experiment
++trainer.devices=XXX \  # Number of GPUs per node
++trainer.num_nodes=XXX \  # Number of nodes
```

## Pretrained models

We release several sets of weights of the models, corresponding to different combinations of parameters and data. The most important combinations are below.

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />params</th>
      <th>VaViM</th>
      <th>VaVAM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VaVAM-S</td>
      <td align="right">185M + 21M</td>
      <td><a href="https://www.github.com/valeoai/VideoActionModel">part 1</a></td>
      <td><a href="https://www.github.com/valeoai/VideoActionModel">part 1</a></td>
    </tr>
    <tr>
      <td>VaVAM-B</td>
      <td align="right">318M + 38M</td>
      <td><a href="https://www.github.com/valeoai/VideoActionModel">part 1</a></td>
      <td><a href="https://www.github.com/valeoai/VideoActionModel">part 1</a></td>
    </tr>
    <tr>
      <td>VaVAM-L</td>
      <td align="right">1.2B + 150M</td>
      <td><a href="https://www.github.com/valeoai/VideoActionModel">part 1</a>, <a href="https://www.github.com/valeoai/VideoActionModel">part 2</a></td>
      <td><a href="https://www.github.com/valeoai/VideoActionModel">part 1</a>, <a href="https://www.github.com/valeoai/VideoActionModel">part 2</a>, <a href="https://www.github.com/valeoai/VideoActionModel">part 3</a></td>
    </tr>
  </tbody>
</table>

The larger models are split into multiple parts due to limitations on large binary attachments of GitHub. Please refer to [MODELS.md](MODELS.md) for the instructions on how to untar them, as well as the complete list with all the weight sets described in our tech report.

**We are releasing the model / weights to the scientific community to foster research advances. Remark that the model / weights license is more restrictive than the code license. Please see below.**

## Inference

### Video generation

To generate a sequence of frames from a sequence of tokens, use the following code as reference:

```python
import torch

from vam.datalib import OpenDVTokensDataset, torch_image_to_plot
from vam.utils import expand_path, plot_multiple_images
from vam.video_pretraining import load_pretrained_gpt

vm_checkpoint_path = "/path/to/trained/vavim"
detokenizer_path = "/path/to/trained/detokenizer"
opendv_data_root_dir = "/path/to/preprocessed/opendv"

# Load the pretrained model and the tokenizer decoder.
gpt = load_pretrained_gpt(expand_path(vm_checkpoint_path))
image_detokenizer = torch.jit.load(expand_path(detokenizer_path)).to("cuda")

# Load the dataset.
dts = OpenDVTokensDataset(
    data_root_dir=opendv_data_root_dir,
    video_list=["5pAf38x5z9Q"],  # This is one of the validation video from OpenDV
    sequence_length=8,
    subsampling_factor=5,
)

# Upper bound quality with ground truth tokens.
visual_tokens = dts[100]["visual_tokens"].to("cuda", non_blocking=True)
gt_images = image_detokenizer(dts[100]["visual_tokens"].to("cuda", non_blocking=True))
gt_images = torch_image_to_plot(gt_images)
_ = plot_multiple_images(gt_images, 2, 4)

# Generate 4 frames in the future from the first 6 frames.
# Note: we can use bloat16 on A100 or H100 GPUs.
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    generated_frames = gpt.forward_inference(
        number_of_future_frames=4,
        burnin_visual_tokens=visual_tokens.unsqueeze(0)[:, :6],
    )

pred_images = image_detokenizer(generated_frames.squeeze(0))
pred_images = torch_image_to_plot(pred_images)
_ = plot_multiple_images(pred_images, 1, 4)
```

### Action generation

To generate multiple trajectories, use the following code as reference:

```python
import pickle

import torch
from einops import rearrange, repeat

from vam.action_expert import load_inference_VAM
from vam.datalib import EgoTrajectoryDataset
from vam.evaluation import min_ade
from vam.utils import expand_path

vam_checkpoint_path = "/path/to/trained/vavam"
nuscenes_pickle_data_path = "/path/to/nuscenes/pickle"
nuscenes_tokens_rootdir = "/path/to/nuscenes/tokens_dir"

# Load the pretrained model.
vam = load_inference_VAM(expand_path(vam_checkpoint_path), "cuda")

with open(nuscenes_pickle_data_path, "rb") as f:
    nuscenes_pickle_data = pickle.load(f)

dataset = EgoTrajectoryDataset(
    pickle_data=nuscenes_pickle_data,
    tokens_rootdir=expand_path(nuscenes_tokens_rootdir),
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

### Examples

Please refer to to the [scripts](scripts) and [notebooks](notebooks) folders for several exaples of VaViM and VaVAM in action.

## Evaluation

### NeuroNCAP

Please follow the instructions on the [NeuroNCAP readme](inference/README.md) for the evaluation on this task.

![teaser](.github/ressources/frontal_0103_run_45.gif)

### Humming Bird

In order to use Humming Bird, you need an additional package:

```bash
pip install scann
```

To run the Humming Bird evaluation, you can use the following code as reference:

```python
import torch
from einops import rearrange
from torch import Tensor

from vam.evaluation.datasets import CityscapesDataset
from vam.evaluation.hbird import hbird_evaluation
from vam.utils import expand_path
from vam.video_pretraining import load_pretrained_gpt

vm_checkpoint_path = "/path/to/trained/vavim"
tokenizer_path = "/path/to/trained/tokenizer"
cityscapes_root = "/path/to/cityscapes/root"

# Load the pretrained model and the tokenizer.
gpt = load_pretrained_gpt(expand_path(vm_checkpoint_path))
image_tokenizer = torch.jit.load(expand_path(tokenizer_path)).to("cuda")
model_info = {
    "patch_size": 16,
    "d_model": gpt.embedding_dim,
}

def forward_fn(x: Tensor, inference: bool) -> Tensor:
    x = image_tokenizer(x)
    x = rearrange(x, "b h w -> b 1 h w")
    x = gpt.get_intermediate_layers(x, 12)
    x = rearrange(x[:, -1], "b h w d -> b (h w) d")
    return x

train_dts = CityscapesDataset(root=expand_path(cityscapes_root), split="train")
val_dts = CityscapesDataset(root=expand_path(cityscapes_root), split="val")

logs, _ = hbird_evaluation(
    ftr_extr_fn=forward_fn,
    model_info=model_info,
    train_dataset=train_dts,
    val_dataset=val_dts,
    batch_size=16,
    batch_size_eval=16,
    augmentation_epoch=1,
    device="cuda",
    dtype="bf16",
    return_labels=False,
    num_neighbour=30,
    nn_params=None,
    memory_size="x10",  # you can set this to reduce memory size
    f_mem_p=None,
    l_mem_p=None,
)

print(logs["mIoU"], logs["IoU"])
```

### Humming Bird depth

To evaluate with Humming Bird depth, first generate the pseudo-depth maps:

```bash
python scripts/depth_anything_a_dataset.py --dataset_name cityscapes
python scripts/depth_anything_a_dataset.py --dataset_name cityscapes --compute_only_issues
```

You can then:

- Pass `evaluation_task="depth"` to the `hbird_evaluation` function above.
- Increase the `memory_size` parameter to `x100` which leads to better results for depth estimation.

## License

We are releasing the code in this repository under the [MIT License](LICENSE).

We are releasing the pre-trained models / weights under the **research-only** [VideoActionModel License](LICENSE_MODEL). The weights were trained with datasets that are subjected to their own licenses and restrictions. Please see below.

This project contains data derived from the [nuScenes dataset](https://www.nuscenes.org) and are licensed under [CC BY-NC-SA 4.0](LICENSE_DATA). Using those data come with the following terms of use:

- Those data can be used for non-commercial purposes only
- Any derivatives must be distributed under the same license
- Attribution must be provided to both this project and nuScenes

## Citation

If you use this code, please cite our technical report:

```bibtex
@misc{}
```

You can also cite the code repository:

```bibtex
@software{Bartoccioni_VaVAM,
    author = {{Florent Bartoccioni} and {Elias Ramzi} and {Victor Besnier} and {Loick Chambon} and {Shashanka Venkataramanan} and {Tuan-Hung Vu} and {Yihong Xu} and {Spyros Gidaris} and {Serkan Odabas} and {David Hurych} and {Renaud Marlet} and {Mickael Chen} and {Eloi Zablocki} and {Alexandre Boulch} and {Andrei Bursuc} and {Eduardo Valle} and {Matthieu Cord}},
    license = {MIT},
    title = {{VaViM and VaVAM: Autonomous Driving through Video Generative Modeling}},
    url = {https://github.com/valeoai/VideoActionModel}
}
```

## Sources

This code was inspired / contains parts of the following repositories:

- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [LLamaGen](https://github.com/FoundationVision/LlamaGen)
- [open-pi-zero](https://github.com/allenzren/open-pi-zero)
- [open-hummingbird-eval](https://github.com/vpariza/open-hummingbird-eval)

## Data sources

We train our VaViM and VaVAM models on the following datasets:

- [OpenDV](https://arxiv.org/abs/2403.09630)
- [nuPlan](https://www.nuscenes.org/nuplan)
- [nuScenes](https://www.nuscenes.org/)

We additionally use the following datasets for evaluation:

- [Cityscapes](https://arxiv.org/abs/1604.01685)
- [KITTI](https://www.cvlibs.net/publications/Geiger2013IJRR.pdf)

## Compute acknowledgements

We thank the following public structures for granting us access to their computing resources:

- This work was performed using HPC resources from GENCI–IDRIS (Grant 2024-GC011015459).
- This work was performed using HPC resources from GENCI–Adastra (Grant 2023-
A0141014181).
- We acknowledge the EuroHPC Joint Undertaking for awarding this project access to the EuroHPC supercomputer LEONARDO, hosted by CINECA (Italy) and the LEONARDO consortium through an EuroHPC AI and Data-Intensive Access call.

## Credits

**Project Lead (Research direction, technical roadmap, project coordination)** <br>
Florent BARTOCCIONI

**Core contributors (All aspects of the codebase, experiments, evaluations)** <br>
Florent BARTOCCIONI, Elias RAMZI

**Contributors**<br>
Victor BESNIER -- Visual Tokenization codebase using pre-trained VQGAN; FID metric code <br>
Loick CHAMBON -- Data download, transfer and extraction; visualization code development <br>
Eduardo VALLE -- OpenDV preprocessing <br>
Shashanka VENKATARAMANAN -- Depth anything pseudo-GT generation <br>
Tuan-Hung VU -- GPT adaptation from nanoGPT <br>
Yihong XU -- nuPlan preprocessing and initial dataloader development <br>

**Technical report (Manuscript preparation, design, visualization, figures)** <br>
Florent BARTOCCIONI, Elias RAMZI, Victor BESNIER, Shashanka VENKATARAMANAN, Eloi ZABLOCKI, Yihong XU, Tuan-Hung VU

**Grant Acquisitions (Grant proposals for Adastra, EuroHPC, and Jean Zay Grand Challenges)** <br>
Florent BARTOCCIONI, Alexandre BOULCH, Eduardo VALLE, Spyros GIDARIS, Eloi ZABLOCKI, Matthieu CORD, Serkan ODABAS, David HURYCH

**Advisory (research and organization guidance)** <br>
Eloi ZABLOCKI, Alexandre BOULCH, Mickael CHEN

**Senior Advisory (research and organization guidance)** <br>
Eduardo VALLE, Andrei BURSUC, Renaud MARLET, Matthieu CORD
