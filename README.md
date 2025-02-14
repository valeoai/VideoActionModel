<div align="center">

# VaViM and VaVAM: Autonomous Driving through Video Generative Modeling

[![Paper](http://img.shields.io/badge/paper-arxiv.0000.0000-B31B1B.svg)](https://arxiv.org/) <br>
[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.4.0-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/) <br>
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/valeoai/VideoActionModel#license)

</div>

This repository contains the code for the VaViM and VaVAM, our video generative model, and video-action model. This repository implements the models detailed in our tech report [tech report](https://arxiv.org/).

```text
This paper explores the potential of large-scale generative video models to enhance
autonomous driving capabilities. We introduce an open-source autoregressive video
model (VaViM) and a companion video-action model (VaVAM) to investigate how
video pre-training can transfer to real-world driving tasks. The video model
VaViM is a simple autoregressive model that predicts future frames by modeling
spatio-temporal token sequences, capturing semantics and dynamics of driving
scenes. The video-action model VaVAM leverages these learned representations
to generate driving trajectories through imitation learning, forming a complete
perception-to-action pipeline. Our study evaluates this approach in both open-
loop and closed-loop driving scenarios, revealing that video-based pre-training
holds promise for autonomous driving. Key insights include the semantic and
geometric richness of learned representations, the benefits of scaling for video
synthesis, and the complex relationship between model size, data, and safety
metrics in closed-loop evaluations. Our code and model weights are released at
github.com/valeoai/VideoActionModel with MIT license for the code and
a RAIL research-only license for the weights.
```

This repository is a research project!

## Install

To use VideoActionModel, install the following dependencies:

```bash
git clone https://www.github.com/valeoai/VideoActionModel
cd VideoActionModel
pip install -e .
# We tested the repo with torch == 2.4.0; to install torch at the same time:
# pip install -e ".[torch]"
# to code:
# pip install -e ".[dev]"
```

Alternatively, on our SLURM cluster we follow the following steps to setup our environment:

```bash
bash scripts/create_vam_env_jeanzay.sh
```

We can then use the repository with the following command:

```bash
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/video_action_model
```

Note: we also have a [Dockerfile](docker/Dockerfile), that we used to run the Neuro-NCAP benchmark, it can provide additional guidance on how to setup an environment.

## Repository structure

```bash
VideoActionModel
|––––inference  # Things related to Neuro-NCAP evaluation
|--------scripts  # Scripts to run the evaluation
|––––mup_shapes  # mup parametrization
|––––notebooks
|--------qualitative*.ipynb  # Qualitative examples for Vavim an Vavam
|--------scaling_laws.ipynb  # notebook to compute the scaling laws of vavim
|––––scripts  # Several useful scripts: fusing deepspeed checkpoints, evaluations, etc...
|––––vam  # Main source code for our project
|--------action_expert  # Implementation of the action expert
|--------datalib  # Data pre-processing and loading
|--------evaluation  # Evaluation utils
|--------video_pretraining  # Implementation of the GPT-style model
```

## Data

Follow the instructions in the [datalib](vam/datalib/README.md) folder.

## Training

The SLURM configurations provided in this repository are specifically tailored for the French Jean-Zay cluster. While it should be compatible with other SLURM clusters, please verify and adapt the scripts, e.g. `jeanzay_slurm_job_submit.py`, to meet your own system requirements.

### Pre-training

To run the VaViM pre-training, you can use the following command:

```bash
python jeanzay_slurm_job_submit.py \
-n VaViM_768_pretraining_OpenDV \
--gpus_per_node 4 --nodes 16 \
-wt 20:00:00 \
-p 'experiment=video_pretraining_GPT2_llamagen_ds16_16384_opendv data.batch_size=6 paths.output_dir=XXXXX model.network.embedding_dim=768 callbacks=callbacks_opendv_training model.optimizer_conf.weight_decay=1e-07 model.network.init_std=0.0289 model.optimizer_conf.lr=0.0041 data.num_workers=6 ++trainer.max_epochs=1 ++trainer.val_check_interval=0.25'
```

### Fine-tuning

You first need to launch the following command to generate the fine-tuning data:

```bash
python scripts/regain_index_from_train.py \
--ckpt /path/to/ckpt.pt \
--outdir $ycy_ALL_CCFRWORK \
--name checkpoint_pretraining
```

Then you can launch the fine-tuning job with the following command:

```bash
python jeanzay_slurm_job_submit.py \
-n VaViM_768_finetuning_data_mix \
--gpus_per_node 4 --nodes 16 \
-wt 02:30:00 \
-p 'experiment=finetune_mix_complet data.batch_size=6 paths.output_dir=XXXXX  model.network.embedding_dim=768 model.optimizer_conf.weight_decay=1e-07 model.optimizer_conf.lr=0.0041 data.num_workers=6 ckpt_path="XXXXX"'
```

### Action learning

Finally, to train VaVAM with imitation learning, you can use the following command:

```bash
python jeanzay_slurm_job_submit.py \
-n VaVAM_768_action_learning_nuPlan_nuScenes \
--gpus_per_node 4 \
--nodes 6 \
-wt 10:00:00 \
-p 'experiment=action_learning data.batch_size=16 model.vam_conf.gpt_config.embedding_dim=768 model.vam_conf.action_config.embedding_dim=192 model.vam_conf.action_config.init_std=0.0086 model.optimizer_conf.lr=0.0194 model.optimizer_conf.weight_decay=1e-07 paths.output_dir=XXXXX ++trainer.max_epochs=1 data.num_workers=6 model.vam_conf.gpt_checkpoint_path="XXXXX" trainer.strategy=ddp +model.grad_logging=100'
```

If you are not using a SLURM environement, you can launch the `vam/train.py` script, and add the following to your command line:

```bash
name=XXX \  # Name of the experiment
++trainer.devices=XXX \  # Number of GPUs per node
++trainer.num_nodes=XXX \  # Number of nodes
```

## Pretrained models

We release all the weights described in our tech report, under a research-only [RAIL Model License](LICENSE_MODEL).

This tables references our main models, in three sizes, you can download both the video generative model (VaViM) and the video-action model (VaVAM) weights.

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

The larger models are split into multiple parts due to the size limit of files for a GitHub release. Please see the [MODELS.md](MODELS.md) on how to untar them.

We also release all the weights described in our tech report, detailed here: [MODELS.md](MODELS.md).

## Inference

### Video generation

To generate a sequence of frames from a sequence of tokens, you can use the following code:

```python
import torch

from vam.datalib import OpenDVTokensDataset, torch_image_to_plot
from vam.utils import expand_path, plot_multiple_images
from vam.video_pretraining import load_pretrained_gpt

vm_checkpoint_path = "XXX"
detokenizer_path = "XXX"
opendv_data_root_dir = "XXX"

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

To generate multiple trajectories, you can use the following code:

```python
import pickle

import torch
from einops import rearrange, repeat

from vam.action_expert import load_inference_VAM
from vam.datalib import EgoTrajectoryDataset
from vam.evaluation import min_ade
from vam.utils import expand_path

vam_checkpoint_path = "XXX"
nuscenes_pickle_data_path = "XXX"
nuscenes_tokens_rootdir = "XXX"

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

There are several example scripts, [detailed here](scripts).

Find notebook examples in the [notebooks](notebooks) folder.

## Evaluation

### Neuro-NCAP

Please follow instruction on: [Neuro-NCAP](inference/README.md).

![teaser](.github/ressources/frontal_0103_run_45.gif)

### Humming bird

In order to use Humming Bird, you need to install an additional package:

```bash
pip install scann
```

To run the Humming Bird evaluation, you can use the following code:

```python
import torch
from einops import rearrange
from torch import Tensor

from vam.evaluation.datasets import CityscapesDataset
from vam.evaluation.hbird import hbird_evaluation
from vam.utils import expand_path
from vam.video_pretraining import load_pretrained_gpt

vm_checkpoint_path = "XXX"
tokenizer_path = "XXX"
cityscapes_root = "XXX"

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

### Humming bird depth

To evaluate with Humming bird depth, you first need to generate pseudo-depth maps using the following code:

```bash
python scripts/depth_anything_a_dataset.py --dataset_name cityscapes
python scripts/depth_anything_a_dataset.py --dataset_name cityscapes --compute_only_issues
```

This should be a standalone script, however it was not exstensively tested.

## TODO

- [x] Remove hard coded paths from the different eval scripts.
- [x] Add the license & model license.
- [x] Finish the compute acknowledgements.
- [x] Update links for the different models.
- [x] Update link for the data_files.tar.gz in [datalib](vam/datalib/README.md).
- [ ] Details of commands to run all different experiments.
- [ ] Upload the pretrained models.
- [ ] Upload the tokenizers (or the script to create JIT files).
- [ ] Upload data tar file.
- [ ] Update citation and the Arxiv link at the top.

## License

This code repository is licensed under [MIT License](LICENSE). The use of pretrained models is subject to the [RAIL Model License](LICENSE_MODEL), which describes the use of our pre-trained weights as academic / research only.

## Citation

If you are using this code, please cite our tech report:

```bibtex
@misc{}
```

You can also cite the code repository:

```bibtex
@software{Bartoccioni_VaVAM,
    author = {{Florent Bartoccioni} and {Elias Ramzi} and {Victor Besnier} and {Loick Chambon} and {Shashanka Venkataramanan} and {Tuan-Hung Vu} and {Yihong Xu} and {Spyros Gidaris} and {Serkan Odabas} and {David Hurych} and {Renaud Marlet} and {Mickael Chen} and {Eloi Zablocki} and {Alexandre Boulch} and {Eduardo Valle} and {Andrei Bursuc} and {Matthieu Cord}},
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

## Compute acknowledgements

We thank the following public structures for granting us access to their computing resources:

- This work was performed using HPC resources from GENCI–IDRIS (Grant 2024-GC011015459).
- This work was performed using HPC resources from GENCI–Adastra (Grant 2024-).
- We acknowledge the EuroHPC Joint Undertaking for awarding this project access to the EuroHPC supercomputer LEONARDO, hosted by CINECA (Italy) and the LEONARDO consortium through an EuroHPC [Extreme/Regular/Benchmark/Development/…] Access call.

## Credits

**Project Lead (Research direction, technical roadmap and project coordination)** <br>
Florent BARTOCCIONI

**Core contributors (Contributed to all aspects of the codebase, ran experiments and evaluations)** <br>
Florent BARTOCCIONI, Elias RAMZI

**Contributors**<br>
Victor BESNIER -- Visual Tokenization codebase using pre-trained VQGAN; FID metric code <br>
Loick CHAMBON -- Data download, transfer and extraction; visualization code development <br>
Eduardo VALLE -- OpenDV preprocessing <br>
Shashanka VENKATARAMANAN -- Depth anything pseudo-GT generation <br>
Tuan-Hung VU -- GPT adaptation from nanoGPT <br>
Yihong XU -- nuPlan preprocessing and initial dataloader development <br>

**Paper (manuscript preparation, designing paper visualization and figures)** <br>
Florent BARTOCCIONI, Victor BESNIER, Shashanka VENKATARAMANAN, Eloi ZABLOCKI, Elias RAMZI, Yihong XU, Tuan-Hung VU

**Public Computing Grant Acquisition (project proposal writing for Adastra, EuroHPC and Jean-Zay grand challenges)** <br>
Florent BARTOCCIONI, Alexandre BOULCH, Eduardo VALLE, Spyros GIDARIS, Eloi ZABLOCKI, Matthieu CORD, Serkan ODABAS, David HURYCH

**Advisory (research and organization guidance)** <br>
Eloi ZABLOCKI, Alexandre BOULCH, Mickael CHEN

**Senior Advisory (research and organization guidance)** <br>
Eduardo VALLE, Andrei BURSUC, Renaud MARLET, Matthieu CORD
