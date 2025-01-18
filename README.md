# World model

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

Follow the instructions in the [opendv](opendv/README.md) folder.

## Training

### Pre-training

### Fine-tuning

### Action learning

## Evaluation

### Video generation

```python
import torch

from world_model.gpt2 import load_pretrained_gpt
from world_model.utils import expand_path, plot_images
from world_model.opendv import RandomTokenizedSequenceOpenDVDataset, torch_image_to_plot

# Load the pretrained model and the tokenizer decoder.
gpt = load_pretrained_gpt(expand_path("XXX"))
image_detokenizer = torch.jit.load(expand_path("XXXX")).to("cuda")

# Load the dataset.
dts = RandomTokenizedSequenceOpenDVDataset(
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
