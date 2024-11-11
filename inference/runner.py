import uuid
import yaml
from typing import List
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from einops import rearrange
from PIL import Image

from hydra.utils import instantiate
from omegaconf import OmegaConf
from world_model.inference import load_model, WorldModelInference, instantiate_samplers

from llamagen import VQ_16

NUSCENES_CAM_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


@dataclass
class WMInferenceInput:
    imgs: List[np.ndarray]
    """shape: (n-cams (1), h (900), w (1600) c (3)) | images without any preprocessing. should be in RGB order as uint8"""
    timestamp: float
    """timestamp of the current frame in seconds"""
    command: int
    """0: right, 1: left, 2: straight"""


@dataclass
class WMAuxOutputs:
    generated_frames: Tensor  # N x T x [3, h, w]

    def to_json(self) -> dict:
        return dict(
            objects_in_bev=self.generated_frames.flaten(start_dim=2).tolist() if self.generated_frames is not None else None,
        )

    @classmethod
    def empty(cls) -> "WMAuxOutputs":
        return cls(
            generated_frames=torch.zeros((0, 0, 3, 0, 0), dtype=torch.float32),
        )


@dataclass
class WMInferenceOutput:
    trajectory: np.ndarray
    """shape: (n-future (6), 2) | predicted trajectory in the ego-frame @ 2Hz"""
    aux_outputs: WMAuxOutputs
    """aux outputs such as objects, tracks, segmentation and motion forecast"""


class WMRunner:
    def __init__(self, config_path: str, checkpoint_path: str, device: torch.device):
        with open(config_path, 'r') as file:
            training_logged_config = yaml.safe_load(file)
        self.inference_config = OmegaConf.create(training_logged_config)

        with open(self.inference_config.model_config_path, 'r') as file:
            model_config = yaml.safe_load(file)
        model_config = model_config.model
        model_config.mup_base_shapes = self.inference_config.mup_base_shapes_path

        wm_checkpoint_path = self.inference_config.world_model_checkpoint_path
        tokenizer_checkpoint_path = self.inference_config.tokenizer_checkpoint_path

        network, sequence_adapter, action_tokenizer = load_model(
            wm_checkpoint_path, model_config, device=device,
        )

        sampler = instantiate_samplers(self.inference_config.samplers)[0]

        self.model = WorldModelInference(
            network,
            sequence_adapter,
            action_tokenizer,
            sampler=sampler,
            return_logits=False,
            max_rolling_context_frames=network.nb_timesteps - 1,
        )

        self.tokenizer = VQ_16()
        keys = self.tokenizer.load_state_dict(torch.load(tokenizer_checkpoint_path, map_location='cpu'), strict=False)
        assert keys.missing_keys == [], f"Missing keys: {keys.missing_keys}"
        for k in keys.unexpected_keys:
            assert any([x in k for x in ["distillation", "temporal"]]), f"Unexpected keys: {keys.unexpected_keys}"

        self.tokenizer.eval()
        self.tokenizer.requires_grad_(False)
        self.tokenizer.to(device)

        self.device = device
        self.preproc_pipeline = instantiate(self.config.data.transform)
        self.reset()

    def reset(self):
        # making a new scene token for each new scene. these are used in the model.
        self.scene_token = str(uuid.uuid4())
        self.prev_frame_info = {
            "scene_token": None,
            "prev_frames": None,
        }

    @torch.no_grad()
    def forward_inference(self, input: WMInferenceInput) -> WMInferenceOutput:
        """Run inference without all the preprocessed dataset stuff."""
        # run it through the inference pipeline (which is same as eval pipeline except not loading annotations)
        preproc_output = torch.stack([self.preproc_pipeline(x) for x in input.imgs], dim=0)

        # first frame
        if self.prev_frame_info["scene_token"] is None:
            self.prev_frame_info["scene_token"] = self.scene_token
            self.prev_frame_info["prev_frames"] = preproc_output
        else:
            # append the current frame to the previous frames
            self.prev_frame_info["prev_frames"] = torch.cat(
                [self.prev_frame_info["prev_frames"], preproc_output], dim=0,
            )

        preproc_output = self.prev_frame_info["prev_frames"].to(self.device)
        T, CAM, *_ = preproc_output.shape
        preproc_output = rearrange(preproc_output, 'T CAM c h w -> (T CAM) c h w')

        quant, _, (_, _, tokens) = self.tokenizer(preproc_output)
        *_, h, w = quant.shape
        visual_tokens = rearrange(tokens, '(T CAM h w) -> 1 T CAM h w', T=T, CAM=CAM, h=h, w=w)

        action_tokens = self.action_tokenizer(visual_tokens=visual_tokens)
        sequence_data = self.sequence_adapter(visual_tokens, action_tokens)
        input_data = {
            'token_sequence': sequence_data['token_sequence'],
            'spatial_positions': sequence_data['spatial_positions'],
            'temporal_positions': sequence_data['temporal_positions'],
        }

        outs_planning = self.model.predict_step(input_data)
        aux_outputs = WMAuxOutputs.empty()
        return WMInferenceOutput(
            trajectory=_format_trajs(outs_planning["sdc_traj"])[0].cpu().numpy(),
            aux_outputs=aux_outputs,
        )


def _format_trajs(trajs: torch.Tensor) -> torch.Tensor:
    """
    Transform the trajector from the WM to the format expected by the server.
    dummy function for now
    """
    return trajs


def _get_sample_input(nusc, sample) -> WMInferenceInput:
    timestamp = sample["timestamp"]

    # get cameras
    camera_tokens = [sample["data"][camera_type] for camera_type in NUSCENES_CAM_ORDER]
    # get the image filepaths
    image_filepaths = [
        nusc.get_sample_data(cam_token)[0] for cam_token in camera_tokens
    ]

    # load the images in rgb hwc format
    images = []
    for filepath in image_filepaths:
        img = Image.open(filepath)
        images.append(img)
    images = np.array(images)

    return WMInferenceInput(
        imgs=images,
        timestamp=timestamp,
        command=0,  # right
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = WMRunner(
        config_path="/UniAD/projects/configs/stage2_e2e/inference_e2e.py",
        checkpoint_path="/UniAD/ckpts/uniad_base_e2e.pth",
        device=torch.device(device),
    )

    # only load this for testing
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    import matplotlib.pyplot as plt

    # load the first surround-cam in nusc mini
    nusc = NuScenes(version="v1.0-mini", dataroot="./data/nuscenes")
    nusc_can = NuScenesCanBus(dataroot="./data/nuscenes")
    scene_name = "scene-0103"
    scene = [s for s in nusc.scene if s["name"] == scene_name][0]
    # get the first sample in the scene
    sample = nusc.get("sample", scene["first_sample_token"])

    for i in range(60):
        inference_input = _get_sample_input(nusc, nusc_can, scene_name, sample)
        if i > 4:
            inference_input.command = 2  # straight
        plan = runner.forward_inference(inference_input)
        # plot in bev
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(inference_input.imgs[0])
        ax[0].axis("off")

        ax[1].plot(plan.trajectory[:, 0], plan.trajectory[:, 1], "r-*")
        ax[1].set_aspect("equal")
        ax[1].set_xlabel("x (m)")
        ax[1].set_ylabel("y (m)")

        # save fig
        fig.savefig(f"{scene_name}_{str(i).zfill(3)}_{sample['timestamp']}.png")
        plt.close(fig)
        if sample["next"] == "":
            break
        sample = nusc.get("sample", sample["next"])
