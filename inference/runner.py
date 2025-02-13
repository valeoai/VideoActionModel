import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch import Tensor

from vam.action_expert import VideoActionModelInference, load_inference_VAM
from vam.datalib.transforms import NeuroNCAPTransform

NUSCENES_CAM_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


@dataclass
class VAMInferenceInput:
    imgs: List[np.ndarray]
    """shape: (n-cams (1), h (900), w (1600) c (3)) | images without any preprocessing. should be in RGB order as uint8"""
    timestamp: float
    """timestamp of the current frame in seconds"""
    command: int
    """0: right, 1: left, 2: straight"""


@dataclass
class VAMAuxOutputs:
    objects_in_bev: np.ndarray  # N x [x, y, width, height, yaw]
    object_classes: List[str]  # (N, )
    object_scores: np.ndarray  # (N, )
    object_ids: np.ndarray  # (N, )
    future_trajs: np.ndarray  # (N, 6 modes, 12 timesteps, [x y])

    def to_json(self) -> dict:
        n_objects = len(self.object_classes)
        return {
            "objects_in_bev": self.objects_in_bev.tolist() if n_objects > 0 else None,
            "object_classes": self.object_classes if n_objects > 0 else None,
            "object_scores": self.object_scores.tolist() if n_objects > 0 else None,
            "object_ids": self.object_ids.tolist() if n_objects > 0 else None,
            "future_trajs": self.future_trajs.tolist() if n_objects > 0 else None,
        }

    @classmethod
    def empty(cls: "VAMAuxOutputs") -> "VAMAuxOutputs":
        return cls(
            objects_in_bev=np.zeros((0, 5)),
            object_classes=[],
            object_scores=np.zeros((0, 1)),
            object_ids=np.zeros((0, 1)),
            future_trajs=np.zeros((0, 6, 12, 2)),
        )


@dataclass
class VAMInferenceOutput:
    trajectory: np.ndarray
    """shape: (n-future (6), 2) | predicted trajectory in the ego-frame @ 2Hz"""
    aux_outputs: VAMAuxOutputs


class VAMRunner:

    def __init__(self, config_path: str, checkpoint_path: str, device: torch.device, dtype: torch.dtype) -> None:
        self.image_tokenizer = torch.jit.load(config_path)
        self.image_tokenizer.to(device)

        self.vam: VideoActionModelInference = load_inference_VAM(checkpoint_path, device)
        self.nb_timesteps = self.vam.context_length

        self.device = device
        self.dtype = dtype
        self.preproc_pipeline = NeuroNCAPTransform()
        self.reset()

    def reset(self) -> None:
        # making a new scene token for each new scene. these are used in the model.
        self.scene_token = str(uuid.uuid4())
        self.prev_frame_info = {
            "scene_token": None,
            "prev_frames": None,
        }

    @torch.no_grad()
    def forward_inference(self, input: VAMInferenceInput) -> VAMInferenceOutput:
        """Run inference without all the preprocessed dataset stuff."""
        # For now we only do single cam
        preproc_output = self.preproc_pipeline(input.imgs[0])
        # run it through the inference pipeline (which is same as eval pipeline except not loading annotations)
        # preproc_output = torch.stack([self.preproc_pipeline(x) for x in input.imgs], dim=0)

        if self.prev_frame_info["scene_token"] is None:
            # first frame
            self.prev_frame_info["scene_token"] = self.scene_token
            self.prev_frame_info["prev_frames"] = preproc_output.unsqueeze(0)
        else:
            # append the current frame to the previous frames
            self.prev_frame_info["prev_frames"] = torch.cat(
                [self.prev_frame_info["prev_frames"], preproc_output.unsqueeze(0)],
                dim=0,
            )[-self.nb_timesteps :]

        # This should (T, c, h, w)
        # So here the temporal frames play the role of batch size
        preproc_output = self.prev_frame_info["prev_frames"].to(self.device)
        # Here we unsqueeze because the input of the VAM is (B, T, h, w)
        visual_tokens = self.image_tokenizer(preproc_output).unsqueeze(0)

        # Get the command tokens
        command_tokens = torch.tensor([input.command]).unsqueeze(0).to(self.device)

        # Get the trajectory
        with torch.amp.autocast("cuda", dtype=self.dtype):
            trajectory = self.vam(visual_tokens, command_tokens, self.dtype)

        return VAMInferenceOutput(
            trajectory=_format_trajs(trajectory),
            aux_outputs=VAMAuxOutputs.empty(),
        )


def _format_trajs(trajs: Tensor) -> Tensor:
    """
    Transform the trajector from the video action model to the format expected by the server.
    dummy function for now
    """
    return rearrange(trajs.float().cpu().numpy(), "1 1 h a -> h a")


if __name__ == "__main__":
    # only load this for testing
    from nuscenes.nuscenes import NuScenes

    from vam.utils import expand_path

    def _get_sample_input(nusc: NuScenes, sample: Dict[str, Any]) -> VAMInferenceInput:
        timestamp = sample["timestamp"]

        # get cameras
        camera_tokens = [sample["data"][camera_type] for camera_type in NUSCENES_CAM_ORDER]
        # get the image filepaths
        image_filepaths = [nusc.get_sample_data(cam_token)[0] for cam_token in camera_tokens]

        # load the images in rgb hwc format
        images = []
        for filepath in image_filepaths:
            img = Image.open(filepath)
            images.append(img)
        images = np.array(images)

        return VAMInferenceInput(
            imgs=images,
            timestamp=timestamp,
            command=0,  # right
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = VAMRunner(
        # config_path=expand_path("~/iveco/scratch_iveco/world_model_JZGC4/jit_models/VQ_ds16_16384_llamagen.jit"),
        config_path=expand_path("$fzh_ALL_CCFRSCRATCH/neuroncap_worldmodel_ckpt/jit_models/VQ_ds16_16384_llamagen.jit"),
        checkpoint_path=expand_path(
            "$fzh_ALL_CCFRSCRATCH/output_data/vaiorbis_grid_search/"
            "Vaiorbis_Nodes6_BSperGPU16_totalBS384_attdim768_actdim192_0119_1044_1737279899/"
            "checkpoints/before_drop_epoch=000_step=0000006525.ckpt"
        ),
        device=torch.device(device),
        dtype=torch.float16,
    )

    # load the first surround-cam in nusc mini
    nusc = NuScenes(version="v1.0-mini", dataroot="/lustre/fsn1/projects/rech/ycy/commun/nuscenes_v2")
    # nusc = NuScenes(version="v1.0-mini", dataroot="/datasets_local/nuscenes")
    # nusc = NuScenes(version="v1.0-mini", dataroot="/model/data/nuscenes")
    scene_name = "scene-0103"
    scene = [s for s in nusc.scene if s["name"] == scene_name][0]
    # get the first sample in the scene
    sample = nusc.get("sample", scene["first_sample_token"])

    inference_input = _get_sample_input(nusc, sample)
    for _ in range(20):
        plan = runner.forward_inference(inference_input)
        assert plan.trajectory.shape == (6, 2), plan.trajectory.shape
    print("All tests passed!")
