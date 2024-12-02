import uuid
import yaml
from typing import List
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from omegaconf import OmegaConf
from world_model.utils.trajectory_inference import load_trajectory_model, WorldModelTrajectoryInference
from world_model.dataloader.components.transforms import CropAndResizeTransform


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
class UniADAuxOutputs:
    objects_in_bev: np.ndarray  # N x [x, y, width, height, yaw]
    object_classes: List[str]  # (N, )
    object_scores: np.ndarray  # (N, )
    object_ids: np.ndarray  # (N, )
    future_trajs: np.ndarray  # (N, 6 modes, 12 timesteps, [x y])

    def to_json(self) -> dict:
        n_objects = len(self.object_classes)
        return dict(
            objects_in_bev=self.objects_in_bev.tolist() if n_objects > 0 else None,
            object_classes=self.object_classes if n_objects > 0 else None,
            object_scores=self.object_scores.tolist() if n_objects > 0 else None,
            object_ids=self.object_ids.tolist() if n_objects > 0 else None,
            future_trajs=self.future_trajs.tolist() if n_objects > 0 else None,
        )

    @classmethod
    def empty(cls) -> "UniADAuxOutputs":
        return cls(
            objects_in_bev=np.zeros((0, 5)),
            object_classes=[],
            object_scores=np.zeros((0, 1)),
            object_ids=np.zeros((0, 1)),
            future_trajs=np.zeros((0, 6, 12, 2)),
        )


@dataclass
class WMInferenceOutput:
    trajectory: np.ndarray
    """shape: (n-future (6), 2) | predicted trajectory in the ego-frame @ 2Hz"""
    aux_outputs: UniADAuxOutputs


class WMRunner:

    def __init__(self, config_path: str, checkpoint_path: str, device: torch.device):
        with open(config_path, 'r') as file:
            inference_config = yaml.safe_load(file)
        self.inference_config = OmegaConf.create(inference_config)

        image_tokenizer_path = self.inference_config.image_tokenizer_path
        self.image_tokenizer = torch.jit.load(image_tokenizer_path)
        self.image_tokenizer.to(device)

        trajectory_tokenizer_path = self.inference_config.trajectory_tokenizer_path
        self.trajectory_tokenizer = torch.jit.load(trajectory_tokenizer_path)
        self.trajectory_tokenizer.to(device)

        network, sequence_adapter = load_trajectory_model(
            ckpt_file_path=self.inference_config.wm_ckpt_path,
            config_file_path=self.inference_config.wm_config_path,
            mup_base_shapes_path=self.inference_config.mup_base_shapes_path,
            device=device
        )
        self.world_model = WorldModelTrajectoryInference(network=network, sequence_adapter=sequence_adapter)

        self.device = device
        self.top_crop = self.inference_config.top_crop
        self.scale_factor = self.inference_config.scale_factor
        self.preproc_pipeline = CropAndResizeTransform(self.top_crop, self.scale_factor)
        self.reset()

    def reset(self):
        # making a new scene token for each new scene. these are used in the model.
        self.scene_token = str(uuid.uuid4())
        self.prev_frame_info = {
            "scene_token": None,
            "prev_frames": None,
            "prev_actions": None,
            "prev_commands": None,
        }

    @torch.no_grad()
    def forward_inference(self, input: WMInferenceInput) -> WMInferenceOutput:
        """Run inference without all the preprocessed dataset stuff."""
        # For now we only do single cam
        preproc_output = self.preproc_pipeline(input.imgs[0])
        # run it through the inference pipeline (which is same as eval pipeline except not loading annotations)
        # preproc_output = torch.stack([self.preproc_pipeline(x) for x in input.imgs], dim=0)

        if self.prev_frame_info["scene_token"] is None:
            # first frame
            self.prev_frame_info["scene_token"] = self.scene_token
            self.prev_frame_info["prev_frames"] = preproc_output.unsqueeze(0)
            self.prev_frame_info["prev_command"] = [input.command]
        else:
            # append the current frame to the previous frames
            self.prev_frame_info["prev_frames"] = torch.cat(
                [self.prev_frame_info["prev_frames"], preproc_output.unsqueeze(0)], dim=0,
            )
            self.prev_frame_info["prev_command"].append(input.command)

        # This should (T, c, h, w)
        # So here the temporal frames play the role of batch size
        preproc_output = self.prev_frame_info["prev_frames"].to(self.device)
        # Here we unsqueeze because the input of the world model is (B, T, h, w)
        visual_tokens = self.image_tokenizer(preproc_output).unsqueeze(0)

        # Get the command tokens
        command_tokens = torch.tensor(self.prev_frame_info["prev_command"]).unsqueeze(0).to(self.device)

        # get the trajectory tokens
        trajectory_tokens = None
        if self.prev_frame_info["prev_actions"] is not None:
            trajectory_tokens = self.prev_frame_info["prev_actions"].unsqueeze(0)

        # Get the predicted trajectory tokens
        predicted_trajectory_tokens = self.world_model(visual_tokens, command_tokens, trajectory_tokens)

        if self.prev_frame_info["prev_actions"] is None:
            self.prev_frame_info["prev_actions"] = predicted_trajectory_tokens
        else:
            self.prev_frame_info["prev_actions"] = torch.cat(
                [self.prev_frame_info["prev_actions"], predicted_trajectory_tokens], dim=0
            )

        # decode the trajectory tokens
        trajectory = self.trajectory_tokenizer(predicted_trajectory_tokens)

        return WMInferenceOutput(
            trajectory=_format_trajs(trajectory),
            aux_outputs=UniADAuxOutputs.empty(),
        )


def _format_trajs(trajs: torch.Tensor) -> torch.Tensor:
    """
    Transform the trajector from the WM to the format expected by the server.
    dummy function for now
    """
    return trajs[0, 1:].cpu().numpy()


if __name__ == "__main__":
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = WMRunner(
        config_path="configs/inference_ncap.yaml",
        checkpoint_path=None,
        device=torch.device(device),
    )

    # only load this for testing
    from nuscenes.nuscenes import NuScenes

    # load the first surround-cam in nusc mini
    # nusc = NuScenes(version="v1.0-mini", dataroot="/datasets_local/nuscenes")
    nusc = NuScenes(version="v1.0-mini", dataroot="/model/data/nuscenes")
    scene_name = "scene-0103"
    scene = [s for s in nusc.scene if s["name"] == scene_name][0]
    # get the first sample in the scene
    sample = nusc.get("sample", scene["first_sample_token"])

    inference_input = _get_sample_input(nusc, sample)
    plan = runner.forward_inference(inference_input)
    assert plan.trajectory.shape == (6, 2)
    plan = runner.forward_inference(inference_input)
    assert plan.trajectory.shape == (6, 2)
    plan = runner.forward_inference(inference_input)
    assert plan.trajectory.shape == (6, 2)
    print("All tests passed!")
