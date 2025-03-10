import argparse
import io
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import uvicorn  # type: ignore
from fastapi import FastAPI  # type: ignore
from pydantic import Base64Bytes, BaseModel  # type: ignore

from inference.runner import NUSCENES_CAM_ORDER, VAMInferenceInput, VAMRunner

app = FastAPI()


class Calibration(BaseModel):
    """Calibration data."""

    camera2image: Dict[str, List[List[float]]]
    """Camera intrinsics. The keys are the camera names."""
    camera2ego: Dict[str, List[List[float]]]
    """Camera extrinsics. The keys are the camera names."""
    lidar2ego: List[List[float]]
    """Lidar extrinsics."""


class InferenceInputs(BaseModel):
    """Input data for inference."""

    images: Dict[str, Base64Bytes]
    """Camera images in PNG format. The keys are the camera names."""
    ego2world: List[List[float]]
    """Ego pose in the world frame."""
    canbus: List[float]
    """CAN bus signals."""
    timestamp: int  # in microseconds
    """Timestamp of the current frame in microseconds."""
    command: Literal[0, 1, 2]
    """Command of the current frame."""
    calibration: Calibration
    """Calibration data.""" ""


class InferenceAuxOutputs(BaseModel):
    objects_in_bev: Optional[List[List[float]]] = None  # N x [x, y, width, height, yaw]
    object_classes: Optional[List[str]] = None  # (N, )
    object_scores: Optional[List[float]] = None  # (N, )
    object_ids: Optional[List[int]] = None  # (N, )
    future_trajs: Optional[List[List[List[List[float]]]]] = None  # N x M x T x [x, y]


class InferenceOutputs(BaseModel):
    """Output / result from running the model."""

    trajectory: List[List[float]]
    """Predicted trajectory in the ego frame. A list of (x, y) points in BEV."""
    aux_outputs: InferenceAuxOutputs


@app.get("/alive")
async def alive() -> bool:
    return True


@app.post("/infer")
async def infer(data: InferenceInputs) -> InferenceOutputs:
    vam_input = _build_vam_input(data)
    vam_output = vam_runner.forward_inference(vam_input)
    return InferenceOutputs(
        trajectory=vam_output.trajectory.tolist(),
        aux_outputs=(InferenceAuxOutputs(**vam_output.aux_outputs.to_json())),
    )


@app.post("/reset")
async def reset_runner() -> bool:
    vam_runner.reset()
    return True


def _build_vam_input(data: InferenceInputs) -> VAMInferenceInput:
    imgs = _bytestr_to_numpy([data.images[c] for c in NUSCENES_CAM_ORDER])
    return VAMInferenceInput(
        imgs=imgs,
        timestamp=data.timestamp / 1e6,  # convert to seconds
        command=data.command,
    )


def _bytestr_to_numpy(pngs: List[bytes]) -> np.ndarray:
    """Convert a list of png bytes to a numpy array of shape (n, h, w, c)."""
    imgs = []
    for png in pngs:
        # using torch load as we use torch save on rendering node
        img = torch.load(io.BytesIO(png)).clone()
        # we convert it to PIL image to have consistency with transform pipeline
        imgs.append(img.numpy())

    return np.stack(imgs, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32", "fp16"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    device = torch.device(args.device)

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[args.dtype]

    vam_runner = VAMRunner(args.config_path, args.checkpoint_path, device, dtype)

    uvicorn.run(app, host=args.host, port=args.port)
