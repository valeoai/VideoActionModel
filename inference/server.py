import argparse
import io
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Base64Bytes

from inference.runner import (
    NUSCENES_CAM_ORDER,
    WMInferenceInput,
    WMRunner,
)


app = FastAPI()


class InferenceInputs(BaseModel):
    """Input data for inference."""

    images: Dict[str, Base64Bytes]
    """Camera images in PNG format. The keys are the camera names."""
    timestamp: int  # in microseconds
    """Timestamp of the current frame in microseconds."""
    command: Literal[0, 1, 2]
    """Command of the current frame."""


class InferenceAuxOutputs(BaseModel):
    generated_frames: Optional[List[List[List[float]]]] = None  # N x T x [width, height, 3]


class InferenceOutputs(BaseModel):
    """Output / result from running the model."""

    trajectory: List[List[float]]
    """Predicted trajectory in the ego frame. A list of (x, y) points in BEV."""
    aux_outputs: InferenceAuxOutputs
    """Auxiliary outputs."""


@app.get("/alive")
async def alive() -> bool:
    return True


@app.post("/infer")
async def infer(data: InferenceInputs) -> InferenceOutputs:
    wm_input = _build_wm_input(data)
    wm_output = wm_runner.forward_inference(wm_input)
    return InferenceOutputs(
        trajectory=wm_output.trajectory.tolist(),
        aux_outputs=(InferenceAuxOutputs(**wm_output.aux_outputs.to_json())),
    )


@app.post("/reset")
async def reset_runner() -> bool:
    wm_runner.reset()
    return True


def _build_wm_input(data: InferenceInputs) -> WMInferenceInput:
    imgs = _bytestr_to_numpy([data.images[c] for c in NUSCENES_CAM_ORDER])
    return WMInferenceInput(
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
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    device = torch.device(args.device)

    wm_runner = WMRunner(args.config_path, args.checkpoint_path, device)

    uvicorn.run(app, host=args.host, port=args.port)
