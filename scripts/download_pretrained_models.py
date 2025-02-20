"""
example usage:

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/video_action_model

srun -A ycy@h100 --pty \
--cpus-per-task=3 --hint=nomultithread \
--partition=prepost --time=00:40:00 \
python scripts/download_pretrained_models.py
"""

import os

import torch

from vam.evaluation.quality import MultiInceptionMetrics

os.environ["TMPDIR"] = os.environ.get("JOBSCRATCH", "/tmp")

try:
    torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
    print("Successfull download of Dinov1-B model")
except Exception as e:
    print(e)

try:
    torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    print("Successfull download of Dinov2-B model")
except Exception as e:
    print(e)

try:
    torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
    print("Successfull download of Dinov2-L model")
except Exception as e:
    print(e)

try:
    torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14_reg")
    print("Successfull download of Dinov2-g model")
except Exception as e:
    print(e)

try:
    _ = MultiInceptionMetrics("cpu", model="dinov2")
    print("Successfull download of Dinov2 model")
except Exception as e:
    print(e)

try:
    _ = MultiInceptionMetrics("cpu", model="i3d")
    print("Successfull download of I3D model")
except Exception as e:
    print(e)
