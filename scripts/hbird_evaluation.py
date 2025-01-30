"""
example usage:
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/world_model

srun -A ycy@h100 -C h100 --pty \
--nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:1 --hint=nomultithread \
--qos=qos_gpu_h100-dev --time=00:40:00 bash

python scripts/hbird_eval.py \
--gpt_checkpoint_path xxx \
--tokenizer_jit_path $ycy_ALL_CCFRWORK/llamagen_jit_models/VQ_ds16_16384_llamagen_encoder.jit \
--outfile ./tmp/test_fn.json \
--num_workers 16 \
--batch_size 16

srun -A ycy@h100 -C h100 --pty \
--nodes=8 --ntasks-per-node=4 --cpus-per-task=24 --gres=gpu:4 --hint=nomultithread \
--qos=qos_gpu_h100-gc --time=10:00:00 \
python scripts/nxt_evaluation.py \
--gpt_checkpoint_path xxxx \
--outfile ./tmp/test_fn.json \
--num_workers 24 \
--batch_size 128

srun -A ycy@h100 -C h100 --pty \
--nodes=1 --ntasks-per-node=4 --cpus-per-task=16 --gres=gpu:4 --hint=nomultithread \
--qos=qos_gpu_h100-dev --time=00:30:00 \
python scripts/nxt_evaluation.py \
--gpt_checkpoint_path xxx \
--outfile ./tmp/test_fn.json \
--num_workers 16 \
--batch_size 96
"""

import argparse
import json
import os
import subprocess
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from vam.evaluation.datasets import CityscapesDataset, KITTIDataset
from vam.evaluation.hbird import hbird_evaluation
from vam.utils import expand_path
from vam.video_pretraining import MupGPT2, load_pretrained_gpt


def get_cityscapes() -> Tuple[CityscapesDataset, ...]:
    return (
        CityscapesDataset(root="$ycy_ALL_CCFRSCRATCH/cityscapes", split="train"),
        CityscapesDataset(root="$ycy_ALL_CCFRSCRATCH/cityscapes", split="val"),
    )


def get_kitti() -> Tuple[KITTIDataset, ...]:
    return (
        KITTIDataset(root="$ycy_ALL_CCFRSCRATCH/KITTI_STEP", split="train", window_size=1),
        KITTIDataset(root="$ycy_ALL_CCFRSCRATCH/KITTI_STEP", split="val", window_size=1),
    )


def get_kitti_video() -> Tuple[KITTIDataset, ...]:
    return (
        KITTIDataset(
            root="$ycy_ALL_CCFRSCRATCH/KITTI_STEP", split="train", window_size=8, frame_stride=5, eval_on_last_frame=True
        ),
        KITTIDataset(
            root="$ycy_ALL_CCFRSCRATCH/KITTI_STEP", split="val", window_size=8, frame_stride=5, eval_on_last_frame=True
        ),
    )


def evaluate_datasets(
    gpt: MupGPT2,
    image_tokenizer: nn.Module,
    datasets: Dict[str, Tuple[CityscapesDataset | KITTIDataset, ...]],
    memory_size: str = "x10",
    num_neighbour: int = 30,
    batch_size: int = 4,
    batch_size_eval: Optional[int] = None,
    num_workers: int = 4,
    dtype: str = "bf16",
    world_size: int = 1,
) -> Dict[str, Dict[str, float | List[float]]]:
    def _forward_fn(x: Tensor, inference: bool) -> Tensor:
        time = 1
        if x.ndim == 5:
            # Our tokenizer does not handle video
            time = x.size(1)
            x = rearrange(x, "b t c h w -> (b t) c h w")
        x = image_tokenizer(x)
        x = rearrange(x, "(b t) h w -> b t h w", t=time)
        x = gpt.get_intermediate_layers(x, 22)
        x = rearrange(x[:, -1], "b h w d -> b (h w) d")
        return x

    def _get_hbird_score(dts: Tuple[CityscapesDataset | KITTIDataset, ...]) -> Dict[str, float | List[float]]:
        logs, _ = hbird_evaluation(
            ftr_extr_fn=_forward_fn,
            model_info={
                "patch_size": 16,
                "d_model": gpt.embedding_dim,
            },
            train_dataset=dts[0],
            val_dataset=dts[1],
            batch_size=batch_size // dts[0].get_window_size(),
            batch_size_eval=(batch_size_eval or batch_size) // dts[0].get_window_size(),
            num_workers=num_workers,
            augmentation_epoch=1,
            device="cuda",
            dtype=dtype,
            is_distributed=world_size > 1,
            return_labels=False,
            num_neighbour=num_neighbour,
            nn_params=None,
            memory_size=memory_size,  # you can set this to reduce memory size
            f_mem_p=None,
            l_mem_p=None,
        )
        return logs

    return {name: _get_hbird_score(ds) for name, ds in datasets.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_checkpoint_path", type=expand_path, required=True)
    parser.add_argument("--tokenizer_jit_path", type=expand_path, required=True)
    parser.add_argument("--outfile", type=expand_path, required=True)
    parser.add_argument("--memory_size", type=str, default="x10")
    parser.add_argument("--num_neighbour", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32", "fp16"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    all_datasets = {
        "cityscapes": get_cityscapes(),
        "kitti": get_kitti(),
        "kitti_video": get_kitti_video(),
    }

    world_size = int(os.environ["SLURM_NTASKS"])
    rank = 0
    if world_size > 1:
        dist_url = "env://"
        dist_backend = "nccl"
        rank = int(os.environ["SLURM_PROCID"])
        node_list = os.environ["SLURM_NODELIST"]
        is_distributed = world_size > 1
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput("scontrol show hostname {} | head -n1".format(node_list))
        local_rank = rank % num_gpus
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        os.environ["MASTER_ADDR"] = addr
        print(f"| distributed init (rank {rank}): {dist_url}")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank)

    gpt = load_pretrained_gpt(args.gpt_checkpoint_path, tempdir=os.environ["JOBSCRATCH"])
    tokenizer = torch.jit.load(args.tokenizer_jit_path).to("cuda")
    metrics = evaluate_datasets(
        gpt,
        tokenizer,
        all_datasets,
        memory_size=args.memory_size,
        num_neighbour=args.num_neighbour,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dtype=args.dtype,
        world_size=world_size,
    )
    metrics["gpt_checkpoint_path"] = args.gpt_checkpoint_path
    metrics["tokenizer_jit_path"] = args.tokenizer_jit_path
    print(metrics)

    if rank == 0:
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
        with open(args.outfile, "w") as f:
            json.dump(metrics, f)
