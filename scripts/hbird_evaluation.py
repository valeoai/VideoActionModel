"""
example usage:
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/video_action_model

srun -A ycy@h100 -C h100 --pty \
--nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:1 --hint=nomultithread \
--qos=qos_gpu_h100-dev --time=00:40:00 bash

python scripts/hbird_eval.py \
--gpt_checkpoint_path xxx \
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
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from vam.evaluation.datasets import CityscapesDataset, KITTIDataset
from vam.evaluation.hbird import hbird_evaluation
from vam.utils import expand_path, read_eval_config
from vam.video_pretraining import MupGPT2, load_pretrained_gpt

Config = Dict[str, Any]


def get_cityscapes(config: Config) -> Tuple[CityscapesDataset, ...]:
    return (
        CityscapesDataset(
            root=config["cityscapes"]["root"],
            split="train",
            pseudo_depth=config["cityscapes"]["pseudo_depth"],
        ),
        CityscapesDataset(
            root=config["cityscapes"]["root"],
            split="val",
            pseudo_depth=config["cityscapes"]["pseudo_depth"],
        ),
    )


def get_kitti(config: Config) -> Tuple[KITTIDataset, ...]:
    return (
        KITTIDataset(
            root=config["kitti"]["root"],
            split="train",
            window_size=1,
            pseudo_depth=config["kitti"]["pseudo_depth"],
        ),
        KITTIDataset(
            root=config["kitti"]["root"],
            split="val",
            window_size=1,
            pseudo_depth=config["kitti"]["pseudo_depth"],
        ),
    )


def get_kitti_video(config: Config) -> Tuple[KITTIDataset, ...]:
    return (
        KITTIDataset(
            root=config["kitti_video"]["root"],
            split="train",
            window_size=8,
            frame_stride=5,
            eval_on_last_frame=True,
            pseudo_depth=config["kitti_video"]["pseudo_depth"],
        ),
        KITTIDataset(
            root=config["kitti_video"]["root"],
            split="val",
            window_size=8,
            frame_stride=5,
            eval_on_last_frame=True,
            pseudo_depth=config["kitti_video"]["pseudo_depth"],
        ),
    )


def evaluate_datasets(
    gpt: MupGPT2,
    layer_idx: int,
    image_tokenizer: nn.Module,
    datasets: Dict[str, Tuple[CityscapesDataset | KITTIDataset, ...]],
    task: str = "segmentation",
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
        x = gpt.get_intermediate_layers(x, layer_idx)
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
            evaluation_task=task,
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
    parser.add_argument("--config", type=read_eval_config, default=read_eval_config("configs/paths/eval_paths_jeanzay.yaml"))
    parser.add_argument("--layer_idx", type=int, default=12)
    parser.add_argument("--outfile", type=expand_path, required=True)
    parser.add_argument("--memory_size", type=str, default="x10")
    parser.add_argument("--task", type=str, default="segmentation", choices=["segmentation", "depth"])
    parser.add_argument("--num_neighbour", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32", "fp16"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    all_datasets = {
        "cityscapes": get_cityscapes(args.config),
        "kitti": get_kitti(args.config),
        "kitti_video": get_kitti_video(args.config),
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

    gpt = load_pretrained_gpt(args.gpt_checkpoint_path, tempdir=os.environ.get("JOBSCRATCH", "/tmp"))
    tokenizer = torch.jit.load(expand_path(args.config["tokenizer_jit_path"])).to("cuda")
    metrics = evaluate_datasets(
        gpt,
        args.layer_idx,
        tokenizer,
        all_datasets,
        task=args.task,
        memory_size=args.memory_size,
        num_neighbour=args.num_neighbour,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dtype=args.dtype,
        world_size=world_size,
    )
    metrics["gpt_checkpoint_path"] = args.gpt_checkpoint_path
    metrics["tokenizer_jit_path"] = args.config["tokenizer_jit_path"]
    print(metrics)

    if rank == 0:
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
        with open(args.outfile, "w") as f:
            json.dump(metrics, f)
