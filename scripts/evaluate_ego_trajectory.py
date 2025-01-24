"""
example usage:

srun -A ycy@h100 -C h100 --pty \
--nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:1 --hint=nomultithread \
--qos=qos_gpu_h100-dev --time=00:15:00 bash

python scripts/evaluate_ego_trajectory.py \
--vai0rbis_checkpoint_path xxx \
--outdir ./tmp/ego_eval \
--batch_size 64 \
--num_workers 16

srun -A ycy@h100 -C h100 --pty \
--nodes=2 --ntasks-per-node=4 --cpus-per-task=24 --gres=gpu:4 --hint=nomultithread \
--qos=qos_gpu_h100-dev --time=00:30:00 \
python scripts/evaluate_ego_trajectory.py \
--vai0rbis_checkpoint_path xxx \
--outdir ./tmp/ego_eval_1024_77k \
--batch_size 64 \
--num_workers 24
"""

import argparse
import json
import os
import pickle
import subprocess
from typing import Dict

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from world_model.evaluation import min_ade
from world_model.gpt2 import Vai0rbisInference, load_inference_vai0rbis
from world_model.opendv import EgoTrajectoryDataset
from world_model.utils import expand_path

plt.style.use("default")
plt.rcParams.update(
    {
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }
)


def get_nuplan() -> EgoTrajectoryDataset:
    with open(expand_path("$ycy_ALL_CCFRWORK/cleaned_trajectory_pickle/nuplan_val_data_cleaned.pkl"), "rb") as f:
        pickle_data = pickle.load(f)

    return EgoTrajectoryDataset(
        pickle_data=pickle_data,
        tokens_rootdir=expand_path("$ycy_ALL_CCFRSCRATCH/nuplan_v2_tokens/tokens"),
        subsampling_factor=5,
        camera="CAM_F0",
    )


def get_nuscenes() -> EgoTrajectoryDataset:
    with open(expand_path("$ycy_ALL_CCFRWORK/cleaned_trajectory_pickle/nuscenes_val_data_cleaned.pkl"), "rb") as f:
        pickle_data = pickle.load(f)

    return EgoTrajectoryDataset(
        pickle_data=pickle_data,
        tokens_rootdir=expand_path("$ycy_ALL_CCFRSCRATCH/nuscenes_v2/tokens"),
    )


@torch.no_grad()
def evaluate_loader(
    vai0rbis: Vai0rbisInference, loader: DataLoader, name: str, outdir: str, rank: int = 0, world_size: int = 1
) -> float:
    _, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("white")
    # Initialize min/max for plot boundaries and yaw rates
    x_min, y_min = float("inf"), float("inf")
    x_max, y_max = float("-inf"), float("-inf")

    num_sampling = 10

    total_loss, total_samples = torch.tensor(0.0).cuda(), torch.tensor(0).cuda()
    iterator = tqdm(loader, "Evaluating", disable=rank != 0)
    for batch in iterator:
        sampled_trajectory = []
        visual_tokens = batch["visual_tokens"].to("cuda", non_blocking=True)
        commands = batch["high_level_command"].to("cuda", non_blocking=True)[:, -1:]
        for _ in range(num_sampling):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                trajectory = vai0rbis(visual_tokens, commands, torch.bfloat16)
            sampled_trajectory.append(trajectory)

        sampled_trajectory = torch.cat(sampled_trajectory, dim=1)
        ground_truth = batch["positions"].to("cuda", non_blocking=True)[:, -1]
        loss, idx = min_ade(sampled_trajectory, ground_truth, return_idx=True, reduction="sum")
        best_sampled_trajectory = sampled_trajectory[torch.arange(len(sampled_trajectory)), idx]
        total_loss += loss
        total_samples += len(ground_truth)
        if rank == 0:
            iterator.set_postfix(minADE=(total_loss / total_samples).item())

        if rank == 0:
            # Update plot boundaries
            x_min = min(x_min, best_sampled_trajectory[..., 0].min().item())
            x_max = max(x_max, best_sampled_trajectory[..., 0].max().item())
            y_min = min(y_min, best_sampled_trajectory[..., 1].min().item())
            y_max = max(y_max, best_sampled_trajectory[..., 1].max().item())

            for traj in best_sampled_trajectory.float().cpu():
                ax.plot(traj[:, 0], traj[:, 1], alpha=0.5, linewidth=1)

    if rank == 0:
        # Add padding to the limits
        padding = 0.05 * max(x_max - x_min, y_max - y_min)
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

        # Add labels and title
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title(f"Trajectory Plot (n={len(loader.dataset)})\nColored by Average Yaw Rate")

        # Equal aspect ratio for proper visualization
        ax.set_aspect("equal")

        # Add grid with light gray color
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")

        # Ensure tight layout
        plt.tight_layout()

        # Save plot
        save_path = os.path.join(outdir, f"{name}.png")
        print(f"Saving plot to {save_path}...")
        plt.savefig(save_path.format(name=name), dpi=300, bbox_inches="tight")

    if world_size > 1:
        torch.distributed.all_reduce(total_loss)
        torch.distributed.all_reduce(total_samples)

    print(f"Evaluated {name} with loss: {total_loss / total_samples}")
    return (total_loss / total_samples).item()


def evaluate_datasets(
    vai0rbis: Vai0rbisInference,
    datasets: Dict[str, Dataset],
    outdir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, float]:
    def _get_loader(ds: Dataset) -> DataLoader:
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(ds, shuffle=False)
        return DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers, sampler=sampler)

    return {
        name: evaluate_loader(vai0rbis, _get_loader(ds), name, outdir, rank=rank, world_size=world_size)
        for name, ds in datasets.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vai0rbis_checkpoint_path", type=expand_path, required=True)
    parser.add_argument("--outdir", type=expand_path, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    dts = {
        "nuplan": get_nuplan(),
        "nuscenes": get_nuscenes(),
    }

    os.makedirs(args.outdir, exist_ok=True)

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

    vai0rbis = load_inference_vai0rbis(args.vai0rbis_checkpoint_path, tempdir=os.environ["JOBSCRATCH"])

    metrics = evaluate_datasets(
        vai0rbis,
        dts,
        outdir=args.outdir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rank=rank,
        world_size=world_size,
    )
    metrics["vai0rbis_checkpoint_path"] = args.vai0rbis_checkpoint_path
    metrics["outdir"] = args.outdir
    print(metrics)

    if rank == 0:
        with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
            json.dump(metrics, f)
