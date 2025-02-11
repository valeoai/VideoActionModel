"""
example usage:

srun -A ycy@h100 -C h100 --pty \
--nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:1 --hint=nomultithread \
--qos=qos_gpu_h100-dev --time=00:15:00 bash

python scripts/evaluate_ego_trajectory.py \
--vam_checkpoint_path ~/iveco/scratch_iveco/VAM_JZGC4/checkpoints/VAM/width_768_pretrained_139k.pt \
--outdir ./tmp/ego_eval_width_768_pretrained_139k \
--batch_size 64 \
--num_workers 16

srun -A ycy@h100 -C h100 --pty \
--nodes=2 --ntasks-per-node=4 --cpus-per-task=24 --gres=gpu:4 --hint=nomultithread \
--qos=qos_gpu_h100-dev --time=00:30:00 \
python scripts/evaluate_ego_trajectory.py \
--vam_checkpoint_path xxx \
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

from vam.action_expert import VideoActionModelInference, load_inference_VAM
from vam.datalib import EgoTrajectoryDataset
from vam.evaluation import min_ade
from vam.utils import boolean_flag, expand_path, torch_dtype

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
    vam: VideoActionModelInference,
    loader: DataLoader,
    name: str,
    outdir: str,
    num_sampled_trajectories: int = 10,
    rank: int = 0,
    world_size: int = 1,
    store_trajectories: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> float:
    _, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("white")
    # Initialize min/max for plot boundaries and yaw rates
    x_min, y_min = float("inf"), float("inf")
    x_max, y_max = float("-inf"), float("-inf")

    stored_trajectories = {}
    total_loss, total_samples = torch.tensor(0.0).cuda(), torch.tensor(0).cuda()
    iterator = tqdm(loader, "Evaluating", disable=rank != 0)
    for batch in iterator:
        sampled_trajectory = []
        visual_tokens = batch["visual_tokens"].to("cuda", non_blocking=True)
        commands = batch["high_level_command"].to("cuda", non_blocking=True)[:, -1:]
        for _ in range(num_sampled_trajectories):
            with torch.amp.autocast("cuda", dtype=dtype):
                trajectory = vam(visual_tokens, commands, dtype)
            sampled_trajectory.append(trajectory)

        sampled_trajectory = torch.cat(sampled_trajectory, dim=1)

        if store_trajectories:
            for idx, window_idx in enumerate(batch["window_idx"]):
                stored_trajectories[window_idx.item()] = sampled_trajectory[idx].cpu().tolist()

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

    if store_trajectories:

        if world_size > 1:
            torch.distributed.barrier()
            all_stored_trajectories = [None] * world_size
            torch.distributed.all_gather_object(all_stored_trajectories, stored_trajectories)

        if rank == 0:
            combine_trajectories = {}
            for stored_trajectories in all_stored_trajectories:
                combine_trajectories.update(stored_trajectories)
            with open(os.path.join(outdir, f"{name}_trajectories.json"), "w") as f:
                json.dump(combine_trajectories, f)

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
    vam: VideoActionModelInference,
    datasets: Dict[str, Dataset],
    outdir: str,
    num_sampled_trajectories: int = 10,
    batch_size: int = 4,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
    store_trajectories: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    def _get_loader(ds: Dataset) -> DataLoader:
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(ds, shuffle=False)
        return DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers, sampler=sampler)

    metrics = {}
    for name, ds in datasets.items():
        metrics[name] = evaluate_loader(
            vam,
            _get_loader(ds),
            name,
            outdir,
            num_sampled_trajectories=num_sampled_trajectories,
            rank=rank,
            world_size=world_size,
            store_trajectories=store_trajectories and name == "nuscenes",
            dtype=dtype,
        )
        print(metrics)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vam_checkpoint_path", type=expand_path, required=True)
    parser.add_argument("--outdir", type=expand_path, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_sampled_trajectories", type=int, default=10)
    parser.add_argument("--store_trajectories", type=boolean_flag, default=False)
    parser.add_argument("--dtype", type=torch_dtype, default=torch.bfloat16)
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

    vam = load_inference_VAM(args.vam_checkpoint_path, tempdir=os.environ.get("JOBSCRATCH", "/tmp"))

    metrics = evaluate_datasets(
        vam,
        dts,
        outdir=args.outdir,
        num_sampled_trajectories=args.num_sampled_trajectories,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rank=rank,
        world_size=world_size,
        store_trajectories=args.store_trajectories,
        dtype=args.dtype,
    )
    metrics["vam_checkpoint_path"] = args.vam_checkpoint_path
    metrics["outdir"] = args.outdir
    print(metrics)

    if rank == 0:
        with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
            json.dump(metrics, f)
