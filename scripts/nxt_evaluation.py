"""
example usage:
python scripts/nxt_evaluation.py \
--gpt_checkpoint_path xxxx \
--outfile ./tmp/test_fn.json \
--num_workers 10

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
import pickle
import subprocess
from typing import Dict

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from world_model.datalib import EgoTrajectoryDataset, OpenDVTokensDataset
from world_model.gpt2 import MupGPT2, load_pretrained_gpt
from world_model.gpt2.prepare_token_sequence import prepare_AR_token_sequences
from world_model.utils import expand_path


def get_opendv() -> OpenDVTokensDataset:
    with open(expand_path("$fzh_ALL_CCFRSCRATCH/OpenDV_processed/val.json")) as f:
        val_videos = json.load(f)

    return OpenDVTokensDataset(
        data_root_dir="$fzh_ALL_CCFRSCRATCH/OpenDV_processed/flat_tokens",
        video_list=val_videos,
        sequence_length=8,
        subsampling_factor=5,
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
def evaluate_loader(gpt: MupGPT2, loader: DataLoader, name: str = "", world_size: int = 1) -> float:
    total_loss, total_samples = torch.tensor(0.0).cuda(), torch.tensor(0).cuda()
    iterator = tqdm(loader, f"Evaluating {name}")
    for batch in iterator:
        visual_tokens = batch["visual_tokens"].to("cuda", non_blocking=True)
        input_data, target_data = prepare_AR_token_sequences(visual_tokens)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits_sequence = gpt(**input_data)
        logits_sequence = rearrange(logits_sequence, "b ... d -> b d ...")
        loss = F.cross_entropy(logits_sequence, target_data["token_sequence"], reduction="none")
        total_loss += loss.sum()
        total_samples += loss.numel()

        iterator.set_postfix(loss=(total_loss / total_samples).item())

    if world_size > 1:
        torch.distributed.all_reduce(total_loss)
        torch.distributed.all_reduce(total_samples)

    print(f"Evaluated {name} with loss: {total_loss / total_samples}")
    return (total_loss / total_samples).item()


def evaluate_datasets(
    gpt: MupGPT2,
    datasets: Dict[str, Dataset],
    batch_size: int = 4,
    num_workers: int = 4,
    world_size: int = 1,
) -> Dict[str, float]:
    def _get_loader(ds: Dataset) -> DataLoader:
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(ds, shuffle=False)
        return DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers, sampler=sampler)

    return {name: evaluate_loader(gpt, _get_loader(ds), name, world_size=world_size) for name, ds in datasets.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_checkpoint_path", type=expand_path, required=True)
    parser.add_argument("--outfile", type=expand_path, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    dts = {
        "opendv": get_opendv(),
        "nuplan": get_nuplan(),
        "nuscenes": get_nuscenes(),
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
    metrics = evaluate_datasets(gpt, dts, batch_size=args.batch_size, num_workers=args.num_workers, world_size=world_size)
    metrics["gpt_checkpoint_path"] = args.gpt_checkpoint_path
    print(metrics)

    if rank == 0:
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
        with open(args.outfile, "w") as f:
            json.dump(metrics, f)
