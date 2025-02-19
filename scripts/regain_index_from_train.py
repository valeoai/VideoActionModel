"""
Example usage:

python scripts/regain_index_from_train.py \
--ckpt /path/to/ckpt.pt \
--outdir tmp \
--name checkpoint_90_pretraining
"""

import argparse
import json
import os
from typing import Any, Dict

import torch
import torch.distributed
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from vam.datalib import OpenDVTokensDataset, StatefulDataLoader
from vam.utils import expand_path

StateDict = Dict[str, Any]


def main(name: str, outdir: str, train_dataset: OpenDVTokensDataset, rank: int, hp: dict) -> None:
    ckpt = hp["loops"]["fit_loop"]["state_dict"]["combined_loader"][0]
    world_size = len(ckpt)

    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=hp["OpenDVTokensDataModule"]["batch_size"],
        shuffle=False,
        num_workers=ckpt[rank]["_snapshot"]["_main_snapshot"]["_num_workers"],
        pin_memory=True,
        sampler=DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            drop_last=True,
            seed=84924,  # Find a way to get that ---> add to the checkpoint
        ),
    )

    train_dataloader.load_state_dict(ckpt[rank])

    if (txt_path := f"{outdir}/indexes_{name}_{rank}.json") and os.path.exists(txt_path):
        os.remove(txt_path)

    all_indexes = []
    for batch in tqdm(train_dataloader, f"Aggregating {rank}", position=1, leave=False):
        indexes = batch["window_idx"].view(-1).tolist()
        all_indexes.extend(indexes)

    with open(txt_path, "w") as f:
        json.dump(all_indexes, f)


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=expand_path, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--outdir", type=expand_path, default="tmp")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

if os.path.isdir(args.ckpt):
    # This is a deepspeed checkpoint
    import tempfile

    from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

    with tempfile.TemporaryDirectory(dir=os.environ.get("JOBSCRATCH", "/tmp")) as tmpdirname:
        hp = convert_zero_checkpoint_to_fp32_state_dict(
            args.ckpt,
            os.path.join(tmpdirname, "fused_ckpt.pt"),
        )
else:
    hp = torch.load(args.ckpt, map_location="cpu")

ckpt = hp["loops"]["fit_loop"]["state_dict"]["combined_loader"][0]

world_size = len(ckpt)

# we should be able to get paths from the checkpoint
data_root_dir = expand_path(hp["OpenDVTokensDataModule"]["data_root_dir"])
with open(expand_path(hp["OpenDVTokensDataModule"]["video_list_path"]), "r") as f:
    video_list = json.load(f)
video_list = [os.path.join(data_root_dir, video) for video in video_list]

# Create datasets
train_dataset = OpenDVTokensDataset(
    data_root_dir,
    video_list,
    hp["OpenDVTokensDataModule"]["sequence_length"],
    hp["OpenDVTokensDataModule"]["subsampling_factor"],
)
train_dataset._idx_only = True

for rank in tqdm(range(world_size), "Creating indexes", position=0):
    main(args.name, args.outdir, train_dataset, rank, hp)

all_indexes = []
for rank in range(world_size):
    with open(f"{args.outdir}/indexes_{args.name}_{rank}.json", "r") as f:
        all_indexes.extend(json.load(f))

with open(f"{args.outdir}/indexes_{args.name}.json", "w") as f:
    json.dump(all_indexes, f)
