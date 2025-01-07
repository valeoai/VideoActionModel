import json
import os
import subprocess

import torch
from torch.utils.data import DistributedSampler

from world_model.opendv.stateful_dataloader import StatefulDataLoader
from world_model.opendv.random_tokenized_sequence_opendv import RandomTokenizedSequenceOpenDVDataset


dist_url = "env://"
dist_backend = "nccl"
os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
os.environ["MASTER_ADDR"] = subprocess.getoutput("scontrol show hostname {} | head -n1".format(os.environ["SLURM_NODELIST"]))
world_size = int(os.environ["SLURM_NTASKS"])
rank = int(os.environ["SLURM_PROCID"])
local_rank = rank % torch.cuda.device_count()
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank)


def _path(path: str) -> str:
    path = os.path.expanduser(os.path.expandvars(path))
    return path


data_root_dir = _path("$fzh_ALL_CCFRSCRATCH/OpenDV_processed/flat_tokens")
with open(_path("$fzh_ALL_CCFRSCRATCH/OpenDV_processed/train.json"), "r") as f:
    video_list = json.load(f)


# Create datasets
train_dataset = RandomTokenizedSequenceOpenDVDataset(
    data_root_dir, video_list, 20
)

train_dataloader = StatefulDataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True,
    sampler=DistributedSampler(train_dataset),
)

for idx, batch in enumerate(train_dataloader):
    print(rank, batch["idx"][:3])
    if idx == 5:
        state_dict = train_dataloader.state_dict()
    if idx == 10:
        break

train_dataloader_2 = StatefulDataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True,
    sampler=DistributedSampler(train_dataset),
)
train_dataloader_2.load_state_dict(state_dict)

for idx, batch in enumerate(train_dataloader_2):
    print(rank, batch["idx"][:3])
    if idx == 5:
        break
