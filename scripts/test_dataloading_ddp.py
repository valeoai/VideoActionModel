import json
import os
import subprocess

import torch
from torch.utils.data import DistributedSampler

from world_model.opendv.random_tokenized_sequence_opendv import RandomTokenizedSequenceOpenDVDataset
from world_model.opendv.stateful_dataloader import StatefulDataLoader


dist_url = "env://"
dist_backend = "nccl"
if os.environ.get("RANK") and os.environ.get("WORLD_SIZE"):
    # launched with torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
else:
    # launched with srun
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["MASTER_ADDR"] = subprocess.getoutput(
        "scontrol show hostname {} | head -n1".format(os.environ["SLURM_NODELIST"])
    )
    world_size = int(os.environ["SLURM_NTASKS"])
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = rank % torch.cuda.device_count()

torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank)
print(f"rank: {rank}, world_size: {world_size}, local_rank: {local_rank}")


def _path(path: str) -> str:
    path = os.path.expanduser(os.path.expandvars(path))
    return path


data_root_dir = _path("$fzh_ALL_CCFRSCRATCH/OpenDV_processed/flat_tokens")
with open(_path("$fzh_ALL_CCFRSCRATCH/OpenDV_processed/train.json"), "r") as f:
    video_list = json.load(f)
video_list = [os.path.join(data_root_dir, video) for video in video_list]

# Create datasets
train_dataset = RandomTokenizedSequenceOpenDVDataset(data_root_dir, video_list, 20)

train_dataloader = StatefulDataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    sampler=DistributedSampler(train_dataset),
)

save_stuff = []
for idx, batch in enumerate(train_dataloader):
    if idx == 4:
        state_dict = train_dataloader.state_dict()
        if rank == 0:
            torch.save(state_dict, "state_dict.pth")
    if idx >= 5:
        save_stuff.append(batch["idx"])
    if idx == 10:
        break


torch.distributed.barrier()
state_dict = torch.load("state_dict.pth")

train_dataloader_2 = StatefulDataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    sampler=DistributedSampler(train_dataset),
)
train_dataloader_2.load_state_dict(state_dict)

save_stuff_2 = []
for idx, batch in enumerate(train_dataloader_2):
    save_stuff_2.append(batch["idx"])
    if idx == 5:
        break

for idx, (s1, s2) in enumerate(zip(save_stuff, save_stuff_2)):
    assert (s1 == s2).all(), f"rank {rank} failed at idx {idx}"

print(f"rank {rank} passed")
