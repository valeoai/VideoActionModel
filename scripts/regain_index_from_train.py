import json
import os

import torch
import torch.distributed

# from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from world_model.opendv.random_tokenized_sequence_opendv import RandomTokenizedSequenceOpenDVDataset
from world_model.opendv.stateful_dataloader import StatefulDataLoader


def main(name: str, train_dataset: RandomTokenizedSequenceOpenDVDataset, rank: int, hp: dict) -> None:
    ckpt = hp["loops"]["fit_loop"]["state_dict"]["combined_loader"][0]
    world_size = len(ckpt)

    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=hp["TokenizedSequenceOpenDVDataModule"]["batch_size"],
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

    if (txt_path := f"tmp/indexes_{name}_{rank}.json") and os.path.exists(txt_path):
        os.remove(txt_path)

    all_indexes = []
    for batch in tqdm(train_dataloader, f"Aggregating {rank}", position=1, leave=False):
        indexes = batch["idx"].view(-1).tolist()
        all_indexes.extend(indexes)

    with open(txt_path, "w") as f:
        json.dump(all_indexes, f)


if __name__ == "__main__":

    def _path(path: str) -> str:
        path = os.path.expanduser(os.path.expandvars(path))
        return path

    # if os.path.exists("tmp/test_data_loader.pt"):
    #     ckpt = torch.load("tmp/test_data_loader.pt")
    #     hp = torch.load("tmp/test_data.pt")
    # else:
    #     hp = convert_zero_checkpoint_to_fp32_state_dict(
    #         "/lustre/fsn1/projects/rech/ycy/commun/WM_debug_restart_deepspeed/default_log_dir/hpc_ckpt_1.ckpt",
    #         "tmp/test_data.pt",
    #     )
    #     ckpt = hp["loops"]["fit_loop"]["state_dict"]["combined_loader"].pop(0)
    #     torch.save(ckpt, "tmp/test_data_loader.pt")

    hp = torch.load(
        "/lustre/fsn1/projects/rech/ycy/commun/output_data/"
        "opendv_gpt2_LlamaGen/wd_sweep/"
        "GPT2_OpenDV_Llamagen_768_Nodes16_BSperGPU6_totalBS384_weight_decay0.001_0116_1241_1737027680/"
        "checkpoints/quarters_epoch=000_step=0000038823_fused.pt"
    )
    ckpt = hp["loops"]["fit_loop"]["state_dict"]["combined_loader"][0]

    world_size = len(ckpt)

    data_root_dir = _path("$fzh_ALL_CCFRSCRATCH/OpenDV_processed/flat_tokens")
    with open(_path("$fzh_ALL_CCFRSCRATCH/OpenDV_processed/train.json"), "r") as f:
        video_list = json.load(f)
    video_list = [os.path.join(data_root_dir, video) for video in video_list]

    # Create datasets
    train_dataset = RandomTokenizedSequenceOpenDVDataset(
        data_root_dir,
        video_list,
        hp["TokenizedSequenceOpenDVDataModule"]["sequence_length"],
    )
    train_dataset._idx_only = True

    for rank in tqdm(range(world_size), "Creating indexes", position=0):
        main("florent_hpc_test", train_dataset, rank, hp)

    all_indexes = []
    for rank in range(world_size):
        with open(f"tmp/indexes_florent_hpc_test_{rank}.json", "r") as f:
            all_indexes.extend(json.load(f))

    with open("tmp/indexes_florent_hpc_test.json", "w") as f:
        json.dump(all_indexes, f)
