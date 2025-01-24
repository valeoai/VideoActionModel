import json
import os
import pickle
from typing import Any, Dict, List, Optional

from lightning import LightningDataModule

from world_model.datalib.data_mixing import all_token_datasets
from world_model.datalib.stateful_dataloader import StatefulDataLoader

StateDict = Dict[str, Any]


def _path(path: str) -> str | None:
    if path is None:
        return None

    path = os.path.expanduser(os.path.expandvars(path))
    return path


def _read_pickle(pickle_path: str) -> List[Dict[str, Any]] | None:
    if pickle_path is None:
        return None

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


def _read_json(json_path: str) -> Dict[str, Any] | None:
    if json_path is None:
        return None

    with open(json_path, "r") as f:
        data = json.load(f)
    return data


class MixFinetuningDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        opendv_tokens_rootdir: str,
        opendv_video_list_path: str,
        opendv_val_video_list_path: str,
        nuplan_tokens_rootdir: str,
        nuplan_train_pickle_path: str,
        nuplan_val_pickle_path: str,
        nuscenes_tokens_rootdir: str,
        nuscenes_train_pickle_path: str,
        nuscenes_val_pickle_path: str,
        sequence_length: int = 8,
        ratios: List[float] | None = None,
        total_number_of_samples: int | None = None,
        fixed_indices_json: Optional[List[str]] = None,
        seed: int = 0,
        batch_size: int = 32,
        num_workers: int = 4,
        is_finetuning: bool = False,
    ) -> None:
        super().__init__()
        self.opendv_tokens_rootdir = _path(opendv_tokens_rootdir)
        self.opendv_video_list_path = _path(opendv_video_list_path)
        self.opendv_val_video_list_path = _path(opendv_val_video_list_path)
        self.nuplan_tokens_rootdir = _path(nuplan_tokens_rootdir)
        self.nuscenes_tokens_rootdir = _path(nuscenes_tokens_rootdir)
        self.nuplan_train_pickle_path = _path(nuplan_train_pickle_path)
        self.nuscenes_train_pickle_path = _path(nuscenes_train_pickle_path)
        self.nuplan_val_pickle_path = _path(nuplan_val_pickle_path)
        self.nuscenes_val_pickle_path = _path(nuscenes_val_pickle_path)

        self.sequence_length = sequence_length
        self.train_ratios = ratios
        self.train_total_number_of_samples = total_number_of_samples
        self.train_fixed_indices_json = fixed_indices_json
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_finetuning = is_finetuning

    def setup(self, stage: Optional[str] = None) -> "MixFinetuningDataModule":
        if hasattr(self, "train_dataset"):
            return

        # Read train and validation video lists
        opendv_video_list = _read_json(self.opendv_video_list_path)
        nuplan_train_data = _read_pickle(self.nuplan_train_pickle_path)
        nuscenes_train_data = _read_pickle(self.nuscenes_train_pickle_path)

        # Create datasets
        if stage == "fit" or stage is None:
            kwargs = {
                "opendv_data_rootdir": self.opendv_tokens_rootdir,
                "nuplan_tokens_rootdir": self.nuplan_tokens_rootdir,
                "nuscenes_tokens_rootdir": self.nuscenes_tokens_rootdir,
                "sequence_length": self.sequence_length,
                "seed": self.seed,
            }

            self.train_dataset = all_token_datasets(
                opendv_video_list=opendv_video_list,
                nuplan_pickle_data=nuplan_train_data,
                nuscenes_pickle_data=nuscenes_train_data,
                ratios=self.train_ratios,
                total_number_of_samples=self.train_total_number_of_samples,
                fixed_indices_json=self.train_fixed_indices_json,
                **kwargs,
            )

            if self.opendv_val_video_list_path is not None:
                opendv_val_video_list = _read_json(self.opendv_val_video_list_path)
                nuplan_val_data = _read_pickle(self.nuplan_val_pickle_path)
                nuscenes_val_data = _read_pickle(self.nuscenes_val_pickle_path)

                # This creates the three datasets in the background
                self.all_val_datasets = all_token_datasets(
                    opendv_video_list=opendv_val_video_list,
                    nuplan_pickle_data=nuplan_val_data,
                    nuscenes_pickle_data=nuscenes_val_data,
                    **kwargs,
                )

        return self

    def train_dataloader(self) -> StatefulDataLoader:
        if not hasattr(self, "train_dataloader_"):
            self.train_dataloader_ = StatefulDataLoader(
                self.train_dataset,
                is_finetuning=self.is_finetuning,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return self.train_dataloader_

    def val_dataloader(self) -> List[StatefulDataLoader]:
        if not hasattr(self, "val_dataloader_"):
            self.val_dataloader_ = [
                StatefulDataLoader(
                    dts, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
                )
                for dts in self.all_val_datasets.datasets
            ]
        return self.val_dataloader_

    def state_dict(self) -> StateDict:
        return {
            "opendv_tokens_rootdir": self.opendv_tokens_rootdir,
            "opendv_video_list_path": self.opendv_video_list_path,
            "opendv_val_video_list_path": self.opendv_val_video_list_path,
            "nuplan_tokens_rootdir": self.nuplan_tokens_rootdir,
            "nuplan_train_pickle_path": self.nuplan_train_pickle_path,
            "nuplan_val_pickle_path": self.nuplan_val_pickle_path,
            "nuscenes_tokens_rootdir": self.nuscenes_tokens_rootdir,
            "nuscenes_train_pickle_path": self.nuscenes_train_pickle_path,
            "nuscenes_val_pickle_path": self.nuscenes_val_pickle_path,
            "sequence_length": self.sequence_length,
            "train_ratios": self.train_ratios,
            "train_total_number_of_samples": self.train_total_number_of_samples,
            "train_fixed_indices_json": self.train_fixed_indices_json,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "train_loader_state_dict": self.train_dataloader_.state_dict(),
        }

    def load_state_dict(self, state_dict: StateDict) -> None:
        self.opendv_tokens_rootdir = state_dict["opendv_tokens_rootdir"]
        self.opendv_video_list_path = state_dict["opendv_video_list_path"]
        self.opendv_val_video_list_path = state_dict["opendv_val_video_list_path"]
        self.nuplan_tokens_rootdir = state_dict["nuplan_tokens_rootdir"]
        self.nuplan_train_pickle_path = state_dict["nuplan_train_pickle_path"]
        self.nuplan_val_pickle_path = state_dict["nuplan_val_pickle_path"]
        self.nuscenes_tokens_rootdir = state_dict["nuscenes_tokens_rootdir"]
        self.nuscenes_train_pickle_path = state_dict["nuscenes_train_pickle_path"]
        self.nuscenes_val_pickle_path = state_dict["nuscenes_val_pickle_path"]
        self.sequence_length = state_dict["sequence_length"]
        self.train_ratios = state_dict["train_ratios"]
        self.train_total_number_of_samples = state_dict["train_total_number_of_samples"]
        self.train_fixed_indices_json = state_dict["train_fixed_indices_json"]
        self.batch_size = state_dict["batch_size"]
        self.num_workers = state_dict["num_workers"]
        self.seed = state_dict["seed"]
        _ = self.setup()
        _ = self.train_dataloader()  # Initialize train dataloader
        _ = self.val_dataloader()  # Initialize val dataloader
        self.train_dataloader_.load_state_dict(state_dict["train_loader_state_dict"])


if __name__ == "__main__":

    dm = MixFinetuningDataModule(
        opendv_tokens_rootdir="$fzh_ALL_CCFRSCRATCH/OpenDV_processed/flat_tokens",
        opendv_video_list_path="$fzh_ALL_CCFRSCRATCH/OpenDV_processed/train.json",
        opendv_val_video_list_path="$fzh_ALL_CCFRSCRATCH/OpenDV_processed/val.json",
        nuplan_tokens_rootdir="$ycy_ALL_CCFRSCRATCH/nuplan_v2_tokens/tokens",
        nuplan_train_pickle_path="$ycy_ALL_CCFRWORK/cleaned_trajectory_pickle/nuplan_train_data_cleaned.pkl",
        nuplan_val_pickle_path="$ycy_ALL_CCFRWORK/cleaned_trajectory_pickle/nuplan_val_data_cleaned.pkl",
        nuscenes_tokens_rootdir="$ycy_ALL_CCFRSCRATCH/nuscenes_v2/tokens",
        nuscenes_train_pickle_path="$ycy_ALL_CCFRWORK/cleaned_trajectory_pickle/nuscenes_train_data_cleaned.pkl",
        nuscenes_val_pickle_path="$ycy_ALL_CCFRWORK/cleaned_trajectory_pickle/nuscenes_val_data_cleaned.pkl",
        ratios=[0.4, 0.587, 0.013],
        total_number_of_samples=5963251,
        fixed_indices_json=["tmp/indexes_florent_hpc_test.json", None, None],
    ).setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    for i, batch in enumerate(train_loader):
        if i > 10:
            break
        print(i, batch["visual_tokens"].shape)

    for val_loader in dm.val_dataloader():
        for i, batch in enumerate(val_loader):
            if i > 10:
                break
            print(i, batch["visual_tokens"].shape)
