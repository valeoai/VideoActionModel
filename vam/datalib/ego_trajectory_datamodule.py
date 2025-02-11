import os
import pickle
from typing import Any, Dict, List, Optional

from lightning import LightningDataModule
from torch.utils.data import default_collate

from vam.datalib.data_mixing import combined_ego_trajectory_dataset
from vam.datalib.stateful_dataloader import StatefulDataLoader

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


class EgoTrajectoryDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        nuplan_tokens_rootdir: Optional[str] = None,
        nuscenes_tokens_rootdir: Optional[str] = None,
        nuplan_train_pickle_path: Optional[str] = None,
        nuscenes_train_pickle_path: Optional[str] = None,
        nuplan_val_pickle_path: Optional[str] = None,
        nuscenes_val_pickle_path: Optional[str] = None,
        sequence_length: int = 8,
        action_length: int = 6,
        batch_size: int = 32,
        num_workers: int = 4,
        sub_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.nuplan_tokens_rootdir = _path(nuplan_tokens_rootdir)
        self.nuscenes_tokens_rootdir = _path(nuscenes_tokens_rootdir)
        self.nuplan_train_pickle_path = _path(nuplan_train_pickle_path)
        self.nuscenes_train_pickle_path = _path(nuscenes_train_pickle_path)
        self.nuplan_val_pickle_path = _path(nuplan_val_pickle_path)
        self.nuscenes_val_pickle_path = _path(nuscenes_val_pickle_path)

        self.sequence_length = sequence_length
        self.action_length = action_length
        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> "EgoTrajectoryDataModule":
        if hasattr(self, "train_dataset"):
            return

        # Read train and validation video lists
        nuplan_train_data = _read_pickle(self.nuplan_train_pickle_path)
        nuscenes_train_data = _read_pickle(self.nuscenes_train_pickle_path)

        # Create datasets
        if stage == "fit" or stage is None:
            kwargs = {
                "nuplan_tokens_rootdir": self.nuplan_tokens_rootdir,
                "nuscenes_tokens_rootdir": self.nuscenes_tokens_rootdir,
                "sequence_length": self.sequence_length,
                "action_length": self.action_length,
            }

            self.train_dataset = combined_ego_trajectory_dataset(
                nuplan_pickle_data=nuplan_train_data,
                nuscenes_pickle_data=nuscenes_train_data,
                with_yaw_rate=self.sub_batch_size is not None,
                **kwargs,
            )

            if self.nuplan_val_pickle_path is not None or self.nuscenes_val_pickle_path is not None:
                nuplan_val_data = _read_pickle(self.nuplan_val_pickle_path)
                nuscenes_val_data = _read_pickle(self.nuscenes_val_pickle_path)

                self.val_dataset = combined_ego_trajectory_dataset(
                    nuplan_pickle_data=nuplan_val_data,
                    nuscenes_pickle_data=nuscenes_val_data,
                    **kwargs,
                )

        return self

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.sub_batch_size is not None:
            # Take the `sub_batch_size` sample with the highest yaw_rate
            batch = sorted(batch, key=lambda x: x["yaw_rate"].max(), reverse=True)[: self.sub_batch_size]

        return default_collate(batch)

    def train_dataloader(self) -> StatefulDataLoader:
        if not hasattr(self, "train_dataloader_"):
            self.train_dataloader_ = StatefulDataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self._collate_fn,
            )
        return self.train_dataloader_

    def val_dataloader(self) -> StatefulDataLoader:
        if not hasattr(self, "val_dataloader_"):
            self.val_dataloader_ = StatefulDataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
            )
        return self.val_dataloader_

    def state_dict(self) -> StateDict:
        return {
            "nuplan_tokens_rootdir": self.nuplan_tokens_rootdir,
            "nuscenes_tokens_rootdir": self.nuscenes_tokens_rootdir,
            "nuplan_train_pickle_path": self.nuplan_train_pickle_path,
            "nuscenes_train_pickle_path": self.nuscenes_train_pickle_path,
            "nuplan_val_pickle_path": self.nuplan_val_pickle_path,
            "nuscenes_val_pickle_path": self.nuscenes_val_pickle_path,
            "sequence_length": self.sequence_length,
            "action_length": self.action_length,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "train_loader_state_dict": self.train_dataloader_.state_dict(),
        }

    def load_state_dict(self, state_dict: StateDict) -> None:
        self.nuplan_tokens_rootdir = state_dict["nuplan_tokens_rootdir"]
        self.nuscenes_tokens_rootdir = state_dict["nuscenes_tokens_rootdir"]
        self.nuplan_train_pickle_path = state_dict["nuplan_train_pickle_path"]
        self.nuscenes_train_pickle_path = state_dict["nuscenes_train_pickle_path"]
        self.nuplan_val_pickle_path = state_dict["nuplan_val_pickle_path"]
        self.nuscenes_val_pickle_path = state_dict["nuscenes_val_pickle_path"]
        self.sequence_length = state_dict["sequence_length"]
        self.action_length = state_dict["action_length"]
        self.batch_size = state_dict["batch_size"]
        self.num_workers = state_dict["num_workers"]
        _ = self.setup()
        _ = self.train_dataloader()  # Initialize train dataloader
        _ = self.val_dataloader()  # Initialize val dataloader
        self.train_dataloader_.load_state_dict(state_dict["train_loader_state_dict"])


if __name__ == "__main__":
    dm = EgoTrajectoryDataModule(
        nuplan_tokens_rootdir="$ycy_ALL_CCFRSCRATCH/nuplan_v2_tokens/tokens",
        nuscenes_tokens_rootdir="$ycy_ALL_CCFRSCRATCH/nuscenes_v2/tokens",
        nuplan_train_pickle_path="$ycy_ALL_CCFRWORK/cleaned_trajectory_pickle/nuplan_train_data_cleaned.pkl",
        nuscenes_train_pickle_path="$ycy_ALL_CCFRWORK/cleaned_trajectory_pickle/nuscenes_train_data_cleaned.pkl",
        nuplan_val_pickle_path="$ycy_ALL_CCFRWORK/cleaned_trajectory_pickle/nuplan_val_data_cleaned.pkl",
        nuscenes_val_pickle_path="$ycy_ALL_CCFRWORK/cleaned_trajectory_pickle/nuscenes_val_data_cleaned.pkl",
    ).setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    for i, batch in enumerate(train_loader):
        if i > 10:
            break
        print(i, batch["visual_tokens"].shape, batch["positions"].shape)

    for i, batch in enumerate(val_loader):
        if i > 10:
            break
        print(i, batch["visual_tokens"].shape, batch["positions"].shape)
