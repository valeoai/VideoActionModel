import os
import pickle
from typing import Any, Dict, List, Optional

from lightning import LightningDataModule

from world_model.opendv.ego_trajectory_dataset import combined_ego_trajectory_dataset
from world_model.opendv.stateful_dataloader import StatefulDataLoader

StateDict = Dict[str, Any]


def _path(path: str) -> str | None:
    if path is None:
        return None

    path = os.path.expanduser(os.path.expandvars(path))
    return path


def read_pickle(pickle_path: str) -> List[Dict[str, Any]] | None:
    if pickle_path is None:
        return None

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


class EgoTrajectoryDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        nuplan_token_rootdir: Optional[str] = None,
        nuscenes_token_rootdir: Optional[str] = None,
        nuplan_train_pickle_path: Optional[str] = None,
        nuscenes_train_pickle_path: Optional[str] = None,
        nuplan_val_pickle_path: Optional[str] = None,
        nuscenes_val_pickle_path: Optional[str] = None,
        sequence_length: int = 8,
        action_length: int = 6,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.nuplan_token_rootdir = _path(nuplan_token_rootdir)
        self.nuscenes_token_rootdir = _path(nuscenes_token_rootdir)
        self.nuplan_train_pickle_path = _path(nuplan_train_pickle_path)
        self.nuscenes_train_pickle_path = _path(nuscenes_train_pickle_path)
        self.nuplan_val_pickle_path = _path(nuplan_val_pickle_path)
        self.nuscenes_val_pickle_path = _path(nuscenes_val_pickle_path)

        self.sequence_length = sequence_length
        self.action_length = action_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> "EgoTrajectoryDataModule":
        if hasattr(self, "train_dataset"):
            return

        # Read train and validation video lists
        nuplan_train_data = read_pickle(self.nuplan_train_pickle_path)
        nuscenes_train_data = read_pickle(self.nuscenes_train_pickle_path)

        # Create datasets
        if stage == "fit" or stage is None:
            kwargs = {
                "nuplan_token_rootdir": self.nuplan_token_rootdir,
                "nuscenes_token_rootdir": self.nuscenes_token_rootdir,
                "sequence_length": self.sequence_length,
                "action_length": self.action_length,
            }

            self.train_dataset = combined_ego_trajectory_dataset(
                nuplan_pickle_data=nuplan_train_data,
                nuscenes_pickle_data=nuscenes_train_data,
                **kwargs,
            )

            if self.nuplan_val_pickle_path is not None or self.nuscenes_val_pickle_path is not None:
                nuplan_val_data = read_pickle(self.nuplan_val_pickle_path)
                nuscenes_val_data = read_pickle(self.nuscenes_val_pickle_path)

                self.val_dataset = combined_ego_trajectory_dataset(
                    nuplan_pickle_data=nuplan_val_data,
                    nuscenes_pickle_data=nuscenes_val_data,
                    **kwargs,
                )

        return self

    def train_dataloader(self) -> StatefulDataLoader:
        if not hasattr(self, "train_dataloader_"):
            self.train_dataloader_ = StatefulDataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
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
            "nuplan_token_rootdir": self.nuplan_token_rootdir,
            "nuscenes_token_rootdir": self.nuscenes_token_rootdir,
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
        self.nuplan_token_rootdir = state_dict["nuplan_token_rootdir"]
        self.nuscenes_token_rootdir = state_dict["nuscenes_token_rootdir"]
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
