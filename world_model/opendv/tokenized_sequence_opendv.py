import json
import os
from typing import Any, Dict, List, Optional, Tuple

from lightning import LightningDataModule
from lightning_utilities.core.rank_zero import rank_zero_only

from world_model.utils import RankedLogger
from world_model.opendv.random_tokenized_sequence_opendv import RandomTokenizedSequenceOpenDVDataset
from world_model.opendv.stateful_dataloader import StatefulDataLoader

logger = RankedLogger(__name__, rank_zero_only=True)

StateDict = Dict[str, Any]


def _path(path: str) -> str:
    path = os.path.expanduser(os.path.expandvars(path))
    return path


class TokenizedSequenceOpenDVDataModule(LightningDataModule):
    def __init__(
        self,
        data_root_dir: str,
        video_list_path: str,
        sequence_length: int,
        val_video_list_path: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.data_root_dir = _path(data_root_dir)
        self.video_list_path = _path(video_list_path)
        self.val_video_list_path = _path(val_video_list_path)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def check_video_existence(self, video_list: List[str]) -> Tuple[List[str], List[str]]:
        existing_videos = []
        missing_videos = []
        for video in video_list:
            video_path = os.path.join(self.data_root_dir, video)
            if os.path.exists(video_path):
                existing_videos.append(video)
            else:
                missing_videos.append(video)
        return existing_videos, missing_videos

    @rank_zero_only
    def print_missing_videos(self, missing_videos: List[str]) -> None:
        if missing_videos:
            logger.warning("The following videos were not found:")
            for video in missing_videos:
                logger.warning(f"- {video}")
        else:
            logger.info("All video folders exist.")

    def setup(self, stage: Optional[str] = None) -> "TokenizedSequenceOpenDVDataModule":
        if hasattr(self, "train_dataset"):
            return

        # Read train and validation video lists
        with open(self.video_list_path, "r") as f:
            video_list = json.load(f)
        self.video_list, missing_videos = self.check_video_existence(video_list)
        self.print_missing_videos(missing_videos)  # Print missing videos only on rank 0

        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = RandomTokenizedSequenceOpenDVDataset(
                self.data_root_dir, self.video_list, self.sequence_length
            )

            if self.val_video_list_path:
                with open(self.val_video_list_path, "r") as f:
                    val_video_list = json.load(f)
                self.val_video_list, missing_val_videos = self.check_video_existence(val_video_list)
                self.print_missing_videos(missing_val_videos)

                self.val_dataset = RandomTokenizedSequenceOpenDVDataset(
                    self.data_root_dir, self.val_video_list, self.sequence_length
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
            "data_root_dir": self.data_root_dir,
            "video_list_path": self.video_list_path,
            "val_video_list_path": self.val_video_list_path,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "train_loader_state_dict": self.train_dataloader_.state_dict(),
        }

    def load_state_dict(self, state_dict: StateDict) -> None:
        self.data_root_dir = state_dict["data_root_dir"]
        self.video_list_path = state_dict["video_list_path"]
        self.val_video_list_path = state_dict["val_video_list_path"]
        self.sequence_length = state_dict["sequence_length"]
        self.batch_size = state_dict["batch_size"]
        self.num_workers = state_dict["num_workers"]
        _ = self.setup()
        _ = self.train_dataloader()  # Initialize train dataloader
        _ = self.val_dataloader()  # Initialize val dataloader
        self.train_dataloader_.load_state_dict(state_dict["train_loader_state_dict"])
