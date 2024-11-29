import os
from typing import List, Optional, Tuple

from lightning import LightningDataModule
from lightning_utilities.core.rank_zero import rank_zero_only
from torch.utils.data import DataLoader

from world_model.dataloader.components.random_tokenized_sequence_opendv import RandomTokenizedSequenceOpenDVDataset


class TokenizedSequenceOpenDVDataModule(LightningDataModule):
    def __init__(
        self,
        data_root_dir: str,
        video_list_path: str,
        sequence_length: int,
        val_video_list_path: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_root_dir = data_root_dir
        self.video_list_path = video_list_path
        self.val_video_list_path = val_video_list_path
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
            print("The following videos were not found:")
            for video in missing_videos:
                print(f"- {video}")
        else:
            print("All video folders exist.")

    def setup(self, stage: Optional[str] = None):
        # Read train and validation video lists
        with open(self.video_list_path, "r") as f:
            video_list = [line.strip() for line in f.readlines()]
        self.video_list, missing_videos = self.check_video_existence(video_list)
        self.print_missing_videos(missing_videos)  # Print missing videos only on rank 0

        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = RandomTokenizedSequenceOpenDVDataset(
                self.data_root_dir, self.video_list, self.sequence_length
            )

            if self.val_video_list_path:
                with open(self.val_video_list_path, "r") as f:
                    val_video_list = [line.strip() for line in f.readlines()]
                self.val_video_list, missing_val_videos = self.check_video_existence(val_video_list)
                self.print_missing_videos(missing_val_videos)

                self.val_dataset = RandomTokenizedSequenceOpenDVDataset(
                    self.data_root_dir, self.val_video_list, self.sequence_length
                )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
