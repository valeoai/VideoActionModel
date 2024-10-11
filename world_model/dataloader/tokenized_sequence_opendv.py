import os
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Optional

# Assuming the VideoFrameDataset is in a file named video_dataset.py
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

        self.train_video_list = None
        self.val_video_list = None

    def setup(self, stage: Optional[str] = None):
        # Read train and validation video lists
        with open(self.video_list_path, 'r') as f:
            self.video_list = [line.strip() for line in f.readlines()]

        # Create datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = RandomTokenizedSequenceOpenDVDataset(
                self.data_root_dir,
                self.train_video_list,
                self.sequence_length
            )
            
            if self.val_video_list_path:
                with open(self.val_video_list_path, 'r') as f:
                    self.val_video_list = [line.strip() for line in f.readlines()]
                
                self.val_dataset = RandomTokenizedSequenceOpenDVDataset(
                    self.data_root_dir,
                    self.val_video_list,
                    self.sequence_length
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
