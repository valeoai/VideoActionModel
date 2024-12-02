import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from world_model.opendv.struct_utils import load_struct


class RandomTokenizedSequenceOpenDVDataset(Dataset):
    def __init__(self, data_root_dir: str, video_list: List[str], sequence_length: int) -> None:
        self.data_root_dir = data_root_dir
        self.video_list = video_list
        self.sequence_length = sequence_length

        self.video_frames = {}
        self.total_windows = 0

        for video_id in self.video_list:
            video_dir = os.path.join(self.data_root_dir, video_id)
            frames = sorted([f for f in os.listdir(video_dir) if f.endswith(".npy")])
            self.video_frames[video_id] = frames
            self.total_windows += max(0, len(frames) - self.sequence_length + 1)

        self.video_windows = []
        for video_id, frames in self.video_frames.items():
            for start_idx in range(len(frames) - self.sequence_length + 1):
                self.video_windows.append((video_id, start_idx))

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id, start_idx = self.video_windows[idx]
        frame_sequence = []

        for i in range(self.sequence_length):
            frame_path = os.path.join(self.data_root_dir, video_id, self.video_frames[video_id][start_idx + i])
            frame_data = load_struct(frame_path).astype(np.uint16)
            frame_tensor = torch.from_numpy(frame_data).long().view(-1)
            frame_sequence.append(frame_tensor)

        return {"visual_tokens": torch.stack(frame_sequence)}
