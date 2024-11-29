import os

import numpy as np
import torch
from torch.utils.data import Dataset


class PickledRandomTokenizedSequenceOpenDVDataset(Dataset):
    def __init__(self, data_root_dir, video_list, video_windows, sequence_length):
        self.data_root_dir = data_root_dir
        self.video_list = video_list
        self.sequence_length = sequence_length

        self.video_frames = {}
        self.video_windows = video_windows
        self.total_windows = len(video_windows)

        for video_id in self.video_list:
            video_dir = os.path.join(self.data_root_dir, video_id)
            frames = sorted([f for f in os.listdir(video_dir) if f.endswith(".npy")])
            self.video_frames[video_id] = frames

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        video_id, start_idx = self.video_windows[idx]
        frame_sequence = []

        for i in range(self.sequence_length):
            frame_path = os.path.join(self.data_root_dir, video_id, self.video_frames[video_id][start_idx + i])
            frame_data = np.load(frame_path)
            frame_tensor = torch.from_numpy(frame_data)
            frame_sequence.append(frame_tensor)

        return {"visual_tokens": torch.stack(frame_sequence)}
