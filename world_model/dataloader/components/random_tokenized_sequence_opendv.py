import os
import numpy as np
import torch
from torch.utils.data import Dataset

class RandomTokenizedSequenceOpenDVDataset(Dataset):
    def __init__(self, data_root_dir, video_list, sequence_length):
        self.data_root_dir = data_root_dir
        self.video_list = video_list
        self.sequence_length = sequence_length
        
        self.video_frames = {}
        self.total_windows = 0
        
        for video_id in self.video_list:
            video_dir = os.path.join(self.data_root_dir, video_id)
            frames = sorted([f for f in os.listdir(video_dir) if f.endswith('.npy')])
            self.video_frames[video_id] = frames
            self.total_windows += max(0, len(frames) - self.sequence_length + 1)
        
        self.video_windows = []
        for video_id, frames in self.video_frames.items():
            for start_idx in range(len(frames) - self.sequence_length + 1):
                self.video_windows.append((video_id, start_idx))
    
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
        
        return torch.stack(frame_sequence)