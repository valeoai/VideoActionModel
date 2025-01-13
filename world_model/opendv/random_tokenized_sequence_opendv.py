import os
from typing import Dict, List

import click
import numpy as np
import torch
from torch.utils.data import Dataset


class RandomTokenizedSequenceOpenDVDataset(Dataset):
    """
    Args:
        data_root_dir: The root directory where the tokens are stored.
        video_list: list of videos name to consider for the given split.
        sequence_length: The number of consecutive frames to include in each data sample.
        load_image: A flag indicating whether to load images from disk. Defaults to True.
        subsampling_factor: only keep one frame every `subsampling_factor` frames. Defaults to 1.
    """

    def __init__(self, data_root_dir: str, video_list: List[str], sequence_length: int, subsampling_factor: int = 1) -> None:
        self.data_root_dir = data_root_dir
        self.video_list = video_list
        self.sequence_length = sequence_length
        self.subsampling_factor = subsampling_factor

        self.video_frames = {}
        self.video_windows = []
        self.total_windows = 0
        self.total_nb_frames = 0

        for video_id in self.video_list:
            video_dir = os.path.join(self.data_root_dir, video_id)
            frames = sorted([f for f in os.listdir(video_dir) if f.endswith(".npy")])
            self.total_nb_frames += len(frames)
            self.video_frames[video_id] = frames
            last_starting_index = len(frames) - 1 - (self.sequence_length - 1) * subsampling_factor
            self.total_windows += max(0, last_starting_index + 1)

            if last_starting_index >= 0:
                for start_idx in range(0, last_starting_index + 1):
                    self.video_windows.append((video_id, start_idx))

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id, start_idx = self.video_windows[idx]
        frame_sequence = []

        for i in range(0, self.sequence_length, self.subsampling_factor):
            frame_path = os.path.join(self.data_root_dir, video_id, self.video_frames[video_id][start_idx + i])
            frame_data = np.load(frame_path)
            frame_tensor = torch.from_numpy(frame_data).long()
            frame_sequence.append(frame_tensor)

        return {"visual_tokens": torch.stack(frame_sequence), "video_id": video_id, "frame_idx": start_idx + i}


if __name__ == "__main__":

    @click.command()
    @click.argument("data_root_dir", type=click.Path(exists=True))
    @click.argument("video_list_file", type=click.Path(exists=True))
    @click.option("--sequence-length", "-sl", default=20, help="Number of consecutive frames per sample")
    @click.option("--subsampling-factor", "-sf", default=5, help="Only keep one frame every N frames")
    def main(data_root_dir: str, video_list_file: str, sequence_length: int, subsampling_factor: int):
        """Analyze and print overview information about the video token dataset."""

        import json

        with open(video_list_file, "r") as f:
            video_list = json.load(f)

        dataset = RandomTokenizedSequenceOpenDVDataset(
            data_root_dir=data_root_dir,
            video_list=video_list,
            sequence_length=sequence_length,
            subsampling_factor=subsampling_factor,
        )

        # Print dataset overview
        click.echo(f"\n=== Dataset Overview ===")
        click.echo(f"Total number of videos: {len(dataset.video_list)}")
        click.echo(f"Total number of frames {dataset.total_nb_frames}")
        click.echo(f"Total number of sequences (i.e., windows): {len(dataset)}")

        # Sample a random sequence and print its properties
        if len(dataset) > 0:
            random_idx = np.random.randint(len(dataset))
            sample = dataset[random_idx]

            click.echo(f"\n=== Random Sample ===")
            click.echo(f"Video ID: {sample['video_id']}")
            click.echo(f"Starting frame index: {sample['frame_idx']}")
            click.echo(f"Token sequence shape: {sample['visual_tokens'].shape}")
            click.echo(f"Token value range: [{sample['visual_tokens'].min()}, {sample['visual_tokens'].max()}]")

    main()
