import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset

from world_model.utils import RankedLogger

terminal_log = RankedLogger(__name__, rank_zero_only=True)

Sample = Dict[str, Any]


class EgoTrajectoryDataset(Dataset):
    """
    Dataset for extracting windows of ego-vehicle trajectories in the ego-car reference frame.
    All trajectories are normalized so that:
    - Initial position is at (0,0)
    - Initial orientation points along the positive x-axis

    Args:
        pickle_data: List of dictionaries containing nuPlan frame metadata
        sequence_length: Number of consecutive frames in each trajectory window
        subsampling_factor: Only keep one frame every `subsampling_factor` frames
        camera: Name of the camera to extract data for

    Returns dictionary containing:
        - positions: Tensor of shape [sequence_length, 2] containing x,y coordinates in ego frame
        - rotations: Tensor of shape [sequence_length, 4] containing quaternions in ego frame
        - timestamps: Tensor of shape [sequence_length] containing relative timestamps
        - scene_names: List of scene identifiers for each window
        - file_paths: List of relative file path for each frame in the sequence
    """

    def __init__(
        self,
        pickle_data: List[dict],
        tokens_rootdir: Optional[str] = None,
        camera: str = "CAM_FRONT",
        sequence_length: int = 8,
        action_length: int = 6,
        subsampling_factor: int = 1,
        transforms: Callable[[Sample], Sample] = None,
    ) -> None:
        self.tokens_rootdir = tokens_rootdir
        self.sequence_length = sequence_length
        self.action_length = action_length
        self.subsampling_factor = subsampling_factor
        self.camera = camera
        self.transforms = transforms

        # Sort by scene and timestamp
        pickle_data.sort(key=lambda x: (x["scene"]["name"], x[self.camera]["timestamp"]))
        self.pickle_data = pickle_data

        # Generate valid sequence indices
        self.sequences_indices = self.get_sequence_indices()

    @staticmethod
    def quaternion_multiply(q1: Tensor, q2: Tensor) -> Tensor:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.tensor([w, x, y, z])

    @staticmethod
    def quaternion_inverse(q: Tensor) -> Tensor:
        """Compute the inverse of a quaternion."""
        return torch.tensor([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def rotate_point(point: Tensor, quaternion: Tensor) -> Tensor:
        """Rotate a point using a quaternion."""
        # Convert point to quaternion format
        point_quat = torch.tensor([0.0, point[0], point[1], 0.0])

        # Apply rotation: q * p * q^(-1)
        q_inv = EgoTrajectoryDataset.quaternion_inverse(quaternion)
        rotated = EgoTrajectoryDataset.quaternion_multiply(
            EgoTrajectoryDataset.quaternion_multiply(quaternion, point_quat), q_inv
        )

        # Return rotated point
        return rotated[1:3]  # Only x,y components

    def get_sequence_indices(self) -> np.ndarray:
        """Generate indices for valid trajectory sequences."""
        indices = []

        for sequence_start_index in range(len(self.pickle_data)):
            is_valid_sequence = True
            previous_sample = None
            sequence_indices = []

            # Check if we can extract a full sequence starting at this index
            max_temporal_index = self.subsampling_factor * (self.sequence_length + self.action_length)
            for t in range(0, max_temporal_index, self.subsampling_factor):
                temporal_index = sequence_start_index + t

                # Check if we've hit the end of the dataset
                if temporal_index >= len(self.pickle_data):
                    is_valid_sequence = False
                    break

                sample = self.pickle_data[temporal_index]

                # Ensure all frames are from the same scene
                if (previous_sample is not None) and (sample["scene"]["name"] != previous_sample["scene"]["name"]):
                    is_valid_sequence = False
                    break

                sequence_indices.append(temporal_index)
                previous_sample = sample

            if is_valid_sequence:
                indices.append(sequence_indices)

        return np.asarray(indices)

    def __len__(self) -> int:
        return len(self.sequences_indices)

    def create_shifted_window_tensor(self, input_tensor: Tensor) -> Tensor:
        """
        Creates a tensor with shifting windows from an input tensor.

        Args:
            input_tensor: Input tensor of shape (self.sequence_length+self.action_length, 2)
            f: Size of the window

        Returns:
            Tensor of shape (self.sequence_length, self.action_length, 2)
        """
        tens = input_tensor.unsqueeze(0).repeat(self.sequence_length, 1, 1)
        result = torch.stack([tens[i, i + 1 : (i + 1) + self.action_length, :] for i in range(self.sequence_length)])
        return result

    def __getitem__(self, index: int) -> dict:
        data = defaultdict(list)
        first_frame_timestamp = None

        # Extract trajectory window
        temporal_indices = self.sequences_indices[index][: (self.sequence_length + self.action_length)]

        # Get initial pose
        initial_sample = self.pickle_data[temporal_indices[0]][self.camera]
        initial_position = torch.tensor(initial_sample["ego_to_world_tran"][:2], dtype=torch.float64)
        initial_rotation = torch.tensor(initial_sample["ego_to_world_rot"], dtype=torch.float64)
        initial_rotation_inv = self.quaternion_inverse(initial_rotation)

        for idx, temporal_index in enumerate(temporal_indices):
            sample = self.pickle_data[temporal_index][self.camera]

            # Get ego vehicle pose
            position = torch.tensor(sample["ego_to_world_tran"][:2], dtype=torch.float64)
            rotation = torch.tensor(sample["ego_to_world_rot"], dtype=torch.float64)

            # Transform to ego frame
            relative_position = position - initial_position
            relative_position = self.rotate_point(relative_position, initial_rotation_inv)

            # Transform rotation to ego frame
            relative_rotation = self.quaternion_multiply(initial_rotation_inv, rotation)

            # Store transformed poses
            data["positions"].append(relative_position)
            data["rotations"].append(relative_rotation)

            if (self.tokens_rootdir is not None) and (idx < self.sequence_length):
                # get visual tokens
                file_path = os.path.join(self.tokens_rootdir, sample["file_path"].replace(".jpg", ".npy"))
                tokens = torch.from_numpy(np.load(file_path))
                data["visual_tokens"].append(tokens)

            # Store metadata
            data["scene_names"].append(self.pickle_data[temporal_index]["scene"]["name"])
            data["file_paths"].append(sample["file_path"])

            # Calculate relative timestamp in seconds
            timestamp = sample["timestamp"]
            if first_frame_timestamp is None:
                first_frame_timestamp = timestamp
            relative_timestamp = (timestamp - first_frame_timestamp) * 1e-6  # Convert microseconds to seconds
            data["timestamps"].append(torch.tensor(relative_timestamp))

        # Stack tensors
        data["positions"] = self.create_shifted_window_tensor(torch.stack(data["positions"], dim=0)).to(dtype=torch.float32)
        data["rotations"] = self.create_shifted_window_tensor(torch.stack(data["rotations"], dim=0)).to(dtype=torch.float32)
        data["timestamps"] = torch.stack(data["timestamps"], dim=0)[: self.sequence_length]
        data["camera"] = self.camera
        if self.tokens_rootdir is not None:
            data["visual_tokens"] = torch.stack(data["visual_tokens"], dim=0)

        data = dict(data)

        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def get_sequence_info(self, index: int) -> str:
        """
        Get a string representation of sequence information for debugging.

        Args:
            index: Index of the sequence

        Returns:
            String containing sequence details
        """
        data = self[index]

        info = [
            f"Sequence {index}:",
            f"Scene: {data['scene_names'][0]}",
            f"Duration: {data['timestamps'][-1]:.2f} seconds",
            f"Number of frames: {len(data['timestamps'])}",
            f"First frame: {data['file_paths'][0]}",
            f"Last frame: {data['file_paths'][-1]}",
            f"Total displacement: {torch.norm(data['positions'][-1] - data['positions'][0]):.2f} meters",
        ]

        return "\n".join(info)


def combined_ego_trajectory_dataset(
    nuplan_pickle_data: Optional[List[dict]] = None,
    nuplan_tokens_rootdir: Optional[str] = None,
    nuscenes_pickle_data: Optional[List[dict]] = None,
    nuscenes_tokens_rootdir: Optional[str] = None,
    **kwargs,
) -> ConcatDataset:
    if nuplan_pickle_data is not None and nuscenes_pickle_data is not None:
        # If both datasets are provided, ensure that either both or none of the tokens rootdirs are provided
        if (nuplan_tokens_rootdir is None and nuscenes_tokens_rootdir is not None) or (
            nuplan_tokens_rootdir is not None and nuscenes_tokens_rootdir is None
        ):
            raise ValueError("Tokens rootdir must be provided for both datasets")

    datasets = []
    if nuplan_pickle_data is not None:
        datasets.append(
            EgoTrajectoryDataset(
                nuplan_pickle_data,
                tokens_rootdir=nuplan_tokens_rootdir,
                camera="CAM_F0",
                subsampling_factor=5,
                **kwargs,
            )
        )

    if nuscenes_pickle_data is not None:
        datasets.append(
            EgoTrajectoryDataset(
                nuscenes_pickle_data,
                tokens_rootdir=nuscenes_tokens_rootdir,
                camera="CAM_FRONT",
                **kwargs,
            )
        )

    assert len(datasets) > 0, "At least one dataset must be provided"
    return ConcatDataset(datasets)


if __name__ == "__main__":
    import pickle

    with open("/lustre/fswork/projects/rech/ycy/commun/nuscenes_pickle/val_data.pkl", "rb") as f:
        nuscenes_pickle_data = pickle.load(f)

    with open("/lustre/fswork/projects/rech/ycy/commun/nuplan_pickling/generated_files/nuplan_val_data.pkl", "rb") as f:
        nuplan_pickle_data = pickle.load(f)

    dataset = combined_ego_trajectory_dataset(
        nuplan_pickle_data=nuplan_pickle_data,
        # nuscenes_pickle_data=nuscenes_pickle_data,
        # nuscenes_tokens_rootdir="/lustre/fsn1/projects/rech/ycy/commun/nuscenes_v2/tokens",
    )
    # import ipdb; ipdb.set_trace()

    print("Length", len(dataset))
    print("Positions", dataset[0]["positions"].shape)
    print("Positions", dataset[0]["positions"])
    # print("Tokens", dataset[0]["visual_tokens"].shape)
