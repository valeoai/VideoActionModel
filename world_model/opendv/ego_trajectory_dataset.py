import enum
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pyquaternion import Quaternion
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset

from world_model.utils import RankedLogger

terminal_log = RankedLogger(__name__, rank_zero_only=True)

Sample = Dict[str, Any]


class Command(int, enum.Enum):
    """Commands for the vehicle."""

    RIGHT = 0
    LEFT = 1
    STRAIGHT = 2
    FOLLOW_REFERENCE = 3


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

    COMMAND_DISTANCE_THRESHOLD: float = 2.0

    def __init__(
        self,
        pickle_data: List[dict],
        tokens_rootdir: Optional[str] = None,
        camera: str = "CAM_FRONT",
        sequence_length: int = 8,
        action_length: int = 6,
        subsampling_factor: int = 1,
        with_yaw_rate: bool = False,
        command_distance_threshold: float = 2.0,
    ) -> None:
        self.tokens_rootdir = tokens_rootdir
        self.sequence_length = sequence_length
        self.action_length = action_length
        self.subsampling_factor = subsampling_factor
        self.with_yaw_rate = with_yaw_rate
        self.camera = camera
        self.command_distance_threshold = command_distance_threshold

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
    def quaternion_to_euler_rates(quaternions: Tensor) -> Tensor:
        """Convert sequence of quaternions to average yaw rate.
        Quaternions are expected in [w, x, y, z] format.
        """
        # Convert to euler angles
        quaternions = quaternions.numpy()
        q0, q1, q2, q3 = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

        # Calculate yaw (rotation around z-axis)
        yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

        # Unwrap angles to handle discontinuities
        yaw_unwrapped = np.unwrap(yaw)

        # Calculate average rate (assuming constant time steps)
        yaw_rate = np.abs(np.mean(np.diff(yaw_unwrapped)))

        return torch.tensor(yaw_rate)

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

    @staticmethod
    def pose_to_matrix(translation: List[float], rotation: List[float]) -> Tensor:
        """Converts a NuScenes ego pose to a transformation matrix."""
        translation = np.array(translation)
        rotation = Quaternion(rotation).rotation_matrix
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        return torch.from_numpy(matrix).double()

    @staticmethod
    def get_high_level_command(
        translation: List[float], rotation: List[float], future_translation: List[float], future_rotation: List[float]
    ) -> Command:
        cur_ego2world = EgoTrajectoryDataset.pose_to_matrix(translation, rotation)
        next_ego2world = EgoTrajectoryDataset.pose_to_matrix(future_translation, future_rotation)

        cur_world2ego = cur_ego2world.inverse()
        future_pos2ego = cur_world2ego @ next_ego2world[:, -1]  # (4x4) x 4, 1
        # we get this in x-forward, y left
        if future_pos2ego[1] > EgoTrajectoryDataset.COMMAND_DISTANCE_THRESHOLD:
            return Command.LEFT
        if future_pos2ego[1] < -EgoTrajectoryDataset.COMMAND_DISTANCE_THRESHOLD:
            return Command.RIGHT

        return Command.STRAIGHT

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

    def sequence_of_positions_to_trajectory(self, positions: Tensor, rotations: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Convert a sequence of positions to a trajectory by computing the relative displacements.

        Args:
            positions: Tensor of shape (self.action_length + 1, 2) containing x,y coordinates
                       from initial position to action_length time steps
            rotations: Tensor of shape (self.action_length + 1, 2) containing quaternions
                       from initial rotation to action_length time steps

        Returns:
            Tensor of shape (self.action_length, 2) containing relative displacements for action_length time steps
        """
        initial_position = torch.tensor(positions[0], dtype=torch.float64)
        initial_rotation = torch.tensor(rotations[0], dtype=torch.float64)
        initial_rotation_inv = self.quaternion_inverse(initial_rotation)

        relative_positions = torch.tensor(positions[1:], dtype=torch.float64) - initial_position
        relative_rotations = torch.tensor(rotations[1:], dtype=torch.float64)

        for i in range(len(relative_positions)):
            relative_positions[i] = self.rotate_point(relative_positions[i], initial_rotation_inv)
            relative_rotations[i] = self.quaternion_multiply(initial_rotation_inv, relative_rotations[i])

        return relative_positions, relative_rotations

    def __getitem__(self, index: int) -> dict:
        data = defaultdict(list)
        first_frame_timestamp = None

        # Extract trajectory window
        temporal_indices = self.sequences_indices[index][: self.sequence_length]

        # Get initial pose
        for temporal_index in temporal_indices:
            sample = self.pickle_data[temporal_index][self.camera]

            positions, rotations = [], []
            for _j in range(1 + self.action_length):
                positions.append(self.pickle_data[temporal_index + _j][self.camera]["ego_to_world_tran"][:2])
                rotations.append(self.pickle_data[temporal_index + _j][self.camera]["ego_to_world_rot"])

            high_level_command = EgoTrajectoryDataset.get_high_level_command(
                self.pickle_data[temporal_index][self.camera]["ego_to_world_tran"],
                self.pickle_data[temporal_index][self.camera]["ego_to_world_rot"],
                self.pickle_data[temporal_index + self.action_length][self.camera]["ego_to_world_tran"],
                self.pickle_data[temporal_index + self.action_length][self.camera]["ego_to_world_rot"],
            )
            data["high_level_commands"].append(high_level_command)

            relative_position, relative_rotation = self.sequence_of_positions_to_trajectory(positions, rotations)

            # Store transformed poses
            data["positions"].append(relative_position)
            data["rotations"].append(relative_rotation)

            if self.with_yaw_rate:
                data["yaw_rate"].append(self.quaternion_to_euler_rates(relative_rotation))

            if self.tokens_rootdir is not None:
                # get visual tokens
                file_path = os.path.join(self.tokens_rootdir, sample["file_path"].replace(".jpg", ".npy"))
                tokens = torch.from_numpy(np.load(file_path)).to(dtype=torch.long)
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
        data["positions"] = torch.stack(data["positions"]).to(dtype=torch.float32)
        data["rotations"] = torch.stack(data["rotations"]).to(dtype=torch.float32)
        data["high_level_command"] = torch.tensor(data["high_level_commands"], dtype=torch.int64)
        data["timestamps"] = torch.stack(data["timestamps"], dim=0)[: self.sequence_length]
        data["camera"] = self.camera
        if self.tokens_rootdir is not None:
            data["visual_tokens"] = torch.stack(data["visual_tokens"], dim=0)
        if self.with_yaw_rate:
            data["yaw_rate"] = torch.stack(data["yaw_rate"], dim=0)

        data = dict(data)
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
    # If both datasets are provided, ensure that either both or none of the tokens rootdirs are provided
    if (nuplan_pickle_data is not None and nuscenes_pickle_data is not None) and (
        (nuplan_tokens_rootdir is None and nuscenes_tokens_rootdir is not None)
        or (nuplan_tokens_rootdir is not None and nuscenes_tokens_rootdir is None)
    ):
        raise ValueError("Tokens rootdir must be provided for both datasets")

    datasets = []
    if nuplan_pickle_data is not None:
        datasets.append(
            EgoTrajectoryDataset(
                nuplan_pickle_data,
                tokens_rootdir=nuplan_tokens_rootdir,
                camera="CAM_F0",
                subsampling_factor=5,  # Nuplan is originally at 10Hz, we subsample to 2Hz
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
    import random

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from torch.utils.data import Subset
    from tqdm import tqdm

    def plot_trajectories(
        dataset: EgoTrajectoryDataset | ConcatDataset,
        figsize: Tuple[int, int] = (12, 8),
        max_trajectories: Optional[int] = None,
        save_path: str = "trajectory_plot.pdf",
    ) -> None:
        plt.style.use("default")
        plt.rcParams.update(
            {
                "axes.facecolor": "white",
                "figure.facecolor": "white",
                "figure.edgecolor": "white",
                "savefig.facecolor": "white",
                "savefig.edgecolor": "white",
            }
        )

        # Random sampling if max_trajectories is specified
        if max_trajectories is not None and max_trajectories < len(dataset):
            indexes = random.sample(range(len(dataset)), max_trajectories)
            print(f"Randomly sampled {max_trajectories} trajectories from {len(dataset)} total")
            dataset = Subset(dataset, indexes)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("white")

        # Initialize min/max for plot boundaries and yaw rates
        x_min, y_min = float("inf"), float("inf")
        x_max, y_max = float("-inf"), float("-inf")

        # First pass: calculate all yaw rates and bounds
        all_positions, all_yaw_rates = [], []
        for idx in tqdm(indexes, desc="Computing yaw rates", leave=True):
            positions = dataset[idx]["positions"]
            yaw_rate = dataset[idx]["yaw_rate"]
            all_positions.append(positions)
            all_yaw_rates.append(yaw_rate)

            # Update plot boundaries
            x_min = min(x_min, positions[..., 0].min())
            x_max = max(x_max, positions[..., 0].max())
            y_min = min(y_min, positions[..., 1].min())
            y_max = max(y_max, positions[..., 1].max())

        colormap = plt.get_cmap("jet")
        norm = Normalize(vmin=np.min(all_yaw_rates), vmax=np.max(all_yaw_rates))

        # Second pass: plot trajectories
        for i, _ in enumerate(tqdm(indexes, desc="Plotting trajectories", leave=True)):
            positions = all_positions[i]
            yaw_rate = all_yaw_rates[i]

            # Use single color or yaw-rate based color
            for j in range(len(yaw_rate)):
                c = colormap(norm(yaw_rate[j]))
                ax.plot(positions[j, :, 0], positions[j, :, 1], color=c, alpha=0.5, linewidth=1)

        # Add padding to the limits
        padding = 0.05 * max(x_max - x_min, y_max - y_min)
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

        # Add labels and title
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title(f"Trajectory Plot (n={len(dataset)})\nColored by Average Yaw Rate")

        # Equal aspect ratio for proper visualization
        ax.set_aspect("equal")

        # Add grid with light gray color
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")

        # Add colorbar if requested and using yaw rate coloring
        sm = ScalarMappable(cmap=colormap, norm=norm)
        cbar = plt.colorbar(
            sm, ax=ax, label="Average Yaw Rate (rad/s)", fraction=0.025, pad=0.02
        )  # Make colorbar thinner and closer

        # Format colorbar ticks to 2 decimal places
        cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

        # Adjust colorbar label padding
        cbar.ax.set_ylabel("Average Yaw Rate (rad/s)", labelpad=-15)

        # Ensure tight layout
        plt.tight_layout()

        print(f"Saving plot to {save_path}...")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    with open("/lustre/fswork/projects/rech/ycy/commun/cleaned_trajectory_pickle/nuscenes_train_data_cleaned.pkl", "rb") as f:
        nuscenes_pickle_data = pickle.load(f)

    with open("/lustre/fswork/projects/rech/ycy/commun/cleaned_trajectory_pickle/nuplan_val_data_cleaned.pkl", "rb") as f:
        nuplan_pickle_data = pickle.load(f)

    dataset = combined_ego_trajectory_dataset(
        nuplan_pickle_data=nuplan_pickle_data,
        # nuplan_tokens_rootdir="/lustre/fsn1/projects/rech/ycy/commun/nuplan_v2_tokens/tokens",
        # nuscenes_pickle_data=nuscenes_pickle_data,
        # with_yaw_rate=True,
        # nuscenes_tokens_rootdir="/lustre/fsn1/projects/rech/ycy/commun/nuscenes_v2/tokens",
    )

    print("Length", len(dataset))
    print("Positions", dataset[0]["positions"].shape)
    print("high_level_command", dataset[0]["high_level_command"].shape)
    # print("Positions", dataset[0]["positions"])
    # print("Tokens", dataset[0]["visual_tokens"].shape)

    # plot_trajectories(dataset, max_trajectories=20000, save_path="trajectory_plot.pdf")
