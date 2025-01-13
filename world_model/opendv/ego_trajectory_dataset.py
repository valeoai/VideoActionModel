from typing import Any, Callable, Dict, List

import torch
from torch import Tensor
from torch.utils.data import Dataset

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
        sequence_length: int = 50,
        subsampling_factor: int = 1,
        camera: str = "CAM_FRONT",
        transforms: Callable[[Sample], Sample] = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.subsampling_factor = subsampling_factor
        self.camera = camera
        self.transforms = transforms

        # Sort by scene and timestamp
        pickle_data.sort(key=lambda x: (x["scene"]["name"], x[self.camera]["timestamp"]))
        self.pickle_data = pickle_data

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
    def quaternion_inverse(q: torch.Tensor) -> torch.Tensor:
        """Compute the inverse of a quaternion."""
        return torch.tensor([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def rotate_point(point: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
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


if __name__ == "__main__":
    import pickle

    with open("/lustre/fswork/projects/rech/ycy/commun/nuscenes_pickle/trainval_data.pkl", "rb") as f:
        pickle_data = pickle.load(f)

    dataset = EgoTrajectoryDataset(pickle_data["val"])
