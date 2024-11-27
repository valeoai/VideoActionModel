import os
import json
import enum
import argparse

import numpy as np
import torch
from torch import Tensor
from pyquaternion import Quaternion

COMMAND_DISTANCE_THRESHOLD = 2.0


NUSCENES_CAM_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


class Command(int, enum.Enum):
    """Commands for the vehicle."""

    RIGHT = 0
    LEFT = 1
    STRAIGHT = 2
    FOLLOW_REFERENCE = 3


def _pose_to_matrix(ego_pose: dict) -> Tensor:
    """Converts a NuScenes ego pose to a transformation matrix."""
    translation = np.array(ego_pose["translation"])
    rotation = Quaternion(ego_pose["rotation"]).rotation_matrix
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return torch.from_numpy(matrix).double()


def _get_last_sample(nusc, scene) -> dict:
    sample = nusc.get("sample", scene["first_sample_token"])
    n_frames = 1
    while sample["next"]:
        sample = nusc.get("sample", sample["next"])
        n_frames += 1
    max_frame_idx = n_frames - 1
    return sample, max_frame_idx


def _get_sample(sample_idx: int, nusc, scene) -> dict:
    """Get the nth sample in the scene."""
    if sample_idx == -1:
        sample, _ = _get_last_sample(nusc, scene)
        return sample
    sample = nusc.get("sample", scene["first_sample_token"])
    for _ in range(sample_idx):
        if not sample["next"]:
            raise ValueError(f"Sample {sample_idx} out of range.")
        sample = nusc.get("sample", sample["next"])
    return sample


def get_ego_pose(frame_idx: int, nusc, scene) -> Tensor:
    """Get the pose of the ego vehicle at the nth sample in the scene."""
    sample = _get_sample(frame_idx, nusc, scene)
    lidar_sample_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    return _pose_to_matrix(ego_pose)


def get_image_path(frame_idx: int, cam_name: str, nusc, scene) -> str:
    """Get the path to the image at the given timestamp."""
    sample = _get_sample(frame_idx, nusc, scene)
    sample_data = nusc.get("sample_data", sample["data"][cam_name])
    return nusc.get_sample_data_path(sample_data["token"])


def get_command(frame_idx: int, max_frame_idx: int, nusc, scene) -> Command:
    cur_ego2world = get_ego_pose(frame_idx, nusc, scene)
    future_frame_idx = min(frame_idx + 6, max_frame_idx)
    next_ego2world = get_ego_pose(future_frame_idx, nusc, scene)

    cur_world2ego = cur_ego2world.inverse()
    future_pos2ego = cur_world2ego @ next_ego2world[:, -1]  # (4x4) x 4, 1
    # we get this in x-forward, y left
    if future_pos2ego[1] > COMMAND_DISTANCE_THRESHOLD:
        return Command.LEFT
    if future_pos2ego[1] < -COMMAND_DISTANCE_THRESHOLD:
        return Command.RIGHT

    return Command.STRAIGHT


def process_a_dataset(nusc):
    db = {}
    stats = []
    scene = nusc.scene[0]
    for scene in tqdm(nusc.scene, 'Processing scenes'):
        _, max_frame_idx = _get_last_sample(nusc, scene)
        for i in tqdm(range(max_frame_idx), 'Processing samples', leave=False, position=1):
            command = get_command(i, max_frame_idx, nusc, scene)
            for cam_name in NUSCENES_CAM_ORDER:
                path = get_image_path(i, cam_name, nusc, scene)
                db[os.path.basename(path)] = command.value
                stats.append(command.value)

    return db, stats


if __name__ == "__main__":
    """
    Example usage:

    python scripts/create_nuscenes_command.py --dataroot /datasets_local/nuscenes

    srun -A ycy@h100 -C h100 --pty --nodes=1 \
    --ntasks-per-node=1 --cpus-per-task=24 \
    --gres=gpu:0 --hint=nomultithread \
    --qos=qos_gpu_h100-dev --time=00:20:00 \
    python scripts/create_nuscenes_command.py --dataroot $ycy_ALL_CCFRSCRATCH/nuscenes_cvpr

    """
    from collections import Counter

    from tqdm import tqdm
    from nuscenes.nuscenes import NuScenes

    def _path(x):
        return os.path.expanduser(os.path.expandvars(x))

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=_path, default="/datasets_local/nuscenes")
    args = parser.parse_args()

    db, stats = {}, []
    for version in ['v1.0-trainval', 'v1.0-test']:
        nusc = NuScenes(version=version, dataroot=args.dataroot, verbose=True)
        tmp_db, tmp_stats = process_a_dataset(nusc)
        db.update(tmp_db)
        stats.extend(tmp_stats)

    print(len(db) // 6)
    print(Counter(stats))
    with open(os.path.join(args.dataroot, "nuscenes_commands.json"), "w") as f:
        json.dump(db, f)
