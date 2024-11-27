import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, Any, List

import numpy as np
import torch

from world_model.utils import RankedLogger

terminal_log = RankedLogger(__name__, rank_zero_only=True)


class TokenizedNuScenesDataset(torch.utils.data.Dataset):
    """
    Args:
        pickle_data: A list of dictionaries, each containing metadata for a single frame in the dataset.
        quantized_visual_tokens_root_dir: The directory where quantized representations of the NuScenes images are stored.
        sequence_length: The number of consecutive frames to include in each data sample. Defaults to 1.
        trajectory_length: The number of **future** steps by which a trajectory is defined. Defaults to 0.
        subsampling_factor: only keep one frame every `subsampling_factor` frames.
        camera: Name of the camera to extract data for (e.g., 'CAM_F0', 'CAM_FRONT' this is dataset dependant)
        quantized_trajectory_root_dir: The directory where quantized representations of the NuScenes trajectory are stored.
        command_db: A dictionary containing the command for each frame in the dataset.

    The `__getitem__` method returns a dictionary containing the following keys:
        - scene_name: A list of string indicating to which scene the frame belongs.
        - images_paths: A list of file paths to the images in the sequence (relative to data root dir).
        - timestamps: The timestamps of each frame in the sequence, in seconds, the first frame's timestamp is 0.
        - visual_tokens: The quantized representations of the images (maps of integers).
        - trajectory_tokens: The quantized representations of the trajectory (maps of integers).
        - commands: The command for each frame in the sequence.
    """

    def __init__(
        self,
        pickle_data: List[Dict[str, Any]],
        quantized_visual_tokens_root_dir: str,
        sequence_length: int = 1,
        trajectory_length: int = 0,
        subsampling_factor: int = 1,
        camera: str = 'CAM_FRONT',
        quantized_trajectory_root_dir: Optional[str] = None,
        command_db: Optional[Dict[str, int]] = None,
    ) -> None:
        self.camera = camera

        self.sequence_length = sequence_length
        self.trajectory_length = trajectory_length
        self.subsampling_factor = subsampling_factor

        self.quantized_visual_tokens_root_dir = Path(quantized_visual_tokens_root_dir)

        self.quantized_trajectory_root_dir = None
        if quantized_trajectory_root_dir is not None:
            self.quantized_trajectory_root_dir = Path(quantized_trajectory_root_dir)

        self.command_db = command_db

        # sort by scene and timestamp
        pickle_data.sort(key=lambda x: (x['scene']['name'], x[self.camera]['timestamp']))
        self.pickle_data = pickle_data

        self.sequences_indices = self.get_sequence_indices()

    def get_sequence_indices(self) -> np.ndarray:
        """
        Generates indices for valid sequences in the dataset based on the specified sequence length.
        A sequence is considered valid if it consists of consecutive frames within the same scene.

        Returns:
            numpy.ndarray: An array of indices, where each element is a list of indices representing
            a valid sequence in the dataset. Each list has a length equal to the specified sequence length + prediction length,
            ensuring that all frames within a sequence belong to the same scene.
        """
        indices = []
        for sequence_start_index in range(len(self.pickle_data)):
            is_valid_data = True
            previous_sample = None
            sequence_indices = []

            # v = visual tokens, t = trajectory tokens
            # the sequence is [v_0, t_0, v_1, t_1]
            # the trajectory tokens actually encodes `trajectory_length` future positions
            # which means that we have to collect `trajectory_length` additional frames
            # to make sure that we extract a valid sequence such that each frames of the sequence
            # have trajectory tokens.
            max_temporal_index = self.sequence_length + self.trajectory_length
            max_temporal_index *= self.subsampling_factor
            for t in range(0, max_temporal_index, self.subsampling_factor):
                
                temporal_index = sequence_start_index + t

                # Going over the dataset size limit.
                if temporal_index >= len(self.pickle_data):
                    is_valid_data = False
                    break

                sample = self.pickle_data[temporal_index]

                # Check if scene is the same
                if (previous_sample is not None) and (sample['scene']['name'] != previous_sample['scene']['name']):
                    is_valid_data = False
                    break

                if t < self.sequence_length * self.subsampling_factor:
                    sequence_indices.append(temporal_index)
                    # even tho we collected `sequence_length` frames
                    # continue looping until reaching `max_temporal_index` to check is sequence is valid
                    # if not, it means not all frames have trajectory tokens, so we ditch the seq
                    
                previous_sample = sample

            if is_valid_data:
                indices.append(sequence_indices)
        return np.asarray(indices)

    def __len__(self) -> int:
        return len(self.sequences_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:

        data = defaultdict(list)
        first_frame_timestamp = 0

        # Loop over all the frames in the temporal extent.
        for i, temporal_index in enumerate(self.sequences_indices[index]):
            sample = self.pickle_data[temporal_index]

            ####### get scene name
            data['scene_names'].append(self.pickle_data[temporal_index]['scene']['name'])

            ####### load command
            if self.command_db is not None:
                # Only a global command and not a per camera command
                try:
                    command = self.command_db[os.path.basename(sample[self.camera]['file_path'])]
                except KeyError as e:
                    if i > self.sequence_length:
                        # For the last frame in NuScenes sequence, we don't have a command
                        command = -1
                    else:
                        raise e
                command = torch.tensor(command)
                data['commands'].append(command)

            ####### get tokens
            relative_img_path = sample[self.camera]['file_path']
            data['images_paths'] = relative_img_path

            ####### load image tokens
            quantized_data_path = (self.quantized_visual_tokens_root_dir / relative_img_path).with_suffix('.npy')
            quantized_data = np.load(quantized_data_path)
            quantized_data = torch.tensor(quantized_data)
            data['visual_tokens'].append(quantized_data)

            ####### load trajectory tokens
            if self.quantized_trajectory_root_dir is not None:
                quantized_trajectory_path = (self.quantized_trajectory_root_dir / relative_img_path).with_suffix('.npy')
                quantized_trajectory = np.load(quantized_trajectory_path)
                quantized_trajectory = torch.tensor(quantized_trajectory)
                data['trajectory_tokens'].append(quantized_trajectory)

            ####### load timestamp
            unix_timestamp = sample[self.camera]['timestamp']
            if i == 0:
                first_frame_timestamp = unix_timestamp
            unix_timestamp = (unix_timestamp - first_frame_timestamp) * 1e-6
            unix_timestamp = torch.tensor(unix_timestamp)
            data['timestamps'].append(unix_timestamp)

        keys_to_stack = ['visual_tokens', 'timestamps']
        if self.quantized_trajectory_root_dir is not None:
            keys_to_stack.append('trajectory_tokens')
        if self.command_db is not None:
            keys_to_stack.append('commands')
            
        for key in keys_to_stack:
            if key in data.keys():
                data[key] = torch.stack(data[key], dim=0)

        return data


if __name__ == '__main__':
    import pickle

    _path = lambda x: os.path.expanduser(os.path.expandvars(x))

    pickle_path = os.path.join(_path('$ycy_ALL_CCFRSCRATCH'), 'nuscenes_cvpr', 'trainval_data.pkl')
    command_db_path = os.path.join(
        _path('$ycy_ALL_CCFRSCRATCH'),
        'nuscenes_cvpr',
        'nuscenes_commands.json',
    )
    visual_tokens_paths = os.path.join(
        _path('$ycy_ALL_CCFRSCRATCH'),
        'nuscenes_tokenized',
        'VQ_ds16_16384_llamagen',
    )
    trajectory_tokens_paths = os.path.join(
        _path('$ycy_ALL_CCFRSCRATCH'),
        'nuscenes_tokenized',
        'TrajectoryFSQ_seqlen6',
        'epoch_029_val_recon_loss_0.0211',
    )

    with open(pickle_path, 'rb') as f:
        pickle_data = pickle.load(f)['train']

    if command_db_path is not None:
        with open(command_db_path) as f:
            command_db = json.load(f)

    dts = TokenizedNuScenesDataset(
        pickle_data,
        visual_tokens_paths,
        sequence_length=3,
        trajectory_length=6,
        quantized_trajectory_root_dir=trajectory_tokens_paths,
        command_db=command_db,
        camera=['CAM_FRONT'],
    )

    print(len(dts))
    sample = dts[0]
    print(sample.keys())
    print(sample['visual_tokens'].shape)
    print(sample['trajectory_tokens'].shape)
    print(sample['commands'].shape)
