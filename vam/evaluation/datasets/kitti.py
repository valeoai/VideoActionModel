import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch import Tensor

from vam.evaluation.datasets.base_dataset import GenericVideoDataset, load_depthMaps


def decode_panoptic_map(panoptic_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decode RGB panoptic map into semantic and instance maps.

    In KITTI-STEP, the panoptic map is encoded in RGB format where:
    - R channel contains the semantic ID
    - G and B channels encode the instance ID as: instance_id = G * 256 + B

    Args:
        panoptic_map: RGB panoptic map of shape [H, W, 3]

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - semantic_map (np.ndarray): Semantic segmentation map of shape [H, W]
            - instance_map (np.ndarray): Instance segmentation map of shape [H, W]
    """
    semantic_map = panoptic_map[0]  # R channel
    instance_map = panoptic_map[1] * 256 + panoptic_map[2]  # G*256 + B
    return semantic_map.int(), instance_map.int()


class KITTIDataset(GenericVideoDataset):
    _NUM_CLASSES = 19
    _IMAGE_SIZE = (288, 512)

    def __init__(
        self,
        root: str,
        split: str,
        window_size: int,
        frame_stride: int = 1,
        target_size: Optional[Tuple[int, int]] = None,
        pseudo_depth: Optional[str] = None,
        **kwargs,
    ) -> None:
        """KITTI-STEP dataset for panoptic segmentation with temporal sequences.

        Args:
            root: Path to the root directory of KITTI-STEP dataset
            split: Dataset split to use. One of ['train', 'val', 'test']
            window_size: Number of consecutive frames to return in each sample
            frame_stride: Stride between consecutive frames (default=1)
                        frame_stride=1 -> 10Hz (original)
                        frame_stride=2 -> 5Hz

        This dataset provides access to temporal sequences of images and their corresponding
        panoptic segmentation maps from the KITTI-STEP dataset. Each sample consists of
        a sequence of consecutive frames of length window_size.

        Directory structure expected:
        root/
        ├── images/
        │   ├── train/
        │   │   └── %04d/ (sequence_idx)
        │   │       └── %06d.png (frame_idx)
        │   ├── val/
        │   └── test/
        └── panoptic_maps/
            ├── train/
            │   └── %04d/ (sequence_idx)
            │       └── %06d.png (frame_idx)
            └── val/

        Note:
            Panoptic maps are encoded in RGB format where:
            - R channel: semantic ID
            - G channel: instance ID // 256
            - B channel: instance ID % 256
        """
        self.root = Path(root)
        self.split = split
        self.window_size = window_size
        self.frame_stride = frame_stride

        if pseudo_depth is not None:
            pseudo_depth = os.path.join(self.root, "kitti_depth", self.split)

        if target_size is not None:
            self._IMAGE_SIZE = target_size

        # Initialize transforms
        self.image_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8, scale=True),
                transforms.Resize(min(self._IMAGE_SIZE), antialias=True),
                transforms.CenterCrop(self._IMAGE_SIZE),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.panoptic_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.Resize(min(self._IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(self._IMAGE_SIZE),
            ]
        )

        # Get all sequence directories
        self.img_dir = self.root / "images" / split
        self.sequences = sorted([d for d in self.img_dir.iterdir() if d.is_dir()])

        # Build frame index with stride
        self.frame_index = []
        for seq_path in self.sequences:
            seq_frames = sorted([f for f in seq_path.iterdir() if f.suffix == ".png"])
            seq_id = int(seq_path.name)

            # Calculate effective sequence length with stride
            total_frames_needed = (window_size - 1) * frame_stride + 1

            # For each valid window in the sequence
            for i in range(0, len(seq_frames) - total_frames_needed + 1):
                # Get frame indices with stride
                frame_indices = range(i, i + total_frames_needed, frame_stride)
                frame_ids = [int(seq_frames[j].stem) for j in frame_indices]
                self.frame_index.append((seq_id, frame_ids))

        self.frames_paths = []
        self.masks_paths = [] if split != "test" else None
        self.depth_paths = [] if pseudo_depth is not None else None
        for seq_id, frame_ids in self.frame_index:
            frames_paths = []
            masks_paths = []
            depth_paths = []
            for frame_id in frame_ids:
                img_path = self.img_dir / f"{seq_id:04d}" / f"{frame_id:06d}.png"
                frames_paths.append(img_path)
                if self.split != "test":
                    pan_path = self.root / "panoptic_maps" / self.split / f"{seq_id:04d}" / f"{frame_id:06d}.png"
                    masks_paths.append(pan_path)
                if pseudo_depth is not None:
                    filename = self.get_unique_identifier_from_path(img_path)
                    filename = filename[: filename.rfind(".")] + "_depth.png"
                    depth_paths.append(os.path.join(pseudo_depth, filename))
            self.frames_paths.append(frames_paths)
            if self.split != "test":
                self.masks_paths.append(masks_paths)
            if pseudo_depth is not None:
                self.depth_paths.append(depth_paths)

        super().__init__(
            windows_paths=self.frames_paths,
            masks_paths=self.masks_paths,
            depth_paths=self.depth_paths,
            target_size=self._IMAGE_SIZE,
            **kwargs,
        )

    def load_rgb_image(self, file_name: str) -> Tensor:
        if self.load_and_transform is not None:
            return self.load_and_transform(file_name)
        img = Image.open(file_name).convert("RGB")
        return self.image_transforms(img)  # Apply transforms to normalize properly

    def load_masks(self, file_name: str) -> Tensor:
        pan_map = self.panoptic_transforms(Image.open(file_name))
        sem_map, inst_map = decode_panoptic_map(pan_map)
        return sem_map.unsqueeze(0)

    def load_depthMaps(self, file_name: str) -> Tensor:
        # as we are using pseudo depth maps
        # they are already at the correct resolution
        return load_depthMaps(file_name, 0, 1.0, self.target_size)

    def get_unique_identifier_from_path(self, path: str) -> str:
        id_ = f"{os.path.basename(os.path.dirname(path))}_{os.path.basename(path)}"
        return id_


if __name__ == "__main__":
    # from torch.utils.data import DataLoader
    # from tqdm import tqdm

    train_dataset = KITTIDataset(
        Path("/datasets_local/KITTI_STEP").expanduser(),
        split="train",
        window_size=30,
        frame_stride=1,
    )

    val_dataset = KITTIDataset(
        Path("/datasets_local/KITTI_STEP").expanduser(),
        split="val",
        window_size=30,
        frame_stride=1,
    )

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    # all_used_paths = []
    # for dts in [train_dataset, val_dataset]:
    #     flatten = lambda l: [item for sublist in l for item in sublist]
    #     all_used_paths.extend(flatten(dts.frames_paths))
    #     all_used_paths.extend(flatten(dts.masks_paths))
    # with open('tmp/kitti_paths.txt', 'w') as f:
    #     for pth in all_used_paths:
    #         pth = str(pth)
    #         pth = pth.replace(os.path.expanduser('~/scania/datasets_scania_raw/'), '')
    #         f.write(f"{pth}\n")
    # # used to rsync with:
    # # rsync --stats -aviur --files-from=tmp/kitti_paths.txt ~/scania/datasets_scania_raw/ jz:xxx

    # uniques = set()
    # print(len(train_dataset))
    # # print(train_dataset[0]['image'].shape)

    # loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=8)

    # for batch in tqdm(loader):
    #     uniques.update(set(batch['mask'].view(-1).unique().tolist()))

    # print(f"Unique classes: {sorted(uniques)}")
    # print(f"Number of unique classes: {len(uniques)}")
