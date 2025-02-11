import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


def top_crop(img: Tensor, trop_crop_size: int) -> Tensor:
    return TF.crop(img, trop_crop_size, 0, img.shape[1] - trop_crop_size, img.shape[2])


def resize_by_factor(img: Tensor, resize_factor: float, mode: str = TF.InterpolationMode.BILINEAR) -> Tensor:
    new_width = int(img.shape[2] / resize_factor)
    new_height = int(img.shape[1] / resize_factor)
    return TF.resize(img, (new_height, new_width), antialias=True, interpolation=mode)


def load_rgb_image(
    file_name: str, top_crop_size: int, resize_factor: float, target_size: Tuple[int] = None, normalize: bool = True
) -> Tensor:
    image = Image.open(file_name)
    image = TF.to_image(image)
    image = TF.to_dtype(image, torch.uint8, scale=True)
    image = top_crop(image, top_crop_size)
    image = resize_by_factor(image, resize_factor)
    if target_size is not None:
        image = TF.center_crop(image, target_size)
    image = TF.to_dtype(image, torch.float32, scale=True)
    if normalize:
        image = 2 * image - 1
    return image


def load_depthMaps(file_name: str, top_crop_size: int, resize_factor: float, target_size: Tuple[int] = None) -> Tensor:
    depth_map = load_rgb_image(file_name, top_crop_size, resize_factor, target_size, normalize=False)
    return depth_map


def load_masks(file_name: str, top_crop_size: int, resize_factor: float, target_size: Optional[Tuple[int]] = None) -> Tensor:
    mask = Image.open(file_name)
    mask = TF.to_image(mask)
    mask = TF.to_dtype(mask, torch.uint8, scale=False)
    mask = top_crop(mask, top_crop_size)
    mask = resize_by_factor(mask, resize_factor, mode=TF.InterpolationMode.NEAREST)
    if target_size is not None:
        mask = TF.center_crop(mask, target_size)
    return mask


def load_token(file_name: str) -> np.ndarray:
    tokens = np.load(file_name)
    return tokens


class GenericDataset(Dataset):

    _NUM_CLASSES = -1
    load_and_transform = None

    def __init__(
        self,
        image_paths: Optional[List[str]] = None,
        depth_paths: Optional[List[str]] = None,
        token_paths: Optional[List[str]] = None,
        masks_paths: Optional[List[str]] = None,
        top_crop_size: int = 0,
        resize_factor: float = 1.0,
        target_size: Optional[Tuple[int]] = None,
    ) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.token_paths = token_paths
        self.masks_paths = masks_paths
        self.top_crop_size = top_crop_size
        self.resize_factor = resize_factor
        self.target_size = target_size

    def load_rgb_image(self, file_name: str) -> Tensor:
        if self.load_and_transform is not None:
            return self.load_and_transform(file_name)
        return load_rgb_image(file_name, self.top_crop_size, self.resize_factor, self.target_size)

    def load_depthMaps(self, file_name: str) -> Tensor:
        return load_depthMaps(file_name, self.top_crop_size, self.resize_factor, self.target_size)

    def load_masks(self, file_name: str) -> Tensor:
        return load_masks(file_name, self.top_crop_size, self.resize_factor, self.target_size)

    def load_token(self, file_name: str) -> np.ndarray:
        return load_token(file_name)

    def get_window_size(self) -> int:
        if not hasattr(self, "window_size"):
            return 1
        return self.window_size

    def get_collate_fn(self) -> Optional[Callable]:
        return None

    def get_unique_identifier_from_path(self, path: str) -> str:
        return os.path.basename(path)

    def post_process(self, output_dict: dict) -> dict:
        return output_dict

    def __len__(self) -> int:
        if hasattr(self, "_length"):
            return self._length

        return len(self.image_paths)

    def get_num_classes(self) -> int:
        return self._NUM_CLASSES  # this includes the background class, at index 0

    def get_image_size(self) -> Tuple[int]:
        return self.target_size or self._IMAGE_SIZE

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        out_dict = {}

        if self.image_paths is not None:
            rgb_filename = self.image_paths[idx]
            out_dict["image"] = self.load_rgb_image(rgb_filename)
            out_dict["image_path"] = rgb_filename

        if self.depth_paths is not None:
            depth_filename = self.depth_paths[idx]
            out_dict["depth"] = self.load_depthMaps(depth_filename)
            out_dict["depth_path"] = depth_filename

        if self.masks_paths is not None:
            mask_filename = self.masks_paths[idx]
            out_dict["mask"] = self.load_masks(mask_filename)
            out_dict["mask_path"] = mask_filename

        if self.token_paths is not None:
            token_filename = self.token_paths[idx]
            out_dict["token"] = self.load_token(token_filename)

        return self.post_process(out_dict)


class GenericVideoDataset(GenericDataset):

    def __init__(
        self,
        windows_paths: Optional[List[List[str]]] = None,
        tokens_paths: Optional[List[List[str]]] = None,
        masks_paths: Optional[List[List[str]]] = None,
        depth_paths: Optional[List[List[str]]] = None,
        top_crop_size: int = 0,
        resize_factor: float = 1.0,
        target_size: Optional[Tuple[int]] = None,
        eval_on_last_frame: bool = False,
    ) -> None:
        super().__init__(
            top_crop_size=top_crop_size,
            resize_factor=resize_factor,
            target_size=target_size,
        )
        self.windows_paths = windows_paths
        self.tokens_paths = tokens_paths
        self.masks_paths = masks_paths
        self.depth_paths = depth_paths

        self.eval_on_last_frame = eval_on_last_frame

        self._length = len(self.windows_paths)
        self.window_size = len(self.windows_paths[0])

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        out_dict = {}

        if self.windows_paths is not None:
            frames_paths = self.windows_paths[idx]
            out_dict["image"] = torch.stack(([self.load_rgb_image(f) for f in frames_paths]), dim=0)
            out_dict["image_path"] = list(map(str, frames_paths))

        if self.depth_paths is not None:
            depth_paths = self.depth_paths[idx]
            if self.eval_on_last_frame:
                out_dict["depth"] = self.load_depthMaps(depth_paths[-1])
                out_dict["depth_path"] = [str(depth_paths[-1])]
            else:
                out_dict["depth"] = torch.stack(([self.load_depthMaps(f) for f in depth_paths]), dim=0)
                out_dict["depth_path"] = list(map(str, depth_paths))

        if self.masks_paths is not None:
            masks_paths = self.masks_paths[idx]
            if self.eval_on_last_frame:
                out_dict["mask"] = self.load_masks(masks_paths[-1])
                out_dict["mask_path"] = [str(masks_paths[-1])]
            else:
                out_dict["mask"] = torch.stack([self.load_masks(f) for f in masks_paths], dim=0)
                out_dict["mask_path"] = list(map(str, masks_paths))

        return self.post_process(out_dict)
