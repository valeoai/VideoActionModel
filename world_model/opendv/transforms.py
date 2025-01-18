from typing import Tuple

import numpy as np
import torch

# https://pytorch.org/vision/0.16/transforms.html#v1-or-v2-which-one-should-i-use
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as TF
from torch import Tensor


class TopCrop:
    def __init__(self, trop_crop_size: int) -> None:
        self.trop_crop_size = trop_crop_size

    def __call__(self, img: Tensor) -> Tensor:
        # Crop the top of the image by the specified amount | top, left, height, width
        return TF.crop(img, self.trop_crop_size, 0, img.shape[1] - self.trop_crop_size, img.shape[2])


class ResizeByFactor:
    def __init__(self, resize_factor: float) -> None:
        self.resize_factor = resize_factor

    def __call__(self, img: Tensor) -> Tensor:
        new_width = int(img.shape[2] / self.resize_factor)
        new_height = int(img.shape[1] / self.resize_factor)
        return TF.resize(img, (new_height, new_width), antialias=True)


class Normalize:
    """
    Expect image input to be in [0;1] rescale in [-1;1]
    """

    def __call__(self, img: Tensor) -> Tensor:
        return 2 * img - 1


class CropAndResizeTransform:
    """
    Wrapper for custom transform
    """

    def __init__(self, trop_crop_size: int, resize_factor: float) -> None:

        self.transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8, scale=True),
                TopCrop(trop_crop_size),
                ResizeByFactor(resize_factor),
                transforms.ToDtype(torch.float32, scale=True),
                Normalize(),  # Normalize to [-1, 1]
            ]
        )

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.transforms(*args, **kwargs)


def torch_image_to_plot(img: Tensor) -> np.ndarray:
    img = torch.clamp(127.5 * img + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return img


class SafeResize:
    def __init__(self, resize_factor: float, size: Tuple[int, int]) -> None:
        self.resize_factor = resize_factor
        self.size = size

    def __call__(self, img: Tensor) -> Tensor:
        height, width = img.shape[1], img.shape[2]
        if (height != self.size[0]) or (width != self.size[1]):
            img = TF.resize(img, self.size, antialias=True)

        new_width = int(img.shape[2] / self.resize_factor)
        new_height = int(img.shape[1] / self.resize_factor)
        return TF.resize(img, (new_height, new_width), antialias=True)


class NeuroNCAPTransform:
    """
    NeuroNCAP transform for nuScenes.

    Some images send by Neuro-NCAP are exactly the correct shape.
    For instance:
    (900, 1599) instead of (900, 1600)
    """

    def __init__(self, resize_factor: float = 3.125, default_size: Tuple[int, int] = (900, 1600)) -> None:

        self.transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8, scale=True),
                SafeResize(resize_factor, default_size),
                transforms.ToDtype(torch.float32, scale=True),
                Normalize(),  # Normalize to [-1, 1]
            ]
        )

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.transforms(*args, **kwargs)
