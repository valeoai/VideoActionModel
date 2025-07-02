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


class CenteredWidthCrop:
    def __init__(self, total_crop_size: int = 0) -> None:
        self.crop_size = total_crop_size // 2

    def __call__(self, img: Tensor) -> Tensor:
        # Crop the top of the image by the specified amount | top, left, height, width
        return TF.crop(img, 0, self.crop_size, img.shape[1], img.shape[2] - self.crop_size)


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


class ImageNetNormalize(transforms.Normalize):
    def __init__(self) -> None:
        super().__init__(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


class CropAndResizeTransform:
    """
    Wrapper for custom transform
    """

    def __init__(
        self, top_crop_size: int, resize_factor: float, width_center_crop: int = 0, imagenet_normalize: bool = False
    ) -> None:

        self.transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8, scale=True),
                TopCrop(top_crop_size),
                CenteredWidthCrop(width_center_crop),
                ResizeByFactor(resize_factor),
                transforms.ToDtype(torch.float32, scale=True),
                ImageNetNormalize() if imagenet_normalize else Normalize(),  # Normalize to [-1, 1]
            ]
        )

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.transforms(*args, **kwargs)


def torch_image_to_plot(img: Tensor, to_numpy: bool = True) -> np.ndarray:
    img = torch.clamp(127.5 * img + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8)
    if to_numpy:
        img = img.numpy()
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

    Some images send by NeuroNCAP are exactly the correct shape.
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


class DINOSafeResize:
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, img: Tensor) -> Tensor:
        height, width = img.shape[1], img.shape[2]
        if (height != self.size[0]) or (width != self.size[1]):
            img = TF.resize(img, self.size, antialias=True)
        return img


class DINONeuroNCAPTransform:
    """
    NeuroNCAP transform for nuScenes.

    Some images send by NeuroNCAP are exactly the correct shape.
    For instance:
    (900, 1599) instead of (900, 1600)
    """

    def __init__(
        self, top_crop_size: int, resize_factor: float, width_center_crop: int = 0, default_size: Tuple[int, int] = (900, 1600)
    ) -> None:

        self.transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8, scale=True),
                DINOSafeResize(default_size),
                TopCrop(top_crop_size),
                CenteredWidthCrop(width_center_crop),
                ResizeByFactor(resize_factor),
                transforms.ToDtype(torch.float32, scale=True),
                ImageNetNormalize(),  # Normalize to [-1, 1]
            ]
        )

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.transforms(*args, **kwargs)


if __name__ == "__main__":
    nuplan_default = CropAndResizeTransform(top_crop_size=30, resize_factor=3.75, width_center_crop=54)
    nuscenes_default = CropAndResizeTransform(top_crop_size=25, resize_factor=3.125, width_center_crop=44)

    nuplan_image = torch.rand(3, 1080, 1920)  # Simulated nuPlan image
    nuscenes_image = torch.rand(3, 900, 1600)  # Simulated nuScenes image

    print("NuPlan transformed shape:", nuplan_default(nuplan_image).shape)
    print("NuScenes transformed shape:", nuscenes_default(nuscenes_image).shape)
