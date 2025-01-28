import os
from typing import List, Optional, Tuple

from torch import Tensor

from vam.evaluation.datasets.base_dataset import GenericDataset, load_depthMaps


class CityscapesDataset(GenericDataset):
    _NUM_CLASSES = 19
    _TOP_CROP_SIZE = 16
    _RESIZE_FACTOR = 3.5
    _IMAGE_SIZE = (288, 512)

    def __init__(
        self,
        root: str = "cityscapes",
        split: str = "train",
        pseudo_depth: Optional[str] = None,
        top_crop_size: Optional[int] = None,
        resize_factor: Optional[float] = None,
        target_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> None:
        self.root = root
        self.split = split
        self.files = {}

        if pseudo_depth is not None:
            pseudo_depth = os.path.join(self.root, "cityscapes_depth", self.split)

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix="leftImg8bit.png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self._NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

        all_img_paths = []
        all_mask_paths = []
        all_depth_paths = [] if pseudo_depth is not None else None
        for pth in self.files[split]:
            img_path = pth.rstrip()
            mask_path = os.path.join(
                self.annotations_base, img_path.split(os.sep)[-2], os.path.basename(img_path)[:-15] + "gtFine_labelIds.png"
            )
            all_img_paths.append(pth)
            all_mask_paths.append(mask_path)
            if pseudo_depth is not None:
                filename = self.get_unique_identifier_from_path(img_path)
                filename = filename[: filename.rfind(".")] + "_depth.png"
                all_depth_paths.append(os.path.join(pseudo_depth, filename))

        super().__init__(
            image_paths=all_img_paths,
            masks_paths=all_mask_paths,
            depth_paths=all_depth_paths,
            top_crop_size=top_crop_size or self._TOP_CROP_SIZE,
            resize_factor=resize_factor or self._RESIZE_FACTOR,
            target_size=target_size or self._IMAGE_SIZE,
            **kwargs,
        )

    def load_masks(self, file_name: str) -> Tensor:
        mask = super().load_masks(file_name)
        mask = self.encode_segmap(mask)
        return mask

    def load_depthMaps(self, file_name: str) -> Tensor:
        # as we are using pseudo depth maps
        # they are already at the correct resolution
        return load_depthMaps(file_name, 0, 1.0, self.target_size)

    def __len__(self) -> int:
        return len(self.files[self.split])

    def encode_segmap(self, mask: Tensor) -> Tensor:
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir: str = ".", suffix: str = "") -> List[str]:
        """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
        """
        return [
            os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames
            if filename.endswith(suffix)
        ]
