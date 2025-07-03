import os

import cv2
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image
from pyquaternion import Quaternion

# Import the transform
from vam.datalib.transforms import CropAndResizeTransform


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension


def get_global_pose(rec, nusc, inverse=False):
    lidar_sample_data = nusc.get("sample_data", rec["data"]["LIDAR_TOP"])

    sd_ep = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
    if inverse is False:
        global_from_ego = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
        ego_from_sensor = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
    else:
        sensor_from_ego = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True)
        ego_from_global = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)
    return pose


class FuturePredictionDataset(torch.utils.data.Dataset):
    SAMPLE_INTERVAL = 0.5  # SECOND

    def __init__(self, nusc, is_train, cfg):
        self.nusc = nusc
        self.dataroot = self.nusc.dataroot
        self.is_train = is_train
        self.cfg = cfg

        if self.is_train == 0:
            self.mode = "train"
        elif self.is_train == 1:
            self.mode = "val"
        elif self.is_train == 2:
            self.mode = "test"
        else:
            raise NotImplementedError

        self.sequence_length = cfg.TIME_RECEPTIVE_FIELD + cfg.N_FUTURE_FRAMES
        self.receptive_field = cfg.TIME_RECEPTIVE_FIELD

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()
        self.indices = self.get_indices()

        # Initialize the transform for image processing
        self.image_transform = CropAndResizeTransform(resize_factor=3.125)

        # Bird's-eye view parameters
        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
        )
        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution.numpy(),
            bev_start_position.numpy(),
            bev_dimension.numpy(),
        )

    def get_scenes(self):
        # filter by scene split
        split = {
            "v1.0-trainval": {0: "train", 1: "val", 2: "test"},
            "v1.0-mini": {0: "mini_train", 1: "mini_val"},
        }[
            self.nusc.version
        ][self.is_train]

        scenes = create_splits_scenes()[split][:]
        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get("scene", samp["scene_token"])["name"] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))

        return samples

    def get_indices(self):
        indices = []
        for index in range(len(self.ixes)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.sequence_length):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.ixes):
                    is_valid_data = False
                    break
                rec = self.ixes[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec["scene_token"] != previous_rec["scene_token"]):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices)

    def get_input_data(self, rec):
        """
        Parameters
        ----------
            rec: nuscenes identifier for a given timestamp

        Returns
        -------
            images: torch.Tensor<float> (N, 3, H, W)
        """
        images = []
        cameras = self.cfg.IMAGE.NAMES

        for cam in cameras:
            camera_sample = self.nusc.get("sample_data", rec["data"][cam])

            # Load image
            image_filename = os.path.join(self.dataroot, camera_sample["filename"])
            img = Image.open(image_filename)

            # Apply the transform
            transformed_img = self.image_transform(img)

            images.append(transformed_img.unsqueeze(0).unsqueeze(0))

        images = torch.cat(images, dim=1)
        return images

    def _get_top_lidar_pose(self, rec):
        egopose = self.nusc.get("ego_pose", self.nusc.get("sample_data", rec["data"]["LIDAR_TOP"])["ego_pose_token"])
        trans = -np.array(egopose["translation"])
        yaw = Quaternion(egopose["rotation"]).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
        return trans, rot

    def get_birds_eye_view_label(self, rec, in_pred):
        translation, rotation = self._get_top_lidar_pose(rec)
        segmentation = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))

        for annotation_token in rec["anns"]:
            # Filter out all non vehicle instances
            annotation = self.nusc.get("sample_annotation", annotation_token)

            if self.cfg.DATASET.FILTER_INVISIBLE_VEHICLES and int(annotation["visibility_token"]) == 1 and in_pred is False:
                continue

            # NuScenes filter
            if "vehicle" in annotation["category_name"]:
                poly_region, z = self._get_poly_region_in_image(annotation, translation, rotation)
                cv2.fillPoly(segmentation, [poly_region], 1.0)

        return segmentation

    def _get_poly_region_in_image(self, instance_annotation, ego_translation, ego_rotation):
        box = Box(instance_annotation["translation"], instance_annotation["size"], Quaternion(instance_annotation["rotation"]))
        box.translate(ego_translation)
        box.rotate(ego_rotation)

        pts = box.bottom_corners()[:2].T
        pts = np.round((pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(
            np.int32
        )
        pts[:, [1, 0]] = pts[:, [0, 1]]

        z = box.bottom_corners()[2, 0]
        return pts, z

    def get_label(self, rec, in_pred):
        segmentation_np = self.get_birds_eye_view_label(rec, in_pred)
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0).unsqueeze(0)

        return segmentation

    def get_gt_trajectory(self, rec, ref_index):
        n_output = self.cfg.N_FUTURE_FRAMES
        gt_trajectory = np.zeros((n_output + 1, 3), np.float64)

        egopose_cur = get_global_pose(rec, self.nusc, inverse=True)

        for i in range(n_output + 1):
            index = ref_index + i
            if index < len(self.ixes):
                rec_future = self.ixes[index]

                egopose_future = get_global_pose(rec_future, self.nusc, inverse=False)

                egopose_future = egopose_cur.dot(egopose_future)
                theta = quaternion_yaw(Quaternion(matrix=egopose_future))

                origin = np.array(egopose_future[:3, 3])

                gt_trajectory[i, :] = [origin[0], origin[1], theta]

        if gt_trajectory[-1][0] >= 2:
            command = "RIGHT"
        elif gt_trajectory[-1][0] <= -2:
            command = "LEFT"
        else:
            command = "FORWARD"

        return gt_trajectory, command

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """
        Returns
        -------
            data: dict with the following keys:
                image: torch.Tensor<float> (T, N, 3, H, W)
                    normalised cameras images with T the sequence length, and N the number of cameras.
                segmentation: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                    (H_bev, W_bev) are the pixel dimensions in bird's-eye view.
                gt_trajectory: torch.Tensor<float> (n_output + 1, 3)
                    Ground truth trajectory
        """
        data = {}
        keys = [
            "image",
            "segmentation",
            "gt_trajectory",
        ]
        for key in keys:
            data[key] = []

        # Loop over all the frames in the sequence.
        for i, index_t in enumerate(self.indices[index]):
            if i >= self.receptive_field:
                in_pred = True
            else:
                in_pred = False
            rec = self.ixes[index_t]

            if i < self.receptive_field:
                images = self.get_input_data(rec)
                data["image"].append(images)

            segmentation = self.get_label(rec, in_pred)
            data["segmentation"].append(segmentation)

            if i == self.cfg.TIME_RECEPTIVE_FIELD - 1:
                gt_trajectory, command = self.get_gt_trajectory(rec, index_t)
                # !
                gt_trajectory[..., 0] = -gt_trajectory[..., 0]  # Invert x-axis for BEV
                # gt_trajectory[..., [0, 1]] = gt_trajectory[..., [1, 0]]

                data["gt_trajectory"] = torch.from_numpy(gt_trajectory).float()
                data["command"] = command

        for key, value in data.items():
            if key in ["image", "segmentation"]:
                data[key] = torch.cat(value, dim=0)

        return data
