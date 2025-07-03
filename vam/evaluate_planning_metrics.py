#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuscenes.nuscenes import NuScenes
from pytorch_lightning.metrics.metric import Metric
from skimage.draw import polygon
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_nuscenes import FuturePredictionDataset

# Import required modules
from vam.action_expert import VideoActionModelInference, load_inference_VAM
from vam.utils import expand_path

# ---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# ---------------------------------------------------------------------------------#


class PlanningMetric(Metric):
    def __init__(
        self,
        cfg,
        n_future=4,
        compute_on_step: bool = False,
    ):
        super().__init__(compute_on_step=compute_on_step)
        dx, bx, _ = gen_dx_bx(cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND)
        dx, bx = dx[:2], bx[:2]
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)

        _, _, self.bev_dimension = calculate_birds_eye_view_parameters(cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND)
        self.bev_dimension = self.bev_dimension.numpy()

        self.W = cfg.EGO.WIDTH
        self.H = cfg.EGO.HEIGHT

        self.n_future = n_future

        self.add_state("obj_col", default=torch.zeros(self.n_future), dist_reduce_fx="sum")
        self.add_state("obj_box_col", default=torch.zeros(self.n_future), dist_reduce_fx="sum")
        self.add_state("L2", default=torch.zeros(self.n_future), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def evaluate_single_coll(self, traj, segmentation):
        """
        gt_segmentation
        traj: torch.Tensor (n_future, 2)
        segmentation: torch.Tensor (n_future, 200, 200)
        """
        pts = np.array(
            [
                [-self.H / 2.0 + 0.5, self.W / 2.0],
                [self.H / 2.0 + 0.5, self.W / 2.0],
                [self.H / 2.0 + 0.5, -self.W / 2.0],
                [-self.H / 2.0 + 0.5, -self.W / 2.0],
            ]
        )
        pts = (pts - self.bx.cpu().numpy()) / (self.dx.cpu().numpy())
        # !
        rr, cc = polygon(pts[:, 0], pts[:, 1])
        # pts[:, [0, 1]] = pts[:, [1, 0]]
        # rr, cc = polygon(pts[:, 1], pts[:, 0])
        rc = np.concatenate([rr[:, None], cc[:, None]], axis=-1)

        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)
        # !

        # trajs[:, :, [0, 1]] = trajs[:, :, [1, 0]]  # can also change original tensor
        trajs = trajs / self.dx
        trajs = trajs.cpu().numpy() + rc  # (n_future, 32, 2)

        r = trajs[:, :, 0].astype(np.int32)
        r = np.clip(r, 0, self.bev_dimension[0] - 1)

        c = trajs[:, :, 1].astype(np.int32)
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        collision = np.full(n_future, False)
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            I = np.logical_and(
                np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
                np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
            )
            collision[t] = np.any(segmentation[t, rr[I], cc[I]].cpu().numpy())

        return torch.from_numpy(collision).to(device=traj.device)

    def evaluate_coll(self, trajs, gt_trajs, segmentation):
        """
        trajs: torch.Tensor (B, n_future, 2)
        gt_trajs: torch.Tensor (B, n_future, 2)
        segmentation: torch.Tensor (B, n_future, 200, 200)
        """
        B, n_future, _ = trajs.shape
        # !

        # trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        # gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum = torch.zeros(n_future, device=segmentation.device)
        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)

        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], segmentation[i])

            xx, yy = trajs[i, :, 0], trajs[i, :, 1]
            # !
            xi = ((xx - self.bx[0]) / self.dx[0]).long()
            yi = ((yy - self.bx[1]) / self.dx[1]).long()
            # yi = ((yy - self.bx[0]) / self.dx[0]).long()
            # xi = ((xx - self.bx[1]) / self.dx[1]).long()

            m1 = torch.logical_and(
                torch.logical_and(yi >= 0, yi < self.bev_dimension[0]),
                torch.logical_and(xi >= 0, xi < self.bev_dimension[1]),
            )
            m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll))

            ti = torch.arange(n_future)
            # !
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], xi[m1], yi[m1]].long()
            # obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], yi[m1], xi[m1]].long()

            m2 = torch.logical_not(gt_box_coll)
            box_coll = self.evaluate_single_coll(trajs[i], segmentation[i])
            obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).long()

        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs):
        """
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        """

        return torch.sqrt(((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2).sum(dim=-1))

    def update(self, trajs, gt_trajs, segmentation):
        """
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        segmentation: torch.Tensor (B, n_future, 200, 200)
        """
        assert trajs.shape == gt_trajs.shape
        L2 = self.compute_L2(trajs, gt_trajs)
        obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(trajs[:, :, :2], gt_trajs[:, :, :2], segmentation)

        self.obj_col += obj_coll_sum
        self.obj_box_col += obj_box_coll_sum
        self.L2 += L2.sum(dim=0)
        self.total += len(trajs)

    def compute(self):
        return {"obj_col": self.obj_col / self.total, "obj_box_col": self.obj_box_col / self.total, "L2": self.L2 / self.total}


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


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx


class SimpleConfig:
    """Simple configuration object for the dataset"""

    def __init__(self):
        self.TIME_RECEPTIVE_FIELD = 3
        self.N_FUTURE_FRAMES = 6

        # Image configuration
        self.IMAGE = type("obj", (object,), {})()
        self.IMAGE.NAMES = ["CAM_FRONT"]
        self.IMAGE.ORIGINAL_HEIGHT = 900
        self.IMAGE.ORIGINAL_WIDTH = 1600
        self.IMAGE.FINAL_DIM = (224, 480)
        self.IMAGE.RESIZE_SCALE = 0.3
        self.IMAGE.TOP_CROP = 46

        # Lift configuration for BEV
        self.LIFT = type("obj", (object,), {})()
        self.LIFT.X_BOUND = [-50.0, 50.0, 0.5]
        self.LIFT.Y_BOUND = [-50.0, 50.0, 0.5]
        self.LIFT.Z_BOUND = [-10.0, 10.0, 20.0]

        # Dataset configuration
        self.DATASET = type("obj", (object,), {})()
        self.DATASET.FILTER_INVISIBLE_VEHICLES = True

        # EGO vehicle configuration (needed for planning metrics)
        self.EGO = type("obj", (object,), {})()
        self.EGO.WIDTH = 1.85  # meters
        self.EGO.HEIGHT = 4.084  # meters


class PlanningEvaluator:
    """Class to handle comprehensive planning evaluation over nuScenes dataset"""

    def __init__(
        self,
        nusc_dataroot: str,
        vam_checkpoint_path: str,
        tokenizer_jit_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        version: str = "v1.0-mini",
    ):
        self.device = device
        self.dtype = dtype

        # Initialize nuScenes
        self.nusc = NuScenes(version=version, dataroot=nusc_dataroot, verbose=True)

        # Create configuration
        self.cfg = SimpleConfig()

        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_jit_path}")
        self.tokenizer = torch.jit.load(expand_path(tokenizer_jit_path)).to(device)
        self.tokenizer.eval()

        # Load VAM model
        print(f"Loading VAM model from {vam_checkpoint_path}")
        self.vam = load_inference_VAM(vam_checkpoint_path, tempdir="/tmp")

        print("Models loaded successfully!")

    def create_datasets(self, train_split: bool = False, val_split: bool = True):
        """Create train and/or validation datasets"""
        datasets = {}

        if train_split:
            datasets["train"] = FuturePredictionDataset(
                nusc=self.nusc,
                is_train=0,  # training split
                cfg=self.cfg,
            )
            print(f"Created training dataset with {len(datasets['train'])} sequences")

        if val_split:
            datasets["val"] = FuturePredictionDataset(
                nusc=self.nusc,
                is_train=1,  # validation split
                cfg=self.cfg,
            )
            print(f"Created validation dataset with {len(datasets['val'])} sequences")

        return datasets

    @torch.no_grad()
    def evaluate_batch(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a single batch and return predictions, BEV data, and GT trajectories

        Returns:
            predictions: (B, N_future, 2)
            bev_data: (B, N_future, 1, H, W)
            gt_trajectories: (B, N_future, 2)
        """
        # Get data from batch
        images = batch["image"].to(self.device, non_blocking=True)  # (B, T, N, 3, H, W)
        bev_data = batch["segmentation"]  # (B, T, 1, H_bev, W_bev)
        gt_trajectories = batch["gt_trajectory"]  # (B, n_output + 1, 3)

        batch_size_actual = images.size(0)

        with torch.amp.autocast("cuda", dtype=self.dtype):
            # Reshape images for tokenization: (B, T, N, 3, H, W) -> (B*T*N, 3, H, W)
            B, T, N, C, H, W = images.shape
            assert N == 1, "Only single camera (front) is supported for now"
            to_tokenize = images.view(B * T * N, C, H, W)

            visual_tokens = self.tokenizer(to_tokenize)
            # Reshape back to (B, T*N, token_dim)
            visual_tokens = visual_tokens.unflatten(0, (B, T))

            # Generate trajectory prediction using VAM
            commands = torch.ones(batch_size_actual, 1, device=self.device, dtype=torch.long) * 2

            trajectory = self.vam(visual_tokens, commands, self.dtype, verbose=False)
            predictions = trajectory.cpu().squeeze(1)[..., :2].to(torch.float32)

            # Extract future frames only
            bev_future = bev_data[:, self.cfg.TIME_RECEPTIVE_FIELD :]
            gt_future = gt_trajectories[:, 1:, :2]

        return predictions, bev_future, gt_future

    def evaluate_single_mode(
        self, gt_trajectories: torch.Tensor, pred_trajectories: torch.Tensor, bev_data: torch.Tensor
    ) -> Dict:
        """
        Evaluate single prediction mode and return metrics at different timesteps

        Args:
            gt_trajectories: (B, N_future, 2)
            pred_trajectories: (B, N_future, 2)
            bev_data: (B, N_future, 1, H, W)

        Returns:
            dict: Metrics for the prediction at different timesteps
        """
        B, N_future, _ = pred_trajectories.shape

        # Convert GT to 3D
        gt_trajs_3d = gt_trajectories
        segmentation = bev_data.squeeze(2)  # (B, N_future, H, W)
        pred_trajs_3d = pred_trajectories

        # Compute metrics at different timesteps (following STP3 logic)
        metrics_by_timestep = {}

        # Compute for each progressive timestep (1s, 2s, 3s, etc.)
        future_second = int(N_future / 2)  # Assuming 2Hz, so N_future/2 gives seconds
        for i in range(future_second):
            cur_time = (i + 1) * 2  # 2 frames per second
            cur_time = min(cur_time, N_future)  # Don't exceed available frames

            # Initialize planning metric for this timestep
            planning_metric = PlanningMetric(self.cfg, n_future=cur_time)

            # Update and compute metrics for this timestep
            planning_metric.update(pred_trajs_3d[:, :cur_time], gt_trajs_3d[:, :cur_time], segmentation[:, :cur_time])
            results = planning_metric.compute()

            # Store results for this timestep
            timestep_key = f"{i+1}s"
            metrics_by_timestep[timestep_key] = {
                "L2": results["L2"].mean().item(),
                "obj_col": results["obj_col"].mean().item(),
                "obj_box_col": results["obj_box_col"].mean().item(),
                "L2_per_frame": results["L2"].cpu().numpy().tolist(),
                "obj_col_per_frame": results["obj_col"].cpu().numpy().tolist(),
                "obj_box_col_per_frame": results["obj_box_col"].cpu().numpy().tolist(),
            }

        # Also compute for the full trajectory
        planning_metric_full = PlanningMetric(self.cfg, n_future=N_future)
        planning_metric_full.update(pred_trajs_3d, gt_trajs_3d, segmentation)
        results_full = planning_metric_full.compute()

        metrics_by_timestep["full"] = {
            "L2_per_timestep": results_full["L2"].cpu().numpy().tolist(),
            "obj_col_per_timestep": results_full["obj_col"].cpu().numpy().tolist(),
            "obj_box_col_per_timestep": results_full["obj_box_col"].cpu().numpy().tolist(),
            "L2_mean": results_full["L2"].mean().item(),
            "obj_col_mean": results_full["obj_col"].mean().item(),
            "obj_box_col_mean": results_full["obj_box_col"].mean().item(),
        }

        return metrics_by_timestep

    def evaluate_dataset(
        self,
        dataset_name: str = "val",
        batch_size: int = 4,
        max_batches: Optional[int] = None,
        save_results: bool = True,
        results_dir: str = "evaluation_results",
    ) -> Dict:
        """
        Evaluate planning metrics over the entire dataset

        Args:
            dataset_name: "train" or "val"
            batch_size: Batch size for evaluation
            max_batches: Maximum number of batches to process (None for all)
            save_results: Whether to save results to disk
            results_dir: Directory to save results

        Returns:
            dict: Comprehensive evaluation results
        """
        # Create dataset
        datasets = self.create_datasets(train_split=(dataset_name == "train"), val_split=(dataset_name == "val"))
        dataset = datasets[dataset_name]

        # Create dataloader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

        print(f"Evaluating {len(dataset)} samples in {len(loader)} batches...")

        # Initialize global metrics for different timesteps
        future_second = int(self.cfg.N_FUTURE_FRAMES / 2)
        global_planning_metrics = {}

        # Initialize metrics for each timestep
        for i in range(future_second):
            cur_time = (i + 1) * 2
            cur_time = min(cur_time, self.cfg.N_FUTURE_FRAMES)
            global_planning_metrics[f"{i+1}s"] = PlanningMetric(self.cfg, n_future=cur_time)

        # Initialize metric for full trajectory
        global_planning_metrics["full"] = PlanningMetric(self.cfg, n_future=self.cfg.N_FUTURE_FRAMES)

        all_batch_results = []
        total_samples = 0

        # Process batches
        max_batches = max_batches or len(loader)
        if max_batches == -1:
            max_batches = len(loader)

        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating batches", total=max_batches)):
            if batch_idx >= max_batches:
                break

            # Generate predictions
            predictions, bev_data, gt_trajectories = self.evaluate_batch(batch)

            # Evaluate predictions for this batch
            batch_metrics = self.evaluate_single_mode(gt_trajectories, predictions, bev_data)
            all_batch_results.append({"batch_idx": batch_idx, "batch_size": predictions.shape[0], "metrics": batch_metrics})

            # Update global metrics for each timestep
            gt_trajs_3d = torch.cat([gt_trajectories, torch.zeros_like(gt_trajectories[:, :, :1])], dim=-1)
            segmentation = bev_data.squeeze(2)
            pred_trajs_3d = torch.cat([predictions, torch.zeros_like(predictions[:, :, :1])], dim=-1)

            # Update each timestep metric
            for i in range(future_second):
                cur_time = (i + 1) * 2
                cur_time = min(cur_time, self.cfg.N_FUTURE_FRAMES)
                timestep_key = f"{i+1}s"
                global_planning_metrics[timestep_key].update(
                    pred_trajs_3d[:, :cur_time], gt_trajs_3d[:, :cur_time], segmentation[:, :cur_time]
                )

            # Update full trajectory metric
            global_planning_metrics["full"].update(pred_trajs_3d, gt_trajs_3d, segmentation)

            total_samples += predictions.shape[0]

        # Compute final global metrics
        final_metrics = {}
        for timestep_key, planning_metric in global_planning_metrics.items():
            results = planning_metric.compute()
            if timestep_key == "full":
                final_metrics[timestep_key] = {
                    "L2_per_timestep": results["L2"].cpu().numpy().tolist(),
                    "obj_col_per_timestep": results["obj_col"].cpu().numpy().tolist(),
                    "obj_box_col_per_timestep": results["obj_box_col"].cpu().numpy().tolist(),
                    "L2_mean": results["L2"].mean().item(),
                    "obj_col_mean": results["obj_col"].mean().item(),
                    "obj_box_col_mean": results["obj_box_col"].mean().item(),
                }
            else:
                final_metrics[timestep_key] = {
                    "L2": results["L2"].mean().item(),
                    "obj_col": results["obj_col"].mean().item(),
                    "obj_box_col": results["obj_box_col"].mean().item(),
                }

        # Compile comprehensive results
        evaluation_results = {
            "dataset": dataset_name,
            "total_samples": total_samples,
            "total_batches": len(all_batch_results),
            "metrics_by_timestep": final_metrics,
            "batch_results": (
                all_batch_results if len(all_batch_results) < 50 else all_batch_results[:50]
            ),  # Limit for JSON size
        }

        # Save results
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir, f"planning_metrics_{dataset_name}.json")
            with open(results_file, "w") as f:
                json.dump(evaluation_results, f, indent=2)
            print(f"Results saved to {results_file}")

        # Print summary
        self.print_evaluation_summary(evaluation_results)

        return evaluation_results

    def print_evaluation_summary(self, results: Dict):
        """Print a summary of evaluation results"""
        print("\n" + "=" * 80)
        print(f"PLANNING METRICS EVALUATION SUMMARY - {results['dataset'].upper()} SET")
        print("=" * 80)
        print(f"Total samples evaluated: {results['total_samples']}")
        print(f"Total batches processed: {results['total_batches']}")

        metrics_by_timestep = results["metrics_by_timestep"]

        print(f"\nPerformance by Timestep:")
        for timestep_key in sorted(metrics_by_timestep.keys()):
            if timestep_key == "full":
                continue
            metrics = metrics_by_timestep[timestep_key]
            print(f"  {timestep_key}:")
            print(f"    L2 Error: {metrics['L2']:.4f} meters")
            print(f"    Object Collision Rate: {metrics['obj_col']:.4f}")
            print(f"    Object Box Collision Rate: {metrics['obj_box_col']:.4f}")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate planning metrics on nuScenes dataset")
    parser.add_argument("--nusc_dataroot", type=str, required=True, help="Path to nuScenes dataset")
    parser.add_argument("--vam_checkpoint", type=str, required=True, help="Path to VAM checkpoint")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer JIT model")
    parser.add_argument("--dataset", type=str, choices=["train", "val"], default="val", help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument(
        "--max_batches", type=int, default=None, help="Maximum number of batches to process (None or -1 for all)"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")
    parser.add_argument("--results_dir", type=str, default="evaluation_results", help="Directory to save results")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="nuScenes version")

    args = parser.parse_args()

    # Create evaluator
    evaluator = PlanningEvaluator(
        nusc_dataroot=args.nusc_dataroot,
        vam_checkpoint_path=args.vam_checkpoint,
        tokenizer_jit_path=args.tokenizer_path,
        device=args.device,
        version=args.version,
    )

    # Run evaluation
    results = evaluator.evaluate_dataset(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        results_dir=args.results_dir,
    )

    print(f"\nEvaluation completed! Results saved to {args.results_dir}")


if __name__ == "__main__":
    main()

# python evaluate_planning_metrics.py \
#     --nusc_dataroot /datasets_local/nuscenes \
#     --vam_checkpoint /home/lchambon/iveco/scratch_iveco/VAM_JZGC4/checkpoints/VAM/VAM_width_768_pretrained_139k.pt \
#     --tokenizer_path /home/lchambon/iveco/scratch_iveco/VAM_JZGC4/llamagen_jit_models/VQ_ds16_16384_llamagen_encoder.jit \
#     --dataset val \
#     --batch_size 8 \
#     --max_batches -1 \
#     --device cuda \
#     --results_dir evaluation_results \
#     --version v1.0-trainval
