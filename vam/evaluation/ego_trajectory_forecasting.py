from typing import Tuple

import torch
from einops import rearrange
from torch import Tensor


def _reduce(tensor: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return tensor.mean()
    elif reduction == "sum":
        return tensor.sum()
    elif reduction == "none":
        return tensor
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def min_ade(
    pred_trajectory: Tensor, gt_trajectory: Tensor, return_idx: bool = False, reduction: str = "mean"
) -> Tensor | Tuple[Tensor, Tensor]:
    assert (
        pred_trajectory.ndim == 4
    ), f"Prediction tensor must have 4 dimensions (bs, modes, future_len, 2), got {pred_trajectory.ndim}"
    assert gt_trajectory.ndim == 3, f"Ground truth tensor must have 3 dimensions (bs, future_len, 2), got {gt_trajectory.ndim}"
    assert len(pred_trajectory) == len(
        gt_trajectory
    ), f"Batch sizes must match, got {len(pred_trajectory)} and {len(gt_trajectory)}"
    assert pred_trajectory.shape[-1] == 2, f"Last dimension of prediction tensor must be 2, got {pred_trajectory.shape[-1]}"
    assert gt_trajectory.shape[-1] == 2, f"Last dimension of ground truth tensor must be 2, got {gt_trajectory.shape[-1]}"
    assert (
        pred_trajectory.shape[2] == gt_trajectory.shape[1]
    ), f"Future lengths must match, got {pred_trajectory.shape[2]} and {gt_trajectory.shape[1]}"
    gt_trajectory = rearrange(gt_trajectory, "b t a -> b 1 t a")  # [bs, 1, future_len, 2]
    # Calculate ADE losses
    ade_diff = torch.norm(pred_trajectory - gt_trajectory, 2, dim=-1)  # [bs, modes, future_len]
    ade_losses = ade_diff.mean(-1)  # [bs, modes]
    ade_losses, ade_indices = ade_losses.min(-1)  # [bs]

    ade_losses = _reduce(ade_losses, reduction)

    if return_idx:
        return ade_losses, ade_indices

    return ade_losses


if __name__ == "__main__":
    pred_trajectory = torch.randn(2, 3, 5, 2)
    gt_trajectory = torch.randn(2, 5, 2)
    minade = min_ade(pred_trajectory, gt_trajectory)
    print(minade)
