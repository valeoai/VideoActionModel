import torch
from einops import rearrange
from torch import Tensor


def min_ade(pred_trajectory: Tensor, gt_trajectory: Tensor) -> Tensor:
    assert pred_trajectory.ndim == 4, "Prediction tensor must have 4 dimensions (bs, modes, future_len, 2)"
    assert gt_trajectory.ndim == 3, "Ground truth tensor must have 3 dimensions (bs, future_len, 2)"
    assert len(pred_trajectory) == len(gt_trajectory), "Batch sizes must match"
    assert pred_trajectory.shape[-1] == 2, "Last dimension of prediction tensor must be 2"
    assert gt_trajectory.shape[-1] == 2, "Last dimension of ground truth tensor must be 2"
    assert pred_trajectory.shape[2] == gt_trajectory.shape[1], "Future lengths must match"
    gt_trajectory = rearrange(gt_trajectory, "b t a -> b 1 t a")  # [bs, 1, future_len, 2]
    # Calculate ADE losses
    ade_diff = torch.norm(pred_trajectory - gt_trajectory, 2, dim=-1)  # [bs, modes, future_len]
    ade_losses = ade_diff.mean(-1)  # [bs, modes]
    ade_losses, _ = ade_losses.min(-1)  # [bs]
    minade = ade_losses.mean()  # scalar

    return minade


if __name__ == "__main__":
    pred_trajectory = torch.randn(2, 3, 5, 2)
    gt_trajectory = torch.randn(2, 5, 2)
    minade = min_ade(pred_trajectory, gt_trajectory)
    print(minade)
