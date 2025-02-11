from typing import Dict

import torch
import torch.distributed as dist
from torch import Tensor


def _compute_errors(gt: Tensor, pred: Tensor, reduce: str = "sum") -> Dict[str, Tensor]:
    assert reduce in ["sum", "none"], f"reduce must be 'sum' or 'none', got {reduce}"
    assert gt.shape == pred.shape, f"gt and pred must have the same shape, got {gt.shape} and {pred.shape}"

    thresh = torch.maximum(gt / pred, pred / gt)

    # this is to reduce over all pixels but not the batch dimension
    reduction = list(range(1, gt.ndim))

    d1 = (thresh < 1.25).float()
    d2 = (thresh < 1.25**2).float()
    d3 = (thresh < 1.25**3).float()

    gt_safe = torch.where(gt == 0, 1e-6, gt)
    abs_rel = torch.abs(gt - pred) / gt_safe
    sq_rel = ((gt - pred) ** 2) / gt_safe

    # we need to take the square root of the mean of the squared errors
    rmse = (gt - pred) ** 2

    logs = {
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
    }

    if reduce == "sum":
        reduce_fn = torch.sum
    elif reduce == "none":
        reduce_fn = lambda x: x  # noqa: E731

    return {k: reduce_fn(v.mean(dim=reduction)) for k, v in logs.items()}


class DepthMetrics:
    """
    Class to compute depth metrics online.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.logs = {
            "d1": torch.tensor(0.0, device=device),
            "d2": torch.tensor(0.0, device=device),
            "d3": torch.tensor(0.0, device=device),
            "abs_rel": torch.tensor(0.0, device=device),
            "sq_rel": torch.tensor(0.0, device=device),
            "rmse": torch.tensor(0.0, device=device),
        }
        self.count = torch.tensor(0, device=device)

    def update(self, gt: Tensor, pred: Tensor) -> None:
        logs = _compute_errors(gt, pred)
        for k in self.logs.keys():
            self.logs[k] += logs[k]
        self.count += len(gt)

    def compute(self) -> Dict[str, float]:
        for k in self.logs.keys():
            self.logs[k] /= self.count
        self.logs["rmse"] = torch.sqrt(self.logs["rmse"])
        return {k: v.item() for k, v in self.logs.items()}

    def dist_all_reduce(self) -> None:
        dist.barrier()
        for k in self.logs.keys():
            dist.all_reduce(self.logs[k], op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.count, op=torch.distributed.ReduceOp.SUM)


def compute_depth_metrics(gt: Tensor, pred: Tensor, reduce: str = "sum") -> Dict[str, float]:
    """
    Compute depth metrics offline.
    """
    logs = _compute_errors(gt, pred, reduce=reduce)
    if reduce == "sum":
        logs = {k: v.item() / len(gt) for k, v in logs.items()}
    logs["rmse"] = torch.sqrt(logs["rmse"])
    return logs
