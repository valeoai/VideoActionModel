"""
Copy-pasted from the following source:
https://github.com/valeoai/bravo_challenge/blob/main/bravo_toolkit/eval/metrics.py
"""

from typing import Optional

import numpy as np
import torch
from torch import Tensor


def batched_bincount(x: Tensor, max_value: int, dim: int = -1) -> Tensor:
    # adapted from
    # https://discuss.pytorch.org/t/batched-bincount/72819/3
    shape = (len(x), max_value)
    target = torch.zeros(*shape, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


def fast_cm_torch(
    y_true: Tensor, y_pred: Tensor, n: int, do_check: bool = True, invalid_value: Optional[int] = None
) -> Tensor:
    """
    Fast computation of a confusion matrix from two arrays of labels.

    Args:
        y_true  (Tensor): array of true labels
        y_pred (Tensor): array of predicted labels
        n (int): number of classes

    Returns:
        Tensor: confusion matrix, where rows are true labels and columns are predicted labels
    """
    y_true = y_true.flatten(start_dim=1).long()
    y_pred = y_pred.flatten(start_dim=1).long()

    if do_check:
        k = (y_true < 0) | (y_true > n) | (y_pred < 0) | (y_pred > n)
        if torch.any(k):
            raise ValueError(
                f"Invalid class values in ground-truth or prediction: {torch.unique(torch.cat((y_true[k], y_pred[k])))}"
            )

    # Convert class numbers into indices of a simulated 2D array of shape (n, n) flattened into 1D, row-major
    effective_indices = n * y_true + y_pred
    max_value = n**2
    if invalid_value is not None:
        max_value = n**2 + 1
        effective_indices[y_true == invalid_value] = n**2
    # Count the occurrences of each index, reshaping the 1D array into a 2D array
    return batched_bincount(effective_indices, max_value)[..., : n**2].view(-1, n, n)


def per_class_iou_torch(cm: Tensor) -> Tensor:
    """'
    Compute the Intersection over Union (IoU) for each class from a confusion matrix.

    Args:
        cm (Tensor): n x n 2D confusion matrix (the orientation is not important, as the formula is symmetric)

    Returns:
        Tensor: 1D array of IoU values for each of the n classes
    """
    # The diagonal contains the intersection of predicted and true labels
    # The sum of rows (columns) is the union of predicted (true) labels (or vice-versa, depending on the orientation)
    return torch.diagonal(cm, dim1=1, dim2=2) / (cm.sum(2) + cm.sum(1) - torch.diagonal(cm, dim1=1, dim2=2))


def fast_cm(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    """
    Fast computation of a confusion matrix from two arrays of labels.

    Args:
        y_true  (np.ndarray): array of true labels
        y_pred (np.ndarray): array of predicted labels
        n (int): number of classes

    Returns:
        np.ndarray: confusion matrix, where rows are true labels and columns are predicted labels
    """
    y_true = y_true.ravel().astype(int)
    y_pred = y_pred.ravel().astype(int)
    k = (y_true < 0) | (y_true > n) | (y_pred < 0) | (y_pred > n)
    if np.any(k):
        raise ValueError(
            "Invalid class values in ground-truth or prediction: " f"{np.unique(np.concatenate((y_true[k], y_pred[k])))}"
        )
    # Convert class numbers into indices of a simulated 2D array of shape (n, n) flattened into 1D, row-major
    effective_indices = n * y_true + y_pred
    # Count the occurrences of each index, reshaping the 1D array into a 2D array
    return np.bincount(effective_indices, minlength=n**2).reshape(n, n)


def per_class_iou(cm: np.ndarray) -> np.ndarray:
    """'
    Compute the Intersection over Union (IoU) for each class from a confusion matrix.

    Args:
        cm (np.ndarray): n x n 2D confusion matrix (the orientation is not important, as the formula is symmetric)

    Returns:
        np.ndarray: 1D array of IoU values for each of the n classes
    """
    # The diagonal contains the intersection of predicted and true labels
    # The sum of rows (columns) is the union of predicted (true) labels (or vice-versa, depending on the orientation)
    return np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm))
