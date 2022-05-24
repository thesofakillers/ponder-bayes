"""Various custom metrics"""
from typing import Optional

import torch


def accuracy(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    threshold: Optional[float] = None,
    dim: Optional[int] = None,
):

    if threshold is None:
        if dim is None:
            return (y_pred == y_true).double().mean()
        else:
            return (y_pred == y_true).double().mean(dim=dim)
    else:
        if dim is None:
            return ((y_pred > threshold) == y_true).double().mean()
        else:
            return ((y_pred > threshold) == y_true).double().mean(dim=dim)
