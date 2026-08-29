"""Focal losses for imbalanced Module 1 detection targets."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Multi-class focal loss for logits shaped ``[..., C]``."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | list[float] | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        alpha_tensor = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha_tensor)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[-1]
        flat_logits = logits.reshape(-1, num_classes)
        flat_targets = targets.reshape(-1).long()
        ce = F.cross_entropy(flat_logits, flat_targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = (1.0 - pt).pow(self.gamma) * ce
        return _reduce(loss, self.reduction, targets.shape)


class BinaryFocalLossWithLogits(nn.Module):
    """Binary focal loss for objectness logits shaped ``[...]``."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.elementwise_loss(logits, targets)
        return _reduce(loss, self.reduction, targets.shape)

    def elementwise_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(targets == 1, prob, 1 - prob)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return alpha_t * (1.0 - pt).pow(self.gamma) * bce


def _reduce(loss: torch.Tensor, reduction: str, target_shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss.reshape(target_shape)
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")
