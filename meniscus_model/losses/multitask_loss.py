"""Classification losses for ROI modules."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .focal_loss import FocalLoss


class ROIClassificationLoss(nn.Module):
    """Loss for Module 2 normal / degeneration / tear classification.

    Args:
        loss_type: ``"focal"`` or ``"weighted_ce"``/``"ce"``.
        class_weights: Optional class weights for CE or focal alpha.
        focal_gamma: Focal gamma.
    """

    def __init__(self, loss_type: str = "focal", class_weights: list[float] | torch.Tensor | None = None, focal_gamma: float = 2.0):
        super().__init__()
        if loss_type not in {"focal", "weighted_ce", "ce"}:
            raise ValueError("loss_type must be focal, weighted_ce, or ce")
        self.loss_type = loss_type
        weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", weights)
        self.focal = FocalLoss(gamma=focal_gamma, alpha=weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        targets = targets.long()
        if self.loss_type == "focal":
            loss = self.focal(logits, targets)
        elif self.loss_type == "weighted_ce":
            loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, targets)
        return {"loss": loss, "loss_cls": loss.detach()}


# Backward-compatible name for older imports; Module 1 now uses detection_loss.
MeniscoViTMultiTaskLoss = ROIClassificationLoss
