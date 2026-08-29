"""Anchor-free Module 1 detection/classification heads.

paper-specified:
    Module 1 performs meniscus detection, medial/lateral classification,
    normal/abnormal classification, and abnormality localization.

paper-inferred implementation:
    The manuscript does not publish the exact detection head. This file uses a
    token-level anchor-free head: every sagittal/coronal patch token predicts
    objectness, a normalized image-plane ``xyxy`` box, meniscus side, and
    abnormality class. This is not a RetinaNet baseline and does not implement
    MeniscoViT-SiP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PlaneTokenInfo:
    """Metadata for mapping flattened token predictions back to planes."""

    sagittal_tokens: int
    coronal_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.sagittal_tokens + self.coronal_tokens


class TokenDetectionTower(nn.Module):
    """Small MLP tower shared by token-level prediction heads."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class MeniscusDetectionHeads(nn.Module):
    """Anchor-free token heads for MeniscoViT-CoP Module 1.

    Input:
        Either a backbone output dict from ``MeniscoViTCoP`` or a tensor
        ``[B, 2*T*N, C]``.

    Output dict:
        ``objectness_logits``: ``[B, 2*T*N]``.
        ``bbox_quality_logits``: optional ``[B, 2*T*N]`` localization-quality score.
        ``bbox_pred``: ``[B, 2*T*N, 4]`` normalized ``xyxy`` in ``[0,1]``.
        ``side_logits``: ``[B, 2*T*N, 2]`` for medial/lateral.
        ``abnormality_logits``: ``[B, 2*T*N, 2]`` for normal/abnormal.
        ``plane_ids``: ``[2*T*N]`` where 0=sagittal, 1=coronal.
        ``shape_metadata``: backbone shape metadata when available.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        bbox_param: str = "xyxy",
        use_bbox_quality_head: bool = False,
    ):
        super().__init__()
        if bbox_param not in {"xyxy", "ltrb_from_token_center"}:
            raise ValueError("bbox_param must be xyxy or ltrb_from_token_center")
        hidden_dim = hidden_dim or embed_dim
        self.bbox_param = bbox_param
        self.use_bbox_quality_head = bool(use_bbox_quality_head)
        self.tower = TokenDetectionTower(embed_dim, hidden_dim, dropout)
        self.objectness = nn.Linear(hidden_dim, 1)
        self.bbox_quality = nn.Linear(hidden_dim, 1) if self.use_bbox_quality_head else None
        self.bbox = nn.Linear(hidden_dim, 4)
        self.side = nn.Linear(hidden_dim, 2)
        self.abnormality = nn.Linear(hidden_dim, 2)

    @staticmethod
    def _plane_ids(sagittal_tokens: int, coronal_tokens: int, device: torch.device) -> torch.Tensor:
        return torch.cat(
            [
                torch.zeros(sagittal_tokens, dtype=torch.long, device=device),
                torch.ones(coronal_tokens, dtype=torch.long, device=device),
            ],
            dim=0,
        )

    @staticmethod
    def _grid_centers(shape: dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rows = (torch.arange(int(shape["grid_height"]), device=device, dtype=dtype) + 0.5) / float(shape["grid_height"])
        cols = (torch.arange(int(shape["grid_width"]), device=device, dtype=dtype) + 0.5) / float(shape["grid_width"])
        y, x = torch.meshgrid(rows, cols, indexing="ij")
        centers_2d = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)
        return centers_2d.repeat(int(shape["slices"]), 1)

    @classmethod
    def _token_centers(cls, shape_metadata: dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if "sagittal" not in shape_metadata or "coronal" not in shape_metadata:
            raise ValueError("bbox_param=ltrb_from_token_center requires sagittal/coronal shape_metadata")
        sagittal_centers = cls._grid_centers(shape_metadata["sagittal"], device=device, dtype=dtype)
        coronal_centers = cls._grid_centers(shape_metadata["coronal"], device=device, dtype=dtype)
        return torch.cat([sagittal_centers, coronal_centers], dim=0)

    def _decode_bbox(self, raw_boxes: torch.Tensor, shape_metadata: dict[str, Any]) -> torch.Tensor:
        if self.bbox_param == "xyxy":
            raw_boxes = raw_boxes.sigmoid()
            x1y1 = torch.minimum(raw_boxes[..., :2], raw_boxes[..., 2:])
            x2y2 = torch.maximum(raw_boxes[..., :2], raw_boxes[..., 2:])
            return torch.cat([x1y1, x2y2], dim=-1)

        # Predict positive distances in feature-grid-cell units (FCOS style).
        # The previous unscaled softplus initialized offsets near 0.69 of the
        # full image and immediately clamped most ltrb boxes to image borders.
        offsets = F.softplus(raw_boxes)
        centers = self._token_centers(shape_metadata, device=raw_boxes.device, dtype=raw_boxes.dtype)
        if centers.shape[0] != raw_boxes.shape[1]:
            raise ValueError(f"token center count {centers.shape[0]} does not match bbox token count {raw_boxes.shape[1]}")
        sagittal_shape = shape_metadata["sagittal"]
        coronal_shape = shape_metadata["coronal"]
        scale_values = []
        for shape in (sagittal_shape, coronal_shape):
            one = raw_boxes.new_tensor(
                [
                    1.0 / float(shape["grid_width"]),
                    1.0 / float(shape["grid_height"]),
                    1.0 / float(shape["grid_width"]),
                    1.0 / float(shape["grid_height"]),
                ]
            )
            scale_values.append(one.expand(int(shape["slices"] * shape["patches_per_slice"]), 4))
        offsets = offsets * torch.cat(scale_values, dim=0).unsqueeze(0)
        centers = centers.unsqueeze(0)
        x1 = centers[..., 0] - offsets[..., 0]
        y1 = centers[..., 1] - offsets[..., 1]
        x2 = centers[..., 0] + offsets[..., 2]
        y2 = centers[..., 1] + offsets[..., 3]
        return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)

    def forward(self, features: torch.Tensor | dict[str, torch.Tensor | dict[str, Any]]) -> dict[str, torch.Tensor | dict[str, Any]]:
        if isinstance(features, dict):
            sagittal_tokens = features["sagittal_tokens_fused"]
            coronal_tokens = features["coronal_tokens_fused"]
            if not isinstance(sagittal_tokens, torch.Tensor) or not isinstance(coronal_tokens, torch.Tensor):
                raise TypeError("backbone output must contain tensor sagittal/coronal tokens")
            tokens = torch.cat([sagittal_tokens, coronal_tokens], dim=1)
            shape_metadata = features.get("shape_metadata", {})
            plane_info = PlaneTokenInfo(sagittal_tokens=sagittal_tokens.shape[1], coronal_tokens=coronal_tokens.shape[1])
        else:
            tokens = features
            shape_metadata = {}
            half = tokens.shape[1] // 2
            plane_info = PlaneTokenInfo(sagittal_tokens=half, coronal_tokens=tokens.shape[1] - half)

        hidden = self.tower(tokens)
        bbox_pred = self._decode_bbox(self.bbox(hidden), shape_metadata)
        outputs = {
            "objectness_logits": self.objectness(hidden).squeeze(-1),
            "bbox_pred": bbox_pred,
            "side_logits": self.side(hidden),
            "abnormality_logits": self.abnormality(hidden),
            "plane_ids": self._plane_ids(plane_info.sagittal_tokens, plane_info.coronal_tokens, tokens.device),
            "shape_metadata": shape_metadata,
        }
        if self.bbox_quality is not None:
            outputs["bbox_quality_logits"] = self.bbox_quality(hidden).squeeze(-1)
        return outputs


class MeniscoViTCoPModule1(nn.Module):
    """Backbone plus paper-inferred anchor-free Module 1 heads."""

    def __init__(self, backbone: nn.Module, heads: MeniscusDetectionHeads):
        super().__init__()
        self.backbone = backbone
        self.heads = heads

    def forward(
        self,
        sagittal_volume: torch.Tensor,
        coronal_volume: torch.Tensor,
        sagittal_coords: torch.Tensor,
        coronal_coords: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        backbone_outputs = self.backbone(sagittal_volume, coronal_volume, sagittal_coords, coronal_coords)
        head_outputs = self.heads(backbone_outputs)
        head_outputs["backbone_outputs"] = backbone_outputs
        return head_outputs
