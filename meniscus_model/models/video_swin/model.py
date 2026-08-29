"""Module 2 Video Swin classifier for meniscus ROI sequences."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .swin3d_blocks import PatchEmbed3D, PatchMerging3D, Swin3DBlock, to_3tuple


class PaperVideoSwinClassifier(nn.Module):
    """Paper Figure-10 Video Swin-T with configurable plane fusion.

    Figure 10 explicitly shows coronal/sagittal channel concatenation followed
    by one standard 3-channel Video Swin-T (27.8M parameters).  The default
    maps the two grayscale planes to ``[coronal, sagittal, mean]``.  A shared
    two-pass feature-concatenation mode is retained because the prose also says
    "resulting features ... are concatenated", enabling a controlled ablation
    without doubling parameters.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = False,
        pretrained_checkpoint: str | None = None,
        normalize_input: bool = True,
        fusion_mode: str = "input_channel_concat",
        dropout: float = 0.0,
    ):
        super().__init__()
        try:
            from torchvision.models.video import Swin3D_T_Weights, swin3d_t
        except ImportError as exc:
            raise ImportError("torchvision>=0.17 is required for the paper Video Swin-T") from exc
        if fusion_mode not in {
            "input_channel_concat",
            "input_channel_difference",
            "shared_feature_concat",
            "shared_feature_mean",
        }:
            raise ValueError("unsupported Video Swin plane fusion mode")
        weights = Swin3D_T_Weights.KINETICS400_V1 if pretrained and not pretrained_checkpoint else None
        self.backbone = swin3d_t(weights=weights, progress=True)
        if pretrained_checkpoint:
            checkpoint_path = Path(pretrained_checkpoint)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Video Swin checkpoint not found: {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.backbone.load_state_dict(state, strict=True)
        feature_dim = int(self.backbone.head.in_features)
        self.backbone.head = nn.Identity()
        head_dim = feature_dim * 2 if fusion_mode == "shared_feature_concat" else feature_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_dim, num_classes))
        self.fusion_mode = fusion_mode
        self.normalize_input = bool(normalize_input)
        # Kinetics-400 normalization used by the official torchvision weights.
        self.register_buffer("input_mean", torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1))
        self.register_buffer("input_std", torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1))

    def _prepare_plane(self, roi: torch.Tensor) -> torch.Tensor:
        if roi.shape[1] == 1:
            roi = roi.repeat(1, 3, 1, 1, 1)
        elif roi.shape[1] != 3:
            raise ValueError("paper Video Swin expects one- or three-channel ROI inputs")
        if self.normalize_input:
            roi = (roi - self.input_mean.to(dtype=roi.dtype)) / self.input_std.to(dtype=roi.dtype)
        return roi

    def _prepare_channel_fusion(self, sagittal: torch.Tensor, coronal: torch.Tensor) -> torch.Tensor:
        if sagittal.shape[1] != 1 or coronal.shape[1] != 1:
            raise ValueError("input channel fusion expects one-channel sagittal/coronal ROIs")
        if self.fusion_mode == "input_channel_difference":
            third = (sagittal - coronal).abs()
        else:
            third = 0.5 * (sagittal + coronal)
        fused = torch.cat([coronal, sagittal, third], dim=1)
        if self.normalize_input:
            fused = (fused - self.input_mean.to(dtype=fused.dtype)) / self.input_std.to(dtype=fused.dtype)
        return fused

    def forward_features(self, sagittal_roi: torch.Tensor, coronal_roi: torch.Tensor | None = None) -> torch.Tensor:
        coronal = sagittal_roi if coronal_roi is None else coronal_roi
        if sagittal_roi.shape != coronal.shape:
            raise ValueError("sagittal/coronal ROI tensors must have identical shapes")
        if self.fusion_mode in {"input_channel_concat", "input_channel_difference"}:
            return self.backbone(self._prepare_channel_fusion(sagittal_roi, coronal))
        sagittal_feature = self.backbone(self._prepare_plane(sagittal_roi))
        coronal_feature = self.backbone(self._prepare_plane(coronal))
        if self.fusion_mode == "shared_feature_mean":
            return 0.5 * (sagittal_feature + coronal_feature)
        return torch.cat([sagittal_feature, coronal_feature], dim=-1)

    def forward(self, sagittal_roi: torch.Tensor, coronal_roi: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.forward_features(sagittal_roi, coronal_roi))


class VideoSwinBackbone3D(nn.Module):
    """Compact hierarchical Video Swin backbone.

    Input:
        Tensor ``[B,1,T,H,W]``.

    Output:
        Global feature tensor ``[B,C_last]``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 96,
        patch_size: int | Sequence[int] = (2, 4, 4),
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int | Sequence[int] = (8, 7, 7),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.1,
        patch_merging_stride: int | Sequence[int] = (1, 2, 2),
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length")
        self.depths = tuple(int(v) for v in depths)
        self.num_heads = tuple(int(v) for v in num_heads)
        self.window_size = to_3tuple(window_size)
        self.use_checkpoint = use_checkpoint
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size=patch_size, dropout=dropout)

        total_blocks = sum(self.depths)
        drop_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        block_idx = 0
        stages = []
        dims = []
        dim = embed_dim
        for stage_idx, depth in enumerate(self.depths):
            blocks = []
            for layer_idx in range(depth):
                shift = tuple(size // 2 for size in self.window_size) if layer_idx % 2 == 1 else (0, 0, 0)
                blocks.append(
                    Swin3DBlock(
                        dim=dim,
                        num_heads=self.num_heads[stage_idx],
                        window_size=self.window_size,
                        shift_size=shift,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attn_drop=attn_drop,
                        drop_path=drop_rates[block_idx],
                    )
                )
                block_idx += 1
            downsample = None
            if stage_idx < len(self.depths) - 1:
                downsample = PatchMerging3D(dim, out_dim=dim * 2, stride=patch_merging_stride)
            stages.append(nn.ModuleDict({"blocks": nn.ModuleList(blocks), "downsample": downsample or nn.Identity()}))
            dims.append(dim)
            if stage_idx < len(self.depths) - 1:
                dim *= 2
        self.stages = nn.ModuleList(stages)
        self.out_dim = dim
        self.norm = nn.LayerNorm(self.out_dim)

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward_features_5d(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for stage in self.stages:
            for block in stage["blocks"]:
                x = self._run_block(block, x)
            x = stage["downsample"](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features_5d(x)
        pooled = x.mean(dim=(2, 3, 4))
        return self.norm(pooled)


class VideoSwinClassifier(nn.Module):
    """Dual-plane Video Swin classifier for Module 2.

    Inputs:
        ``sagittal_roi``: Tensor ``[B,1,T,H,W]``.
        ``coronal_roi``: Tensor ``[B,1,T,H,W]``.

    Output:
        ``logits``: Tensor ``[B,3]`` for normal / degeneration / tear.

    If ``share_branch_weights=True``, sagittal and coronal ROIs use the same
    backbone module. If ``coronal_roi`` is omitted, the sagittal branch feature
    is reused for compatibility with single-stream callers.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        embed_dim: int = 96,
        patch_size: int | Sequence[int] = (2, 4, 4),
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int | Sequence[int] = (8, 7, 7),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.1,
        patch_merging_stride: int | Sequence[int] = (1, 2, 2),
        use_checkpoint: bool = False,
        share_branch_weights: bool = False,
    ):
        super().__init__()
        branch = VideoSwinBackbone3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_drop=attn_drop,
            drop_path_rate=drop_path_rate,
            patch_merging_stride=patch_merging_stride,
            use_checkpoint=use_checkpoint,
        )
        self.sagittal_branch = branch
        self.coronal_branch = branch if share_branch_weights else deepcopy(branch)
        self.share_branch_weights = share_branch_weights
        feature_dim = branch.out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, num_classes),
        )

    def forward_features(self, sagittal_roi: torch.Tensor, coronal_roi: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        sagittal_feature = self.sagittal_branch(sagittal_roi)
        coronal_feature = self.coronal_branch(coronal_roi) if coronal_roi is not None else sagittal_feature
        return sagittal_feature, coronal_feature

    def forward(self, sagittal_roi: torch.Tensor, coronal_roi: torch.Tensor | None = None) -> torch.Tensor:
        sagittal_feature, coronal_feature = self.forward_features(sagittal_roi, coronal_roi)
        return self.head(torch.cat([sagittal_feature, coronal_feature], dim=1))
