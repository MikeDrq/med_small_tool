"""MeniscoViT-CoP backbone for Module 1.

paper-specified:
    Module 1 uses sagittal and coronal knee MRI, DICOM real-world coordinates,
    Real Coordinate Position Embedding (RCPE), and sagittal-coronal regional
    cross-attention where each query attends to the nearest spatial tokens.

paper-inferred implementation:
    The manuscript does not publish exact backbone block counts, block internals,
    pre/post cross-attention split, or global pooling details. This file uses a
    conservative engineering implementation:
    1. 2D per-slice patch embedding for each plane.
    2. RCPE added to each plane's patch tokens.
    3. A first stack of shallow/local UniFormer-like blocks.
    4. Bidirectional regional cross-attention, sagittal->coronal and
       coronal->sagittal.
    5. A second stack of deep/global UniFormer-like blocks.
    6. Mean pooling over both planes for a global feature.

This is a backbone only. Final detection/classification heads and training loss
details are intentionally outside this file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .patch_embed import MeniscusPatchEmbed, PatchEmbedShape
from .rcpe import RealCoordinatePositionEmbedding
from .regional_cross_attention import RegionalCrossAttention
from .uniformer_blocks import TokenGridShape, UniFormerBlock


@dataclass(frozen=True)
class MeniscoViTShapeMetadata:
    """Shape metadata emitted by the backbone."""

    sagittal: PatchEmbedShape
    coronal: PatchEmbedShape
    tokens_per_plane: int
    fused_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sagittal": asdict(self.sagittal),
            "coronal": asdict(self.coronal),
            "tokens_per_plane": self.tokens_per_plane,
            "fused_tokens": self.fused_tokens,
        }


class MeniscoViTCoP(nn.Module):
    """Backbone for sagittal-coronal MeniscoViT-CoP.

    Inputs:
        ``sagittal_volume``: Tensor ``[B, 1, T, H, W]``.
        ``coronal_volume``: Tensor ``[B, 1, T, H, W]``.
        ``sagittal_coords``: Tensor ``[B, T*N, 3]``.
        ``coronal_coords``: Tensor ``[B, T*N, 3]``.

    Outputs dict:
        ``sagittal_tokens_fused``: Tensor ``[B, T*N, C]``.
        ``coronal_tokens_fused``: Tensor ``[B, T*N, C]``.
        ``fused_global_feature``: Tensor ``[B, C]``.
        ``per_token_features``: Tensor ``[B, 2*T*N, C]``.
        ``shape_metadata``: Dict with patch grid metadata.
    """

    def __init__(
        self,
        image_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_channels: int = 1,
        embed_dim: int = 192,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        cross_attention_k: int = 32,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        use_checkpoint: bool = False,
        num_slices: int = 32,
        rcpe_normalization: str = "per_case",
        rcpe_coord_mean: torch.Tensor | list[float] | tuple[float, float, float] | None = None,
        rcpe_coord_std: torch.Tensor | list[float] | tuple[float, float, float] | None = None,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2 so the backbone has pre- and post-cross-attention blocks")
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.cross_attention_k = cross_attention_k
        self.use_checkpoint = use_checkpoint
        self.num_slices = num_slices

        self.sagittal_patch_embed = MeniscusPatchEmbed(in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size)
        self.coronal_patch_embed = MeniscusPatchEmbed(in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size)
        self.rcpe = RealCoordinatePositionEmbedding(
            embed_dim=embed_dim,
            normalization=rcpe_normalization,
            coord_mean=rcpe_coord_mean,
            coord_std=rcpe_coord_std,
            dropout=dropout,
        )
        self.pos_drop = nn.Dropout(dropout)

        pre_depth = depth // 2
        post_depth = depth - pre_depth
        drop_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        def make_blocks(start: int, count: int, block_type: str) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    UniFormerBlock(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        block_type=block_type,
                        dropout=dropout,
                        drop_path=drop_rates[start + idx],
                    )
                    for idx in range(count)
                ]
            )

        # paper-inferred implementation: sagittal and coronal branches use the
        # same architecture but separate weights before/after cross-attention.
        self.sagittal_pre_blocks = make_blocks(0, pre_depth, "local")
        self.coronal_pre_blocks = make_blocks(0, pre_depth, "local")
        self.sagittal_to_coronal = RegionalCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            k_nearest=cross_attention_k,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.coronal_to_sagittal = RegionalCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            k_nearest=cross_attention_k,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.sagittal_post_blocks = make_blocks(pre_depth, post_depth, "global")
        self.coronal_post_blocks = make_blocks(pre_depth, post_depth, "global")
        self.final_norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _grid_shape(shape: PatchEmbedShape) -> TokenGridShape:
        return TokenGridShape(slices=shape.slices, grid_height=shape.grid_height, grid_width=shape.grid_width)

    @staticmethod
    def _validate_coords(tokens: torch.Tensor, coords: torch.Tensor, name: str) -> None:
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(f"{name} must be shaped [B,N,3], got {tuple(coords.shape)}")
        if coords.shape[:2] != tokens.shape[:2]:
            raise ValueError(f"{name} shape {tuple(coords.shape)} does not match token shape {tuple(tokens.shape)}")

    def _run_blocks(self, tokens: torch.Tensor, grid_shape: TokenGridShape, blocks: nn.ModuleList) -> torch.Tensor:
        for block in blocks:
            if self.use_checkpoint and self.training:
                tokens = checkpoint(lambda x, b=block: b(x, grid_shape), tokens, use_reentrant=False)
            else:
                tokens = block(tokens, grid_shape)
        return tokens

    def _embed_plane(self, volume: torch.Tensor, coords: torch.Tensor, patch_embed: MeniscusPatchEmbed, plane_name: str) -> tuple[torch.Tensor, PatchEmbedShape]:
        tokens, shape = patch_embed(volume)
        if shape.slices != self.num_slices:
            # paper-inferred implementation: num_slices is a config contract for
            # reproducible training/inference shapes, but variable T still works
            # if callers intentionally configure it differently.
            pass
        self._validate_coords(tokens, coords, f"{plane_name}_coords")
        tokens = self.pos_drop(tokens + self.rcpe(coords))
        return tokens, shape

    def forward(
        self,
        sagittal_volume: torch.Tensor,
        coronal_volume: torch.Tensor,
        sagittal_coords: torch.Tensor,
        coronal_coords: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        sagittal_tokens, sagittal_shape = self._embed_plane(sagittal_volume, sagittal_coords, self.sagittal_patch_embed, "sagittal")
        coronal_tokens, coronal_shape = self._embed_plane(coronal_volume, coronal_coords, self.coronal_patch_embed, "coronal")

        sagittal_grid = self._grid_shape(sagittal_shape)
        coronal_grid = self._grid_shape(coronal_shape)
        sagittal_tokens = self._run_blocks(sagittal_tokens, sagittal_grid, self.sagittal_pre_blocks)
        coronal_tokens = self._run_blocks(coronal_tokens, coronal_grid, self.coronal_pre_blocks)

        # Bidirectional co-plane regional cross-attention. Both directions use
        # the pre-cross tokens to avoid order-dependent fusion.
        sagittal_cross = self.sagittal_to_coronal(
            query_tokens=sagittal_tokens,
            query_coords=sagittal_coords,
            key_value_tokens=coronal_tokens,
            key_value_coords=coronal_coords,
        )
        coronal_cross = self.coronal_to_sagittal(
            query_tokens=coronal_tokens,
            query_coords=coronal_coords,
            key_value_tokens=sagittal_tokens,
            key_value_coords=sagittal_coords,
        )

        sagittal_tokens = self._run_blocks(sagittal_cross, sagittal_grid, self.sagittal_post_blocks)
        coronal_tokens = self._run_blocks(coronal_cross, coronal_grid, self.coronal_post_blocks)
        sagittal_tokens = self.final_norm(sagittal_tokens)
        coronal_tokens = self.final_norm(coronal_tokens)

        per_token_features = torch.cat([sagittal_tokens, coronal_tokens], dim=1)
        fused_global_feature = per_token_features.mean(dim=1)
        metadata = MeniscoViTShapeMetadata(
            sagittal=sagittal_shape,
            coronal=coronal_shape,
            tokens_per_plane=sagittal_tokens.shape[1],
            fused_tokens=per_token_features.shape[1],
        )
        return {
            "sagittal_tokens_fused": sagittal_tokens,
            "coronal_tokens_fused": coronal_tokens,
            "fused_global_feature": fused_global_feature,
            "per_token_features": per_token_features,
            "shape_metadata": metadata.to_dict(),
        }
