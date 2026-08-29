"""UniFormer-like intra-plane blocks for MeniscoViT-CoP.

paper-specified:
    The manuscript describes a ViT-based model for anisotropic MRI sequences and
    a sagittal-coronal coordinate-aware attention mechanism.

paper-inferred implementation:
    The public manuscript does not specify exact intra-plane block internals.
    This file implements a conservative UniFormer-like backbone block:
    shallow/local blocks use depthwise 3D convolution over the token grid
    ``[T, Hp, Wp]`` for local relation aggregation; deep/global blocks use
    global multi-head self-attention over all tokens in one plane.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TokenGridShape:
    """Token grid metadata used by local relation aggregation.

    Attributes:
        slices: T.
        grid_height: Hp.
        grid_width: Wp.
    """

    slices: int
    grid_height: int
    grid_width: int

    @property
    def num_tokens(self) -> int:
        return self.slices * self.grid_height * self.grid_width


class DropPath(nn.Module):
    """Stochastic depth per sample.

    This is a standard implementation used only during training. At eval time or
    when ``drop_prob=0``, it is an identity.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    """Transformer feed-forward network."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalRelationAggregation(nn.Module):
    """Depthwise 3D convolution over ``[T, Hp, Wp]`` token grids.

    Input/Output:
        ``tokens``: Tensor ``[B, T*Hp*Wp, C]``.
    """

    def __init__(self, embed_dim: int, kernel_size: tuple[int, int, int] = (3, 3, 3), dropout: float = 0.0):
        super().__init__()
        padding = tuple(size // 2 for size in kernel_size)
        self.depthwise = nn.Conv3d(embed_dim, embed_dim, kernel_size=kernel_size, padding=padding, groups=embed_dim)
        self.pointwise = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, grid_shape: TokenGridShape) -> torch.Tensor:
        batch_size, num_tokens, channels = tokens.shape
        if num_tokens != grid_shape.num_tokens:
            raise ValueError(f"token count {num_tokens} does not match grid_shape {grid_shape}")
        x = tokens.transpose(1, 2).reshape(batch_size, channels, grid_shape.slices, grid_shape.grid_height, grid_shape.grid_width)
        x = self.pointwise(self.act(self.depthwise(x)))
        x = x.reshape(batch_size, channels, num_tokens).transpose(1, 2)
        return self.drop(x)


class GlobalSelfAttention(nn.Module):
    """Global multi-head self-attention over all tokens in one plane."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, grid_shape: TokenGridShape) -> torch.Tensor:
        del grid_shape
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return self.drop(out)


class UniFormerBlock(nn.Module):
    """UniFormer-like token block.

    Args:
        embed_dim: Token embedding dimension C.
        num_heads: Attention heads for global blocks.
        mlp_ratio: FFN hidden ratio.
        block_type: ``"local"`` for shallow local relation aggregation or
            ``"global"`` for deep global self-attention.
        dropout: Dropout probability for attention/relation and MLP.
        drop_path: Stochastic depth probability.

    Input:
        ``tokens``: Tensor ``[B, T*Hp*Wp, C]``.
        ``grid_shape``: TokenGridShape.

    Output:
        Tensor ``[B, T*Hp*Wp, C]``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        block_type: str = "local",
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if block_type not in {"local", "global"}:
            raise ValueError("block_type must be 'local' or 'global'")
        self.block_type = block_type
        self.norm1 = nn.LayerNorm(embed_dim)
        if block_type == "local":
            self.relation = LocalRelationAggregation(embed_dim, dropout=dropout)
        else:
            self.relation = GlobalSelfAttention(embed_dim, num_heads=num_heads, dropout=dropout)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim, hidden_dim=int(embed_dim * mlp_ratio), dropout=dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, tokens: torch.Tensor, grid_shape: TokenGridShape) -> torch.Tensor:
        tokens = tokens + self.drop_path1(self.relation(self.norm1(tokens), grid_shape))
        tokens = tokens + self.drop_path2(self.mlp(self.norm2(tokens)))
        return tokens
