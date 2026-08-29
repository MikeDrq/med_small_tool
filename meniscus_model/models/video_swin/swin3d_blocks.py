"""Compact Video Swin / Swin3D building blocks.

paper-specified:
    Module 2 uses a Video Swin Transformer to classify meniscus ROI sequences as
    normal / degeneration / tear.

paper-inferred implementation:
    The manuscript does not publish exact Video Swin code. This compact
    implementation keeps the required structure: 3D patch embedding,
    window-based spatio-temporal self-attention, shifted windows, hierarchical
    stages, patch merging/downsampling, dropout, and checkpoint-friendly blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def to_3tuple(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError("Expected an int or a 3-item sequence")
    return int(value[0]), int(value[1]), int(value[2])


class DropPath(nn.Module):
    """Stochastic depth."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep_prob) * mask


class Mlp(nn.Module):
    """Transformer feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchEmbed3D(nn.Module):
    """3D patch embedding for ROI sequences.

    Input:
        Tensor ``[B,C,T,H,W]``.

    Output:
        Tensor ``[B,C_embed,Tp,Hp,Wp]``.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 96, patch_size: int | Sequence[int] = (2, 4, 4), dropout: float = 0.0):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,T,H,W], got {tuple(x.shape)}")
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.drop(self.norm(x))
        return x.permute(0, 4, 1, 2, 3).contiguous()


def window_partition(x: torch.Tensor, window_size: tuple[int, int, int]) -> torch.Tensor:
    """Partition channels-last tensor into windows.

    Args:
        x: Tensor ``[B,T,H,W,C]``.

    Returns:
        Tensor ``[num_windows*B, window_volume, C]``.
    """

    bsz, depth, height, width, channels = x.shape
    wd, wh, ww = window_size
    x = x.view(bsz, depth // wd, wd, height // wh, wh, width // ww, ww, channels)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, wd * wh * ww, channels)


def window_reverse(windows: torch.Tensor, window_size: tuple[int, int, int], batch_size: int, depth: int, height: int, width: int) -> torch.Tensor:
    """Reverse ``window_partition`` to ``[B,T,H,W,C]``."""

    wd, wh, ww = window_size
    channels = windows.shape[-1]
    x = windows.view(batch_size, depth // wd, height // wh, width // ww, wd, wh, ww, channels)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(batch_size, depth, height, width, channels)


def pad_to_window_size(x: torch.Tensor, window_size: tuple[int, int, int]) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Pad channels-last tensor ``[B,T,H,W,C]`` to multiples of window size."""

    _, depth, height, width, _ = x.shape
    wd, wh, ww = window_size
    pad_d = (wd - depth % wd) % wd
    pad_h = (wh - height % wh) % wh
    pad_w = (ww - width % ww) % ww
    if pad_d or pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
    return x, (pad_d, pad_h, pad_w)


class WindowAttention3D(nn.Module):
    """Multi-head self-attention inside 3D windows."""

    def __init__(self, dim: int, num_heads: int, window_size: int | Sequence[int], attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = to_3tuple(window_size)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(b_windows, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(b_windows // num_windows, num_windows, self.num_heads, tokens, tokens)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, tokens, tokens)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b_windows, tokens, channels)
        return self.proj_drop(self.proj(out))


def compute_attention_mask(
    depth: int,
    height: int,
    width: int,
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor | None:
    """Build shifted-window attention mask."""

    if not any(shift_size):
        return None
    img_mask = torch.zeros((1, depth, height, width, 1), device=device)
    wd, wh, ww = window_size
    sd, sh, sw = shift_size
    d_slices = (slice(0, -wd), slice(-wd, -sd), slice(-sd, None)) if sd > 0 else (slice(0, depth),)
    h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None)) if sh > 0 else (slice(0, height),)
    w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None)) if sw > 0 else (slice(0, width),)
    count = 0
    for d in d_slices:
        for h in h_slices:
            for w in w_slices:
                img_mask[:, d, h, w, :] = count
                count += 1
    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))


class Swin3DBlock(nn.Module):
    """Video Swin block with optional shifted-window attention.

    Input/Output:
        Tensor ``[B,C,T,H,W]``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int | Sequence[int] = (8, 7, 7),
        shift_size: int | Sequence[int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = to_3tuple(window_size)
        raw_shift = to_3tuple(shift_size)
        self.shift_size = tuple(min(raw_shift[idx], self.window_size[idx] - 1) for idx in range(3))
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, num_heads, self.window_size, attn_drop=attn_drop, proj_drop=dropout)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, depth, height, width = x.shape
        shortcut = x
        x_cl = x.permute(0, 2, 3, 4, 1).contiguous()
        x_cl = self.norm1(x_cl)
        x_pad, pads = pad_to_window_size(x_cl, self.window_size)
        _, dp, hp, wp, _ = x_pad.shape

        shift = tuple(0 if dim <= win else s for dim, win, s in zip((dp, hp, wp), self.window_size, self.shift_size))
        shifted = torch.roll(x_pad, shifts=tuple(-s for s in shift), dims=(1, 2, 3)) if any(shift) else x_pad
        windows = window_partition(shifted, self.window_size)
        mask = compute_attention_mask(dp, hp, wp, self.window_size, shift, x.device)
        attn_windows = self.attn(windows, mask)
        shifted = window_reverse(attn_windows, self.window_size, bsz, dp, hp, wp)
        x_pad = torch.roll(shifted, shifts=shift, dims=(1, 2, 3)) if any(shift) else shifted

        pad_d, pad_h, pad_w = pads
        if pad_d or pad_h or pad_w:
            x_pad = x_pad[:, :depth, :height, :width, :]
        x_attn = x_pad.permute(0, 4, 1, 2, 3).contiguous()
        x = shortcut + self.drop_path(x_attn)

        mlp_in = self.norm2(x.permute(0, 2, 3, 4, 1))
        mlp_out = self.mlp(mlp_in).permute(0, 4, 1, 2, 3).contiguous()
        return x + self.drop_path(mlp_out)


class PatchMerging3D(nn.Module):
    """Hierarchical downsampling between Swin stages."""

    def __init__(self, dim: int, out_dim: int | None = None, stride: int | Sequence[int] = (1, 2, 2)):
        super().__init__()
        out_dim = out_dim or dim * 2
        self.reduction = nn.Conv3d(dim, out_dim, kernel_size=to_3tuple(stride), stride=to_3tuple(stride))
        self.norm = nn.BatchNorm3d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.reduction(x))


@dataclass(frozen=True)
class StageConfig:
    dim: int
    depth: int
    num_heads: int
