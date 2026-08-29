"""2D per-slice patch embedding for anisotropic knee MRI volumes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PatchEmbedShape:
    """Shape metadata needed to restore token grids.

    Attributes:
        batch_size: B.
        channels: Input channel count.
        slices: T.
        height: Input H.
        width: Input W.
        patch_height: 2D patch height.
        patch_width: 2D patch width.
        grid_height: Hp = H // patch_height.
        grid_width: Wp = W // patch_width.
        patches_per_slice: N = Hp * Wp.
    """

    batch_size: int
    channels: int
    slices: int
    height: int
    width: int
    patch_height: int
    patch_width: int
    grid_height: int
    grid_width: int
    patches_per_slice: int


class MeniscusPatchEmbed(nn.Module):
    """Embed each slice of a volume with the same 2D patch projection.

    Input:
        ``volume``: Tensor ``[B, 1, T, H, W]``.

    Output:
        ``tokens``: Tensor ``[B, T * N, C]`` where
        ``N = (H // patch_h) * (W // patch_w)``.
        ``shape``: PatchEmbedShape with ``T``, ``Hp``, ``Wp`` and related info.

    Notes:
        This deliberately uses 2D patch embedding per slice. It does not merge
        adjacent slices, preserving anisotropic MRI slice structure for later
        sagittal-coronal coordinate-aware attention.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 192, patch_size: int | tuple[int, int] | tuple[int, int, int] = 16):
        super().__init__()
        patch_h, patch_w = self._normalize_patch_size(patch_size)
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = (patch_h, patch_w)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))

    @staticmethod
    def _normalize_patch_size(patch_size: int | tuple[int, int] | tuple[int, int, int]) -> tuple[int, int]:
        if isinstance(patch_size, int):
            return patch_size, patch_size
        if len(patch_size) == 2:
            return int(patch_size[0]), int(patch_size[1])
        if len(patch_size) == 3:
            return int(patch_size[1]), int(patch_size[2])
        raise ValueError("patch_size must be an int, (patch_h, patch_w), or legacy (t, patch_h, patch_w)")

    def forward(self, volume: torch.Tensor) -> tuple[torch.Tensor, PatchEmbedShape]:
        """Patchify a volume.

        Args:
            volume: Tensor ``[B, C, T, H, W]``.

        Returns:
            ``tokens``: Tensor ``[B, T * Hp * Wp, embed_dim]``.
            ``shape``: PatchEmbedShape.
        """

        if volume.ndim != 5:
            raise ValueError(f"volume must be shaped [B,C,T,H,W], got {tuple(volume.shape)}")
        batch_size, channels, slices, height, width = volume.shape
        if channels != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {channels}")
        patch_h, patch_w = self.patch_size
        if height < patch_h or width < patch_w:
            raise ValueError(f"patch_size {self.patch_size} is larger than input image {(height, width)}")

        # [B,C,T,H,W] -> [B*T,C,H,W], applying one shared 2D projection per slice.
        frames = volume.permute(0, 2, 1, 3, 4).reshape(batch_size * slices, channels, height, width)
        features = self.proj(frames)  # [B*T,Cembed,Hp,Wp]
        grid_height, grid_width = features.shape[-2:]
        patches_per_slice = grid_height * grid_width

        # [B*T,Cembed,Hp,Wp] -> [B,T,Hp*Wp,Cembed] -> [B,T*N,Cembed].
        tokens = features.flatten(2).transpose(1, 2).reshape(batch_size, slices * patches_per_slice, self.embed_dim)
        shape = PatchEmbedShape(
            batch_size=batch_size,
            channels=channels,
            slices=slices,
            height=height,
            width=width,
            patch_height=patch_h,
            patch_width=patch_w,
            grid_height=grid_height,
            grid_width=grid_width,
            patches_per_slice=patches_per_slice,
        )
        return tokens, shape
