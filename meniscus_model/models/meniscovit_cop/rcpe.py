"""Real Coordinate Position Embedding (RCPE)."""

from __future__ import annotations

import torch
from torch import nn


class RealCoordinatePositionEmbedding(nn.Module):
    """Embed DICOM patient-world patch-center coordinates.

    Input:
        ``coords_xyz``: Tensor ``[B, T * N, 3]`` in millimeters.

    Output:
        Positional embedding tensor ``[B, T * N, C]``.

    Coordinate normalization modes:
        ``"none"``:
            Feed coordinates directly to the MLP.
        ``"per_case"``:
            Normalize each case independently over its tokens:
            ``(coords - mean_B) / std_B``.
        ``"dataset"``:
            Normalize with dataset-level ``coord_mean`` and ``coord_std``.

    Usage:
        ``tokens = tokens + rcpe(coords)``.
    """

    VALID_NORMALIZATION = {"none", "per_case", "dataset"}

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        normalization: str = "per_case",
        coord_mean: torch.Tensor | list[float] | tuple[float, float, float] | None = None,
        coord_std: torch.Tensor | list[float] | tuple[float, float, float] | None = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if normalization not in self.VALID_NORMALIZATION:
            raise ValueError(f"normalization must be one of {sorted(self.VALID_NORMALIZATION)}, got {normalization}")
        if normalization == "dataset" and (coord_mean is None or coord_std is None):
            raise ValueError("coord_mean and coord_std are required when normalization='dataset'")

        hidden_dim = hidden_dim or embed_dim
        self.embed_dim = embed_dim
        self.normalization = normalization
        self.eps = eps

        mean = torch.zeros(3, dtype=torch.float32) if coord_mean is None else torch.as_tensor(coord_mean, dtype=torch.float32)
        std = torch.ones(3, dtype=torch.float32) if coord_std is None else torch.as_tensor(coord_std, dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError("coord_mean and coord_std must each have shape [3]")
        self.register_buffer("coord_mean", mean)
        self.register_buffer("coord_std", std.clamp_min(eps))

        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def normalize_coords(self, coords_xyz: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates according to the configured mode.

        Args:
            coords_xyz: Tensor ``[B, T*N, 3]``.

        Returns:
            Tensor ``[B, T*N, 3]``.
        """

        if coords_xyz.ndim != 3 or coords_xyz.shape[-1] != 3:
            raise ValueError(f"coords_xyz must be shaped [B,N,3], got {tuple(coords_xyz.shape)}")
        coords = coords_xyz.float()
        if self.normalization == "none":
            return coords
        if self.normalization == "per_case":
            mean = coords.mean(dim=1, keepdim=True)
            std = coords.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
            return (coords - mean) / std
        return (coords - self.coord_mean.view(1, 1, 3)) / self.coord_std.view(1, 1, 3)

    def forward(self, coords_xyz: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_coords(coords_xyz)
        return self.mlp(normalized)
