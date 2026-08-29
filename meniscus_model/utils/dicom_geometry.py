"""DICOM affine and coordinate-grid helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from datasets.coordinate_utils import voxel_indices_to_world, voxel_to_world


def make_coordinate_grid(depth: int, height: int, width: int, affine: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """Create a dense world-coordinate grid for a volume.

    Args:
        depth, height, width: Volume dimensions.
        affine: Tensor ``[4,4]`` mapping ``[slice,row,col]`` to world mm.

    Returns:
        Tensor ``[D,H,W,3]``.
    """

    z, y, x = torch.meshgrid(
        torch.arange(depth, device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    indices = torch.stack([z, y, x], dim=-1).float()
    return voxel_indices_to_world(indices, affine.to(device=device))


def make_coordinate_grid_from_metadata(dicom_metadata: dict[str, Any]) -> np.ndarray:
    """Create a dense world-coordinate grid from per-slice DICOM metadata."""

    depth, height, width = dicom_metadata["processed_shape"]
    rows = np.arange(height, dtype=np.float32)
    cols = np.arange(width, dtype=np.float32)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    slices = [voxel_to_world(row_grid, col_grid, slice_index, dicom_metadata) for slice_index in range(depth)]
    return np.stack(slices, axis=0).astype(np.float32)


def build_affine_from_dicom_tags(
    image_orientation_patient: list[float],
    image_position_patient: list[float],
    pixel_spacing: list[float],
    slice_spacing: float,
) -> torch.Tensor:
    """Build voxel-to-world affine from common DICOM tags.

    The affine maps ``[slice,row,col,1]`` to DICOM patient coordinates.
    ``ImageOrientationPatient`` stores row and column direction cosines.
    ``PixelSpacing`` is ``[row_spacing, column_spacing]``.
    """

    iop = torch.tensor(image_orientation_patient, dtype=torch.float32)
    ipp = torch.tensor(image_position_patient, dtype=torch.float32)
    spacing = torch.tensor(pixel_spacing, dtype=torch.float32)
    row_direction = iop[:3]
    column_direction = iop[3:]
    normal = torch.cross(row_direction, column_direction, dim=0)

    affine = torch.eye(4, dtype=torch.float32)
    affine[:3, 0] = normal * float(slice_spacing)
    affine[:3, 1] = column_direction * spacing[0]
    affine[:3, 2] = row_direction * spacing[1]
    affine[:3, 3] = ipp
    return affine


def bbox_to_world_corners(bbox_xyxy: list[float], slice_index: int, dicom_metadata: dict[str, Any]) -> np.ndarray:
    """Project a 2D bbox on one slice into four world-space corner points."""

    x1, y1, x2, y2 = bbox_xyxy
    rows = np.asarray([y1, y1, y2, y2], dtype=np.float32)
    cols = np.asarray([x1, x2, x2, x1], dtype=np.float32)
    return voxel_to_world(rows, cols, slice_index, dicom_metadata)
