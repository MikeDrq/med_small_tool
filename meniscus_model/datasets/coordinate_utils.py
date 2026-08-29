"""Coordinate transforms and nearest-neighbor helpers for DICOM world space."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _slice_metadata(dicom_metadata: dict[str, Any], slice_index: int) -> dict[str, Any]:
    slices = dicom_metadata.get("slices")
    if not slices:
        raise ValueError("dicom_metadata must contain a non-empty 'slices' list")
    if slice_index < 0 or slice_index >= len(slices):
        raise IndexError(f"slice_index {slice_index} outside available range [0, {len(slices) - 1}]")
    return slices[slice_index]


def voxel_to_world(row: float | np.ndarray, col: float | np.ndarray, slice_index: int, dicom_metadata: dict[str, Any]) -> np.ndarray:
    """Convert DICOM pixel indices to patient-world coordinates.

    Args:
        row: Pixel row index or array of row indices.
        col: Pixel column index or array of column indices.
        slice_index: Processed slice index in the loaded ``[T,H,W]`` volume.
        dicom_metadata: Metadata returned by ``read_dicom_series``.

    Returns:
        Array shaped ``[..., 3]`` in patient-world millimeters.

    DICOM formula:
        ``IPP + col * column_spacing * row_direction + row * row_spacing * column_direction``.
    """

    meta = _slice_metadata(dicom_metadata, slice_index)
    iop = np.asarray(meta["ImageOrientationPatient"], dtype=np.float64)
    ipp = np.asarray(meta["ImagePositionPatient"], dtype=np.float64)
    pixel_spacing = np.asarray(meta["PixelSpacing"], dtype=np.float64)
    row_direction = iop[:3]
    column_direction = iop[3:]

    row_array = np.asarray(row, dtype=np.float64)
    col_array = np.asarray(col, dtype=np.float64)
    world = (
        ipp
        + col_array[..., None] * pixel_spacing[1] * row_direction
        + row_array[..., None] * pixel_spacing[0] * column_direction
    )
    return world.astype(np.float32)


def patch_grid_to_world_coordinates(
    image_size: tuple[int, int],
    patch_size: int | tuple[int, int],
    slice_index: int,
    dicom_metadata: dict[str, Any],
) -> np.ndarray:
    """Return world coordinates for every patch center on one slice.

    Args:
        image_size: ``(H, W)`` after preprocessing.
        patch_size: Patch size as integer or ``(patch_h, patch_w)``.
        slice_index: Processed slice index.
        dicom_metadata: Metadata returned by ``read_dicom_series``.

    Returns:
        Float32 array ``[num_patches, 3]`` in raster patch order.
    """

    height, width = image_size
    if isinstance(patch_size, int):
        patch_h = patch_w = patch_size
    else:
        patch_h, patch_w = patch_size
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError("patch_size must be positive")
    n_h = height // patch_h
    n_w = width // patch_w
    if n_h == 0 or n_w == 0:
        raise ValueError(f"patch_size {patch_size} is larger than image_size {image_size}")
    rows = np.arange(n_h, dtype=np.float32) * patch_h + (patch_h - 1) / 2.0
    cols = np.arange(n_w, dtype=np.float32) * patch_w + (patch_w - 1) / 2.0
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    return voxel_to_world(row_grid.reshape(-1), col_grid.reshape(-1), slice_index, dicom_metadata)


def volume_patch_grid_to_world_coordinates(
    image_size: tuple[int, int],
    patch_size: int | tuple[int, int],
    num_slices: int,
    dicom_metadata: dict[str, Any],
) -> np.ndarray:
    """Return patch-center coordinates for all slices.

    Returns:
        Float32 array ``[T * num_patches_per_slice, 3]``.
    """

    coords = [
        patch_grid_to_world_coordinates(image_size=image_size, patch_size=patch_size, slice_index=t, dicom_metadata=dicom_metadata)
        for t in range(num_slices)
    ]
    return np.concatenate(coords, axis=0).astype(np.float32)


def voxel_indices_to_world(indices_zyx: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    """Map voxel indices to DICOM patient-world coordinates.

    Args:
        indices_zyx: Tensor ``[..., 3]`` in ``[slice,row,col]`` order.
        affine: Tensor ``[..., 4, 4]`` or ``[4, 4]`` mapping voxel to world mm.

    Returns:
        Tensor ``[..., 3]`` in world millimeters.
    """

    ones = torch.ones_like(indices_zyx[..., :1])
    homogeneous = torch.cat([indices_zyx.to(affine.dtype), ones], dim=-1)
    world = torch.matmul(homogeneous, affine.transpose(-1, -2))
    return world[..., :3]


def world_to_voxel_indices(world_xyz: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    """Map DICOM patient-world coordinates to continuous voxel indices ``[slice,row,col]``."""

    inv_affine = torch.linalg.inv(affine)
    ones = torch.ones_like(world_xyz[..., :1])
    homogeneous = torch.cat([world_xyz.to(inv_affine.dtype), ones], dim=-1)
    voxel = torch.matmul(homogeneous, inv_affine.transpose(-1, -2))
    return voxel[..., :3]


def pairwise_world_distance(query_xyz: torch.Tensor, key_xyz: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean distances in millimeters.

    Args:
        query_xyz: Tensor ``[B, Nq, 3]``.
        key_xyz: Tensor ``[B, Nk, 3]``.

    Returns:
        Tensor ``[B, Nq, Nk]``.
    """

    return torch.cdist(query_xyz, key_xyz, p=2)


def nearest_k_indices(query_xyz: torch.Tensor, key_xyz: torch.Tensor, k: int = 32) -> torch.Tensor:
    """Return nearest token indices by real-world distance."""

    distances = pairwise_world_distance(query_xyz, key_xyz)
    return torch.topk(distances, k=min(k, key_xyz.shape[1]), dim=-1, largest=False).indices
