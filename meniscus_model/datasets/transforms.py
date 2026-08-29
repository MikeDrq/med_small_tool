"""Sample-level transforms for MRI volumes and 2D bbox labels.

Photometric-only augmentation is the safe default for the world-coordinate
backbone. The old independent per-plane spatial transform remains available as
an explicit legacy mode for checkpoint/experiment reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import get_worker_info


TRAINING_AUGMENTATION_MODES = {
    "photometric",
    "spatial_photometric",
    "legacy_spatial_photometric",
}


class Compose:
    """Compose sample-level transforms operating on dict samples."""

    def __init__(self, transforms: list[Callable[[dict[str, Any]], dict[str, Any]]]):
        self.transforms = transforms

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def _bbox_corners(bbox: list[float]) -> torch.Tensor:
    x1, y1, x2, y2 = bbox
    return torch.tensor([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=torch.float32)


def _corners_to_bbox(corners: torch.Tensor, width: int, height: int) -> list[float]:
    xs = corners[:, 0].clamp(0, width - 1)
    ys = corners[:, 1].clamp(0, height - 1)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _rotate_points(points_xy: torch.Tensor, angle_degrees: float, width: int, height: int) -> torch.Tensor:
    angle = math.radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    center = torch.tensor([(width - 1) / 2.0, (height - 1) / 2.0], dtype=torch.float32)
    shifted = points_xy - center
    rotation = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
    return shifted @ rotation.T + center


def transform_bbox(
    bbox: list[float],
    width: int,
    height: int,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
    rotation_degrees: float = 0.0,
) -> list[float]:
    """Apply spatial transforms to a bbox in ``[x1,y1,x2,y2]`` format."""

    corners = _bbox_corners(bbox)
    if horizontal_flip:
        corners[:, 0] = (width - 1) - corners[:, 0]
    if vertical_flip:
        corners[:, 1] = (height - 1) - corners[:, 1]
    if abs(rotation_degrees) > 1e-6:
        corners = _rotate_points(corners, rotation_degrees, width, height)
    return _corners_to_bbox(corners, width, height)


@dataclass(frozen=True)
class PlaneAugmentParams:
    horizontal_flip: bool
    vertical_flip: bool
    rotation_degrees: float
    brightness_shift: float
    gaussian_noise_sigma: float


class RandomPlaneAugmentation:
    """Apply requested 3D-volume augmentation to one plane key.

    Args:
        volume_key: ``sagittal_volume`` or ``coronal_volume``.
        plane: Label plane name whose bbox annotations should be updated.
    """

    def __init__(
        self,
        volume_key: str,
        plane: str,
        coordinate_key: str | None = None,
        probability: float = 0.5,
        max_rotation_degrees: float = 30.0,
        max_brightness_shift: float = 0.2,
        noise_sigma: float = 0.02,
        enable_spatial: bool = True,
        enable_photometric: bool = True,
        update_world_coordinates: bool = False,
        rng: random.Random | None = None,
        worker_seed_offset: int = 0,
    ):
        self.volume_key = volume_key
        self.plane = plane
        self.coordinate_key = coordinate_key
        self.probability = probability
        self.max_rotation_degrees = max_rotation_degrees
        self.max_brightness_shift = max_brightness_shift
        self.noise_sigma = noise_sigma
        self.enable_spatial = bool(enable_spatial)
        self.enable_photometric = bool(enable_photometric)
        self.update_world_coordinates = bool(update_world_coordinates)
        self.rng = rng or random.Random()
        self.worker_seed_offset = int(worker_seed_offset)
        self._worker_seed_signature: tuple[int, int] | None = None

    def _worker_safe_rng(self) -> random.Random:
        """Reseed copied transform state once per DataLoader worker.

        Without this, every forked worker starts from the same private
        ``random.Random`` state and emits duplicate augmentation sequences.
        PyTorch assigns each worker a stable distinct seed, which we preserve
        across persistent-worker epochs.
        """

        worker = get_worker_info()
        if worker is not None:
            signature = (int(worker.id), int(worker.seed))
            if signature != self._worker_seed_signature:
                self.rng.seed(int(worker.seed) + self.worker_seed_offset)
                self._worker_seed_signature = signature
        return self.rng

    def _params(self) -> PlaneAugmentParams:
        # The supplement applies the *pipeline* with 50% probability per
        # volume.  Once selected, flip, rotation, brightness and noise are all
        # applied using their published ranges (rather than four independent
        # Bernoulli draws).
        rng = self._worker_safe_rng()
        apply_pipeline = rng.random() < self.probability
        horizontal_flip = False
        vertical_flip = False
        if self.enable_spatial and apply_pipeline:
            flip_mode = rng.choice(("horizontal", "vertical", "both"))
            horizontal_flip = flip_mode in {"horizontal", "both"}
            vertical_flip = flip_mode in {"vertical", "both"}
        return PlaneAugmentParams(
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rotation_degrees=(
                rng.uniform(-self.max_rotation_degrees, self.max_rotation_degrees)
                if self.enable_spatial and apply_pipeline
                else 0.0
            ),
            brightness_shift=(
                rng.uniform(-self.max_brightness_shift, self.max_brightness_shift)
                if self.enable_photometric and apply_pipeline
                else 0.0
            ),
            gaussian_noise_sigma=(
                self.noise_sigma
                if self.enable_photometric and apply_pipeline
                else 0.0
            ),
        )

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.volume_key not in sample:
            return sample
        volume = sample[self.volume_key].float()
        if volume.ndim != 4:
            raise ValueError(f"{self.volume_key} must be shaped [1,T,H,W], got {tuple(volume.shape)}")
        _, _, height, width = volume.shape
        params = self._params()

        if params.horizontal_flip:
            volume = torch.flip(volume, dims=[-1])
        if params.vertical_flip:
            volume = torch.flip(volume, dims=[-2])
        if abs(params.rotation_degrees) > 1e-6:
            volume = rotate_volume(volume, params.rotation_degrees)
        if abs(params.brightness_shift) > 1e-6:
            volume = volume + params.brightness_shift
        if params.gaussian_noise_sigma > 0:
            volume = volume + torch.randn_like(volume) * params.gaussian_noise_sigma
        sample[self.volume_key] = volume.clamp(0.0, 1.0)

        spatial_changed = (
            params.horizontal_flip
            or params.vertical_flip
            or abs(params.rotation_degrees) > 1e-6
        )
        if spatial_changed:
            labels = sample.get("labels", {})
            for label in labels.get("module1_labels", []):
                if label.get("plane") != self.plane or "bbox" not in label:
                    continue
                label["bbox"] = transform_bbox(
                    label["bbox"],
                    width=width,
                    height=height,
                    horizontal_flip=params.horizontal_flip,
                    vertical_flip=params.vertical_flip,
                    rotation_degrees=params.rotation_degrees,
                )
            if self.update_world_coordinates and self.coordinate_key in sample:
                sample[self.coordinate_key] = transform_world_coordinate_grid(
                    sample[self.coordinate_key],
                    depth=volume.shape[1],
                    image_height=height,
                    image_width=width,
                    horizontal_flip=params.horizontal_flip,
                    vertical_flip=params.vertical_flip,
                    rotation_degrees=params.rotation_degrees,
                )
        sample.setdefault("augmentation", {})[self.plane] = params
        return sample


def rotate_volume(volume: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    """Rotate all slices in a ``[1,T,H,W]`` volume with one shared angle."""

    channels, depth, height, width = volume.shape
    x = volume.permute(1, 0, 2, 3)
    angle = math.radians(angle_degrees)
    theta = torch.tensor(
        [[math.cos(angle), math.sin(angle), 0.0], [-math.sin(angle), math.cos(angle), 0.0]],
        dtype=x.dtype,
        device=x.device,
    ).unsqueeze(0).repeat(depth, 1, 1)
    grid = F.affine_grid(theta, size=(depth, channels, height, width), align_corners=False)
    rotated = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return rotated.permute(1, 0, 2, 3)


def _infer_patch_grid(tokens_per_slice: int, image_height: int, image_width: int) -> tuple[int, int]:
    """Infer the rectangular patch grid from token count and image aspect."""

    candidates: list[tuple[float, int, int]] = []
    target_ratio = float(image_height) / max(float(image_width), 1.0)
    for grid_h in range(1, int(math.sqrt(tokens_per_slice)) + 1):
        if tokens_per_slice % grid_h:
            continue
        grid_w = tokens_per_slice // grid_h
        candidates.append((abs(grid_h / float(grid_w) - target_ratio), grid_h, grid_w))
        candidates.append((abs(grid_w / float(grid_h) - target_ratio), grid_w, grid_h))
    if not candidates:
        raise ValueError(f"cannot infer patch grid for {tokens_per_slice} tokens")
    _, grid_h, grid_w = min(candidates, key=lambda item: item[0])
    return grid_h, grid_w


def transform_world_coordinate_grid(
    coordinates: torch.Tensor,
    depth: int,
    image_height: int,
    image_width: int,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
    rotation_degrees: float = 0.0,
) -> torch.Tensor:
    """Transform RCPE patch-center coordinates with the augmented pixels.

    The values remain DICOM patient-world millimeters; only their positions on
    the token grid change.  This preserves the voxel/world correspondence that
    independent image-space augmentation previously broke.
    """

    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError(f"coordinates must be [tokens,3], got {tuple(coordinates.shape)}")
    if depth <= 0 or coordinates.shape[0] % int(depth):
        raise ValueError("coordinate token count must be divisible by volume depth")
    tokens_per_slice = int(coordinates.shape[0] // int(depth))
    grid_h, grid_w = _infer_patch_grid(tokens_per_slice, image_height, image_width)
    grid = coordinates.reshape(depth, grid_h, grid_w, 3).permute(0, 3, 1, 2)
    if horizontal_flip:
        grid = torch.flip(grid, dims=[-1])
    if vertical_flip:
        grid = torch.flip(grid, dims=[-2])
    if abs(rotation_degrees) > 1e-6:
        angle = math.radians(rotation_degrees)
        theta = torch.tensor(
            [[math.cos(angle), math.sin(angle), 0.0], [-math.sin(angle), math.cos(angle), 0.0]],
            dtype=grid.dtype,
            device=grid.device,
        ).unsqueeze(0).repeat(depth, 1, 1)
        sampling_grid = F.affine_grid(theta, size=grid.shape, align_corners=False)
        grid = F.grid_sample(
            grid,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
    return grid.permute(0, 2, 3, 1).reshape(-1, 3).contiguous()


class RandomTrainingAugmentation:
    """Apply configured per-plane augmentation.

    ``photometric`` preserves voxel-to-world/token-coordinate correspondence.
    ``legacy_spatial_photometric`` reproduces the previous independent spatial
    transforms, which do not update DICOM world coordinates.
    """

    def __init__(
        self,
        seed: int | None = None,
        mode: str = "photometric",
        probability: float = 0.5,
        max_rotation_degrees: float = 30.0,
        max_brightness_shift: float = 0.2,
        noise_sigma: float = 0.02,
    ):
        normalized_mode = str(mode).lower()
        if normalized_mode not in TRAINING_AUGMENTATION_MODES:
            raise ValueError(
                f"augmentation mode must be one of {sorted(TRAINING_AUGMENTATION_MODES)}, "
                f"got {mode!r}"
            )
        self.mode = normalized_mode
        enable_spatial = normalized_mode in {"spatial_photometric", "legacy_spatial_photometric"}
        update_world_coordinates = normalized_mode == "spatial_photometric"
        rng = random.Random(seed)
        common = {
            "probability": probability,
            "max_rotation_degrees": max_rotation_degrees,
            "max_brightness_shift": max_brightness_shift,
            "noise_sigma": noise_sigma,
            "enable_spatial": enable_spatial,
            "enable_photometric": True,
            "update_world_coordinates": update_world_coordinates,
        }
        self.sagittal = RandomPlaneAugmentation(
            "sagittal_volume",
            "sagittal",
            coordinate_key="sagittal_coords",
            rng=random.Random(rng.random()),
            worker_seed_offset=0,
            **common,
        )
        self.coronal = RandomPlaneAugmentation(
            "coronal_volume",
            "coronal",
            coordinate_key="coronal_coords",
            rng=random.Random(rng.random()),
            worker_seed_offset=1_000_003,
            **common,
        )

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample = self.sagittal(sample)
        return self.coronal(sample)


class RandomROIAugmentation:
    """Paper-aligned paired augmentation for Module 2/3 ROI videos.

    Sagittal and coronal ROI sequences are fused as channels by the paper
    Video Swin model, so one set of spatial parameters is applied to both
    planes.  One Bernoulli draw gates the complete pipeline per ROI volume,
    using the manuscript values: p=0.5, rotation +/-30 degrees, brightness
    +/-0.2, and Gaussian noise sigma=0.02.
    """

    def __init__(
        self,
        probability: float = 0.5,
        max_rotation_degrees: float = 30.0,
        max_brightness_shift: float = 0.2,
        noise_sigma: float = 0.02,
    ):
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        self.probability = float(probability)
        self.max_rotation_degrees = float(max_rotation_degrees)
        self.max_brightness_shift = float(max_brightness_shift)
        self.noise_sigma = float(noise_sigma)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        keys = [key for key in ("sagittal_roi", "coronal_roi") if key in sample]
        if not keys:
            return sample
        for key in keys:
            if sample[key].ndim != 4:
                raise ValueError(f"{key} must be shaped [1,T,H,W], got {tuple(sample[key].shape)}")

        # The supplement describes horizontal, vertical, or both as one flip
        # augmentation.  Draw that mode once and preserve cross-plane pairing.
        apply_pipeline = float(torch.rand(())) < self.probability
        flip_mode = 0
        if apply_pipeline:
            flip_mode = int(torch.randint(1, 4, ()).item())
        angle = 0.0
        if apply_pipeline:
            angle = float(
                (torch.rand(()) * 2.0 - 1.0) * self.max_rotation_degrees
            )
        brightness = 0.0
        if apply_pipeline:
            brightness = float(
                (torch.rand(()) * 2.0 - 1.0) * self.max_brightness_shift
            )
        add_noise = apply_pipeline and self.noise_sigma > 0.0

        for key in keys:
            volume = sample[key].float()
            if flip_mode & 1:
                volume = torch.flip(volume, dims=[-1])
            if flip_mode & 2:
                volume = torch.flip(volume, dims=[-2])
            if abs(angle) > 1e-6:
                volume = rotate_volume(volume, angle)
            if abs(brightness) > 1e-6:
                volume = volume + brightness
            if add_noise:
                # Noise is independent per acquisition while spatial and
                # brightness parameters remain paired.
                volume = volume + torch.randn_like(volume) * self.noise_sigma
            sample[key] = volume.clamp(0.0, 1.0)

        sample["augmentation"] = {
            "flip_mode": flip_mode,
            "rotation_degrees": angle,
            "brightness_shift": brightness,
            "gaussian_noise_sigma": self.noise_sigma if add_noise else 0.0,
        }
        return sample
