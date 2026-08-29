"""Tests for coordinate-safe Module 1 augmentation modes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from datasets.transforms import RandomROIAugmentation, RandomTrainingAugmentation
from engine.train_module1 import build_training_transform


def _sample() -> dict:
    volume = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) / 63.0
    return {
        "sagittal_volume": volume.clone(),
        "coronal_volume": volume.clone(),
        "sagittal_coords": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "coronal_coords": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "labels": {
            "module1_labels": [
                {
                    "plane": "sagittal",
                    "slice_index": 0,
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                }
            ]
        },
    }


def test_photometric_mode_preserves_bbox_and_world_coordinates() -> None:
    sample = _sample()
    original_volume = sample["sagittal_volume"].clone()
    original_coords = sample["sagittal_coords"].clone()
    original_bbox = list(sample["labels"]["module1_labels"][0]["bbox"])
    transform = RandomTrainingAugmentation(
        seed=7,
        mode="photometric",
        probability=1.0,
        max_brightness_shift=0.2,
        noise_sigma=0.0,
    )

    transformed = transform(sample)

    params = transformed["augmentation"]["sagittal"]
    assert not params.horizontal_flip
    assert not params.vertical_flip
    assert params.rotation_degrees == 0.0
    assert transformed["labels"]["module1_labels"][0]["bbox"] == original_bbox
    assert torch.equal(transformed["sagittal_coords"], original_coords)
    assert not torch.equal(transformed["sagittal_volume"], original_volume)


def test_legacy_spatial_mode_is_explicit_and_transforms_bbox() -> None:
    sample = _sample()
    original_coords = sample["sagittal_coords"].clone()
    transform = RandomTrainingAugmentation(
        seed=7,
        mode="legacy_spatial_photometric",
        probability=1.0,
        max_rotation_degrees=0.0,
        max_brightness_shift=0.0,
        noise_sigma=0.0,
    )

    transformed = transform(sample)

    params = transformed["augmentation"]["sagittal"]
    assert params.horizontal_flip or params.vertical_flip
    assert transformed["labels"]["module1_labels"][0]["bbox"] != [1.0, 2.0, 3.0, 4.0]
    assert torch.equal(transformed["sagittal_coords"], original_coords)


def test_coordinate_safe_spatial_mode_transforms_world_grid_with_pixels() -> None:
    sample = _sample()
    original_coords = sample["sagittal_coords"].clone()
    transform = RandomTrainingAugmentation(
        seed=7,
        mode="spatial_photometric",
        probability=1.0,
        max_rotation_degrees=0.0,
        max_brightness_shift=0.0,
        noise_sigma=0.0,
    )

    transformed = transform(sample)

    params = transformed["augmentation"]["sagittal"]
    expected = original_coords.reshape(1, 2, 2, 3)
    if params.horizontal_flip:
        expected = expected.flip(2)
    if params.vertical_flip:
        expected = expected.flip(1)
    expected = expected.reshape(-1, 3)
    assert torch.equal(transformed["sagittal_coords"], expected)
    assert params.horizontal_flip or params.vertical_flip


def test_roi_augmentation_keeps_cross_plane_spatial_pairing() -> None:
    roi = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) / 63.0
    sample = {"sagittal_roi": roi.clone(), "coronal_roi": roi.clone()}
    transform = RandomROIAugmentation(
        probability=1.0,
        max_rotation_degrees=0.0,
        max_brightness_shift=0.0,
        noise_sigma=0.0,
    )

    transformed = transform(sample)

    assert transformed["augmentation"]["flip_mode"] in {1, 2, 3}
    assert torch.equal(transformed["sagittal_roi"], transformed["coronal_roi"])
    assert not torch.equal(transformed["sagittal_roi"], roi)


def test_training_transform_config_defaults_to_safe_mode_and_can_be_disabled() -> None:
    safe = build_training_transform({"seed": 42, "data": {"train_augmentation": True}})
    disabled = build_training_transform({"seed": 42, "data": {"train_augmentation": False}})

    assert safe is not None
    assert safe.mode == "photometric"
    assert disabled is None

    with pytest.raises(ValueError):
        build_training_transform(
            {
                "seed": 42,
                "data": {
                    "train_augmentation": True,
                    "train_augmentation_mode": "unsafe_unknown_mode",
                },
            }
        )


def test_main_config_uses_paper_spatial_augmentation_with_coordinate_updates() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "configs/module1_meniscovit_cop.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["data"]["train_augmentation"] is True
    assert config["data"]["train_augmentation_mode"] == "spatial_photometric"
    assert config["data"]["train_slice_selection"]["mode"] == "label_chunks"
    transform = build_training_transform(config)
    assert transform is not None
    assert transform.mode == "spatial_photometric"
