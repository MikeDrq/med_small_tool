"""Tests for the paper-aligned hierarchical MeniscoViT-CoP backbone."""

from __future__ import annotations

import torch

from models.meniscovit_cop.hierarchical_model import HierarchicalMeniscoViTCoP


def _coords(batch: int, depth: int, image_size: int) -> torch.Tensor:
    grid = image_size // 4
    z, y, x = torch.meshgrid(
        torch.arange(depth, dtype=torch.float32),
        torch.arange(grid, dtype=torch.float32),
        torch.arange(grid, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([x, y, z], dim=-1).reshape(1, -1, 3).repeat(batch, 1, 1)


def test_hierarchical_backbone_restores_original_depth_for_detection() -> None:
    model = HierarchicalMeniscoViTCoP(
        image_size=64,
        in_channels=3,
        embed_dims=[8, 16, 24, 32],
        depths=[1, 1, 1, 1],
        head_dim=8,
        cross_attention_k=4,
        cross_attention_max_tokens=64,
        use_checkpoint=False,
        drop_path_rate=0.0,
    ).eval()
    volume = torch.rand(1, 1, 4, 64, 64)
    sagittal_coords = _coords(1, 4, 64)
    coronal_coords = _coords(1, 4, 64) + torch.tensor([0.1, 0.2, 0.3])

    with torch.no_grad():
        outputs = model(volume, volume, sagittal_coords, coronal_coords)

    assert outputs["sagittal_tokens_fused"].shape == (1, 4 * 2 * 2, 32)
    assert outputs["coronal_tokens_fused"].shape == (1, 4 * 2 * 2, 32)
    assert outputs["shape_metadata"]["sagittal"]["slices"] == 4
    assert outputs["shape_metadata"]["sagittal"]["patch_height"] == 32
    assert outputs["shape_metadata"]["architecture"] == "paper_hierarchical_uniformer"


def test_paper_configuration_is_at_reported_parameter_scale() -> None:
    model = HierarchicalMeniscoViTCoP(
        image_size=512,
        embed_dims=[64, 128, 320, 512],
        depths=[3, 4, 8, 3],
        head_dim=64,
        use_checkpoint=True,
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    # Supplement reports 51.3M including Module-1 heads. Keep a narrow enough
    # bound to catch accidental reversion to the previous compact backbone.
    assert 45_000_000 <= parameters <= 60_000_000
