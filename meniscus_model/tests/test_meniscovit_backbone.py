"""Forward tests for the MeniscoViT-CoP backbone."""

from __future__ import annotations

import unittest

import torch

from models.meniscovit_cop.model import MeniscoViTCoP


class MeniscoViTBackboneTest(unittest.TestCase):
    def test_forward_shapes_with_224_inputs(self) -> None:
        batch_size = 1
        slices = 16
        height = width = 224
        patch_size = 16
        embed_dim = 32
        patches_per_slice = (height // patch_size) * (width // patch_size)
        num_tokens = slices * patches_per_slice

        model = MeniscoViTCoP(
            image_size=224,
            patch_size=patch_size,
            in_channels=1,
            embed_dim=embed_dim,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            cross_attention_k=8,
            dropout=0.0,
            use_checkpoint=False,
            num_slices=slices,
        )
        sagittal_volume = torch.randn(batch_size, 1, slices, height, width)
        coronal_volume = torch.randn(batch_size, 1, slices, height, width)
        sagittal_coords = torch.randn(batch_size, num_tokens, 3)
        coronal_coords = torch.randn(batch_size, num_tokens, 3)

        with torch.inference_mode():
            outputs = model(sagittal_volume, coronal_volume, sagittal_coords, coronal_coords)

        self.assertEqual(outputs["sagittal_tokens_fused"].shape, (batch_size, num_tokens, embed_dim))
        self.assertEqual(outputs["coronal_tokens_fused"].shape, (batch_size, num_tokens, embed_dim))
        self.assertEqual(outputs["per_token_features"].shape, (batch_size, num_tokens * 2, embed_dim))
        self.assertEqual(outputs["fused_global_feature"].shape, (batch_size, embed_dim))
        self.assertEqual(outputs["shape_metadata"]["sagittal"]["slices"], slices)
        self.assertEqual(outputs["shape_metadata"]["sagittal"]["grid_height"], height // patch_size)
        self.assertEqual(outputs["shape_metadata"]["sagittal"]["grid_width"], width // patch_size)

    def test_amp_forward(self) -> None:
        batch_size = 1
        slices = 16
        height = width = 224
        patch_size = 16
        embed_dim = 32
        num_tokens = slices * (height // patch_size) * (width // patch_size)
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
        device = torch.device(device_type)

        model = MeniscoViTCoP(
            image_size=224,
            patch_size=patch_size,
            in_channels=1,
            embed_dim=embed_dim,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            cross_attention_k=8,
            dropout=0.0,
            use_checkpoint=False,
            num_slices=slices,
        ).to(device)
        sagittal_volume = torch.randn(batch_size, 1, slices, height, width, device=device)
        coronal_volume = torch.randn(batch_size, 1, slices, height, width, device=device)
        sagittal_coords = torch.randn(batch_size, num_tokens, 3, device=device)
        coronal_coords = torch.randn(batch_size, num_tokens, 3, device=device)

        with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=dtype):
            outputs = model(sagittal_volume, coronal_volume, sagittal_coords, coronal_coords)

        self.assertEqual(outputs["sagittal_tokens_fused"].shape, (batch_size, num_tokens, embed_dim))
        self.assertEqual(outputs["coronal_tokens_fused"].shape, (batch_size, num_tokens, embed_dim))
        self.assertEqual(outputs["fused_global_feature"].shape, (batch_size, embed_dim))


if __name__ == "__main__":
    unittest.main()
