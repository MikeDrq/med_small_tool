"""Unit tests for MeniscoViT-CoP core building blocks."""

from __future__ import annotations

import unittest

import torch

from models.meniscovit_cop.patch_embed import MeniscusPatchEmbed
from models.meniscovit_cop.rcpe import RealCoordinatePositionEmbedding
from models.meniscovit_cop.regional_cross_attention import RegionalCrossAttention


class MeniscoViTCoreTest(unittest.TestCase):
    def test_patch_embed_uses_2d_per_slice_patches(self) -> None:
        batch_size, channels, slices, height, width = 2, 1, 4, 64, 80
        embed_dim = 32
        patch_size = 16
        volume = torch.randn(batch_size, channels, slices, height, width)

        patch_embed = MeniscusPatchEmbed(in_channels=channels, embed_dim=embed_dim, patch_size=patch_size)
        tokens, shape = patch_embed(volume)

        self.assertEqual(shape.slices, slices)
        self.assertEqual(shape.grid_height, height // patch_size)
        self.assertEqual(shape.grid_width, width // patch_size)
        self.assertEqual(shape.patches_per_slice, (height // patch_size) * (width // patch_size))
        self.assertEqual(tokens.shape, (batch_size, slices * shape.patches_per_slice, embed_dim))

    def test_rcpe_shape_and_normalization_modes(self) -> None:
        batch_size, num_tokens, embed_dim = 2, 128, 48
        coords = torch.randn(batch_size, num_tokens, 3) * 40.0 + 100.0

        per_case_rcpe = RealCoordinatePositionEmbedding(embed_dim=embed_dim, normalization="per_case")
        per_case_pos = per_case_rcpe(coords)
        self.assertEqual(per_case_pos.shape, (batch_size, num_tokens, embed_dim))

        dataset_rcpe = RealCoordinatePositionEmbedding(
            embed_dim=embed_dim,
            normalization="dataset",
            coord_mean=[100.0, 100.0, 100.0],
            coord_std=[40.0, 40.0, 40.0],
        )
        dataset_pos = dataset_rcpe(coords)
        self.assertEqual(dataset_pos.shape, (batch_size, num_tokens, embed_dim))

    def test_regional_cross_attention_forward_shape_and_local_attention(self) -> None:
        batch_size = 2
        num_query = 96
        num_key_value = 160
        embed_dim = 64
        num_heads = 8
        k_nearest = 32
        query_tokens = torch.randn(batch_size, num_query, embed_dim)
        key_value_tokens = torch.randn(batch_size, num_key_value, embed_dim)
        query_coords = torch.randn(batch_size, num_query, 3) * 50.0
        key_value_coords = torch.randn(batch_size, num_key_value, 3) * 50.0

        attention = RegionalCrossAttention(embed_dim=embed_dim, num_heads=num_heads, k_nearest=k_nearest, attn_drop=0.1, proj_drop=0.1)
        fused, info = attention(
            query_tokens=query_tokens,
            query_coords=query_coords,
            key_value_tokens=key_value_tokens,
            key_value_coords=key_value_coords,
            return_attention=True,
        )

        self.assertEqual(fused.shape, (batch_size, num_query, embed_dim))
        self.assertEqual(info.neighbor_indices.shape, (batch_size, num_query, k_nearest))
        self.assertEqual(info.attention_weights.shape, (batch_size, num_heads, num_query, k_nearest))

        local_attention_elements = info.attention_weights.numel()
        full_attention_elements = batch_size * num_heads * num_query * num_key_value
        self.assertLess(local_attention_elements, full_attention_elements)
        self.assertEqual(local_attention_elements, batch_size * num_heads * num_query * k_nearest)

    def test_regional_cross_attention_supports_autocast_when_available(self) -> None:
        batch_size, num_query, num_key_value, embed_dim, num_heads = 1, 32, 48, 32, 4
        module = RegionalCrossAttention(embed_dim=embed_dim, num_heads=num_heads, k_nearest=16)
        query_tokens = torch.randn(batch_size, num_query, embed_dim)
        key_value_tokens = torch.randn(batch_size, num_key_value, embed_dim)
        query_coords = torch.randn(batch_size, num_query, 3)
        key_value_coords = torch.randn(batch_size, num_key_value, 3)

        if torch.cuda.is_available():
            module = module.cuda()
            query_tokens = query_tokens.cuda()
            key_value_tokens = key_value_tokens.cuda()
            query_coords = query_coords.cuda()
            key_value_coords = key_value_coords.cuda()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                fused = module(query_tokens, query_coords, key_value_tokens, key_value_coords)
        else:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                fused = module(query_tokens, query_coords, key_value_tokens, key_value_coords)

        self.assertEqual(fused.shape, (batch_size, num_query, embed_dim))


if __name__ == "__main__":
    unittest.main()
