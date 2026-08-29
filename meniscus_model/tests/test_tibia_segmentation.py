"""Tests for the coronal U2Net preprocessing and label restoration adapter."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from models.tibia_segmentation import CoronalTibiaU2Net


class FixedThreeClassModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = images.shape
        output = torch.zeros(batch, 3, height, width, device=images.device)
        output[:, 0] = 0.1
        output[:, 2, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 1.0
        return output


class TibiaSegmentationTest(unittest.TestCase):
    def test_predict_mask_restores_original_shape_and_class_two(self) -> None:
        segmenter = CoronalTibiaU2Net(
            FixedThreeClassModel(),
            device="cpu",
            input_size=16,
            tibia_class=2,
            batch_size=2,
            amp=False,
        )
        mask = segmenter.predict_mask(torch.rand(1, 3, 20, 30))
        self.assertEqual(tuple(mask.shape), (3, 20, 30))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertTrue(mask[:, 7:13, 10:20].all())
        self.assertFalse(mask[:, 0, 0].any())


if __name__ == "__main__":
    unittest.main()
