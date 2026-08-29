"""Tests for Module 3 bucket-handle Video Swin classifier."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.roi_classification_dataset import ROIClassificationDataset
from engine.train_module3 import build_sampler
from engine.validate_module3 import evaluate_module3, roi_binary_collate, select_f1_threshold
from losses.focal_loss import BinaryFocalLossWithLogits
from models.video_swin.model import VideoSwinClassifier


class Module3BucketTest(unittest.TestCase):
    def test_validation_threshold_maximizes_f1(self) -> None:
        threshold = select_f1_threshold(
            labels=[1, 1, 0, 0],
            scores=[0.9, 0.8, 0.7, 0.1],
        )
        self.assertAlmostEqual(threshold, 0.8)

    def test_sampler_uses_configured_positive_fraction(self) -> None:
        dataset = type(
            "Dataset",
            (),
            {"records": [{"module3_label": 0}] * 9 + [{"module3_label": 1}]},
        )()
        sampler = build_sampler(dataset, positive_fraction=0.25)
        negative_mass = float(sampler.weights[:9].sum())
        positive_mass = float(sampler.weights[9:].sum())
        self.assertAlmostEqual(positive_mass / (positive_mass + negative_mass), 0.25)

    def test_binary_video_swin_forward(self) -> None:
        model = VideoSwinClassifier(
            in_channels=1,
            num_classes=1,
            embed_dim=12,
            patch_size=[2, 4, 4],
            depths=[2, 1],
            num_heads=[3, 6],
            window_size=[2, 4, 4],
            mlp_ratio=2.0,
            dropout=0.0,
            drop_path_rate=0.0,
        )
        sagittal = torch.randn(2, 1, 8, 32, 32)
        coronal = torch.randn(2, 1, 8, 32, 32)
        logits = model(sagittal, coronal).squeeze(-1)
        self.assertEqual(tuple(logits.shape), (2,))
        loss = BinaryFocalLossWithLogits()(logits, torch.tensor([0.0, 1.0]))
        self.assertTrue(torch.isfinite(loss))

    def test_side_specific_validation_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, label in enumerate([0, 1]):
                payload = {
                    "case_id": f"case_{idx}",
                    "medial": {"sagittal_roi": torch.rand(1, 8, 32, 32), "coronal_roi": torch.rand(1, 8, 32, 32), "boxes": {}},
                    "lateral": {"sagittal_roi": torch.rand(1, 8, 32, 32), "coronal_roi": torch.rand(1, 8, 32, 32), "boxes": {}},
                    "labels": {
                        "medial_status": 2,
                        "lateral_status": 2,
                        "medial_bucket_handle": label,
                        "lateral_bucket_handle": 1 - label,
                        "lateral_discoid": 0,
                    },
                }
                torch.save(payload, Path(tmpdir) / f"case_{idx}.pt")
            dataset = ROIClassificationDataset(tmpdir, task="module3", sides=("medial", "lateral"))
            loader = DataLoader(dataset, batch_size=1, collate_fn=roi_binary_collate)
            model = VideoSwinClassifier(
                in_channels=1,
                num_classes=1,
                embed_dim=12,
                patch_size=[2, 4, 4],
                depths=[1],
                num_heads=[3],
                window_size=[2, 4, 4],
                mlp_ratio=2.0,
                dropout=0.0,
                drop_path_rate=0.0,
            )
            metrics = evaluate_module3(model, loader, torch.device("cpu"), threshold=0.5, amp=False, bootstrap_samples=10)
            self.assertIn("overall_roc_auc", metrics)
            self.assertIn("medial_f1", metrics)
            self.assertIn("lateral_f1", metrics)
            self.assertIn("overall_specificity_ci_low", metrics)
            self.assertEqual(metrics["overall_bootstrap_group_count"], 2.0)


if __name__ == "__main__":
    unittest.main()
