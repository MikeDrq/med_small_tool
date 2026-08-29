"""Tests for compact Module 2 Video Swin classifier."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from datasets.roi_classification_dataset import ROIClassificationDataset
from engine.validate_module2 import evaluate_module2, roi_collate
from losses.multitask_loss import ROIClassificationLoss
from models.video_swin.model import PaperVideoSwinClassifier, VideoSwinClassifier
from torch.utils.data import DataLoader


class Module2VideoSwinTest(unittest.TestCase):
    def test_paper_video_swin_is_single_27_8m_backbone(self) -> None:
        model = PaperVideoSwinClassifier(num_classes=3, pretrained=False).eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreater(parameter_count, 27_000_000)
        self.assertLess(parameter_count, 29_000_000)
        logits = model(torch.rand(1, 1, 4, 64, 64), torch.rand(1, 1, 4, 64, 64))
        self.assertEqual(tuple(logits.shape), (1, 3))

    def test_video_swin_forward_and_amp(self) -> None:
        model = VideoSwinClassifier(
            in_channels=1,
            num_classes=3,
            embed_dim=12,
            patch_size=[2, 4, 4],
            depths=[2, 1],
            num_heads=[3, 6],
            window_size=[2, 4, 4],
            mlp_ratio=2.0,
            dropout=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
        )
        sagittal = torch.randn(2, 1, 8, 32, 32)
        coronal = torch.randn(2, 1, 8, 32, 32)
        logits = model(sagittal, coronal)
        self.assertEqual(tuple(logits.shape), (2, 3))
        loss = ROIClassificationLoss(loss_type="focal")(logits, torch.tensor([0, 2]))["loss"]
        self.assertTrue(torch.isfinite(loss))

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            amp_logits = model(sagittal, coronal)
        self.assertEqual(tuple(amp_logits.shape), (2, 3))

    def test_validate_metrics_on_roi_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, label in enumerate([0, 1, 2]):
                payload = {
                    "case_id": f"case_{idx}",
                    "medial": {"sagittal_roi": torch.rand(1, 8, 32, 32), "coronal_roi": torch.rand(1, 8, 32, 32), "boxes": {}},
                    "lateral": {"sagittal_roi": torch.rand(1, 8, 32, 32), "coronal_roi": torch.rand(1, 8, 32, 32), "boxes": {}},
                    "labels": {
                        "medial_status": label,
                        "lateral_status": label,
                        "medial_bucket_handle": 0,
                        "lateral_bucket_handle": 0,
                        "lateral_discoid": 0,
                    },
                }
                torch.save(payload, Path(tmpdir) / f"case_{idx}.pt")
            dataset = ROIClassificationDataset(tmpdir, task="module2", sides=("medial",))
            loader = DataLoader(dataset, batch_size=1, collate_fn=roi_collate)
            model = VideoSwinClassifier(
                in_channels=1,
                num_classes=3,
                embed_dim=12,
                patch_size=[2, 4, 4],
                depths=[1],
                num_heads=[3],
                window_size=[2, 4, 4],
                mlp_ratio=2.0,
                dropout=0.0,
                drop_path_rate=0.0,
            )
            metrics = evaluate_module2(model, loader, torch.device("cpu"), ["normal", "degeneration", "tear"], amp=False, bootstrap_samples=10)
            self.assertIn("accuracy", metrics)
            self.assertIn("confusion_matrix", metrics)
            self.assertIn("normal_f1", metrics)
            self.assertIn("normal_accuracy", metrics)
            self.assertIn("normal_specificity_ci_low", metrics)
            self.assertEqual(metrics["bootstrap_group_count"], 3.0)


if __name__ == "__main__":
    unittest.main()
