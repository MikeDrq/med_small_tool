"""Tests for ROI generation helpers and ROIClassificationDataset."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from datasets.roi_classification_dataset import ROIClassificationDataset
from scripts.generate_rois import crop_roi_sequence, fill_missing_boxes, resolve_score_threshold


class ROIGenerationTest(unittest.TestCase):
    def test_fill_missing_boxes_and_crop_shape(self) -> None:
        boxes = [
            {"box": torch.tensor([0.25, 0.25, 0.75, 0.75]), "score": 1.0, "source": "gt"},
            None,
            {"box": torch.tensor([0.30, 0.30, 0.80, 0.80]), "score": 0.9, "source": "gt"},
        ]
        filled = fill_missing_boxes(boxes, strategy="interpolate")
        self.assertEqual(len(filled), 3)
        volume = torch.rand(1, 3, 64, 64)
        roi = crop_roi_sequence(volume, filled, crop_size=32)
        self.assertEqual(tuple(roi.shape), (1, 3, 32, 32))

    def test_vectorized_roi_crop_matches_slice_then_interpolate(self) -> None:
        volume = torch.arange(3 * 20 * 24, dtype=torch.float32).reshape(1, 3, 20, 24)
        boxes = [
            {"box": torch.tensor([0.0, 0.0, 1.0, 1.0])},
            {"box": torch.tensor([0.12, 0.23, 0.81, 0.91])},
            {"box": torch.tensor([0.52, 0.10, 0.53, 0.12])},
        ]
        actual = crop_roi_sequence(volume, boxes, crop_size=11)

        # Compare against the legacy per-slice crop/interpolate implementation.
        expected_square = []
        for slice_index, item in enumerate(boxes):
            box = item["box"]
            x1 = max(0, int(torch.floor(box[0] * 24)))
            y1 = max(0, int(torch.floor(box[1] * 20)))
            x2 = min(24, max(x1 + 1, int(torch.ceil(box[2] * 24))))
            y2 = min(20, max(y1 + 1, int(torch.ceil(box[3] * 20))))
            crop = volume[:, slice_index, y1:y2, x1:x2].unsqueeze(0)
            expected_square.append(
                F.interpolate(crop, size=(11, 11), mode="bilinear", align_corners=False).squeeze(0)
            )
        self.assertTrue(torch.allclose(actual, torch.stack(expected_square, dim=1), atol=1e-4, rtol=1e-5))

    def test_roi_classification_dataset_reads_pt_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "case_001.pt"
            payload = {
                "case_id": "case_001",
                "group_id": "patient_001",
                "medial": {
                    "sagittal_roi": torch.rand(1, 4, 32, 32),
                    "coronal_roi": torch.rand(1, 4, 32, 32),
                    "boxes": {},
                },
                "lateral": {
                    "sagittal_roi": torch.rand(1, 4, 32, 32),
                    "coronal_roi": torch.rand(1, 4, 32, 32),
                    "boxes": {},
                },
                "labels": {
                    "medial_status": 2,
                    "lateral_status": 0,
                    "medial_bucket_handle": 1,
                    "lateral_bucket_handle": 0,
                    "lateral_discoid": 0,
                },
            }
            torch.save(payload, path)
            dataset = ROIClassificationDataset(tmpdir, task="module2")
            self.assertEqual(len(dataset), 2)
            sample = dataset[0]
            self.assertEqual(tuple(sample["sagittal_roi"].shape), (1, 4, 32, 32))
            self.assertIn(sample["side"], {"medial", "lateral"})
            self.assertEqual(sample["label"].ndim, 0)
            self.assertEqual(sample["group_id"], "patient_001")

            bucket_dataset = ROIClassificationDataset(tmpdir, task="module3")
            self.assertEqual(len(bucket_dataset), 1)
            self.assertEqual(bucket_dataset[0]["side"], "medial")

    def test_score_threshold_prefers_cli_then_config(self) -> None:
        config = {"validate": {"score_threshold": 0.1}}
        self.assertEqual(resolve_score_threshold(config, None), 0.1)
        self.assertEqual(resolve_score_threshold(config, 0.25), 0.25)
        self.assertEqual(resolve_score_threshold({}, None), 0.5)


if __name__ == "__main__":
    unittest.main()
