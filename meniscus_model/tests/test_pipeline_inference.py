"""Tests for the end-to-end pipeline orchestration with dummy modules."""

from __future__ import annotations

import tempfile
import unittest

import torch
from torch import nn

from models.pipeline import MeniscusDiagnosticPipeline, PipelineCase


class DummyModule1(nn.Module):
    def forward(self, sagittal_volume, coronal_volume, sagittal_coords, coronal_coords):
        del sagittal_volume, coronal_volume, sagittal_coords, coronal_coords
        shape = {
            "sagittal": {"slices": 2, "height": 32, "width": 32, "grid_height": 2, "grid_width": 2, "patches_per_slice": 4},
            "coronal": {"slices": 2, "height": 32, "width": 32, "grid_height": 2, "grid_width": 2, "patches_per_slice": 4},
            "tokens_per_plane": 8,
            "fused_tokens": 16,
        }
        objectness = torch.full((1, 16), -10.0)
        objectness[0, 10] = 10.0
        bbox = torch.full((1, 16, 4), 0.25)
        bbox[0, :, :] = torch.tensor([0.2, 0.2, 0.8, 0.8])
        side_logits = torch.zeros(1, 16, 2)
        side_logits[..., 1] = 5.0
        abnormality_logits = torch.zeros(1, 16, 2)
        abnormality_logits[..., 1] = 5.0
        return {
            "objectness_logits": objectness,
            "bbox_pred": bbox,
            "side_logits": side_logits,
            "abnormality_logits": abnormality_logits,
            "shape_metadata": shape,
            "plane_ids": torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)]),
        }


class DummyModule2(nn.Module):
    def forward(self, sagittal_roi, coronal_roi):
        del sagittal_roi, coronal_roi
        return torch.tensor([[0.0, 0.0, 5.0]])


class DummyModule3(nn.Module):
    def forward(self, sagittal_roi, coronal_roi):
        del sagittal_roi, coronal_roi
        return torch.tensor([[5.0]])


class DummyTibiaSegmenter:
    source = "dummy_u2net"

    def predict_mask(self, volume):
        mask = torch.zeros(volume.shape[-3:], dtype=torch.bool)
        mask[:, 8:24, 4:28] = True
        return mask


class PipelineInferenceTest(unittest.TestCase):
    def test_pipeline_output_schema_and_fallback(self) -> None:
        config = {
            "pipeline": {"amp": False, "roi_crop_size": 16, "roi_fill_strategy": "nearest", "input_size": 32},
            "thresholds": {"detection_score": 0.5, "bucket_handle_score": 0.5, "discoid_beta": 0.4},
            "module4": {"threshold": 0.4, "probability_temperature": 0.05},
            "outputs": {"save_visualizations": False},
        }
        pipeline = MeniscusDiagnosticPipeline(DummyModule1(), DummyModule2(), DummyModule3(), config=config, device="cpu")
        case = PipelineCase(
            case_id="CASE001",
            sagittal_volume=torch.rand(1, 2, 32, 32),
            coronal_volume=torch.rand(1, 2, 32, 32),
            sagittal_coords=torch.rand(8, 3),
            coronal_coords=torch.rand(8, 3),
            sagittal_metadata={"slices": [{"PixelSpacing": [1.0, 1.0]}]},
            coronal_metadata={"slices": [{"PixelSpacing": [1.0, 1.0]}]},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = pipeline.infer_case(case, output_dir=tmpdir, tibia_width=100.0)
        self.assertEqual(result["case_id"], "CASE001")
        self.assertFalse(result["menisci"]["medial"]["detected"])
        self.assertTrue(result["menisci"]["lateral"]["detected"])
        self.assertEqual(result["menisci"]["lateral"]["status"], "tear")
        self.assertTrue(result["menisci"]["lateral"]["bucket_handle"])
        self.assertTrue(result["menisci"]["lateral"]["discoid_available"])
        self.assertAlmostEqual(result["menisci"]["lateral"]["discoid_beta"], 0.192, places=3)
        self.assertIn("boxes", result["menisci"]["lateral"])
        self.assertIn("visualizations", result)

    def test_pipeline_automatically_runs_tibia_segmenter(self) -> None:
        config = {
            "pipeline": {"amp": False, "roi_crop_size": 16, "roi_fill_strategy": "nearest", "input_size": 32},
            "thresholds": {"detection_score": 0.5, "bucket_handle_score": 0.5, "discoid_beta": 0.4},
            "module4": {"threshold": 0.4, "probability_temperature": 0.05, "min_tibia_slice_pixels": 1},
            "outputs": {"save_visualizations": False},
        }
        pipeline = MeniscusDiagnosticPipeline(
            DummyModule1(),
            DummyModule2(),
            DummyModule3(),
            config=config,
            device="cpu",
            tibia_segmenter=DummyTibiaSegmenter(),
        )
        case = PipelineCase(
            case_id="CASE_U2NET",
            sagittal_volume=torch.rand(1, 2, 32, 32),
            coronal_volume=torch.rand(1, 2, 32, 32),
            sagittal_coords=torch.rand(8, 3),
            coronal_coords=torch.rand(8, 3),
            sagittal_metadata={"slices": [{"PixelSpacing": [1.0, 0.5]}]},
            coronal_metadata={"slices": [{"PixelSpacing": [1.0, 0.5]}]},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = pipeline.infer_case(case, output_dir=tmpdir)
        lateral = result["menisci"]["lateral"]
        self.assertTrue(lateral["discoid_available"])
        self.assertEqual(lateral["tibia_width_source"], "dummy_u2net")
        self.assertAlmostEqual(lateral["tibia_width"], 12.0)
        self.assertIsNotNone(lateral["discoid_beta"])


if __name__ == "__main__":
    unittest.main()
