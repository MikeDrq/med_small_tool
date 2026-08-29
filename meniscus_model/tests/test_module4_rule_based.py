"""Tests for Module 4 rule-based discoid detection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from engine.validate_module4 import evaluate_records
from models.module4_rule_based import (
    PairedWidthMeasurement,
    WidthMeasurement,
    detect_discoid_meniscus,
    lateral_coronal_predicted_meniscus_width_from_roi_cache,
    paired_widths_from_mask_and_roi_cache,
    select_paired_width_candidate,
    tibia_width_from_mask,
    tibia_width_from_unet,
)


class DummyTibiaSegmenter:
    source = "dummy_u2net"

    def predict_mask(self, volume: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(volume.shape[-3:], dtype=torch.bool)
        mask[0, 4:12, 3:23] = True
        return mask


class Module4RuleBasedTest(unittest.TestCase):
    def test_beta_rule(self) -> None:
        measurement = detect_discoid_meniscus(45.0, 100.0, threshold=0.4)
        self.assertAlmostEqual(measurement.beta, 0.45)
        self.assertEqual(measurement.predicted_label, 1)
        self.assertTrue(measurement.probability > 0.5)

    def test_mask_width_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mask = np.zeros((2, 20, 30), dtype=np.uint8)
            mask[0, 5:10, 3:23] = 1
            mask_path = Path(tmpdir) / "mask.npy"
            np.save(mask_path, mask)
            tibia = tibia_width_from_mask(mask_path, pixel_spacing_col=0.5)
            self.assertAlmostEqual(tibia.width, 10.0)

        config = {
            "module4": {
                "threshold": 0.4,
                "probability_temperature": 0.05,
                "tibia_width_priority": ["annotation"],
                "meniscus_width_priority": ["gt_bbox"],
                "bootstrap_samples": 10,
            },
            "outputs": {"output_dir": "outputs", "save_visualizations": False},
        }
        records = [
            {
                "case_id": "case_positive",
                "module1_labels": [{"plane": "coronal", "slice_index": 3, "bbox": [0, 0, 45, 10], "meniscus_side": "lateral"}],
                "module4_labels": {"lateral_discoid": 1, "tibia_width": 100.0, "pixel_spacing_col": 1.0},
            },
            {
                "case_id": "case_negative",
                "module1_labels": [{"plane": "coronal", "slice_index": 3, "bbox": [0, 0, 30, 10], "meniscus_side": "lateral"}],
                "module4_labels": {"lateral_discoid": 0, "tibia_width": 100.0, "pixel_spacing_col": 1.0},
            },
        ]
        metrics, results = evaluate_records(config, records)
        self.assertEqual(len(results), 2)
        self.assertEqual(metrics["evaluated_cases"], 2.0)
        self.assertEqual(metrics["threshold"], 0.4)
        self.assertIn("roc_auc", metrics)
        self.assertIn("specificity_ci_low", metrics)
        self.assertEqual(metrics["bootstrap_group_count"], 2.0)
        self.assertEqual(results[0]["predicted_label"], 1)
        self.assertEqual(results[1]["predicted_label"], 0)

    def test_unet_width_adapter(self) -> None:
        volume = torch.rand(1, 2, 20, 30)
        tibia = tibia_width_from_unet(volume, DummyTibiaSegmenter(), pixel_spacing_col=0.5)
        self.assertAlmostEqual(tibia.width, 10.0)
        self.assertEqual(tibia.source, "dummy_u2net")

    def test_tibia_width_ignores_disconnected_outlier(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mask = np.zeros((1, 20, 30), dtype=np.uint8)
            mask[0, 5:12, 8:23] = 1
            mask[0, 2, 1] = 1
            mask_path = Path(tmpdir) / "mask.npy"
            np.save(mask_path, mask)
            tibia = tibia_width_from_mask(mask_path, pixel_spacing_col=0.5)
        self.assertAlmostEqual(tibia.width, 7.5)

    def test_roi_cache_uses_widest_coronal_box_not_highest_score(self) -> None:
        payload = {
            "lateral": {
                "boxes": {
                    "coronal": [
                        {"box": [0.1, 0.2, 0.2, 0.3], "score": 0.9, "source": "module1"},
                        {"box": [0.1, 0.2, 0.5, 0.3], "score": 0.5, "source": "module1"},
                    ]
                }
            }
        }
        measurement = lateral_coronal_predicted_meniscus_width_from_roi_cache(
            payload,
            image_width=100,
            pixel_spacing_col=0.5,
        )
        self.assertIsNotNone(measurement)
        self.assertAlmostEqual(measurement.width, 20.0)
        self.assertEqual(measurement.slice_index, 1)

    def test_roi_cache_rejects_filled_and_fallback_boxes(self) -> None:
        payload = {
            "lateral": {
                "boxes": {
                    "coronal": [
                        {"box": [0.0, 0.0, 1.0, 1.0], "score": 0.0, "source": "fallback_center"},
                        {"box": [0.1, 0.2, 0.9, 0.3], "score": 0.4, "source": "filled_nearest"},
                        {"box": [0.1, 0.2, 0.4, 0.3], "score": 0.5, "source": "pred"},
                    ]
                }
            }
        }
        measurement = lateral_coronal_predicted_meniscus_width_from_roi_cache(payload, image_width=100)
        self.assertIsNotNone(measurement)
        self.assertAlmostEqual(measurement.width, 30.0)
        self.assertEqual(measurement.slice_index, 2)

    def test_roi_cache_spacing_accounts_for_original_image_width(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            torch.save(
                {
                    "lateral": {
                        "boxes": {
                            "coronal": [
                                {"box": [0.1, 0.2, 0.6, 0.3], "score": 0.8, "source": "pred"},
                            ]
                        }
                    }
                },
                cache_dir / "case_256.pt",
            )
            config = {
                "module4": {
                    "threshold": 0.4,
                    "probability_temperature": 0.05,
                    "tibia_width_priority": ["annotation"],
                    "meniscus_width_priority": ["roi_cache"],
                    "roi_cache_dir": str(cache_dir),
                    "image_width": 512,
                    "bootstrap_samples": 0,
                },
                "outputs": {"output_dir": tmpdir, "save_visualizations": False},
            }
            records = [
                {
                    "case_id": "case_256",
                    "module4_labels": {
                        "lateral_discoid": 1,
                        "tibia_width": 100.0,
                        "pixel_spacing_col": 0.5,
                        "image_columns": 256,
                    },
                }
            ]
            _, results = evaluate_records(config, records)
        self.assertAlmostEqual(results[0]["meniscus_width"], 64.0, places=4)

    def test_same_slice_pairing_selects_fullest_tibial_plateau(self) -> None:
        mask = np.zeros((2, 30, 120), dtype=np.uint8)
        mask[0, 5:20, 10:90] = 1
        mask[1, 5:20, 10:110] = 1
        payload = {
            "lateral": {
                "boxes": {
                    "coronal": [
                        {"box": [0.1, 0.2, 0.5, 0.3], "score": 0.8, "source": "pred"},
                        {"box": [0.1, 0.2, 0.55, 0.3], "score": 0.8, "source": "pred"},
                    ]
                }
            }
        }
        paired = paired_widths_from_mask_and_roi_cache(mask, payload, image_width=100)
        self.assertIsNotNone(paired)
        self.assertEqual(paired.meniscus.slice_index, 1)
        self.assertAlmostEqual(paired.meniscus.width, 45.0)
        self.assertAlmostEqual(paired.tibia.width, 100.0)
        self.assertAlmostEqual(paired.beta, 0.45)

    def test_same_slice_pairing_keeps_near_max_plateau_with_wider_meniscus(self) -> None:
        mask = np.zeros((2, 30, 120), dtype=np.uint8)
        mask[0, 5:20, 10:105] = 1
        mask[1, 5:20, 10:110] = 1
        payload = {
            "lateral": {
                "boxes": {
                    "coronal": [
                        {"box": [0.1, 0.2, 0.6, 0.3], "score": 0.8, "source": "pred"},
                        {"box": [0.1, 0.2, 0.3, 0.3], "score": 0.8, "source": "pred"},
                    ]
                }
            }
        }
        paired = paired_widths_from_mask_and_roi_cache(mask, payload, image_width=100)
        self.assertIsNotNone(paired)
        self.assertEqual(paired.meniscus.slice_index, 0)
        self.assertAlmostEqual(paired.tibia.width, 95.0)
        self.assertAlmostEqual(paired.meniscus.width, 50.0, places=5)

    def test_configurable_same_slice_aggregation_strategies(self) -> None:
        def candidate(slice_index: int, w1: float, beta: float) -> PairedWidthMeasurement:
            return PairedWidthMeasurement(
                meniscus=WidthMeasurement(
                    width=w1 * beta,
                    source="pred",
                    slice_index=slice_index,
                ),
                tibia=WidthMeasurement(
                    width=w1,
                    source="mask",
                    slice_index=slice_index,
                ),
            )

        candidates = [
            candidate(10, 100.0, 0.20),
            candidate(11, 99.0, 0.50),
            candidate(12, 98.0, 0.40),
            candidate(13, 70.0, 0.90),
        ]
        selected = {
            strategy: select_paired_width_candidate(
                candidates,
                aggregation_strategy=strategy,
                min_plateau_width_fraction=0.95,
            )
            for strategy in ("max_w2", "max_w1", "median_beta", "q25_beta", "central3_median")
        }
        self.assertEqual(selected["max_w2"].meniscus.slice_index, 11)
        self.assertEqual(selected["max_w1"].meniscus.slice_index, 10)
        self.assertEqual(selected["median_beta"].meniscus.slice_index, 12)
        self.assertEqual(selected["q25_beta"].meniscus.slice_index, 12)
        self.assertEqual(selected["central3_median"].meniscus.slice_index, 11)
        self.assertEqual(selected["max_w2"].candidate_count, 4)
        self.assertEqual(selected["max_w2"].eligible_candidate_count, 3)

    def test_minimum_candidate_slice_requirement(self) -> None:
        candidates = [
            PairedWidthMeasurement(
                meniscus=WidthMeasurement(40.0, source="pred", slice_index=0),
                tibia=WidthMeasurement(100.0, source="mask", slice_index=0),
            )
        ]
        selected = select_paired_width_candidate(
            candidates,
            aggregation_strategy="median_beta",
            min_candidate_slices=2,
        )
        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
