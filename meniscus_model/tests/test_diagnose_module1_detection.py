"""Pure tests for the read-only Module 1 diagnostic entrypoint."""

from __future__ import annotations

import unittest

import torch

from scripts.module1_debug.diagnose_module1_detection import (
    classify_failure,
    correlation_report,
    resolve_diagnostic_options,
    select_case_indices,
    token_iou_targets,
)


class DiagnoseModule1DetectionTest(unittest.TestCase):
    def test_resolve_options_inherits_validation_contract(self) -> None:
        base = {
            "seed": 7,
            "validate": {
                "score_threshold": 0.2,
                "topk_per_image": 123,
                "nms_threshold": 0.6,
                "final_score_mode": "objectness_quality",
            },
        }
        diagnostic = {"diagnostics": {"score_threshold": None, "topk_per_image": None, "nms_threshold": None}}
        resolved = resolve_diagnostic_options(base, diagnostic)
        self.assertEqual(resolved["seed"], 7)
        self.assertEqual(resolved["score_threshold"], 0.2)
        self.assertEqual(resolved["topk_per_image"], 123)
        self.assertEqual(resolved["nms_threshold"], 0.6)
        self.assertEqual(resolved["final_score_mode"], "objectness_quality")

    def test_case_subset_is_seeded(self) -> None:
        first = select_case_indices(20, 5, seed=42)
        second = select_case_indices(20, 5, seed=42)
        different = select_case_indices(20, 5, seed=43)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(first, sorted(first))

    def test_geometry_and_strict_iou_separate_side_error(self) -> None:
        boxes = torch.tensor([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.4, 0.4], [0.7, 0.7, 0.9, 0.9]])
        plane_ids = torch.tensor([0, 0, 0])
        slice_indices = torch.tensor([1, 1, 0])
        predicted_side_ids = torch.tensor([1, 0, 0])
        gt_items = [
            {
                "plane_id": 0,
                "processed_slice": 1,
                "side_id": 0,
                "box": torch.tensor([0.1, 0.1, 0.4, 0.4]),
            }
        ]
        targets = token_iou_targets(gt_items, boxes, plane_ids, slice_indices, predicted_side_ids)
        self.assertAlmostEqual(float(targets["geometry"][0]), 1.0)
        self.assertAlmostEqual(float(targets["strict"][0]), 0.0)
        self.assertAlmostEqual(float(targets["strict"][1]), 1.0)
        self.assertTrue(bool(targets["relevant"][0]))
        self.assertFalse(bool(targets["relevant"][2]))

    def test_failure_taxonomy_is_causal(self) -> None:
        threshold = 0.3
        self.assertEqual(classify_failure(0.2, 0.0, 0.0, 0.0, 0.0, threshold), "localization_error")
        self.assertEqual(classify_failure(0.8, 0.2, 0.0, 0.0, 0.0, threshold), "side_classification_error")
        self.assertEqual(classify_failure(0.8, 0.8, 0.2, 0.0, 0.0, threshold), "score_threshold_error")
        self.assertEqual(classify_failure(0.8, 0.8, 0.8, 0.2, 0.0, threshold), "ranking_topk_error")
        self.assertEqual(classify_failure(0.8, 0.8, 0.8, 0.8, 0.2, threshold), "nms_error")
        self.assertEqual(classify_failure(0.8, 0.8, 0.8, 0.8, 0.8, threshold), "retained")

    def test_correlation_report_has_relevant_subset(self) -> None:
        scores = torch.tensor([0.1, 0.9, 0.2, 0.8])
        strict = torch.tensor([0.0, 1.0, 0.0, 0.8])
        geometry = strict.clone()
        relevant = torch.tensor([False, True, False, True])
        report = correlation_report(scores, strict, geometry, relevant)
        self.assertIn("all_token_strict_spearman", report)
        self.assertIn("gt_plane_slice_geometry_spearman", report)
        self.assertGreater(report["gt_plane_slice_geometry_spearman"], 0.99)


if __name__ == "__main__":
    unittest.main()
