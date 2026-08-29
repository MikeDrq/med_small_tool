"""Tests for Module 1 anchor-free detection heads and loss."""

from __future__ import annotations

import unittest

import torch

from losses.detection_loss import (
    DetectionTargetConfig,
    Module1DetectionLoss,
    build_bbox_quality_targets,
    build_module1_targets,
    build_quality_iou_objectness_targets,
    build_task_aligned_objectness_targets,
)
from models.meniscovit_cop.detection_heads import MeniscusDetectionHeads
from utils.metrics import (
    compute_module1_final_scores,
    detection_match_counts,
    filter_module1_predictions,
    match_final_predictions,
    module1_token_metrics,
    objectness_iou_alignment_metrics,
    oracle_positive_bbox_ious,
    summarize_iou_detection_metrics,
    topk_detection_diagnostics,
)


class Module1DetectionTest(unittest.TestCase):
    def _backbone_outputs(self) -> dict:
        batch_size, tokens, embed_dim = 1, 8, 16
        shape = {
            "sagittal": {
                "slices": 2,
                "height": 64,
                "width": 64,
                "grid_height": 2,
                "grid_width": 2,
                "patches_per_slice": 4,
            },
            "coronal": {
                "slices": 2,
                "height": 64,
                "width": 64,
                "grid_height": 2,
                "grid_width": 2,
                "patches_per_slice": 4,
            },
            "tokens_per_plane": tokens,
            "fused_tokens": tokens * 2,
        }
        return {
            "sagittal_tokens_fused": torch.randn(batch_size, tokens, embed_dim),
            "coronal_tokens_fused": torch.randn(batch_size, tokens, embed_dim),
            "shape_metadata": shape,
        }

    def _labels(self) -> list[dict]:
        return [
            {
                "case_id": "case_001",
                "module1_labels": [
                    {
                        "plane": "sagittal",
                        "slice_index": 0,
                        "bbox": [0, 0, 40, 40],
                        "meniscus_side": "medial",
                        "abnormality": "abnormal",
                    }
                ],
            }
        ]

    def test_detection_head_shapes(self) -> None:
        outputs = self._backbone_outputs()
        head = MeniscusDetectionHeads(embed_dim=16, hidden_dim=16)
        predictions = head(outputs)
        self.assertEqual(predictions["objectness_logits"].shape, (1, 16))
        self.assertNotIn("bbox_quality_logits", predictions)
        self.assertEqual(predictions["bbox_pred"].shape, (1, 16, 4))
        self.assertEqual(predictions["side_logits"].shape, (1, 16, 2))
        self.assertEqual(predictions["abnormality_logits"].shape, (1, 16, 2))
        self.assertEqual(predictions["plane_ids"].shape, (16,))
        self.assertTrue(torch.all(predictions["bbox_pred"] >= 0))
        self.assertTrue(torch.all(predictions["bbox_pred"] <= 1))

    def test_bbox_quality_head_shape_when_enabled(self) -> None:
        outputs = self._backbone_outputs()
        head = MeniscusDetectionHeads(embed_dim=16, hidden_dim=16, use_bbox_quality_head=True)
        predictions = head(outputs)

        self.assertEqual(predictions["bbox_quality_logits"].shape, (1, 16))
        self.assertEqual(predictions["objectness_logits"].shape, (1, 16))

    def test_ltrb_from_token_center_bbox_param(self) -> None:
        outputs = self._backbone_outputs()
        head = MeniscusDetectionHeads(embed_dim=16, hidden_dim=16, bbox_param="ltrb_from_token_center")
        predictions = head(outputs)

        self.assertEqual(predictions["bbox_pred"].shape, (1, 16, 4))
        self.assertTrue(torch.all(predictions["bbox_pred"] >= 0))
        self.assertTrue(torch.all(predictions["bbox_pred"] <= 1))
        self.assertTrue(torch.all(predictions["bbox_pred"][..., 2] >= predictions["bbox_pred"][..., 0]))
        self.assertTrue(torch.all(predictions["bbox_pred"][..., 3] >= predictions["bbox_pred"][..., 1]))

    def test_target_assignment_and_loss(self) -> None:
        outputs = self._backbone_outputs()
        head = MeniscusDetectionHeads(embed_dim=16, hidden_dim=16)
        predictions = head(outputs)
        labels = self._labels()
        targets = build_module1_targets(labels, predictions["shape_metadata"], device=torch.device("cpu"))
        self.assertTrue(targets["positive_mask"].any())

        criterion = Module1DetectionLoss()
        loss_dict = criterion(predictions, labels)
        self.assertIn("loss", loss_dict)
        self.assertIn("loss_objectness_positive", loss_dict)
        self.assertIn("loss_objectness_negative", loss_dict)
        self.assertIn("objectness_positive_tokens", loss_dict)
        self.assertIn("objectness_negative_tokens", loss_dict)
        self.assertTrue(torch.isfinite(loss_dict["loss"]))
        self.assertGreater(int(loss_dict["objectness_positive_tokens"].item()), 0)
        self.assertGreater(int(loss_dict["objectness_negative_tokens"].item()), 0)
        reconstructed_objectness = (
            loss_dict["loss_objectness_positive_contribution"]
            + loss_dict["loss_objectness_negative_contribution"]
            + loss_dict["loss_objectness_soft_positive_contribution"]
        )
        self.assertTrue(torch.allclose(loss_dict["loss_objectness"], reconstructed_objectness))

        token_loss = criterion._objectness_elementwise_loss(predictions["objectness_logits"], targets["objectness"])
        expected = token_loss.sum() / targets["positive_mask"].sum().clamp_min(1)
        self.assertTrue(torch.allclose(loss_dict["loss_objectness"], expected))
        self.assertEqual(int(loss_dict["objectness_normalizer"].item()), int(targets["positive_mask"].sum().item()))

    def test_legacy_all_token_objectness_normalization_is_explicit(self) -> None:
        outputs = self._backbone_outputs()
        predictions = MeniscusDetectionHeads(embed_dim=16, hidden_dim=16)(outputs)
        labels = self._labels()
        targets = build_module1_targets(labels, predictions["shape_metadata"], device=torch.device("cpu"))
        criterion = Module1DetectionLoss(objectness_normalization="all_tokens")

        loss_dict = criterion(predictions, labels)
        token_loss = criterion._objectness_elementwise_loss(predictions["objectness_logits"], targets["objectness"])

        self.assertTrue(torch.allclose(loss_dict["loss_objectness"], token_loss.mean()))
        self.assertEqual(int(loss_dict["objectness_normalizer"].item()), token_loss.numel())

    def test_positive_radius_expands_targets_on_same_slice(self) -> None:
        outputs = self._backbone_outputs()
        head = MeniscusDetectionHeads(embed_dim=16, hidden_dim=16)
        predictions = head(outputs)
        labels = self._labels()

        base_targets = build_module1_targets(labels, predictions["shape_metadata"], device=torch.device("cpu"))
        expanded_targets = build_module1_targets(
            labels,
            predictions["shape_metadata"],
            device=torch.device("cpu"),
            target_config=DetectionTargetConfig(positive_radius_tokens=1),
        )

        self.assertEqual(int(base_targets["positive_mask"].sum()), 1)
        self.assertEqual(int(expanded_targets["positive_mask"].sum()), 4)

    def test_detection_metric_requires_matching_slice(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        labels = [
            {
                "case_id": "case_001",
                "module1_labels": [
                    {
                        "plane": "sagittal",
                        "slice_index": 1,
                        "bbox": [0, 0, 40, 40],
                        "meniscus_side": "medial",
                        "abnormality": "abnormal",
                    }
                ],
            }
        ]
        predictions = {
            "objectness_logits": torch.full((1, 16), -10.0),
            "bbox_pred": torch.full((1, 16, 4), 0.0),
            "side_logits": torch.zeros(1, 16, 2),
            "plane_ids": torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)]),
            "shape_metadata": shape,
        }
        predictions["objectness_logits"][0, 0] = 10.0
        predictions["bbox_pred"][0, 0] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])
        predictions["side_logits"][0, :, 0] = 5.0

        counts = detection_match_counts(predictions, labels, score_threshold=0.5, iou_threshold=0.3)

        self.assertEqual(counts["tp"], 0)
        self.assertEqual(counts["fp"], 1)
        self.assertEqual(counts["fn"], 1)

    def test_detection_metric_requires_matching_side(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        predictions = {
            "objectness_logits": torch.full((1, 16), -10.0),
            "bbox_pred": torch.full((1, 16, 4), 0.0),
            "side_logits": torch.zeros(1, 16, 2),
            "plane_ids": torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)]),
            "shape_metadata": shape,
        }
        predictions["objectness_logits"][0, 0] = 10.0
        predictions["bbox_pred"][0, 0] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])
        predictions["side_logits"][0, :, 1] = 5.0

        counts = detection_match_counts(predictions, labels, score_threshold=0.5, iou_threshold=0.3)

        self.assertEqual(counts["tp"], 0)
        self.assertEqual(counts["fp"], 1)
        self.assertEqual(counts["fn"], 1)

    def test_objectness_token_pr_auc_name(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        predictions = {
            "objectness_logits": torch.full((1, 16), -10.0),
            "bbox_pred": torch.zeros(1, 16, 4),
            "side_logits": torch.zeros(1, 16, 2),
            "abnormality_logits": torch.zeros(1, 16, 2),
            "shape_metadata": shape,
        }
        predictions["objectness_logits"][0, 0] = 10.0

        metrics = module1_token_metrics(predictions, labels)

        self.assertIn("objectness_token_pr_auc", metrics)
        self.assertNotIn("detection_pr_auc", metrics)

    def test_final_prediction_filtering_topk_and_nms(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        predictions = {
            "objectness_logits": torch.full((1, 16), -10.0),
            "bbox_pred": torch.zeros(1, 16, 4),
            "side_logits": torch.zeros(1, 16, 2),
            "shape_metadata": shape,
        }
        predictions["side_logits"][0, :, 0] = 5.0
        predictions["objectness_logits"][0, 0] = 10.0
        predictions["objectness_logits"][0, 1] = 9.0
        predictions["objectness_logits"][0, 2] = 8.0
        predictions["bbox_pred"][0, 0] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])
        predictions["bbox_pred"][0, 1] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])
        predictions["bbox_pred"][0, 2] = torch.tensor([0.6, 0.6, 0.9, 0.9])

        final_predictions = filter_module1_predictions(
            predictions,
            score_threshold=0.5,
            topk_per_image=2,
            nms_threshold=0.5,
        )
        counts = final_predictions[0]["counts"]

        self.assertEqual(counts["num_raw_predictions"], 16)
        self.assertEqual(counts["num_after_score_threshold"], 3)
        self.assertEqual(counts["num_after_topk"], 2)
        self.assertEqual(counts["num_after_nms"], 1)
        self.assertEqual(counts["num_final_predictions_for_matching"], 1)

        match_counts = match_final_predictions(final_predictions, labels, shape, iou_threshold=0.3)
        self.assertEqual(match_counts["tp"], 1)
        self.assertEqual(match_counts["fp"], 0)
        self.assertEqual(match_counts["fn"], 0)

    def test_topk_and_oracle_bbox_diagnostics(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        predictions = {
            "objectness_logits": torch.full((1, 16), -10.0),
            "bbox_pred": torch.zeros(1, 16, 4),
            "side_logits": torch.zeros(1, 16, 2),
            "abnormality_logits": torch.zeros(1, 16, 2),
            "shape_metadata": shape,
        }
        predictions["side_logits"][0, :, 0] = 5.0
        predictions["objectness_logits"][0, 0] = 10.0
        predictions["bbox_pred"][0, 0] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])

        diagnostics = topk_detection_diagnostics(predictions, labels, topk=4, detail_count=2)
        oracle_ious = oracle_positive_bbox_ious(predictions, labels)

        self.assertEqual(diagnostics[0]["topk"], 4)
        self.assertGreaterEqual(diagnostics[0]["top100_positive_target_tokens"], 1)
        self.assertEqual(len(diagnostics[0]["top10_predictions"]), 2)
        self.assertGreaterEqual(max(oracle_ious), 0.99)

    def test_iou_detection_summary_uses_primary_threshold(self) -> None:
        metrics = summarize_iou_detection_metrics(
            {
                0.1: {"tp": 1, "fp": 1, "fn": 0},
                0.3: {"tp": 0, "fp": 2, "fn": 1},
                0.45: {"tp": 0, "fp": 2, "fn": 1},
            },
            best_ious=[0.2, 0.4],
            primary_iou_threshold=0.3,
        )

        self.assertEqual(metrics["iou_detection_tp"], 0.0)
        self.assertEqual(metrics["iou_detection_fp"], 2.0)
        self.assertEqual(metrics["iou_detection_fn"], 1.0)
        self.assertIn("iou_detection_f1@0.1", metrics)
        self.assertAlmostEqual(metrics["mean_best_iou"], 0.3)

    def test_weighted_bce_objectness_loss(self) -> None:
        outputs = self._backbone_outputs()
        head = MeniscusDetectionHeads(embed_dim=16, hidden_dim=16)
        predictions = head(outputs)
        criterion = Module1DetectionLoss(
            objectness_loss_type="weighted_bce",
            objectness_positive_weight=1.0,
            objectness_negative_weight=3.0,
        )

        loss_dict = criterion(predictions, self._labels())

        self.assertTrue(torch.isfinite(loss_dict["loss"]))
        self.assertGreater(float(loss_dict["loss_objectness_negative"]), 0.0)

    def test_iou_aware_objectness_target_does_not_penalize_good_non_center_box(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        predictions = {
            "objectness_logits": torch.full((1, 16), -10.0),
            "bbox_pred": torch.zeros(1, 16, 4),
            "side_logits": torch.zeros(1, 16, 2),
            "abnormality_logits": torch.zeros(1, 16, 2),
            "shape_metadata": shape,
        }
        predictions["objectness_logits"][0, 3] = 10.0
        predictions["bbox_pred"][0, 3] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])

        assigned_loss = Module1DetectionLoss(objectness_loss_type="weighted_bce")
        iou_aware_loss = Module1DetectionLoss(
            objectness_loss_type="weighted_bce",
            objectness_target_mode="assigned_or_pred_iou",
        )

        assigned = assigned_loss(predictions, labels)
        iou_aware = iou_aware_loss(predictions, labels)

        self.assertLess(float(iou_aware["loss_objectness"]), float(assigned["loss_objectness"]))
        self.assertGreaterEqual(float(iou_aware["objectness_target_max"]), 1.0)
        self.assertGreater(int(iou_aware["objectness_soft_positive_tokens"].item()), int(assigned["objectness_soft_positive_tokens"].item()))

    def test_quality_iou_objectness_uses_strict_key_and_warmup(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        predictions = {
            "objectness_logits": torch.full((1, 16), -10.0),
            "bbox_pred": torch.zeros(1, 16, 4, requires_grad=True),
            "side_logits": torch.zeros(1, 16, 2),
            "abnormality_logits": torch.zeros(1, 16, 2),
            "shape_metadata": shape,
        }
        predictions["side_logits"][0, :, 0] = 5.0
        predictions["objectness_logits"][0, 3] = 10.0
        predictions["bbox_pred"].data[0, 3] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])

        criterion = Module1DetectionLoss(
            objectness_loss_type="weighted_bce",
            objectness_target_mode="quality_iou",
            quality_iou_threshold=0.3,
            quality_iou_start_epoch=30,
        )
        criterion.set_epoch(29)
        warmup = criterion(predictions, labels)
        criterion.set_epoch(30)
        quality = criterion(predictions, labels)

        self.assertEqual(int(warmup["num_quality_positive"].item()), 0)
        self.assertGreater(int(quality["num_quality_positive"].item()), 0)
        self.assertGreater(float(quality["quality_target_max"]), 0.99)
        self.assertIsNone(predictions["bbox_pred"].grad)

    def test_bce_objectness_loss_accepts_quality_soft_targets(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        predictions = {
            "objectness_logits": torch.full((1, 16), -2.0),
            "bbox_pred": torch.zeros(1, 16, 4),
            "side_logits": torch.zeros(1, 16, 2),
            "abnormality_logits": torch.zeros(1, 16, 2),
            "shape_metadata": shape,
        }
        predictions["side_logits"][0, :, 0] = 5.0
        predictions["bbox_pred"][0, 3] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])
        criterion = Module1DetectionLoss(
            objectness_loss_type="bce",
            objectness_target_mode="quality_iou",
            quality_iou_threshold=0.3,
        )
        criterion.set_epoch(30)

        loss_dict = criterion(predictions, labels)

        self.assertTrue(torch.isfinite(loss_dict["loss_objectness"]))
        self.assertGreater(int(loss_dict["num_quality_positive"].item()), 0)

    def test_objectness_iou_alignment_and_topk_quality_diagnostics(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        predictions = {
            "objectness_logits": torch.full((1, 16), -10.0),
            "bbox_pred": torch.zeros(1, 16, 4),
            "side_logits": torch.zeros(1, 16, 2),
            "abnormality_logits": torch.zeros(1, 16, 2),
            "shape_metadata": shape,
        }
        predictions["side_logits"][0, :, 0] = 5.0
        predictions["objectness_logits"][0, 3] = 10.0
        predictions["bbox_pred"][0, 3] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])

        diagnostics = topk_detection_diagnostics(predictions, labels, topk=4, quality_iou_threshold=0.3)
        alignment = objectness_iou_alignment_metrics(predictions, labels, quality_iou_threshold=0.3)

        self.assertGreaterEqual(diagnostics[0]["top100_quality_positive_tokens"], 1)
        self.assertIn("objectness_iou_pearson_corr", alignment)
        self.assertIn("objectness_iou_spearman_corr", alignment)

    def test_quality_iou_target_promotes_good_non_assigned_token(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        assigned = build_module1_targets(labels, shape, device=torch.device("cpu"))
        pred_boxes = torch.zeros(1, 16, 4)
        pred_boxes[0, 3] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])
        side_logits = torch.zeros(1, 16, 2)
        side_logits[0, :, 0] = 5.0

        result = build_quality_iou_objectness_targets(
            labels,
            shape,
            pred_boxes,
            side_logits,
            assigned["objectness"],
            assigned["positive_mask"],
            device=torch.device("cpu"),
            quality_iou_threshold=0.3,
        )

        self.assertFalse(bool(assigned["positive_mask"][0, 3].item()))
        self.assertGreater(float(result["objectness_targets"][0, 3].item()), 0.99)

    def test_quality_iou_target_is_detached_from_pred_bbox(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        assigned = build_module1_targets(labels, shape, device=torch.device("cpu"))
        pred_boxes = torch.zeros(1, 16, 4)
        pred_boxes[0, 3] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])
        pred_boxes.requires_grad_()
        side_logits = torch.zeros(1, 16, 2)
        side_logits[0, :, 0] = 5.0
        objectness_logits = torch.zeros(1, 16, requires_grad=True)

        result = build_quality_iou_objectness_targets(
            labels,
            shape,
            pred_boxes,
            side_logits,
            assigned["objectness"],
            assigned["positive_mask"],
            device=torch.device("cpu"),
            quality_iou_threshold=0.3,
        )
        criterion = Module1DetectionLoss(objectness_loss_type="bce")
        objectness_loss = criterion._objectness_elementwise_loss(objectness_logits, result["objectness_targets"]).mean()
        objectness_loss.backward()

        self.assertIsNone(pred_boxes.grad)
        self.assertIsNotNone(objectness_logits.grad)

    def test_assigned_positive_target_priority_over_quality_iou(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        assigned = build_module1_targets(labels, shape, device=torch.device("cpu"))
        assigned_index = int(torch.nonzero(assigned["positive_mask"][0], as_tuple=False).flatten()[0].item())
        pred_boxes = torch.zeros(1, 16, 4)
        side_logits = torch.zeros(1, 16, 2)
        side_logits[0, :, 0] = 5.0

        result = build_quality_iou_objectness_targets(
            labels,
            shape,
            pred_boxes,
            side_logits,
            assigned["objectness"],
            assigned["positive_mask"],
            device=torch.device("cpu"),
            quality_iou_threshold=0.3,
        )

        self.assertEqual(float(result["objectness_targets"][0, assigned_index].item()), 1.0)

    def test_soft_target_bce_accepts_fractional_objectness_targets(self) -> None:
        logits = torch.zeros(3, requires_grad=True)
        targets = torch.tensor([0.3, 0.5, 0.8])
        criterion = Module1DetectionLoss(objectness_loss_type="weighted_bce")

        loss = criterion._objectness_elementwise_loss(logits, targets).mean()
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)

    def test_task_aligned_topk_selects_tokens_per_gt(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        assigned = build_module1_targets(labels, shape, device=torch.device("cpu"))
        pred_boxes = torch.zeros(1, 16, 4)
        pred_boxes[0, 0] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])
        pred_boxes[0, 1] = torch.tensor([0.0, 0.0, 38.0 / 64.0, 38.0 / 64.0])
        pred_boxes[0, 2] = torch.tensor([0.0, 0.0, 36.0 / 64.0, 36.0 / 64.0])
        objectness_logits = torch.zeros(1, 16)
        objectness_logits[0, 0] = 1.0
        objectness_logits[0, 1] = 0.8
        objectness_logits[0, 2] = 0.6
        side_logits = torch.zeros(1, 16, 2)
        side_logits[0, :, 0] = 5.0

        result = build_task_aligned_objectness_targets(
            labels,
            shape,
            pred_boxes,
            objectness_logits,
            side_logits,
            assigned["objectness"],
            assigned["positive_mask"],
            device=torch.device("cpu"),
            alpha=1.0,
            beta=6.0,
            topk=3,
        )

        self.assertGreaterEqual(int(result["task_aligned_mask"].sum().item()), 3)
        self.assertEqual(float(result["objectness_targets"][0, 0].item()), 1.0)

    def test_bbox_quality_loss_does_not_backprop_to_pred_bbox(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        pred_boxes = torch.zeros(1, 16, 4)
        pred_boxes[0, 3] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])
        pred_boxes.requires_grad_()
        predictions = {
            "objectness_logits": torch.zeros(1, 16),
            "bbox_quality_logits": torch.zeros(1, 16, requires_grad=True),
            "bbox_pred": pred_boxes,
            "side_logits": torch.zeros(1, 16, 2),
            "abnormality_logits": torch.zeros(1, 16, 2),
            "shape_metadata": shape,
        }
        predictions["side_logits"][0, :, 0] = 5.0
        criterion = Module1DetectionLoss(
            objectness_weight=0.0,
            bbox_weight=0.0,
            side_weight=0.0,
            abnormality_weight=0.0,
            bbox_quality_loss_weight=1.0,
            bbox_quality_start_epoch=0,
            bbox_quality_train_mask="iou_above_threshold",
            bbox_quality_iou_threshold=0.3,
        )

        loss_dict = criterion(predictions, labels)
        loss_dict["loss"].backward()

        self.assertIsNone(pred_boxes.grad)
        self.assertIsNotNone(predictions["bbox_quality_logits"].grad)
        self.assertGreater(int(loss_dict["num_bbox_quality_tokens"].item()), 0)

    def test_final_score_mode_quality_uses_bbox_quality_logits(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        predictions = {
            "objectness_logits": torch.tensor([[10.0, 9.0, -10.0] + [-10.0] * 13]),
            "bbox_quality_logits": torch.tensor([[-10.0, 10.0, -10.0] + [-10.0] * 13]),
            "bbox_pred": torch.zeros(1, 16, 4),
            "side_logits": torch.zeros(1, 16, 2),
            "shape_metadata": shape,
        }
        predictions["side_logits"][0, :, 0] = 5.0
        predictions["bbox_pred"][0, 0] = torch.tensor([0.6, 0.6, 0.9, 0.9])
        predictions["bbox_pred"][0, 1] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])

        final_predictions = filter_module1_predictions(
            predictions,
            score_threshold=0.5,
            topk_per_image=1,
            nms_threshold=None,
            final_score_mode="quality",
        )

        self.assertEqual(int(final_predictions[0]["indices"][0].item()), 1)

    def test_final_score_mode_objectness_quality_multiplies_scores(self) -> None:
        predictions = {
            "objectness_logits": torch.tensor([[0.0, 2.0]]),
            "bbox_quality_logits": torch.tensor([[2.0, 0.0]]),
        }

        scores = compute_module1_final_scores(predictions, final_score_mode="objectness_quality")

        expected = torch.sigmoid(predictions["objectness_logits"]) * torch.sigmoid(predictions["bbox_quality_logits"])
        self.assertTrue(torch.allclose(scores, expected))

    def test_bbox_quality_topk_by_alignment_mask_limits_tokens(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        assigned = build_module1_targets(labels, shape, device=torch.device("cpu"))
        pred_boxes = torch.zeros(1, 16, 4)
        pred_boxes[0, 0] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])
        pred_boxes[0, 1] = torch.tensor([0.0, 0.0, 38.0 / 64.0, 38.0 / 64.0])
        pred_boxes[0, 2] = torch.tensor([0.0, 0.0, 36.0 / 64.0, 36.0 / 64.0])
        objectness_logits = torch.zeros(1, 16)
        objectness_logits[0, 0] = 1.0
        objectness_logits[0, 1] = 0.8
        objectness_logits[0, 2] = 0.6
        side_logits = torch.zeros(1, 16, 2)
        side_logits[0, :, 0] = 5.0

        result = build_bbox_quality_targets(
            labels,
            shape,
            pred_boxes,
            objectness_logits,
            side_logits,
            assigned["positive_mask"],
            device=torch.device("cpu"),
            train_mask="topk_by_alignment",
            topk=2,
            alpha=1.0,
            beta=6.0,
        )

        self.assertEqual(int(result["quality_mask"].sum().item()), 2)
        self.assertGreater(float(result["quality_targets"][result["quality_mask"]].max().item()), 0.9)

    def test_bbox_quality_hard_pos_neg_sampling_selects_hard_negative(self) -> None:
        torch.manual_seed(7)
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        assigned = build_module1_targets(labels, shape, device=torch.device("cpu"))
        pred_boxes = torch.zeros(1, 16, 4)
        pred_boxes[0, 0] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])
        pred_boxes[0, 1] = torch.tensor([0.0, 0.0, 38.0 / 64.0, 38.0 / 64.0])
        pred_boxes[0, 2] = torch.tensor([0.0, 0.0, 20.0 / 64.0, 20.0 / 64.0])
        pred_boxes[0, 3] = torch.tensor([0.7, 0.7, 0.9, 0.9])
        objectness_logits = torch.zeros(1, 16)
        objectness_logits[0, 3] = 8.0
        side_logits = torch.zeros(1, 16, 2)
        side_logits[0, :, 0] = 5.0

        result = build_bbox_quality_targets(
            labels,
            shape,
            pred_boxes,
            objectness_logits,
            side_logits,
            assigned["positive_mask"],
            device=torch.device("cpu"),
            train_mask="hard_pos_neg_iou_sampling",
            quality_pos_iou_threshold=0.3,
            quality_neg_iou_threshold=0.1,
            quality_num_pos=2,
            quality_num_neg=1,
            quality_num_mid=1,
            quality_hard_negative_source="objectness",
            quality_mid_weight=0.25,
            quality_neg_target_mode="zero",
            quality_sampling_per_group=False,
        )

        self.assertTrue(bool(result["quality_mask"][0, 0].item()))
        self.assertTrue(bool(result["quality_mask"][0, 1].item()))
        self.assertTrue(bool(result["quality_mask"][0, 3].item()))
        self.assertEqual(int(result["num_quality_pos_samples"].item()), 2)
        self.assertEqual(int(result["num_quality_neg_samples"].item()), 1)
        self.assertEqual(int(result["num_quality_mid_samples"].item()), 1)
        self.assertAlmostEqual(float(result["quality_targets"][0, 3].item()), 0.0, places=6)
        self.assertAlmostEqual(float(result["quality_weights"][0, 2].item()), 0.25, places=6)

    def test_bbox_quality_hard_negative_can_use_quality_score_source(self) -> None:
        torch.manual_seed(13)
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        assigned = build_module1_targets(labels, shape, device=torch.device("cpu"))
        pred_boxes = torch.zeros(1, 16, 4)
        pred_boxes[0, 0] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])
        pred_boxes[0, 3] = torch.tensor([0.7, 0.7, 0.9, 0.9])
        objectness_logits = torch.zeros(1, 16)
        bbox_quality_logits = torch.zeros(1, 16)
        bbox_quality_logits[0, 3] = 8.0
        side_logits = torch.zeros(1, 16, 2)
        side_logits[0, :, 0] = 5.0

        result = build_bbox_quality_targets(
            labels,
            shape,
            pred_boxes,
            objectness_logits,
            side_logits,
            assigned["positive_mask"],
            device=torch.device("cpu"),
            train_mask="hard_pos_neg_iou_sampling",
            quality_pos_iou_threshold=0.3,
            quality_neg_iou_threshold=0.1,
            quality_num_pos=1,
            quality_num_neg=1,
            quality_num_mid=0,
            quality_hard_negative_source="quality",
            quality_neg_target_mode="zero",
            quality_sampling_per_group=False,
            bbox_quality_logits=bbox_quality_logits,
        )

        self.assertTrue(bool(result["quality_mask"][0, 3].item()))
        self.assertEqual(int(result["num_quality_neg_samples"].item()), 1)
        self.assertAlmostEqual(float(result["quality_targets"][0, 3].item()), 0.0, places=6)

    def test_bbox_quality_sampled_soft_iou_uses_configured_ratios(self) -> None:
        torch.manual_seed(11)
        shape = self._backbone_outputs()["shape_metadata"]
        labels = [
            {
                "case_id": "case_001",
                "module1_labels": [
                    {"plane": "sagittal", "slice_index": 0, "bbox": [0, 0, 40, 40], "meniscus_side": "medial", "abnormality": "abnormal"},
                    {"plane": "sagittal", "slice_index": 1, "bbox": [0, 0, 40, 40], "meniscus_side": "medial", "abnormality": "abnormal"},
                    {"plane": "coronal", "slice_index": 0, "bbox": [0, 0, 40, 40], "meniscus_side": "medial", "abnormality": "abnormal"},
                    {"plane": "coronal", "slice_index": 1, "bbox": [0, 0, 40, 40], "meniscus_side": "medial", "abnormality": "abnormal"},
                ],
            }
        ]
        assigned = build_module1_targets(labels, shape, device=torch.device("cpu"))
        pred_boxes = torch.zeros(1, 16, 4)
        pred_boxes[0, 0:4] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])
        pred_boxes[0, 4:8] = torch.tensor([0.0, 0.0, 20.0 / 64.0, 20.0 / 64.0])
        pred_boxes[0, 8:16] = torch.tensor([0.7, 0.7, 0.9, 0.9])
        objectness_logits = torch.zeros(1, 16)
        side_logits = torch.zeros(1, 16, 2)
        side_logits[0, :, 0] = 5.0

        result = build_bbox_quality_targets(
            labels,
            shape,
            pred_boxes,
            objectness_logits,
            side_logits,
            assigned["positive_mask"],
            device=torch.device("cpu"),
            train_mask="sampled_soft_iou",
            quality_pos_iou_threshold=0.3,
            quality_neg_iou_threshold=0.1,
            quality_sample_total=8,
            quality_high_iou_fraction=0.25,
            quality_mid_iou_fraction=0.25,
            quality_low_iou_fraction=0.50,
        )

        self.assertEqual(int(result["quality_mask"].sum().item()), 8)
        self.assertEqual(int(result["num_quality_pos_samples"].item()), 2)
        self.assertEqual(int(result["num_quality_mid_samples"].item()), 2)
        self.assertEqual(int(result["num_quality_neg_samples"].item()), 4)

    def test_quality_ranking_loss_penalizes_low_positive_score_and_detaches_bbox(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        pred_boxes = torch.zeros(1, 16, 4)
        pred_boxes[0, 0] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])
        pred_boxes[0, 3] = torch.tensor([0.7, 0.7, 0.9, 0.9])
        pred_boxes.requires_grad_()
        predictions = {
            "objectness_logits": torch.zeros(1, 16),
            "bbox_quality_logits": torch.zeros(1, 16, requires_grad=True),
            "bbox_pred": pred_boxes,
            "side_logits": torch.zeros(1, 16, 2),
            "abnormality_logits": torch.zeros(1, 16, 2),
            "shape_metadata": shape,
        }
        predictions["objectness_logits"][0, 3] = 8.0
        predictions["bbox_quality_logits"].data[0, 0] = -2.0
        predictions["bbox_quality_logits"].data[0, 3] = 2.0
        predictions["side_logits"][0, :, 0] = 5.0
        criterion = Module1DetectionLoss(
            objectness_weight=0.0,
            bbox_weight=0.0,
            side_weight=0.0,
            abnormality_weight=0.0,
            bbox_quality_loss_weight=0.0,
            bbox_quality_start_epoch=0,
            use_quality_ranking_loss=True,
            quality_ranking_loss_weight=1.0,
            ranking_pos_iou_threshold=0.3,
            ranking_neg_iou_threshold=0.1,
            ranking_margin=0.2,
            ranking_num_pairs=1,
            quality_hard_negative_source="objectness",
        )

        loss_dict = criterion(predictions, labels)
        loss_dict["loss"].backward()

        self.assertGreater(float(loss_dict["loss_quality_ranking"].item()), 0.0)
        self.assertEqual(int(loss_dict["num_quality_ranking_pairs"].item()), 1)
        self.assertIsNone(pred_boxes.grad)
        self.assertIsNotNone(predictions["bbox_quality_logits"].grad)

    def test_bbox_quality_varifocal_and_qfl_modulate_soft_iou_targets(self) -> None:
        varifocal = Module1DetectionLoss(bbox_quality_loss_type="varifocal")
        high_score_negative = varifocal._bbox_quality_loss(torch.tensor([4.0]), torch.tensor([0.0]), torch.ones(1))
        low_score_negative = varifocal._bbox_quality_loss(torch.tensor([-4.0]), torch.tensor([0.0]), torch.ones(1))
        self.assertGreater(float(high_score_negative.item()), float(low_score_negative.item()))

        qfl = Module1DetectionLoss(bbox_quality_loss_type="quality_focal", bbox_quality_qfl_beta=2.0)
        matched = qfl._bbox_quality_loss(torch.logit(torch.tensor([0.8])), torch.tensor([0.8]), torch.ones(1))
        mismatched = qfl._bbox_quality_loss(torch.logit(torch.tensor([0.2])), torch.tensor([0.8]), torch.ones(1))
        self.assertLess(float(matched.item()), float(mismatched.item()))

    def test_bbox_quality_pairwise_ranking_loss_type_detaches_bbox(self) -> None:
        shape = self._backbone_outputs()["shape_metadata"]
        labels = self._labels()
        pred_boxes = torch.zeros(1, 16, 4)
        pred_boxes[0, 0] = torch.tensor([0.0, 0.0, 40.0 / 64.0, 40.0 / 64.0])
        pred_boxes[0, 3] = torch.tensor([0.7, 0.7, 0.9, 0.9])
        pred_boxes.requires_grad_()
        predictions = {
            "objectness_logits": torch.zeros(1, 16),
            "bbox_quality_logits": torch.zeros(1, 16, requires_grad=True),
            "bbox_pred": pred_boxes,
            "side_logits": torch.zeros(1, 16, 2),
            "abnormality_logits": torch.zeros(1, 16, 2),
            "shape_metadata": shape,
        }
        predictions["objectness_logits"][0, 3] = 8.0
        predictions["bbox_quality_logits"].data[0, 0] = -2.0
        predictions["bbox_quality_logits"].data[0, 3] = 2.0
        predictions["side_logits"][0, :, 0] = 5.0
        criterion = Module1DetectionLoss(
            objectness_weight=0.0,
            bbox_weight=0.0,
            side_weight=0.0,
            abnormality_weight=0.0,
            bbox_quality_loss_weight=1.0,
            bbox_quality_loss_type="pairwise_ranking",
            bbox_quality_start_epoch=0,
            ranking_pos_iou_threshold=0.3,
            ranking_neg_iou_threshold=0.1,
            ranking_margin=0.2,
            ranking_num_pairs=1,
            quality_hard_negative_source="objectness",
        )

        loss_dict = criterion(predictions, labels)
        loss_dict["loss"].backward()

        self.assertGreater(float(loss_dict["loss_bbox_quality"].item()), 0.0)
        self.assertGreater(float(loss_dict["loss_quality_ranking"].item()), 0.0)
        self.assertIsNone(pred_boxes.grad)
        self.assertIsNotNone(predictions["bbox_quality_logits"].grad)

    def test_use_bbox_quality_head_false_keeps_objectness_flow(self) -> None:
        outputs = self._backbone_outputs()
        head = MeniscusDetectionHeads(embed_dim=16, hidden_dim=16, use_bbox_quality_head=False)
        predictions = head(outputs)
        self.assertNotIn("bbox_quality_logits", predictions)

        final_predictions = filter_module1_predictions(
            predictions,
            score_threshold=0.0,
            topk_per_image=1,
            nms_threshold=None,
            final_score_mode="objectness",
        )

        self.assertEqual(final_predictions[0]["counts"]["num_after_topk"], 1)


if __name__ == "__main__":
    unittest.main()
