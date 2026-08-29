"""Reproduce deterministic Module 1 head/loss initialization scale probes.

The probe uses synthetic fused tokens and one synthetic GT at the production
16x14x14 token shape. It is not a trained-model result. Its purpose is to make
initial bbox saturation and loss-scale claims in the algorithm audit exactly
reproducible without private MRI data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.detection_loss import Module1DetectionLoss, build_module1_targets, generalized_box_iou_loss
from models.meniscovit_cop.detection_heads import MeniscusDetectionHeads


def make_inputs(seed: int = 42) -> tuple[dict, dict, list[dict]]:
    torch.manual_seed(int(seed))
    tokens_per_plane = 16 * 14 * 14
    shape = {
        "sagittal": {"slices": 16, "height": 224, "width": 224, "grid_height": 14, "grid_width": 14, "patches_per_slice": 196},
        "coronal": {"slices": 16, "height": 224, "width": 224, "grid_height": 14, "grid_width": 14, "patches_per_slice": 196},
    }
    features = {
        "sagittal_tokens_fused": torch.randn(1, tokens_per_plane, 32),
        "coronal_tokens_fused": torch.randn(1, tokens_per_plane, 32),
        "shape_metadata": shape,
    }
    labels = [
        {
            "case_id": "synthetic_probe",
            "module1_labels": [
                {
                    "plane": "sagittal",
                    "slice_index": 7,
                    "bbox": [64, 96, 160, 144],
                    "meniscus_side": "medial",
                    "abnormality": "abnormal",
                }
            ],
        }
    ]
    return shape, features, labels


def run_probe(seed: int = 42) -> dict:
    shape, features, labels = make_inputs(seed)
    xyxy_head = MeniscusDetectionHeads(32, 32, bbox_param="xyxy")
    predictions = xyxy_head(features)
    boxes = predictions["bbox_pred"].detach()
    wh = boxes[..., 2:] - boxes[..., :2]
    areas = wh[..., 0] * wh[..., 1]

    targets = build_module1_targets(labels, shape, torch.device("cpu"))
    positive = targets["positive_mask"]
    smooth_l1 = F.smooth_l1_loss(predictions["bbox_pred"][positive], targets["bbox_targets"][positive])
    giou = generalized_box_iou_loss(predictions["bbox_pred"][positive], targets["bbox_targets"][positive])
    default_loss = Module1DetectionLoss(
        objectness_weight=1.0,
        bbox_weight=1.0,
        side_weight=0.5,
        abnormality_weight=1.0,
        bbox_loss_type="smooth_l1_giou",
    )(predictions, labels)

    shape_ltrb, features_ltrb, _ = make_inputs(seed)
    del shape_ltrb
    ltrb_boxes = MeniscusDetectionHeads(32, 32, bbox_param="ltrb_from_token_center")(features_ltrb)["bbox_pred"].detach()
    ltrb_boundary = (ltrb_boxes == 0) | (ltrb_boxes == 1)
    ltrb_full = (ltrb_boxes[..., 0] == 0) & (ltrb_boxes[..., 1] == 0) & (ltrb_boxes[..., 2] == 1) & (ltrb_boxes[..., 3] == 1)
    ltrb_area = (ltrb_boxes[..., 2] - ltrb_boxes[..., 0]) * (ltrb_boxes[..., 3] - ltrb_boxes[..., 1])

    _, weighted_features, weighted_labels = make_inputs(seed)
    weighted_head = MeniscusDetectionHeads(32, 32, bbox_param="xyxy")
    torch.nn.init.constant_(weighted_head.objectness.bias, math.log(0.01 / 0.99))
    weighted_predictions = weighted_head(weighted_features)
    weighted = Module1DetectionLoss(
        objectness_weight=1.0,
        bbox_weight=20.0,
        side_weight=0.5,
        abnormality_weight=0.5,
        bbox_loss_type="smooth_l1_giou",
        objectness_loss_type="weighted_bce",
        objectness_positive_weight=1.0,
        objectness_negative_weight=1.0,
    )(weighted_predictions, weighted_labels)
    weighted_bbox = 20.0 * float(weighted["loss_bbox"].detach())
    weighted_total = float(weighted["loss"].detach())

    return {
        "contract": {
            "kind": "synthetic_initialization_probe_not_training_log",
            "seed": int(seed),
            "torch_version": torch.__version__,
            "tokens_total": int(boxes.shape[1]),
            "embed_dim": 32,
        },
        "xyxy_initialization": {
            "mean_width": float(wh[..., 0].mean().detach()),
            "mean_height": float(wh[..., 1].mean().detach()),
            "mean_area": float(areas.mean().detach()),
            "area_lt_0.01_fraction": float((areas < 0.01).float().mean().detach()),
        },
        "ltrb_initialization": {
            "boundary_coordinate_fraction": float(ltrb_boundary.float().mean()),
            "full_image_box_fraction": float(ltrb_full.float().mean()),
            "mean_area": float(ltrb_area.mean()),
        },
        "default_loss": {
            "positive_tokens": int(positive.sum()),
            "positive_fraction": float(positive.float().mean()),
            "smooth_l1": float(smooth_l1.detach()),
            "giou": float(giou.detach()),
            "smooth_l1_plus_giou": float(default_loss["loss_bbox"]),
            "objectness": float(default_loss["loss_objectness"]),
            "side_unweighted": float(default_loss["loss_side"]),
            "abnormality_unweighted": float(default_loss["loss_abnormality"]),
            "total_weighted": float(default_loss["loss"].detach()),
        },
        "bbox_weight_20_probe": {
            "objectness": float(weighted["loss_objectness"]),
            "bbox_unweighted": float(weighted["loss_bbox"]),
            "bbox_weighted": weighted_bbox,
            "side_weighted": 0.5 * float(weighted["loss_side"]),
            "abnormality_weighted": 0.5 * float(weighted["loss_abnormality"]),
            "total_weighted": weighted_total,
            "bbox_fraction_of_total": weighted_bbox / max(weighted_total, 1e-12),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = run_probe(args.seed)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
