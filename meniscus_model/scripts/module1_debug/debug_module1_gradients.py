"""Inspect Module 1 objectness and bbox gradient signals on one case.

The script runs a few in-memory optimization steps and does not save model
weights. It is meant for checking whether positive/negative objectness logits
separate and whether positive-token bbox predictions move toward the GT boxes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import warnings

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.meniscus_detection_dataset import MeniscusDetectionDataset
from engine.validate_module1 import build_module1_model, module1_collate, move_batch_to_device
from losses.detection_loss import DetectionTargetConfig, Module1DetectionLoss, box_iou, build_module1_targets
from utils.checkpoint import load_checkpoint


warnings.filterwarnings("ignore", message="Invalid value for VR UI:*")


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _select_sample(dataset: MeniscusDetectionDataset, case_id: str | None) -> dict[str, Any]:
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if case_id is None or str(sample["case_id"]) == str(case_id):
            return sample
    raise ValueError(f"case_id={case_id} not found in {dataset.index_jsonl}")


def _build_criterion(config: dict[str, Any], args: argparse.Namespace) -> Module1DetectionLoss:
    loss_cfg = config["loss"]
    return Module1DetectionLoss(
        objectness_weight=float(args.objectness_weight if args.objectness_weight is not None else loss_cfg.get("objectness_weight", 1.0)),
        bbox_weight=float(args.bbox_weight if args.bbox_weight is not None else loss_cfg.get("bbox_weight", 1.0)),
        side_weight=float(loss_cfg.get("side_weight", 0.5)),
        abnormality_weight=float(loss_cfg.get("abnormality_weight", 1.0)),
        bbox_loss_type=str(loss_cfg.get("bbox_loss_type", "smooth_l1")),
        objectness_loss_type=str(loss_cfg.get("objectness_loss", loss_cfg.get("objectness_loss_type", "focal"))),
        objectness_positive_weight=float(loss_cfg.get("objectness_positive_weight", 1.0)),
        objectness_negative_weight=float(loss_cfg.get("objectness_negative_weight", 1.0)),
        focal_alpha=float(loss_cfg.get("focal_alpha", 0.25)),
        focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
        positive_radius_tokens=int(args.positive_radius_tokens if args.positive_radius_tokens is not None else loss_cfg.get("positive_radius_tokens", 0)),
    )


def _stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu().flatten()
    if values.numel() == 0:
        return {"count": 0.0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": float(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)) if values.numel() > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _positive_bbox_stats(predictions: dict[str, torch.Tensor | dict[str, Any]], targets: dict[str, torch.Tensor]) -> dict[str, float]:
    positive_mask = targets["positive_mask"]
    if not positive_mask.any():
        return {"positive_count": 0.0, "bbox_l1_mean": float("nan"), "bbox_l1_max": float("nan"), "bbox_iou_mean": float("nan")}
    pred_boxes = predictions["bbox_pred"][positive_mask].detach().float()
    target_boxes = targets["bbox_targets"][positive_mask].detach().float()
    l1 = (pred_boxes - target_boxes).abs().mean(dim=1)
    ious = box_iou(pred_boxes, target_boxes).diag()
    return {
        "positive_count": float(pred_boxes.shape[0]),
        "bbox_l1_mean": float(l1.mean().cpu()),
        "bbox_l1_max": float(l1.max().cpu()),
        "bbox_iou_mean": float(ious.mean().cpu()),
        "bbox_iou_max": float(ious.max().cpu()),
    }


def _grad_norms(model: torch.nn.Module) -> dict[str, float]:
    groups = {
        "total": 0.0,
        "backbone": 0.0,
        "heads_objectness": 0.0,
        "heads_bbox": 0.0,
        "heads_side": 0.0,
        "heads_abnormality": 0.0,
        "heads_tower": 0.0,
    }
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_sq = float(param.grad.detach().float().pow(2).sum().cpu())
        groups["total"] += grad_sq
        if name.startswith("backbone."):
            groups["backbone"] += grad_sq
        elif name.startswith("heads.objectness."):
            groups["heads_objectness"] += grad_sq
        elif name.startswith("heads.bbox."):
            groups["heads_bbox"] += grad_sq
        elif name.startswith("heads.side."):
            groups["heads_side"] += grad_sq
        elif name.startswith("heads.abnormality."):
            groups["heads_abnormality"] += grad_sq
        elif name.startswith("heads.tower."):
            groups["heads_tower"] += grad_sq
    return {key: value ** 0.5 for key, value in groups.items()}


def _step_report(
    step: int,
    predictions: dict[str, torch.Tensor | dict[str, Any]],
    targets: dict[str, torch.Tensor],
    loss_dict: dict[str, torch.Tensor],
    grad_norms: dict[str, float],
) -> dict[str, Any]:
    positive_mask = targets["positive_mask"]
    objectness_logits = predictions["objectness_logits"].detach()
    report = {
        "step": step,
        "loss": float(loss_dict["loss"].detach().cpu()),
        "loss_objectness": float(loss_dict["loss_objectness"].detach().cpu()),
        "loss_objectness_positive": float(loss_dict.get("loss_objectness_positive", torch.tensor(float("nan"))).detach().cpu()),
        "loss_objectness_negative": float(loss_dict.get("loss_objectness_negative", torch.tensor(float("nan"))).detach().cpu()),
        "loss_objectness_positive_contribution": float(loss_dict.get("loss_objectness_positive_contribution", torch.tensor(float("nan"))).detach().cpu()),
        "loss_objectness_negative_contribution": float(loss_dict.get("loss_objectness_negative_contribution", torch.tensor(float("nan"))).detach().cpu()),
        "objectness_positive_tokens": int(loss_dict.get("objectness_positive_tokens", torch.tensor(0)).detach().cpu()),
        "objectness_negative_tokens": int(loss_dict.get("objectness_negative_tokens", torch.tensor(0)).detach().cpu()),
        "loss_bbox": float(loss_dict["loss_bbox"].detach().cpu()),
        "loss_side": float(loss_dict["loss_side"].detach().cpu()),
        "loss_abnormality": float(loss_dict["loss_abnormality"].detach().cpu()),
        "positive_objectness_logits": _stats(objectness_logits[positive_mask]),
        "negative_objectness_logits": _stats(objectness_logits[~positive_mask]),
        "positive_objectness_scores": _stats(torch.sigmoid(objectness_logits[positive_mask])),
        "negative_objectness_scores": _stats(torch.sigmoid(objectness_logits[~positive_mask])),
        "bbox": _positive_bbox_stats(predictions, targets),
        "grad_norms": grad_norms,
    }
    print(
        f"step={step:02d} loss={report['loss']:.4f} obj={report['loss_objectness']:.4f} "
        f"obj_pos={report['loss_objectness_positive']:.4f} obj_neg={report['loss_objectness_negative']:.4f} "
        f"bbox={report['loss_bbox']:.4f} pos_logit_mean={report['positive_objectness_logits']['mean']:.4f} "
        f"neg_logit_mean={report['negative_objectness_logits']['mean']:.4f} "
        f"bbox_l1={report['bbox']['bbox_l1_mean']:.4f} bbox_iou={report['bbox']['bbox_iou_mean']:.4f} "
        f"grad_total={report['grad_norms']['total']:.4f}"
    )
    return report


def debug_gradients(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    data_cfg = config["data"]
    index_jsonl = args.index_jsonl or data_cfg["train_index"]
    dataset = MeniscusDetectionDataset(
        index_jsonl=index_jsonl,
        output_size=data_cfg["output_size"],
        target_slices=data_cfg["num_slices"],
        patch_size=data_cfg["patch_size"],
        transform=None,
    )
    sample = _select_sample(dataset, args.case_id)
    batch = module1_collate([sample])

    device_name = config.get("device", "cuda")
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    batch = move_batch_to_device(batch, device)
    model = build_module1_model(config).to(device)
    if args.checkpoint:
        metadata = load_checkpoint(args.checkpoint, model, map_location=device)
        print(f"loaded checkpoint={args.checkpoint} metadata={metadata}")
    model.train()

    criterion = _build_criterion(config, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.0)
    target_config = DetectionTargetConfig(positive_radius_tokens=criterion.target_config.positive_radius_tokens)

    print(f"case_id={sample['case_id']} device={device} optimization_steps={args.steps} lr={args.lr}")
    print(
        f"debug weights: objectness={criterion.objectness_weight} bbox={criterion.bbox_weight} "
        f"positive_radius_tokens={criterion.target_config.positive_radius_tokens}"
    )

    reports = []
    for step in range(int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch["sagittal_volume"], batch["coronal_volume"], batch["sagittal_coords"], batch["coronal_coords"])
        targets = build_module1_targets(batch["labels"], predictions["shape_metadata"], device=device, target_config=target_config)
        loss_dict = criterion(predictions, batch["labels"])
        loss_dict["loss"].backward()
        grad_norms = _grad_norms(model)
        reports.append(_step_report(step, predictions, targets, loss_dict, grad_norms))
        if step < int(args.steps):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.clip_grad_norm))
            optimizer.step()

    summary = {
        "case_id": sample["case_id"],
        "config": str(args.config),
        "index_jsonl": str(index_jsonl),
        "checkpoint": args.checkpoint,
        "steps": int(args.steps),
        "lr": float(args.lr),
        "reports": reports,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "gradient_debug.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved JSON report: {report_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/module1_meniscovit_cop.yaml")
    parser.add_argument("--index-jsonl", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/module1_gradient_debug")
    parser.add_argument("--steps", type=int, default=3, help="In-memory optimizer steps; weights are not saved.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--positive-radius-tokens", type=int, default=None)
    parser.add_argument("--objectness-weight", type=float, default=None)
    parser.add_argument("--bbox-weight", type=float, default=None)
    args = parser.parse_args()
    debug_gradients(args)


if __name__ == "__main__":
    main()
