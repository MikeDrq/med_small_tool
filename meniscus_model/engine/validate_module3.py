"""Validation entrypoint for Module 3 bucket-handle Video Swin classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.roi_classification_dataset import ROIClassificationDataset
from datasets.loader_utils import worker_loader_kwargs
from engine.validate_module2 import build_module2_model
from utils.checkpoint import load_checkpoint
from utils.logging import format_metrics, get_logger
from utils.bootstrap import grouped_bootstrap_ci
from utils.metrics import precision_recall_specificity_f1, safe_average_precision, safe_roc_auc


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def roi_binary_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_id": [item["case_id"] for item in batch],
        "group_id": [item.get("group_id", item["case_id"]) for item in batch],
        "side": [item["side"] for item in batch],
        "sagittal_roi": torch.stack([item["sagittal_roi"] for item in batch], dim=0),
        "coronal_roi": torch.stack([item["coronal_roi"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0).float(),
        "module3_label": torch.stack([item["module3_label"] for item in batch], dim=0).float(),
    }


def create_val_loader(config: dict[str, Any], roi_index: str | None = None) -> DataLoader:
    dataset = ROIClassificationDataset(
        roi_index or config["data"]["val_rois"],
        task="module3",
        sides=tuple(config["data"].get("sides", ["medial", "lateral"])),
        module3_tear_only=bool(config["data"].get("tear_only", True)),
    )
    return DataLoader(
        dataset,
        batch_size=config.get("validate", {}).get("batch_size", 2),
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=roi_binary_collate,
        **worker_loader_kwargs(config["data"]),
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key in ("sagittal_roi", "coronal_roi", "label", "module3_label"):
        out[key] = batch[key].to(device, non_blocking=True)
    return out


def select_f1_threshold(
    labels: list[int],
    scores: list[float],
    minimum: float = 0.05,
    maximum: float = 0.95,
) -> float:
    """Select one operating threshold on validation data.

    The model is shared by both sides, so the threshold is selected from the
    pooled medial/lateral validation samples.  F1 ties prefer higher precision
    and then the higher threshold, which is useful for this rare subtype.
    """

    candidates = {float(minimum), float(maximum)}
    candidates.update(float(score) for score in scores if minimum <= float(score) <= maximum)
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    for threshold in sorted(candidates):
        preds = [int(score >= threshold) for score in scores]
        tp = sum(int(pred == 1 and target == 1) for pred, target in zip(preds, labels))
        fp = sum(int(pred == 1 and target == 0) for pred, target in zip(preds, labels))
        fn = sum(int(pred == 0 and target == 1) for pred, target in zip(preds, labels))
        rates = precision_recall_specificity_f1(tp=tp, fp=fp, tn=0, fn=fn)
        key = (rates["f1"], rates["precision"], threshold)
        if key > best_key:
            best_key = key
            best_threshold = threshold
    return float(best_threshold)


def _binary_metrics(
    labels: list[int],
    scores: list[float],
    threshold: float,
    prefix: str,
    bootstrap_samples: int = 0,
    case_ids: list[str] | None = None,
    group_ids: list[str] | None = None,
) -> dict[str, float]:
    preds = [int(score >= threshold) for score in scores]
    tp = sum(int(pred == 1 and target == 1) for pred, target in zip(preds, labels))
    tn = sum(int(pred == 0 and target == 0) for pred, target in zip(preds, labels))
    fp = sum(int(pred == 1 and target == 0) for pred, target in zip(preds, labels))
    fn = sum(int(pred == 0 and target == 1) for pred, target in zip(preds, labels))
    rates = precision_recall_specificity_f1(tp=tp, fp=fp, tn=tn, fn=fn)
    metrics = {
        f"{prefix}_accuracy": (tp + tn) / max(len(labels), 1),
        f"{prefix}_roc_auc": safe_roc_auc(labels, scores),
        f"{prefix}_pr_auc": safe_average_precision(labels, scores),
        f"{prefix}_precision": rates["precision"],
        f"{prefix}_recall": rates["recall"],
        f"{prefix}_specificity": rates["specificity"],
        f"{prefix}_f1": rates["f1"],
        f"{prefix}_tp": float(tp),
        f"{prefix}_fp": float(fp),
        f"{prefix}_tn": float(tn),
        f"{prefix}_fn": float(fn),
    }
    if bootstrap_samples > 0:
        pairs = list(zip(labels, scores))
        bootstrap_groups = group_ids or case_ids or [str(index) for index in range(len(pairs))]

        def task_metric(items: list[tuple[int, float]], name: str) -> float:
            item_labels = [item[0] for item in items]
            item_scores = [item[1] for item in items]
            if name == "roc_auc":
                return safe_roc_auc(item_labels, item_scores)
            if name == "pr_auc":
                return safe_average_precision(item_labels, item_scores)
            item_preds = [int(score >= threshold) for score in item_scores]
            item_tp = sum(int(pred == 1 and target == 1) for pred, target in zip(item_preds, item_labels))
            item_tn = sum(int(pred == 0 and target == 0) for pred, target in zip(item_preds, item_labels))
            item_fp = sum(int(pred == 1 and target == 0) for pred, target in zip(item_preds, item_labels))
            item_fn = sum(int(pred == 0 and target == 1) for pred, target in zip(item_preds, item_labels))
            if name == "accuracy":
                return (item_tp + item_tn) / max(len(item_labels), 1)
            return precision_recall_specificity_f1(item_tp, item_fp, item_tn, item_fn)[name]

        for metric_name in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"):
            _, lo, hi = grouped_bootstrap_ci(
                pairs,
                bootstrap_groups,
                lambda items, name=metric_name: task_metric(items, name),
                n_boot=bootstrap_samples,
            )
            metrics[f"{prefix}_{metric_name}_ci_low"] = lo
            metrics[f"{prefix}_{metric_name}_ci_high"] = hi
        metrics[f"{prefix}_bootstrap_case_count"] = float(len(set(case_ids or [])))
        metrics[f"{prefix}_bootstrap_group_count"] = float(len(set(bootstrap_groups)))
    return metrics


@torch.inference_mode()
def evaluate_module3(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    amp: bool = True,
    bootstrap_samples: int = 0,
    optimize_threshold: bool = False,
    threshold_minimum: float = 0.05,
    threshold_maximum: float = 0.95,
) -> dict[str, float]:
    model.eval()
    labels_by_group: dict[str, list[int]] = {"overall": [], "medial": [], "lateral": []}
    scores_by_group: dict[str, list[float]] = {"overall": [], "medial": [], "lateral": []}
    cases_by_group: dict[str, list[str]] = {"overall": [], "medial": [], "lateral": []}
    patients_by_group: dict[str, list[str]] = {"overall": [], "medial": [], "lateral": []}
    device_type = "cuda" if device.type == "cuda" else "cpu"
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(batch["sagittal_roi"], batch["coronal_roi"]).squeeze(-1)
        scores = torch.sigmoid(logits).detach().cpu().float().tolist()
        labels = batch["label"].detach().cpu().int().tolist()
        for case_id, group_id, side, label, score in zip(
            batch["case_id"], batch["group_id"], batch["side"], labels, scores
        ):
            labels_by_group["overall"].append(int(label))
            scores_by_group["overall"].append(float(score))
            cases_by_group["overall"].append(str(case_id))
            patients_by_group["overall"].append(str(group_id))
            labels_by_group[side].append(int(label))
            scores_by_group[side].append(float(score))
            cases_by_group[side].append(str(case_id))
            patients_by_group[side].append(str(group_id))

    if optimize_threshold:
        threshold = select_f1_threshold(
            labels_by_group["overall"],
            scores_by_group["overall"],
            minimum=threshold_minimum,
            maximum=threshold_maximum,
        )

    metrics: dict[str, float] = {"decision_threshold": float(threshold)}
    for group in ("overall", "medial", "lateral"):
        metrics.update(
            _binary_metrics(
                labels_by_group[group],
                scores_by_group[group],
                threshold,
                group,
                bootstrap_samples=bootstrap_samples,
                case_ids=cases_by_group[group],
                group_ids=patients_by_group[group],
            )
        )
    return metrics


def validate_module3(
    config_path: str,
    checkpoint: str,
    roi_index: str | None = None,
    bootstrap_samples: int | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    config = load_config(config_path)
    logger = get_logger("validate_module3")
    device_name = config.get("device", "cuda")
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_module2_model(config).to(device)
    metadata = load_checkpoint(checkpoint, model, map_location=device)
    decision_threshold = threshold
    if decision_threshold is None:
        decision_threshold = float(
            metadata.get(
                "decision_threshold",
                config.get("validate", {}).get("threshold", 0.5),
            )
        )
    loader = create_val_loader(config, roi_index=roi_index)
    metrics = evaluate_module3(
        model,
        loader,
        device,
        threshold=decision_threshold,
        amp=config.get("train", {}).get("amp", True),
        bootstrap_samples=int(
            config.get("validate", {}).get("bootstrap_samples", 1000)
            if bootstrap_samples is None
            else bootstrap_samples
        ),
    )
    logger.info(f"decision_threshold={decision_threshold:.4f}")
    for group in ("overall", "medial", "lateral"):
        group_metrics = {key: value for key, value in metrics.items() if key.startswith(f"{group}_")}
        logger.info(f"{group} {format_metrics(group_metrics)}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/module3_bucket_swin.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--roi-index", default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Override the threshold stored in the checkpoint.")
    args = parser.parse_args()
    validate_module3(
        args.config,
        args.checkpoint,
        roi_index=args.roi_index,
        bootstrap_samples=args.bootstrap,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
