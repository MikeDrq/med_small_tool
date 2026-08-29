"""Validation entrypoint for Module 2 Video Swin ROI classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.roi_classification_dataset import ROIClassificationDataset
from datasets.loader_utils import worker_loader_kwargs
from models.video_swin.model import PaperVideoSwinClassifier, VideoSwinClassifier
from utils.checkpoint import load_checkpoint
from utils.logging import format_metrics, get_logger
from utils.bootstrap import grouped_bootstrap_ci
from utils.metrics import precision_recall_specificity_f1


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def roi_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_id": [item["case_id"] for item in batch],
        "group_id": [item.get("group_id", item["case_id"]) for item in batch],
        "side": [item["side"] for item in batch],
        "sagittal_roi": torch.stack([item["sagittal_roi"] for item in batch], dim=0),
        "coronal_roi": torch.stack([item["coronal_roi"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "module2_label": torch.stack([item["module2_label"] for item in batch], dim=0),
    }


def build_module2_model(config: dict[str, Any]) -> torch.nn.Module:
    cfg = config["model"]
    implementation = str(cfg.get("implementation", "compact_dual_branch")).lower()
    if implementation in {"paper", "paper_swin_t", "torchvision_swin3d_t", "swin3d_t"}:
        return PaperVideoSwinClassifier(
            num_classes=cfg["num_classes"],
            pretrained=bool(cfg.get("pretrained", False)),
            pretrained_checkpoint=cfg.get("pretrained_checkpoint"),
            normalize_input=bool(cfg.get("normalize_input", True)),
            fusion_mode=str(cfg.get("fusion_mode", "input_channel_concat")),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    return VideoSwinClassifier(
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
        embed_dim=cfg["embed_dim"],
        patch_size=cfg["patch_size"],
        depths=cfg["depths"],
        num_heads=cfg["num_heads"],
        window_size=cfg["window_size"],
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        dropout=cfg.get("dropout", 0.0),
        attn_drop=cfg.get("attn_drop", 0.0),
        drop_path_rate=cfg.get("drop_path_rate", 0.0),
        patch_merging_stride=cfg.get("patch_merging_stride", [1, 2, 2]),
        use_checkpoint=cfg.get("use_checkpoint", False),
        share_branch_weights=cfg.get("share_branch_weights", False),
    )


def create_val_loader(config: dict[str, Any], roi_index: str | None = None) -> DataLoader:
    dataset = ROIClassificationDataset(
        roi_index or config["data"]["val_rois"],
        task="module2",
        sides=tuple(config["data"].get("sides", ["medial", "lateral"])),
    )
    return DataLoader(
        dataset,
        batch_size=config.get("validate", {}).get("batch_size", 2),
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=roi_collate,
        **worker_loader_kwargs(config["data"]),
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key in ("sagittal_roi", "coronal_roi", "label", "module2_label"):
        out[key] = batch[key].to(device, non_blocking=True)
    return out


def confusion_matrix(labels: list[int], preds: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(labels, preds):
        matrix[int(target), int(pred)] += 1
    return matrix


def class_metrics_from_confusion(matrix: np.ndarray, class_names: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    total = int(matrix.sum())
    for class_idx, class_name in enumerate(class_names):
        tp = int(matrix[class_idx, class_idx])
        fp = int(matrix[:, class_idx].sum() - tp)
        fn = int(matrix[class_idx, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        values = precision_recall_specificity_f1(tp=tp, fp=fp, tn=tn, fn=fn)
        metrics[f"{class_name}_accuracy"] = (tp + tn) / max(total, 1)
        for key, value in values.items():
            metrics[f"{class_name}_{key}"] = value
    metrics["macro_f1"] = float(np.mean([metrics[f"{name}_f1"] for name in class_names])) if class_names else float("nan")
    return metrics


@torch.inference_mode()
def evaluate_module2(model: torch.nn.Module, loader: DataLoader, device: torch.device, class_names: list[str], amp: bool = True, bootstrap_samples: int = 0) -> dict[str, Any]:
    model.eval()
    labels: list[int] = []
    preds: list[int] = []
    case_ids: list[str] = []
    group_ids: list[str] = []
    device_type = "cuda" if device.type == "cuda" else "cpu"
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(batch["sagittal_roi"], batch["coronal_roi"])
        preds.extend(logits.argmax(dim=1).detach().cpu().int().tolist())
        labels.extend(batch["label"].detach().cpu().int().tolist())
        case_ids.extend(str(case_id) for case_id in batch["case_id"])
        group_ids.extend(str(group_id) for group_id in batch["group_id"])

    matrix = confusion_matrix(labels, preds, len(class_names))
    accuracy = float(np.trace(matrix) / max(matrix.sum(), 1))
    metrics: dict[str, Any] = {"accuracy": accuracy, "confusion_matrix": matrix.tolist()}
    metrics.update(class_metrics_from_confusion(matrix, class_names))
    if bootstrap_samples > 0:
        pairs = list(zip(labels, preds))

        def metrics_from_pairs(items: list[tuple[int, int]]) -> dict[str, float]:
            item_matrix = confusion_matrix(
                [item[0] for item in items],
                [item[1] for item in items],
                len(class_names),
            )
            values = {
                "accuracy": float(np.trace(item_matrix) / max(item_matrix.sum(), 1)),
            }
            values.update(class_metrics_from_confusion(item_matrix, class_names))
            return values

        ci_names = ["accuracy", "macro_f1"]
        for class_name in class_names:
            ci_names.extend(
                f"{class_name}_{name}"
                for name in ("accuracy", "precision", "recall", "specificity", "f1")
            )
        for metric_name in ci_names:
            _, lo, hi = grouped_bootstrap_ci(
                pairs,
                group_ids,
                lambda items, name=metric_name: metrics_from_pairs(items)[name],
                n_boot=bootstrap_samples,
            )
            metrics[f"{metric_name}_ci_low"] = lo
            metrics[f"{metric_name}_ci_high"] = hi
        metrics["bootstrap_case_count"] = float(len(set(str(case_id) for case_id in case_ids)))
        metrics["bootstrap_group_count"] = float(len(set(group_ids)))
    return metrics


def validate_module2(
    config_path: str,
    checkpoint: str,
    roi_index: str | None = None,
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    logger = get_logger("validate_module2")
    device_name = config.get("device", "cuda")
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_module2_model(config).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    loader = create_val_loader(config, roi_index=roi_index)
    metrics = evaluate_module2(
        model,
        loader,
        device,
        class_names=list(config["data"]["classes"]),
        amp=config.get("train", {}).get("amp", True),
        bootstrap_samples=int(
            config.get("validate", {}).get("bootstrap_samples", 1000)
            if bootstrap_samples is None
            else bootstrap_samples
        ),
    )
    scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, float)}
    logger.info(format_metrics(scalar_metrics))
    logger.info(f"confusion_matrix={metrics['confusion_matrix']}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/module2_swin.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--roi-index", default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    args = parser.parse_args()
    validate_module2(
        args.config,
        args.checkpoint,
        roi_index=args.roi_index,
        bootstrap_samples=args.bootstrap,
    )


if __name__ == "__main__":
    main()
