"""Validation entrypoint for Module 1 MeniscoViT-CoP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.meniscus_detection_dataset import MeniscusDetectionDataset
from datasets.loader_utils import worker_loader_kwargs
from losses.detection_loss import DetectionTargetConfig, build_module1_targets, build_strict_pred_iou_targets
from models.meniscovit_cop.detection_heads import MeniscoViTCoPModule1, MeniscusDetectionHeads
from models.meniscovit_cop.hierarchical_model import HierarchicalMeniscoViTCoP
from models.meniscovit_cop.model import MeniscoViTCoP
from utils.bootstrap import grouped_bootstrap_ci
from utils.checkpoint import load_checkpoint
from utils.logging import format_metrics, get_logger
from utils.metrics import (
    binary_task_metrics,
    detection_filter_count_sums,
    filter_module1_predictions,
    final_prediction_best_ious,
    match_final_predictions,
    objectness_iou_alignment_metrics,
    oracle_positive_bbox_ious,
    precision_recall_specificity_f1,
    safe_average_precision,
    safe_roc_auc,
    score_mode_detection_diagnostics,
    summarize_iou_detection_metrics,
    summarize_oracle_positive_bbox_metrics,
    topk_detection_diagnostics,
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def module1_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length JSON labels while stacking tensors."""

    return {
        "case_id": [item["case_id"] for item in batch],
        "group_id": [item.get("group_id", item["case_id"]) for item in batch],
        "sagittal_volume": torch.stack([item["sagittal_volume"] for item in batch], dim=0),
        "coronal_volume": torch.stack([item["coronal_volume"] for item in batch], dim=0),
        "sagittal_coords": torch.stack([item["sagittal_coords"] for item in batch], dim=0),
        "coronal_coords": torch.stack([item["coronal_coords"] for item in batch], dim=0),
        "labels": [item["labels"] for item in batch],
    }


def build_module1_model(config: dict[str, Any]) -> MeniscoViTCoPModule1:
    model_cfg = config["model"]
    architecture = str(model_cfg.get("architecture", "legacy_single_scale")).lower()
    if architecture in {"paper", "paper_hierarchical", "hierarchical", "uniformer"}:
        backbone = HierarchicalMeniscoViTCoP(
            image_size=model_cfg["image_size"],
            in_channels=model_cfg.get("in_channels", 3),
            embed_dims=model_cfg.get("embed_dims", [64, 128, 320, 512]),
            depths=model_cfg.get("depths", [3, 4, 8, 3]),
            head_dim=model_cfg.get("head_dim", 64),
            mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
            cross_attention_k=model_cfg.get("cross_attention_k", 32),
            cross_attention_max_tokens=model_cfg.get("cross_attention_max_tokens", 4096),
            dropout=model_cfg.get("dropout", 0.0),
            attn_dropout=model_cfg.get("attn_dropout", 0.0),
            drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
            use_checkpoint=model_cfg.get("use_checkpoint", True),
            factorized_global_attention=model_cfg.get("factorized_global_attention", True),
            output_original_depth=model_cfg.get("output_original_depth", True),
            normalize_input=model_cfg.get("normalize_input", False),
            input_mean=model_cfg.get("input_mean", [0.45, 0.45, 0.45]),
            input_std=model_cfg.get("input_std", [0.225, 0.225, 0.225]),
        )
        pretrained_checkpoint = model_cfg.get("pretrained_checkpoint")
        if pretrained_checkpoint:
            backbone.load_uniformer_pretrained(
                pretrained_checkpoint,
                expected_sha256=model_cfg.get("pretrained_sha256"),
                strict=bool(model_cfg.get("pretrained_strict", True)),
            )
        head_embed_dim = backbone.out_dim
    else:
        backbone = MeniscoViTCoP(
            image_size=model_cfg["image_size"],
            patch_size=model_cfg["patch_size"],
            in_channels=model_cfg["in_channels"],
            embed_dim=model_cfg["embed_dim"],
            depth=model_cfg["depth"],
            num_heads=model_cfg["num_heads"],
            mlp_ratio=model_cfg["mlp_ratio"],
            cross_attention_k=model_cfg["cross_attention_k"],
            dropout=model_cfg.get("dropout", 0.0),
            drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
            use_checkpoint=model_cfg.get("use_checkpoint", False),
            num_slices=model_cfg["num_slices"],
            rcpe_normalization=model_cfg.get("rcpe_normalization", "per_case"),
        )
        head_embed_dim = int(model_cfg["embed_dim"])
    heads = MeniscusDetectionHeads(
        embed_dim=head_embed_dim,
        hidden_dim=model_cfg.get("head_hidden_dim", head_embed_dim),
        dropout=model_cfg.get("dropout", 0.0),
        bbox_param=model_cfg.get("bbox_param", config.get("loss", {}).get("bbox_param", "xyxy")),
        use_bbox_quality_head=bool(model_cfg.get("use_bbox_quality_head", False)),
    )
    model = MeniscoViTCoPModule1(backbone, heads)
    prior_prob = config.get("loss", {}).get("objectness_prior_prob")
    if prior_prob is not None:
        prior = float(prior_prob)
        if not 0.0 < prior < 1.0:
            raise ValueError("loss.objectness_prior_prob must be in (0, 1)")
        bias = torch.logit(torch.tensor(prior, dtype=model.heads.objectness.bias.dtype))
        torch.nn.init.constant_(model.heads.objectness.bias, float(bias.item()))
    return model


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key in ("sagittal_volume", "coronal_volume", "sagittal_coords", "coronal_coords"):
        out[key] = batch[key].to(device, non_blocking=True)
    return out


def create_val_loader(config: dict[str, Any], index_jsonl: str | None = None) -> DataLoader:
    data_cfg = config["data"]
    val_cfg = config.get("validate", {})
    dataset = MeniscusDetectionDataset(
        index_jsonl=index_jsonl or data_cfg["val_index"],
        output_size=data_cfg["output_size"],
        target_slices=data_cfg["num_slices"],
        patch_size=data_cfg["patch_size"],
        slice_selection=data_cfg.get("val_slice_selection", data_cfg.get("slice_selection")),
        transform=None,
    )
    return DataLoader(
        dataset,
        batch_size=val_cfg.get("batch_size", 1),
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=module1_collate,
        **worker_loader_kwargs(data_cfg),
    )


@torch.inference_mode()
def evaluate_module1(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    iou_thresholds: list[float] | tuple[float, ...] | None = None,
    topk_per_image: int | None = None,
    nms_threshold: float | None = None,
    final_score_mode: str = "objectness",
    quality_score_alpha: float = 1.0,
    quality_score_beta: float = 1.0,
    amp: bool = True,
    bootstrap_samples: int = 0,
    positive_radius_tokens: int = 0,
    quality_iou_threshold: float = 0.3,
    logger: Any | None = None,
    log_detection_diagnostics: bool = False,
    compute_extended_diagnostics: bool = True,
    progress_interval: int = 0,
    side_threshold: float = 0.5,
    abnormality_threshold: float = 0.4,
) -> dict[str, float]:
    model.eval()
    object_labels: list[int] = []
    object_scores: list[float] = []
    side_labels: list[int] = []
    side_scores: list[float] = []
    abnormal_labels: list[int] = []
    abnormal_scores: list[float] = []
    object_case_pr_aucs: list[float] = []
    object_case_group_ids: list[str] = []
    side_group_ids: list[str] = []
    abnormal_group_ids: list[str] = []
    detection_group_ids: list[str] = []
    iou_threshold_values = _resolve_iou_thresholds(iou_threshold, iou_thresholds)
    counts_by_threshold = {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in iou_threshold_values}
    case_counts_by_threshold: dict[float, list[dict[str, int]]] = {
        threshold: [] for threshold in iou_threshold_values
    }
    oracle_iou_strict_counts_by_threshold = {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in iou_threshold_values}
    filter_counts = {
        "num_raw_predictions": 0,
        "num_after_score_threshold": 0,
        "num_after_topk": 0,
        "num_after_nms": 0,
        "num_final_predictions_for_matching": 0,
    }
    best_ious: list[float] = []
    oracle_bbox_ious: list[float] = []
    top100_positive_target_tokens: list[int] = []
    top100_quality_positive_tokens: list[int] = []
    top100_mean_ious: list[float] = []
    top100_max_ious: list[float] = []
    top100_iou_gt_0_1: list[int] = []
    top100_iou_gt_0_3: list[int] = []
    top100_iou_gt_0_45: list[int] = []
    objectness_iou_pearson: list[float] = []
    objectness_iou_spearman: list[float] = []
    score_mode_names = ["objectness"]
    score_mode_metric_sums: dict[str, dict[str, float]] = {}
    score_mode_metric_counts: dict[str, int] = {}
    label_mapping_counts = {
        "raw_label_count": 0,
        "matched_source_label_count": 0,
        "exact_source_label_count": 0,
        "remapped_source_label_count": 0,
        "dropped_source_label_count": 0,
        "processed_label_count": 0,
        "cases_with_dropped_source_labels": 0,
    }

    target_config = DetectionTargetConfig(positive_radius_tokens=int(positive_radius_tokens))
    device_type = "cuda" if device.type == "cuda" else "cpu"
    for batch_index, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        for case_labels in batch["labels"]:
            mapping = case_labels.get("module1_label_mapping", {})
            for key in (
                "raw_label_count",
                "matched_source_label_count",
                "exact_source_label_count",
                "remapped_source_label_count",
                "dropped_source_label_count",
                "processed_label_count",
            ):
                label_mapping_counts[key] += int(mapping.get(key, 0))
            if int(mapping.get("dropped_source_label_count", 0)) > 0:
                label_mapping_counts["cases_with_dropped_source_labels"] += 1
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            predictions = model(batch["sagittal_volume"], batch["coronal_volume"], batch["sagittal_coords"], batch["coronal_coords"])

        targets = build_module1_targets(batch["labels"], predictions["shape_metadata"], device=device, target_config=target_config)
        object_label_batch = targets["objectness"].detach().cpu().int()
        object_score_batch = torch.sigmoid(predictions["objectness_logits"]).detach().cpu()
        object_labels.extend(object_label_batch.flatten().tolist())
        object_scores.extend(object_score_batch.flatten().tolist())
        positive_mask = targets["positive_mask"]
        side_probability = torch.softmax(predictions["side_logits"], dim=-1)[..., 1].detach()
        abnormal_probability = torch.softmax(predictions["abnormality_logits"], dim=-1)[..., 1].detach()
        for case_index, _case_id_value in enumerate(batch["case_id"]):
            group_id = str(batch["group_id"][case_index])
            case_object_labels = object_label_batch[case_index].flatten().tolist()
            case_object_scores = object_score_batch[case_index].flatten().tolist()
            object_case_pr_aucs.append(safe_average_precision(case_object_labels, case_object_scores))
            object_case_group_ids.append(group_id)
            case_positive_mask = positive_mask[case_index]
            if case_positive_mask.any():
                case_side_labels = targets["side_targets"][case_index][case_positive_mask].detach().cpu().int().tolist()
                case_side_scores = side_probability[case_index][case_positive_mask].cpu().tolist()
                case_abnormal_labels = targets["abnormality_targets"][case_index][case_positive_mask].detach().cpu().int().tolist()
                case_abnormal_scores = abnormal_probability[case_index][case_positive_mask].cpu().tolist()
                side_labels.extend(case_side_labels)
                side_scores.extend(case_side_scores)
                side_group_ids.extend([group_id] * len(case_side_labels))
                abnormal_labels.extend(case_abnormal_labels)
                abnormal_scores.extend(case_abnormal_scores)
                abnormal_group_ids.extend([group_id] * len(case_abnormal_labels))
        final_predictions = filter_module1_predictions(
            predictions,
            score_threshold=score_threshold,
            topk_per_image=topk_per_image,
            nms_threshold=nms_threshold,
            final_score_mode=final_score_mode,
            quality_score_alpha=quality_score_alpha,
            quality_score_beta=quality_score_beta,
        )
        batch_filter_counts = detection_filter_count_sums(final_predictions)
        for key in filter_counts:
            filter_counts[key] += batch_filter_counts[key]
        detection_group_ids.extend(str(value) for value in batch["group_id"])
        for threshold in iou_threshold_values:
            batch_counts = match_final_predictions(
                final_predictions,
                batch["labels"],
                predictions["shape_metadata"],
                iou_threshold=threshold,
            )
            for key in counts_by_threshold[threshold]:
                counts_by_threshold[threshold][key] += batch_counts[key]
            for case_index in range(len(final_predictions)):
                case_counts_by_threshold[threshold].append(
                    match_final_predictions(
                        [final_predictions[case_index]],
                        [batch["labels"][case_index]],
                        predictions["shape_metadata"],
                        iou_threshold=threshold,
                    )
                )
        best_ious.extend(final_prediction_best_ious(final_predictions, batch["labels"], predictions["shape_metadata"]))

        topk_items: list[dict[str, Any]] = []
        if compute_extended_diagnostics:
            strict_iou_scores = build_strict_pred_iou_targets(
                batch["labels"],
                predictions["shape_metadata"],
                predictions["bbox_pred"].detach(),
                predictions["side_logits"],
                device=device,
            )
            oracle_predictions = dict(predictions)
            oracle_predictions["objectness_logits"] = _scores_to_logits(strict_iou_scores)
            oracle_final_predictions = filter_module1_predictions(
                oracle_predictions,
                score_threshold=score_threshold,
                topk_per_image=topk_per_image,
                nms_threshold=nms_threshold,
                final_score_mode="objectness",
                quality_score_alpha=quality_score_alpha,
                quality_score_beta=quality_score_beta,
            )
            for threshold in iou_threshold_values:
                oracle_counts = match_final_predictions(
                    oracle_final_predictions,
                    batch["labels"],
                    predictions["shape_metadata"],
                    iou_threshold=threshold,
                )
                for key in oracle_iou_strict_counts_by_threshold[threshold]:
                    oracle_iou_strict_counts_by_threshold[threshold][key] += oracle_counts[key]
            oracle_bbox_ious.extend(oracle_positive_bbox_ious(predictions, batch["labels"], target_config=target_config))
            alignment = objectness_iou_alignment_metrics(
                predictions,
                batch["labels"],
                quality_iou_threshold=float(quality_iou_threshold),
            )
            objectness_iou_pearson.append(float(alignment["objectness_iou_pearson_corr"]))
            objectness_iou_spearman.append(float(alignment["objectness_iou_spearman_corr"]))
            current_score_modes = ["objectness"]
            if "bbox_quality_logits" in predictions:
                current_score_modes = ["objectness", "quality", "objectness_quality", "task_aligned_score"]
                score_mode_names = current_score_modes
            mode_diagnostics = score_mode_detection_diagnostics(
                predictions,
                batch["labels"],
                score_modes=current_score_modes,
                score_threshold=float(score_threshold),
                topk_per_image=topk_per_image,
                nms_threshold=nms_threshold,
                iou_thresholds=iou_threshold_values,
                quality_score_alpha=float(quality_score_alpha),
                quality_score_beta=float(quality_score_beta),
                stage_iou_threshold=0.3,
            )
            for mode, mode_metrics in mode_diagnostics.items():
                sums = score_mode_metric_sums.setdefault(mode, {})
                score_mode_metric_counts[mode] = score_mode_metric_counts.get(mode, 0) + 1
                for key, value in mode_metrics.items():
                    sums[key] = sums.get(key, 0.0) + float(value)
            topk_items = topk_detection_diagnostics(
                predictions,
                batch["labels"],
                target_config=target_config,
                quality_iou_threshold=float(quality_iou_threshold),
            )
            for item in topk_items:
                top100_positive_target_tokens.append(int(item["top100_positive_target_tokens"]))
                top100_quality_positive_tokens.append(int(item["top100_quality_positive_tokens"]))
                top100_mean_ious.append(float(item["top100_mean_iou"]))
                top100_max_ious.append(float(item["top100_max_iou"]))
                top100_iou_gt_0_1.append(int(item["top100_iou_gt_0.1"]))
                top100_iou_gt_0_3.append(int(item["top100_iou_gt_0.3"]))
                top100_iou_gt_0_45.append(int(item["top100_iou_gt_0.45"]))
        if log_detection_diagnostics and logger is not None:
            logger.info(f"detection_filter_counts {format_metrics({k: float(v) for k, v in batch_filter_counts.items()})}")
            for item in topk_items:
                logger.info(f"topk_detection_diagnostics {json.dumps(item, ensure_ascii=False)}")
        if progress_interval > 0 and logger is not None and (
            batch_index % int(progress_interval) == 0 or batch_index == len(loader)
        ):
            logger.info(f"validation_progress={batch_index}/{len(loader)} extended={compute_extended_diagnostics}")

    iou_metrics = summarize_iou_detection_metrics(
        counts_by_threshold,
        best_ious,
        primary_iou_threshold=float(iou_threshold),
    )
    oracle_metrics = summarize_oracle_positive_bbox_metrics(oracle_bbox_ious) if compute_extended_diagnostics else {}
    metrics = {
        "objectness_token_pr_auc": safe_average_precision(object_labels, object_scores),
        "side_roc_auc": safe_roc_auc(side_labels, side_scores),
        "abnormality_roc_auc": safe_roc_auc(abnormal_labels, abnormal_scores),
    }
    for prefix, labels, scores, threshold in (
        ("side", side_labels, side_scores, float(side_threshold)),
        ("abnormality", abnormal_labels, abnormal_scores, float(abnormality_threshold)),
    ):
        task_metrics = binary_task_metrics(labels, scores, threshold=threshold)
        metrics[f"{prefix}_threshold"] = threshold
        for name in ("accuracy", "precision", "recall", "specificity", "f1"):
            metrics[f"{prefix}_{name}"] = float(task_metrics[name])
    metrics.update({key: float(value) for key, value in filter_counts.items()})
    metrics.update({f"input_{key}": float(value) for key, value in label_mapping_counts.items()})
    metrics["input_gt_slice_sampling_recall"] = float(
        label_mapping_counts["exact_source_label_count"]
        / max(label_mapping_counts["raw_label_count"], 1)
    )
    metrics.update(iou_metrics)
    metrics.update(oracle_metrics)
    if compute_extended_diagnostics:
        metrics.update({
            "top100_positive_target_tokens": float(sum(top100_positive_target_tokens)),
            "top100_assigned_positive_tokens": float(sum(top100_positive_target_tokens)),
            "top100_quality_positive_tokens": float(sum(top100_quality_positive_tokens)),
            "top100_mean_iou": float(sum(top100_mean_ious) / max(len(top100_mean_ious), 1)),
            "top100_max_iou": float(max(top100_max_ious)) if top100_max_ious else 0.0,
            "top100_iou_gt_0.1": float(sum(top100_iou_gt_0_1)),
            "top100_iou_gt_0.3": float(sum(top100_iou_gt_0_3)),
            "top100_iou_gt_0.45": float(sum(top100_iou_gt_0_45)),
            "objectness_iou_pearson_corr": _mean_finite(objectness_iou_pearson),
            "objectness_iou_spearman_corr": _mean_finite(objectness_iou_spearman),
        })
        for mode in score_mode_names:
            sums = score_mode_metric_sums.get(mode, {})
            count = max(score_mode_metric_counts.get(mode, 0), 1)
            prefix = f"score_mode_{mode}"
            for key, value in sums.items():
                metrics[f"{prefix}_{key}"] = float(value) / count
        for threshold, counts in sorted(oracle_iou_strict_counts_by_threshold.items()):
            rates = precision_recall_specificity_f1(
                tp=int(counts["tp"]),
                fp=int(counts["fp"]),
                tn=0,
                fn=int(counts["fn"]),
            )
            suffix = f"{float(threshold):g}"
            metrics[f"oracle_iou_strict_precision@{suffix}"] = rates["precision"]
            metrics[f"oracle_iou_strict_recall@{suffix}"] = rates["recall"]
            metrics[f"oracle_iou_strict_f1@{suffix}"] = rates["f1"]
            metrics[f"oracle_iou_strict_tp@{suffix}"] = float(counts["tp"])
            metrics[f"oracle_iou_strict_fp@{suffix}"] = float(counts["fp"])
            metrics[f"oracle_iou_strict_fn@{suffix}"] = float(counts["fn"])
    _add_quality_ranking_aliases(metrics, final_score_mode=final_score_mode)
    metrics["tp"] = metrics["iou_detection_tp"]
    metrics["fp"] = metrics["iou_detection_fp"]
    metrics["fn"] = metrics["iou_detection_fn"]
    if bootstrap_samples > 0:
        side_pairs = list(zip(side_labels, side_scores))
        abnormal_pairs = list(zip(abnormal_labels, abnormal_scores))

        finite_case_items = [
            (value, group_id)
            for value, group_id in zip(object_case_pr_aucs, object_case_group_ids)
            if value == value
        ]
        point, lo, hi = grouped_bootstrap_ci(
            [item[0] for item in finite_case_items],
            [item[1] for item in finite_case_items],
            lambda values: float(sum(values) / max(len(values), 1)),
            n_boot=bootstrap_samples,
        )
        metrics["objectness_case_macro_pr_auc"] = point
        metrics["objectness_case_macro_pr_auc_ci_low"] = lo
        metrics["objectness_case_macro_pr_auc_ci_high"] = hi

        for prefix, pairs, group_ids, threshold in (
            ("side", side_pairs, side_group_ids, float(side_threshold)),
            ("abnormality", abnormal_pairs, abnormal_group_ids, float(abnormality_threshold)),
        ):
            def classification_metric(
                items: list[tuple[int, float]],
                name: str,
            ) -> float:
                item_labels = [item[0] for item in items]
                item_scores = [item[1] for item in items]
                if name == "roc_auc":
                    return safe_roc_auc(item_labels, item_scores)
                item_preds = [int(score >= threshold) for score in item_scores]
                tp = sum(int(pred == 1 and target == 1) for pred, target in zip(item_preds, item_labels))
                tn = sum(int(pred == 0 and target == 0) for pred, target in zip(item_preds, item_labels))
                fp = sum(int(pred == 1 and target == 0) for pred, target in zip(item_preds, item_labels))
                fn = sum(int(pred == 0 and target == 1) for pred, target in zip(item_preds, item_labels))
                if name == "accuracy":
                    return (tp + tn) / max(len(item_labels), 1)
                return precision_recall_specificity_f1(tp=tp, fp=fp, tn=tn, fn=fn)[name]

            for metric_name in ("roc_auc", "accuracy", "precision", "recall", "specificity", "f1"):
                _, lo, hi = grouped_bootstrap_ci(
                    pairs,
                    group_ids,
                    lambda items, name=metric_name: classification_metric(items, name),
                    n_boot=bootstrap_samples,
                )
                metrics[f"{prefix}_{metric_name}_ci_low"] = lo
                metrics[f"{prefix}_{metric_name}_ci_high"] = hi
        primary_case_counts = case_counts_by_threshold[float(iou_threshold)]

        def detection_rate(items: list[dict[str, int]], name: str) -> float:
            totals = {
                key: sum(int(item[key]) for item in items)
                for key in ("tp", "fp", "fn")
            }
            rates = precision_recall_specificity_f1(
                tp=totals["tp"],
                fp=totals["fp"],
                tn=0,
                fn=totals["fn"],
            )
            return rates[name]

        for metric_name in ("precision", "recall", "f1"):
            _, lo, hi = grouped_bootstrap_ci(
                primary_case_counts,
                detection_group_ids,
                lambda items, name=metric_name: detection_rate(items, name),
                n_boot=bootstrap_samples,
            )
            metrics[f"iou_detection_{metric_name}_ci_low"] = lo
            metrics[f"iou_detection_{metric_name}_ci_high"] = hi
        metrics["bootstrap_case_count"] = float(len(primary_case_counts))
        metrics["bootstrap_group_count"] = float(len(set(detection_group_ids)))
    return metrics


def validate_module1(
    config_path: str,
    checkpoint: str,
    index_jsonl: str | None = None,
    bootstrap_samples: int | None = None,
) -> dict[str, float]:
    config = load_config(config_path)
    logger = get_logger("validate_module1")
    device_name = config.get("device", "cuda")
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_module1_model(config).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    loader = create_val_loader(config, index_jsonl=index_jsonl)
    validate_cfg = config.get("validate", {})
    metrics = evaluate_module1(
        model,
        loader,
        device,
        score_threshold=config.get("validate", {}).get("score_threshold", 0.5),
        iou_threshold=config.get("validate", {}).get("iou_threshold", 0.45),
        iou_thresholds=config.get("validate", {}).get("iou_thresholds"),
        topk_per_image=config.get("validate", {}).get("topk_per_image"),
        nms_threshold=config.get("validate", {}).get("nms_threshold"),
        final_score_mode=config.get("validate", {}).get("final_score_mode", "objectness"),
        quality_score_alpha=config.get("validate", {}).get("quality_score_alpha", 1.0),
        quality_score_beta=config.get("validate", {}).get("quality_score_beta", 1.0),
        amp=config.get("train", {}).get("amp", True),
        bootstrap_samples=int(
            validate_cfg.get("bootstrap_samples", 1000)
            if bootstrap_samples is None
            else bootstrap_samples
        ),
        positive_radius_tokens=config.get("loss", {}).get("positive_radius_tokens", 0),
        quality_iou_threshold=config.get("loss", {}).get("quality_iou_threshold", 0.3),
        logger=logger,
        log_detection_diagnostics=bool(validate_cfg.get("log_detection_diagnostics", True)),
        compute_extended_diagnostics=bool(validate_cfg.get("compute_extended_diagnostics", True)),
        progress_interval=int(validate_cfg.get("progress_interval", 50)),
        side_threshold=float(validate_cfg.get("side_threshold", 0.5)),
        abnormality_threshold=float(validate_cfg.get("abnormality_threshold", 0.4)),
    )
    logger.info(format_metrics(metrics))
    return metrics


def _resolve_iou_thresholds(
    primary_iou_threshold: float,
    iou_thresholds: list[float] | tuple[float, ...] | None,
) -> list[float]:
    thresholds = [0.1, 0.3, 0.45] if iou_thresholds is None else [float(value) for value in iou_thresholds]
    thresholds.append(float(primary_iou_threshold))
    thresholds.append(0.3)
    return sorted(set(float(value) for value in thresholds))


def _mean_finite(values: list[float]) -> float:
    finite = [float(value) for value in values if value == value]
    return sum(finite) / max(len(finite), 1)


def _scores_to_logits(scores: torch.Tensor) -> torch.Tensor:
    return torch.logit(scores.detach().float().clamp(1e-6, 1.0 - 1e-6)).to(device=scores.device)


def _add_quality_ranking_aliases(metrics: dict[str, float], final_score_mode: str) -> None:
    final_prefix = f"score_mode_{final_score_mode}"
    quality_prefix = "score_mode_quality"
    objectness_quality_prefix = "score_mode_objectness_quality"
    aliases = {
        "final_precision@0.3": "iou_detection_precision@0.3",
        "final_recall@0.3": "iou_detection_recall@0.3",
        "final_f1@0.3": "iou_detection_f1@0.3",
        "final_P@0.3": "iou_detection_precision@0.3",
        "final_R@0.3": "iou_detection_recall@0.3",
        "final_F1@0.3": "iou_detection_f1@0.3",
        "oracle_iou_strict_P@0.3": "oracle_iou_strict_precision@0.3",
        "oracle_iou_strict_R@0.3": "oracle_iou_strict_recall@0.3",
        "oracle_iou_strict_F1@0.3": "oracle_iou_strict_f1@0.3",
        "after_topk_recall@0.3": f"{final_prefix}_after_topk_recall@0.3",
        "after_topk_R@0.3": f"{final_prefix}_after_topk_recall@0.3",
        "quality_only_precision@0.3": f"{quality_prefix}_precision@0.3",
        "quality_only_recall@0.3": f"{quality_prefix}_recall@0.3",
        "quality_only_f1@0.3": f"{quality_prefix}_f1@0.3",
        "objectness_quality_precision@0.3": f"{objectness_quality_prefix}_precision@0.3",
        "objectness_quality_recall@0.3": f"{objectness_quality_prefix}_recall@0.3",
        "objectness_quality_f1@0.3": f"{objectness_quality_prefix}_f1@0.3",
        "good_quality_score_mean": f"{quality_prefix}_good_candidate_quality_score_mean",
        "bad_quality_score_mean": f"{quality_prefix}_bad_candidate_quality_score_mean",
        "good_bad_score_gap": f"{quality_prefix}_good_bad_quality_score_gap",
        "quality_score_spearman": f"{quality_prefix}_quality_score_spearman_iou",
        "quality_score_pearson": f"{quality_prefix}_quality_score_pearson_iou",
    }
    for alias, source in aliases.items():
        if source in metrics:
            metrics[alias] = metrics[source]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/module1_meniscovit_cop.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    args = parser.parse_args()
    validate_module1(args.config, args.checkpoint, index_jsonl=args.index, bootstrap_samples=args.bootstrap)


if __name__ == "__main__":
    main()
