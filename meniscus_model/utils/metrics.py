"""Evaluation metrics for Module 1 detection and classification."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import random
from typing import Any

import numpy as np
import torch

from losses.detection_loss import DetectionTargetConfig, box_iou, build_module1_targets, build_strict_pred_iou_targets
from utils.bootstrap import bootstrap_ci as generic_bootstrap_ci, grouped_bootstrap_ci


def binary_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute binary accuracy from logits ``[N]`` or ``[N,2]``."""

    if logits.ndim == 2 and logits.shape[1] == 2:
        preds = logits.argmax(dim=1)
    else:
        preds = (logits.sigmoid() >= threshold).long()
    return (preds.cpu() == targets.cpu().long()).float().mean().item()


def multiclass_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute multi-class accuracy from logits ``[N,C]``."""

    return (logits.argmax(dim=1).cpu() == targets.cpu().long()).float().mean().item()


def confusion_counts(preds: torch.Tensor, targets: torch.Tensor) -> dict[str, int]:
    """Return TP/TN/FP/FN for binary predictions and targets."""

    preds = preds.cpu().bool()
    targets = targets.cpu().bool()
    return {
        "tp": int((preds & targets).sum()),
        "tn": int((~preds & ~targets).sum()),
        "fp": int((preds & ~targets).sum()),
        "fn": int((~preds & targets).sum()),
    }


def safe_average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Average precision / PR-AUC with graceful fallback for single-class data."""

    y = np.asarray(labels)
    s = np.asarray(scores)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s))


def safe_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """ROC-AUC with graceful fallback for single-class data."""

    y = np.asarray(labels)
    s = np.asarray(scores)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def safe_pearson_corr(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def safe_spearman_corr(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    return safe_pearson_corr(_average_ranks(a), _average_ranks(b))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def precision_recall_specificity_f1(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "specificity": specificity, "f1": f1}


def bootstrap_ci(
    values: Sequence[Any],
    metric_fn: Callable[[list[Any]], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap estimate and percentile 95% CI by default."""

    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = []
    values_list = list(values)
    for _ in range(n_boot):
        sample = [rng.choice(values_list) for _ in values_list]
        estimates.append(metric_fn(sample))
    estimates = [v for v in estimates if not np.isnan(v)]
    point = metric_fn(values_list)
    if not estimates:
        return point, float("nan"), float("nan")
    estimates.sort()
    lo_idx = max(0, int((alpha / 2) * len(estimates)))
    hi_idx = min(len(estimates) - 1, int((1 - alpha / 2) * len(estimates)) - 1)
    return point, float(estimates[lo_idx]), float(estimates[hi_idx])


def binary_task_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = 0.5,
    predictions: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Compute paper-style binary metrics from labels and scores."""

    y_true = [int(v) for v in labels]
    y_score = [float(v) for v in scores]
    if len(y_true) == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "specificity": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }
    preds = (
        [int(value) for value in predictions]
        if predictions is not None
        else [int(score >= threshold) for score in y_score]
    )
    if not len(y_true) == len(y_score) == len(preds):
        raise ValueError("labels, scores, and predictions must have the same length")
    tp = sum(int(pred == 1 and target == 1) for pred, target in zip(preds, y_true))
    tn = sum(int(pred == 0 and target == 0) for pred, target in zip(preds, y_true))
    fp = sum(int(pred == 1 and target == 0) for pred, target in zip(preds, y_true))
    fn = sum(int(pred == 0 and target == 1) for pred, target in zip(preds, y_true))
    rates = precision_recall_specificity_f1(tp=tp, fp=fp, tn=tn, fn=fn)
    return {
        "n": len(y_true),
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "precision": rates["precision"],
        "recall": rates["recall"],
        "specificity": rates["specificity"],
        "f1": rates["f1"],
        "roc_auc": safe_roc_auc(y_true, y_score),
        "pr_auc": safe_average_precision(y_true, y_score),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def binary_task_metrics_with_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 42,
    group_ids: Sequence[Any] | None = None,
    predictions: Sequence[int] | None = None,
) -> dict[str, float | int | dict[str, float]]:
    """Binary task metrics with patient/case-clustered bootstrap CIs."""

    resolved_predictions = (
        [int(value) for value in predictions]
        if predictions is not None
        else [int(float(score) >= threshold) for score in scores]
    )
    pairs = list(zip(
        [int(v) for v in labels],
        [float(v) for v in scores],
        resolved_predictions,
        strict=True,
    ))
    if group_ids is not None and len(group_ids) != len(pairs):
        raise ValueError("group_ids must have the same length as labels and scores")
    metrics = binary_task_metrics(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        threshold,
        predictions=[p[2] for p in pairs],
    )
    for key in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"):
        def metric_fn(items: list[tuple[int, float, int]], metric_key: str = key) -> float:
            item_metrics = binary_task_metrics(
                [item[0] for item in items],
                [item[1] for item in items],
                threshold,
                predictions=[item[2] for item in items],
            )
            return float(item_metrics[metric_key])

        if group_ids is None:
            _, lo, hi = generic_bootstrap_ci(pairs, metric_fn, n_boot=n_boot, seed=seed)
        else:
            _, lo, hi = grouped_bootstrap_ci(
                pairs,
                list(group_ids),
                metric_fn,
                n_boot=n_boot,
                seed=seed,
            )
        metrics[f"{key}_ci"] = {"low": lo, "high": hi}
    return metrics


def module1_token_metrics(
    predictions: dict[str, torch.Tensor | dict[str, Any]],
    labels: list[dict[str, Any]],
    target_config: DetectionTargetConfig | None = None,
) -> dict[str, float]:
    """Token-level PR-AUC and classification ROC-AUC for positive tokens."""

    device = predictions["objectness_logits"].device
    targets = build_module1_targets(labels, predictions["shape_metadata"], device=device, target_config=target_config)
    objectness_scores = torch.sigmoid(predictions["objectness_logits"]).detach().cpu().flatten().numpy().tolist()
    objectness_labels = targets["objectness"].detach().cpu().flatten().int().numpy().tolist()
    positive_mask = targets["positive_mask"]
    metrics = {"objectness_token_pr_auc": safe_average_precision(objectness_labels, objectness_scores)}
    if positive_mask.any():
        side_scores = torch.softmax(predictions["side_logits"][positive_mask], dim=-1)[:, 1].detach().cpu().numpy().tolist()
        side_labels = targets["side_targets"][positive_mask].detach().cpu().numpy().tolist()
        abnormal_scores = torch.softmax(predictions["abnormality_logits"][positive_mask], dim=-1)[:, 1].detach().cpu().numpy().tolist()
        abnormal_labels = targets["abnormality_targets"][positive_mask].detach().cpu().numpy().tolist()
        metrics["side_roc_auc"] = safe_roc_auc(side_labels, side_scores)
        metrics["abnormality_roc_auc"] = safe_roc_auc(abnormal_labels, abnormal_scores)
    else:
        metrics["side_roc_auc"] = float("nan")
        metrics["abnormality_roc_auc"] = float("nan")
    return metrics


def detection_match_counts(
    predictions: dict[str, torch.Tensor | dict[str, Any]],
    labels: list[dict[str, Any]],
    score_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    topk_per_image: int | None = None,
    nms_threshold: float | None = None,
    final_score_mode: str = "objectness",
    quality_score_alpha: float = 1.0,
    quality_score_beta: float = 1.0,
    target_config: DetectionTargetConfig | None = None,
) -> dict[str, int]:
    """Compute IoU-based detection TP/FP/FN from final filtered predictions.

    Predictions are token-level normalized boxes. Predicted boxes above
    ``score_threshold`` are optionally top-k filtered and NMS-filtered before
    greedy matching. Matching is only within the same case, plane, processed
    slice, and meniscus side.
    """

    device = predictions["objectness_logits"].device
    shape_metadata = predictions["shape_metadata"]
    targets = build_module1_targets(labels, shape_metadata, device=device, target_config=target_config)
    scores = compute_module1_final_scores(
        predictions,
        final_score_mode=final_score_mode,
        quality_score_alpha=quality_score_alpha,
        quality_score_beta=quality_score_beta,
    )
    pred_positive = scores >= score_threshold

    final_predictions = filter_module1_predictions(
        predictions,
        score_threshold=score_threshold,
        topk_per_image=topk_per_image,
        nms_threshold=nms_threshold,
        final_score_mode=final_score_mode,
        quality_score_alpha=quality_score_alpha,
        quality_score_beta=quality_score_beta,
    )
    iou_counts = match_final_predictions(
        final_predictions,
        labels,
        shape_metadata,
        iou_threshold=iou_threshold,
    )

    token_counts = confusion_counts(pred_positive.cpu(), targets["objectness"].cpu().bool())
    # Keep token-level TN only for backward-compatible callers; TP/FP/FN are final-detection counts.
    return {"tp": iou_counts["tp"], "fp": iou_counts["fp"], "tn": token_counts["tn"], "fn": iou_counts["fn"]}


def compute_module1_final_scores(
    predictions: dict[str, torch.Tensor | dict[str, Any]],
    final_score_mode: str = "objectness",
    quality_score_alpha: float = 1.0,
    quality_score_beta: float = 1.0,
) -> torch.Tensor:
    """Return token scores used for Module 1 post-processing."""

    if final_score_mode not in {"objectness", "quality", "objectness_quality", "task_aligned_score"}:
        raise ValueError("final_score_mode must be objectness, quality, objectness_quality, or task_aligned_score")
    objectness = torch.sigmoid(predictions["objectness_logits"]).detach()
    if final_score_mode == "objectness":
        return objectness
    if "bbox_quality_logits" not in predictions:
        raise KeyError(f"bbox_quality_logits is required for final_score_mode={final_score_mode}")
    quality = torch.sigmoid(predictions["bbox_quality_logits"]).detach()
    if final_score_mode == "quality":
        return quality
    if final_score_mode == "objectness_quality":
        return objectness * quality
    return objectness.pow(float(quality_score_alpha)) * quality.pow(float(quality_score_beta))


def filter_module1_predictions(
    predictions: dict[str, torch.Tensor | dict[str, Any]],
    score_threshold: float = 0.5,
    topk_per_image: int | None = None,
    nms_threshold: float | None = None,
    final_score_mode: str = "objectness",
    quality_score_alpha: float = 1.0,
    quality_score_beta: float = 1.0,
) -> list[dict[str, Any]]:
    """Decode raw token predictions and apply score threshold, top-k, and NMS."""

    shape_metadata = predictions["shape_metadata"]
    device = predictions["objectness_logits"].device
    plane_ids, slice_indices = _token_plane_and_slice_ids(shape_metadata, device)
    scores = compute_module1_final_scores(
        predictions,
        final_score_mode=final_score_mode,
        quality_score_alpha=quality_score_alpha,
        quality_score_beta=quality_score_beta,
    )
    boxes = predictions["bbox_pred"].detach()
    side_ids = predictions["side_logits"].detach().argmax(dim=-1)

    filtered: list[dict[str, Any]] = []
    for batch_idx in range(scores.shape[0]):
        case_scores = scores[batch_idx]
        case_boxes = boxes[batch_idx]
        raw_count = int(case_scores.numel())

        threshold_indices = torch.nonzero(case_scores >= float(score_threshold), as_tuple=False).flatten()
        num_after_score = int(threshold_indices.numel())
        if threshold_indices.numel() > 0:
            ordered = case_scores[threshold_indices].argsort(descending=True)
            threshold_indices = threshold_indices[ordered]
        if topk_per_image is not None:
            topk = max(0, int(topk_per_image))
            threshold_indices = threshold_indices[:topk]
        num_after_topk = int(threshold_indices.numel())

        if threshold_indices.numel() > 0 and nms_threshold is not None:
            keep_indices: list[torch.Tensor] = []
            group_keys = torch.stack(
                [plane_ids[threshold_indices], slice_indices[threshold_indices], side_ids[batch_idx, threshold_indices]],
                dim=1,
            )
            unique_keys = torch.unique(group_keys, dim=0)
            for key in unique_keys:
                group_mask = (group_keys == key).all(dim=1)
                group_indices = threshold_indices[group_mask]
                keep_local = _nms_boxes(case_boxes[group_indices], case_scores[group_indices], float(nms_threshold))
                keep_indices.append(group_indices[keep_local])
            final_indices = torch.cat(keep_indices, dim=0) if keep_indices else threshold_indices.new_empty((0,))
            if final_indices.numel() > 0:
                final_indices = final_indices[case_scores[final_indices].argsort(descending=True)]
        else:
            final_indices = threshold_indices
        num_after_nms = int(final_indices.numel())

        counts = {
            "num_raw_predictions": raw_count,
            "num_after_score_threshold": num_after_score,
            "num_after_topk": num_after_topk,
            "num_after_nms": num_after_nms,
            "num_final_predictions_for_matching": num_after_nms,
        }
        filtered.append(
            {
                "indices": final_indices,
                "scores": case_scores[final_indices],
                "boxes": case_boxes[final_indices],
                "plane_ids": plane_ids[final_indices],
                "slice_indices": slice_indices[final_indices],
                "side_ids": side_ids[batch_idx, final_indices],
                "counts": counts,
            }
        )
    return filtered


def match_final_predictions(
    final_predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    shape_metadata: dict[str, Any],
    iou_threshold: float = 0.45,
) -> dict[str, int]:
    """Strict IoU matching for already-filtered final predictions."""

    device = final_predictions[0]["boxes"].device if final_predictions else torch.device("cpu")
    tp = fp = fn = 0
    for batch_idx, case_predictions in enumerate(final_predictions):
        case_gt = _case_gt_boxes(labels[batch_idx], shape_metadata, device)
        case_pred = _final_predictions_by_key(case_predictions)
        case_tp = case_fp = case_fn = 0
        for key in sorted(set(case_gt) | set(case_pred)):
            pred_items = case_pred.get(key, [])
            gt_boxes = case_gt.get(key, torch.zeros((0, 4), device=device))
            if not pred_items:
                case_fn += int(gt_boxes.shape[0])
                continue
            pred_rows = torch.tensor([item[0] for item in pred_items], device=device, dtype=torch.long)
            pred_scores = torch.tensor([item[1] for item in pred_items], device=device, dtype=torch.float32)
            if gt_boxes.numel() == 0:
                case_fp += int(pred_rows.numel())
                continue
            pred_boxes = case_predictions["boxes"][pred_rows]
            ious = box_iou(pred_boxes, gt_boxes)
            used_gt: set[int] = set()
            order = pred_scores.argsort(descending=True)
            for pred_order_idx in order:
                row = int(pred_order_idx)
                best_iou, best_gt = ious[row].max(dim=0)
                best_gt_int = int(best_gt)
                if float(best_iou) >= float(iou_threshold) and best_gt_int not in used_gt:
                    case_tp += 1
                    used_gt.add(best_gt_int)
                else:
                    case_fp += 1
            case_fn += max(gt_boxes.shape[0] - len(used_gt), 0)

        final_count = int(case_predictions["counts"]["num_final_predictions_for_matching"])
        assert case_fp <= final_count
        if final_count == 0:
            assert case_fp == 0 and case_tp == 0
        tp += case_tp
        fp += case_fp
        fn += case_fn
    return {"tp": tp, "fp": fp, "fn": fn}


def final_prediction_best_ious(
    final_predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    shape_metadata: dict[str, Any],
) -> list[float]:
    """Best strict-key GT IoU for each final prediction."""

    device = final_predictions[0]["boxes"].device if final_predictions else torch.device("cpu")
    best_ious: list[float] = []
    for batch_idx, case_predictions in enumerate(final_predictions):
        case_gt = _case_gt_boxes(labels[batch_idx], shape_metadata, device)
        for row in range(int(case_predictions["boxes"].shape[0])):
            key = (
                int(case_predictions["plane_ids"][row].item()),
                int(case_predictions["slice_indices"][row].item()),
                int(case_predictions["side_ids"][row].item()),
            )
            gt_boxes = case_gt.get(key)
            if gt_boxes is None or gt_boxes.numel() == 0:
                best_ious.append(0.0)
                continue
            iou = box_iou(case_predictions["boxes"][row].view(1, 4), gt_boxes).max()
            best_ious.append(float(iou.item()))
    return best_ious


def detection_filter_count_sums(final_predictions: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "num_raw_predictions": 0,
        "num_after_score_threshold": 0,
        "num_after_topk": 0,
        "num_after_nms": 0,
        "num_final_predictions_for_matching": 0,
    }
    for case_predictions in final_predictions:
        for key in counts:
            counts[key] += int(case_predictions["counts"][key])
    return counts


def topk_detection_diagnostics(
    predictions: dict[str, torch.Tensor | dict[str, Any]],
    labels: list[dict[str, Any]],
    target_config: DetectionTargetConfig | None = None,
    quality_iou_threshold: float = 0.3,
    topk: int = 100,
    detail_count: int = 10,
) -> list[dict[str, Any]]:
    """Per-case top-k objectness diagnostics before final filtering."""

    device = predictions["objectness_logits"].device
    shape_metadata = predictions["shape_metadata"]
    targets = build_module1_targets(labels, shape_metadata, device=device, target_config=target_config)
    plane_ids, slice_indices = _token_plane_and_slice_ids(shape_metadata, device)
    scores = torch.sigmoid(predictions["objectness_logits"]).detach()
    boxes = predictions["bbox_pred"].detach()
    side_ids = predictions["side_logits"].detach().argmax(dim=-1)
    strict_iou_targets = build_strict_pred_iou_targets(labels, shape_metadata, boxes, predictions["side_logits"], device=device)
    diagnostics: list[dict[str, Any]] = []

    for batch_idx in range(scores.shape[0]):
        case_scores = scores[batch_idx]
        k = min(max(0, int(topk)), int(case_scores.numel()))
        top_indices = torch.topk(case_scores, k=k, largest=True).indices if k > 0 else case_scores.new_empty((0,), dtype=torch.long)
        case_gt = _case_gt_boxes(labels[batch_idx], shape_metadata, device)
        best_iou_values: list[float] = []
        details: list[dict[str, Any]] = []
        for rank, token_idx_tensor in enumerate(top_indices):
            token_idx = int(token_idx_tensor.item())
            key = (
                int(plane_ids[token_idx].item()),
                int(slice_indices[token_idx].item()),
                int(side_ids[batch_idx, token_idx].item()),
            )
            gt_boxes = case_gt.get(key)
            nearest_gt_box: list[float] | None = None
            nearest_gt_iou = 0.0
            if gt_boxes is not None and gt_boxes.numel() > 0:
                ious = box_iou(boxes[batch_idx, token_idx].view(1, 4), gt_boxes).flatten()
                nearest_gt_iou_tensor, nearest_gt_idx = ious.max(dim=0)
                nearest_gt_iou = float(nearest_gt_iou_tensor.item())
                nearest_gt_box = [float(v) for v in gt_boxes[int(nearest_gt_idx)].detach().cpu().tolist()]
            best_iou_values.append(nearest_gt_iou)
            if rank < int(detail_count):
                details.append(
                    {
                        "score": float(case_scores[token_idx].item()),
                        "token_index": token_idx,
                        "plane": _plane_name(key[0]),
                        "processed_slice": key[1],
                        "side": _side_name(key[2]),
                        "token_center": _token_center_for_index(token_idx, shape_metadata, device),
                        "is_positive_target": bool(targets["positive_mask"][batch_idx, token_idx].item()),
                        "is_quality_target": bool(
                            (not targets["positive_mask"][batch_idx, token_idx].item())
                            and strict_iou_targets[batch_idx, token_idx].item() >= float(quality_iou_threshold)
                        ),
                        "pred_box": [float(v) for v in boxes[batch_idx, token_idx].detach().cpu().tolist()],
                        "target_box": nearest_gt_box,
                        "nearest_gt_box": nearest_gt_box,
                        "target_iou": nearest_gt_iou,
                        "nearest_gt_iou": nearest_gt_iou,
                    }
                )

        diagnostics.append(
            {
                "case_id": labels[batch_idx].get("case_id", str(batch_idx)),
                "topk": k,
                "top100_positive_target_tokens": int(targets["positive_mask"][batch_idx, top_indices].sum().item()) if k > 0 else 0,
                "top100_quality_positive_tokens": int(
                    (
                        (~targets["positive_mask"][batch_idx, top_indices])
                        & (strict_iou_targets[batch_idx, top_indices] >= float(quality_iou_threshold))
                    ).sum().item()
                )
                if k > 0
                else 0,
                "top100_mean_iou": float(np.mean(best_iou_values)) if best_iou_values else 0.0,
                "top100_max_iou": float(np.max(best_iou_values)) if best_iou_values else 0.0,
                "top100_iou_gt_0.1": int(sum(iou > 0.1 for iou in best_iou_values)),
                "top100_iou_gt_0.3": int(sum(iou > 0.3 for iou in best_iou_values)),
                "top100_iou_gt_0.45": int(sum(iou > 0.45 for iou in best_iou_values)),
                "top10_predictions": details,
            }
        )
    return diagnostics


def objectness_iou_alignment_metrics(
    predictions: dict[str, torch.Tensor | dict[str, Any]],
    labels: list[dict[str, Any]],
    quality_iou_threshold: float = 0.3,
) -> dict[str, float]:
    """Correlation between token objectness score and strict predicted-box IoU."""

    device = predictions["objectness_logits"].device
    strict_ious = build_strict_pred_iou_targets(
        labels,
        predictions["shape_metadata"],
        predictions["bbox_pred"].detach(),
        predictions["side_logits"],
        device=device,
    )
    scores = torch.sigmoid(predictions["objectness_logits"]).detach()
    score_values = scores.cpu().flatten().tolist()
    iou_values = strict_ious.cpu().flatten().tolist()
    return {
        "objectness_iou_pearson_corr": safe_pearson_corr(score_values, iou_values),
        "objectness_iou_spearman_corr": safe_spearman_corr(score_values, iou_values),
        "num_quality_iou_tokens": float((strict_ious >= float(quality_iou_threshold)).sum().item()),
    }


def score_mode_detection_diagnostics(
    predictions: dict[str, torch.Tensor | dict[str, Any]],
    labels: list[dict[str, Any]],
    score_modes: Sequence[str],
    score_threshold: float,
    topk_per_image: int | None,
    nms_threshold: float | None,
    iou_thresholds: Sequence[float],
    quality_score_alpha: float = 1.0,
    quality_score_beta: float = 1.0,
    stage_iou_threshold: float = 0.3,
) -> dict[str, dict[str, float]]:
    """Evaluate final detection and candidate stages for several score modes."""

    device = predictions["objectness_logits"].device
    plane_ids, slice_indices = _token_plane_and_slice_ids(predictions["shape_metadata"], device)
    boxes = predictions["bbox_pred"].detach()
    side_ids = predictions["side_logits"].detach().argmax(dim=-1)
    strict_ious = build_strict_pred_iou_targets(
        labels,
        predictions["shape_metadata"],
        boxes,
        predictions["side_logits"],
        device=device,
    )
    results: dict[str, dict[str, float]] = {}
    for mode in score_modes:
        scores = compute_module1_final_scores(
            predictions,
            final_score_mode=mode,
            quality_score_alpha=quality_score_alpha,
            quality_score_beta=quality_score_beta,
        )
        final_predictions = filter_module1_predictions(
            predictions,
            score_threshold=score_threshold,
            topk_per_image=topk_per_image,
            nms_threshold=nms_threshold,
            final_score_mode=mode,
            quality_score_alpha=quality_score_alpha,
            quality_score_beta=quality_score_beta,
        )
        mode_metrics: dict[str, float] = {}
        for threshold in iou_thresholds:
            counts = match_final_predictions(
                final_predictions,
                labels,
                predictions["shape_metadata"],
                iou_threshold=float(threshold),
            )
            rates = precision_recall_specificity_f1(tp=counts["tp"], fp=counts["fp"], tn=0, fn=counts["fn"])
            suffix = _threshold_suffix(float(threshold))
            mode_metrics[f"precision@{suffix}"] = rates["precision"]
            mode_metrics[f"recall@{suffix}"] = rates["recall"]
            mode_metrics[f"f1@{suffix}"] = rates["f1"]

        stage_hits = {"raw": 0, "threshold": 0, "topk": 0, "nms": 0}
        total_gt = 0
        for batch_idx in range(scores.shape[0]):
            trace = _postprocess_indices_for_scores(
                scores=scores[batch_idx],
                boxes=boxes[batch_idx],
                plane_ids=plane_ids,
                slice_indices=slice_indices,
                side_ids=side_ids[batch_idx],
                score_threshold=score_threshold,
                topk_per_image=topk_per_image,
                nms_threshold=nms_threshold,
            )
            case_total, case_raw = _stage_gt_hits(
                labels[batch_idx],
                predictions["shape_metadata"],
                torch.arange(scores.shape[1], device=device, dtype=torch.long),
                boxes[batch_idx],
                plane_ids,
                slice_indices,
                side_ids[batch_idx],
                float(stage_iou_threshold),
            )
            _, case_threshold = _stage_gt_hits(
                labels[batch_idx],
                predictions["shape_metadata"],
                trace["threshold_indices"],
                boxes[batch_idx],
                plane_ids,
                slice_indices,
                side_ids[batch_idx],
                float(stage_iou_threshold),
            )
            _, case_topk = _stage_gt_hits(
                labels[batch_idx],
                predictions["shape_metadata"],
                trace["topk_indices"],
                boxes[batch_idx],
                plane_ids,
                slice_indices,
                side_ids[batch_idx],
                float(stage_iou_threshold),
            )
            _, case_nms = _stage_gt_hits(
                labels[batch_idx],
                predictions["shape_metadata"],
                trace["final_indices"],
                boxes[batch_idx],
                plane_ids,
                slice_indices,
                side_ids[batch_idx],
                float(stage_iou_threshold),
            )
            total_gt += case_total
            stage_hits["raw"] += case_raw
            stage_hits["threshold"] += case_threshold
            stage_hits["topk"] += case_topk
            stage_hits["nms"] += case_nms

        score_values = scores.detach().cpu().flatten().tolist()
        iou_values = strict_ious.detach().cpu().flatten().tolist()
        good_mask = strict_ious >= 0.3
        bad_mask = strict_ious < 0.1
        score_pearson = safe_pearson_corr(score_values, iou_values)
        score_spearman = safe_spearman_corr(score_values, iou_values)
        good_score_mean = float(scores[good_mask].mean().item()) if good_mask.any() else 0.0
        bad_score_mean = float(scores[bad_mask].mean().item()) if bad_mask.any() else 0.0
        mode_metrics.update(
            {
                "score_iou_pearson_corr": score_pearson,
                "score_iou_spearman_corr": score_spearman,
                "good_candidate_score_mean_iou_gt_0.3": good_score_mean,
                "bad_candidate_score_mean_iou_lt_0.1": bad_score_mean,
                "good_candidate_quality_score_mean": good_score_mean,
                "bad_candidate_quality_score_mean": bad_score_mean,
                "good_bad_quality_score_gap": good_score_mean - bad_score_mean,
                "quality_score_spearman_iou": score_spearman,
                "quality_score_pearson_iou": score_pearson,
                "raw_upper_recall@0.3": float(stage_hits["raw"]) / max(total_gt, 1),
                "after_threshold_recall@0.3": float(stage_hits["threshold"]) / max(total_gt, 1),
                "after_topk_recall@0.3": float(stage_hits["topk"]) / max(total_gt, 1),
                "after_nms_recall@0.3": float(stage_hits["nms"]) / max(total_gt, 1),
                "final_recall@0.3": mode_metrics.get("recall@0.3", 0.0),
            }
        )
        results[mode] = mode_metrics
    return results


def oracle_positive_bbox_ious(
    predictions: dict[str, torch.Tensor | dict[str, Any]],
    labels: list[dict[str, Any]],
    target_config: DetectionTargetConfig | None = None,
) -> list[float]:
    """IoUs between bbox predictions and assigned GT boxes at target-positive tokens."""

    device = predictions["objectness_logits"].device
    targets = build_module1_targets(labels, predictions["shape_metadata"], device=device, target_config=target_config)
    positive_mask = targets["positive_mask"]
    if not positive_mask.any():
        return []
    pred_boxes = predictions["bbox_pred"].detach()[positive_mask]
    target_boxes = targets["bbox_targets"].detach()[positive_mask]
    ious = box_iou(pred_boxes, target_boxes).diag()
    return [float(v) for v in ious.detach().cpu().tolist()]


def summarize_iou_detection_metrics(
    counts_by_threshold: dict[float, dict[str, int]],
    best_ious: Sequence[float],
    primary_iou_threshold: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for threshold, counts in sorted(counts_by_threshold.items()):
        rates = precision_recall_specificity_f1(
            tp=int(counts["tp"]),
            fp=int(counts["fp"]),
            tn=0,
            fn=int(counts["fn"]),
        )
        suffix = _threshold_suffix(threshold)
        metrics[f"iou_detection_precision@{suffix}"] = rates["precision"]
        metrics[f"iou_detection_recall@{suffix}"] = rates["recall"]
        metrics[f"iou_detection_f1@{suffix}"] = rates["f1"]
        metrics[f"iou_detection_tp@{suffix}"] = float(counts["tp"])
        metrics[f"iou_detection_fp@{suffix}"] = float(counts["fp"])
        metrics[f"iou_detection_fn@{suffix}"] = float(counts["fn"])
        metrics[f"final_detection_precision@{suffix}"] = rates["precision"]
        metrics[f"final_detection_recall@{suffix}"] = rates["recall"]

    primary_key = min(counts_by_threshold, key=lambda key: abs(float(key) - float(primary_iou_threshold)))
    primary_counts = counts_by_threshold[primary_key]
    primary_rates = precision_recall_specificity_f1(
        tp=int(primary_counts["tp"]),
        fp=int(primary_counts["fp"]),
        tn=0,
        fn=int(primary_counts["fn"]),
    )
    metrics["iou_detection_tp"] = float(primary_counts["tp"])
    metrics["iou_detection_fp"] = float(primary_counts["fp"])
    metrics["iou_detection_fn"] = float(primary_counts["fn"])
    metrics["iou_detection_precision"] = primary_rates["precision"]
    metrics["iou_detection_recall"] = primary_rates["recall"]
    metrics["iou_detection_f1"] = primary_rates["f1"]
    metrics["mean_best_iou"] = float(np.mean(best_ious)) if best_ious else 0.0
    metrics["max_best_iou"] = float(np.max(best_ious)) if best_ious else 0.0
    return metrics


def summarize_oracle_positive_bbox_metrics(ious: Sequence[float]) -> dict[str, float]:
    return {
        "oracle_pos_bbox_mean_iou": float(np.mean(ious)) if ious else 0.0,
        "oracle_pos_bbox_max_iou": float(np.max(ious)) if ious else 0.0,
        "oracle_pos_bbox_recall@0.1": float(np.mean([iou >= 0.1 for iou in ious])) if ious else 0.0,
        "oracle_pos_bbox_recall@0.3": float(np.mean([iou >= 0.3 for iou in ious])) if ious else 0.0,
        "oracle_pos_bbox_recall@0.45": float(np.mean([iou >= 0.45 for iou in ious])) if ious else 0.0,
    }


def _token_slices(shape: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.arange(int(shape["slices"]), device=device).repeat_interleave(int(shape["patches_per_slice"]))


def _token_plane_and_slice_ids(shape_metadata: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sagittal_shape = shape_metadata["sagittal"]
    coronal_shape = shape_metadata["coronal"]
    sag_slices = _token_slices(sagittal_shape, device=device)
    cor_slices = _token_slices(coronal_shape, device=device)
    plane_ids = torch.cat(
        [
            torch.zeros_like(sag_slices, dtype=torch.long),
            torch.ones_like(cor_slices, dtype=torch.long),
        ],
        dim=0,
    )
    slice_indices = torch.cat([sag_slices, cor_slices], dim=0)
    return plane_ids, slice_indices


def _nms_boxes(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), device=boxes.device, dtype=torch.long)
    order = scores.argsort(descending=True)
    keep: list[torch.Tensor] = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious = box_iou(boxes[current].view(1, 4), boxes[remaining]).flatten()
        order = remaining[ious <= float(iou_threshold)]
    return torch.stack(keep).long() if keep else torch.zeros((0,), device=boxes.device, dtype=torch.long)


def _postprocess_indices_for_scores(
    scores: torch.Tensor,
    boxes: torch.Tensor,
    plane_ids: torch.Tensor,
    slice_indices: torch.Tensor,
    side_ids: torch.Tensor,
    score_threshold: float,
    topk_per_image: int | None,
    nms_threshold: float | None,
) -> dict[str, torch.Tensor]:
    threshold_indices = torch.nonzero(scores >= float(score_threshold), as_tuple=False).flatten()
    if threshold_indices.numel() > 0:
        threshold_indices = threshold_indices[scores[threshold_indices].argsort(descending=True)]
    topk_indices = threshold_indices
    if topk_per_image is not None:
        topk_indices = topk_indices[: max(0, int(topk_per_image))]
    if topk_indices.numel() > 0 and nms_threshold is not None:
        keep_indices: list[torch.Tensor] = []
        group_keys = torch.stack([plane_ids[topk_indices], slice_indices[topk_indices], side_ids[topk_indices]], dim=1)
        for key in torch.unique(group_keys, dim=0):
            group_mask = (group_keys == key).all(dim=1)
            group_indices = topk_indices[group_mask]
            keep_local = _nms_boxes(boxes[group_indices], scores[group_indices], float(nms_threshold))
            keep_indices.append(group_indices[keep_local])
        final_indices = torch.cat(keep_indices, dim=0) if keep_indices else topk_indices.new_empty((0,))
        if final_indices.numel() > 0:
            final_indices = final_indices[scores[final_indices].argsort(descending=True)]
    else:
        final_indices = topk_indices
    return {
        "threshold_indices": threshold_indices,
        "topk_indices": topk_indices,
        "final_indices": final_indices,
    }


def _stage_gt_hits(
    case_labels: dict[str, Any],
    shape_metadata: dict[str, Any],
    stage_indices: torch.Tensor,
    boxes: torch.Tensor,
    plane_ids: torch.Tensor,
    slice_indices: torch.Tensor,
    side_ids: torch.Tensor,
    iou_threshold: float,
) -> tuple[int, int]:
    case_gt = _case_gt_boxes(case_labels, shape_metadata, boxes.device)
    total = int(sum(int(gt_boxes.shape[0]) for gt_boxes in case_gt.values()))
    if stage_indices.numel() == 0 or total == 0:
        return total, 0
    hits = 0
    for key, gt_boxes in case_gt.items():
        mask = (
            (plane_ids[stage_indices] == int(key[0]))
            & (slice_indices[stage_indices] == int(key[1]))
            & (side_ids[stage_indices] == int(key[2]))
        )
        pred_indices = stage_indices[mask]
        if pred_indices.numel() == 0:
            continue
        ious = box_iou(boxes[pred_indices], gt_boxes)
        hits += int((ious.max(dim=0).values >= float(iou_threshold)).sum().item())
    return total, hits


def _final_predictions_by_key(case_predictions: dict[str, Any]) -> dict[tuple[int, int, int], list[tuple[int, float]]]:
    boxes_by_key: dict[tuple[int, int, int], list[tuple[int, float]]] = {}
    for row in range(int(case_predictions["boxes"].shape[0])):
        key = (
            int(case_predictions["plane_ids"][row].item()),
            int(case_predictions["slice_indices"][row].item()),
            int(case_predictions["side_ids"][row].item()),
        )
        boxes_by_key.setdefault(key, []).append((row, float(case_predictions["scores"][row].item())))
    return boxes_by_key


def _threshold_suffix(threshold: float) -> str:
    return f"{float(threshold):g}"


def _plane_name(plane_id: int) -> str:
    return "sagittal" if int(plane_id) == 0 else "coronal"


def _side_name(side_id: int) -> str:
    return "medial" if int(side_id) == 0 else "lateral"


def _token_center_for_index(token_idx: int, shape_metadata: dict[str, Any], device: torch.device) -> list[float]:
    _, slice_indices = _token_plane_and_slice_ids(shape_metadata, device)
    sagittal_shape = shape_metadata["sagittal"]
    sag_tokens = int(sagittal_shape["slices"] * sagittal_shape["patches_per_slice"])
    if int(token_idx) < sag_tokens:
        shape = sagittal_shape
        local_idx = int(token_idx)
    else:
        shape = shape_metadata["coronal"]
        local_idx = int(token_idx) - sag_tokens
    per_slice = int(shape["patches_per_slice"])
    index_in_slice = local_idx - int(slice_indices[token_idx].item()) * per_slice
    row = index_in_slice // int(shape["grid_width"])
    col = index_in_slice % int(shape["grid_width"])
    return [
        (float(col) + 0.5) / float(shape["grid_width"]),
        (float(row) + 0.5) / float(shape["grid_height"]),
    ]


def _case_pred_boxes(
    pred_positive: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_side_ids: torch.Tensor,
    shape_metadata: dict[str, Any],
) -> dict[tuple[int, int, int], list[tuple[int, float]]]:
    device = pred_positive.device
    sagittal_shape = shape_metadata["sagittal"]
    coronal_shape = shape_metadata["coronal"]
    sag_tokens = int(sagittal_shape["slices"] * sagittal_shape["patches_per_slice"])
    plane_slices = {
        0: _token_slices(sagittal_shape, device=device),
        1: _token_slices(coronal_shape, device=device),
    }
    plane_offsets = {0: 0, 1: sag_tokens}
    boxes_by_key: dict[tuple[int, int, int], list[tuple[int, float]]] = {}
    for plane_id in (0, 1):
        offset = plane_offsets[plane_id]
        local_positive = torch.nonzero(
            pred_positive[offset : offset + plane_slices[plane_id].numel()],
            as_tuple=False,
        ).flatten()
        for local_idx_tensor in local_positive:
            local_idx = int(local_idx_tensor)
            global_idx = offset + local_idx
            slice_index = int(plane_slices[plane_id][local_idx].item())
            side_id = int(pred_side_ids[global_idx].item())
            key = (plane_id, slice_index, side_id)
            boxes_by_key.setdefault(key, []).append((global_idx, float(pred_scores[global_idx].item())))
    return boxes_by_key


def _case_gt_boxes(case_labels: dict[str, Any], shape_metadata: dict[str, Any], device: torch.device) -> dict[tuple[int, int, int], torch.Tensor]:
    boxes_by_key: dict[tuple[int, int, int], list[torch.Tensor]] = {}
    for label in case_labels.get("module1_labels", []):
        plane = label.get("plane")
        if plane not in {"sagittal", "coronal"}:
            continue
        shape = shape_metadata[plane]
        plane_id = 0 if plane == "sagittal" else 1
        slice_index = int(label["slice_index"])
        side_id = 0 if label.get("meniscus_side") == "medial" else 1
        x1, y1, x2, y2 = [float(v) for v in label["bbox"]]
        box = torch.tensor(
            [x1 / float(shape["width"]), y1 / float(shape["height"]), x2 / float(shape["width"]), y2 / float(shape["height"])],
            device=device,
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        x1y1 = torch.minimum(box[:2], box[2:])
        x2y2 = torch.maximum(box[:2], box[2:])
        boxes_by_key.setdefault((plane_id, slice_index, side_id), []).append(torch.cat([x1y1, x2y2]))
    return {
        key: torch.stack(items, dim=0) if items else torch.zeros((0, 4), device=device)
        for key, items in boxes_by_key.items()
    }
