"""Run a reproducible, read-only audit of Module 1 detection candidates.

This entrypoint deliberately lives outside the training loop. It never updates
weights and writes every artifact under one diagnostics directory. The audit
separates four questions that the normal validation metrics currently mix:

1. Can any token on the correct plane/slice localize a GT (geometry recall)?
2. Does the candidate also predict the correct meniscus side (strict recall)?
3. Does the learned score retain it through threshold/top-k/NMS?
4. Would oracle or random ranking retain a different candidate set?

The script also exports per-GT assignment counts, high-IoU/low-score and
low-IoU/high-score candidates, score/IoU correlations, branch distributions,
plots, overlays, one-factor post-processing sweeps, and optional per-loss
gradient norms. It does not change the model, target builder, or default config.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.meniscus_detection_dataset import MeniscusDetectionDataset
from engine.train_module1 import build_loss
from engine.validate_module1 import build_module1_model, module1_collate, move_batch_to_device
from losses.detection_loss import (
    DetectionTargetConfig,
    box_iou,
    build_module1_targets,
    generalized_box_iou_loss,
)
from scripts.module1_debug.runtime_utils import set_diagnostic_seed
from utils.checkpoint import load_checkpoint
from utils.metrics import (
    _postprocess_indices_for_scores,
    _token_plane_and_slice_ids,
    compute_module1_final_scores,
    filter_module1_predictions,
    match_final_predictions,
    precision_recall_specificity_f1,
    safe_pearson_corr,
    safe_spearman_corr,
)


PLANE_TO_ID = {"sagittal": 0, "coronal": 1}
ID_TO_PLANE = {value: key for key, value in PLANE_TO_ID.items()}
SIDE_TO_ID = {"medial": 0, "lateral": 1}
ID_TO_SIDE = {value: key for key, value in SIDE_TO_ID.items()}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _inherit(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def resolve_diagnostic_options(
    base_config: dict[str, Any],
    diagnostic_config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve nullable diagnostic values against ``base_config.validate``."""

    options = copy.deepcopy(diagnostic_config.get("diagnostics", diagnostic_config))
    overrides = overrides or {}
    for key, value in overrides.items():
        if value is not None:
            options[key] = value
    validate = base_config.get("validate", {})
    options["seed"] = int(_inherit(options.get("seed"), base_config.get("seed", 42)))
    options["score_threshold"] = float(_inherit(options.get("score_threshold"), validate.get("score_threshold", 0.5)))
    options["topk_per_image"] = _inherit(options.get("topk_per_image"), validate.get("topk_per_image"))
    if options["topk_per_image"] is not None:
        options["topk_per_image"] = int(options["topk_per_image"])
    options["nms_threshold"] = _inherit(options.get("nms_threshold"), validate.get("nms_threshold"))
    if options["nms_threshold"] is not None:
        options["nms_threshold"] = float(options["nms_threshold"])
    options["final_score_mode"] = str(_inherit(options.get("final_score_mode"), validate.get("final_score_mode", "objectness")))
    options["quality_score_alpha"] = float(_inherit(options.get("quality_score_alpha"), validate.get("quality_score_alpha", 1.0)))
    options["quality_score_beta"] = float(_inherit(options.get("quality_score_beta"), validate.get("quality_score_beta", 1.0)))
    options["max_cases"] = None if options.get("max_cases") is None else int(options["max_cases"])
    options["primary_iou_threshold"] = float(options.get("primary_iou_threshold", 0.3))
    options["iou_thresholds"] = [float(value) for value in options.get("iou_thresholds", [0.1, 0.3, 0.45])]
    return options


def select_case_indices(total: int, max_cases: int | None, seed: int) -> list[int]:
    """Choose a deterministic random subset and return it in dataset order."""

    if max_cases is None or max_cases <= 0 or max_cases >= total:
        return list(range(total))
    rng = random.Random(int(seed))
    return sorted(rng.sample(range(total), int(max_cases)))


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "min": None, "p05": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def normalize_gt_items(case_labels: dict[str, Any], shape_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for gt_index, label in enumerate(case_labels.get("module1_labels", [])):
        plane = str(label.get("plane", ""))
        side = str(label.get("meniscus_side", ""))
        if plane not in PLANE_TO_ID or side not in SIDE_TO_ID:
            continue
        shape = shape_metadata[plane]
        x1, y1, x2, y2 = [float(value) for value in label["bbox"]]
        raw = torch.tensor(
            [x1 / float(shape["width"]), y1 / float(shape["height"]), x2 / float(shape["width"]), y2 / float(shape["height"])],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        box = torch.cat([torch.minimum(raw[:2], raw[2:]), torch.maximum(raw[:2], raw[2:])])
        items.append(
            {
                "gt_index": int(gt_index),
                "label": label,
                "plane": plane,
                "plane_id": PLANE_TO_ID[plane],
                "processed_slice": int(label["slice_index"]),
                "source_slice": int(label.get("source_slice_index", label["slice_index"])),
                "side": side,
                "side_id": SIDE_TO_ID[side],
                "box": box,
                "box_area": float(((box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)).item()),
            }
        )
    return items


def token_iou_targets(
    gt_items: Sequence[dict[str, Any]],
    boxes: torch.Tensor,
    plane_ids: torch.Tensor,
    slice_indices: torch.Tensor,
    predicted_side_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return geometry-only and predicted-side-strict max IoU per token."""

    geometry = torch.zeros(boxes.shape[0], device=boxes.device, dtype=torch.float32)
    strict = torch.zeros_like(geometry)
    relevant = torch.zeros(boxes.shape[0], device=boxes.device, dtype=torch.bool)
    for item in gt_items:
        geometry_mask = (plane_ids == int(item["plane_id"])) & (slice_indices == int(item["processed_slice"]))
        indices = torch.nonzero(geometry_mask, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        relevant[indices] = True
        ious = box_iou(boxes[indices].float(), item["box"].to(boxes.device).view(1, 4).float()).flatten()
        geometry[indices] = torch.maximum(geometry[indices], ious)
        correct_side = predicted_side_ids[indices] == int(item["side_id"])
        strict_indices = indices[correct_side]
        if strict_indices.numel() > 0:
            strict[strict_indices] = torch.maximum(strict[strict_indices], ious[correct_side])
    return {"geometry": geometry.clamp(0, 1), "strict": strict.clamp(0, 1), "relevant": relevant}


def best_gt_iou(
    item: dict[str, Any],
    indices: torch.Tensor,
    boxes: torch.Tensor,
    plane_ids: torch.Tensor,
    slice_indices: torch.Tensor,
    side_ids: torch.Tensor,
    strict_side: bool,
) -> float:
    if indices.numel() == 0:
        return 0.0
    mask = (plane_ids[indices] == int(item["plane_id"])) & (slice_indices[indices] == int(item["processed_slice"]))
    if strict_side:
        mask &= side_ids[indices] == int(item["side_id"])
    selected = indices[mask]
    if selected.numel() == 0:
        return 0.0
    return float(box_iou(boxes[selected].float(), item["box"].to(boxes.device).view(1, 4).float()).max().item())


def classify_failure(
    raw_geometry_iou: float,
    raw_strict_iou: float,
    threshold_iou: float,
    topk_iou: float,
    nms_iou: float,
    iou_threshold: float,
) -> str:
    """Assign one mutually-exclusive failure class in causal stage order."""

    if raw_geometry_iou < iou_threshold:
        return "localization_error"
    if raw_strict_iou < iou_threshold:
        return "side_classification_error"
    if threshold_iou < iou_threshold:
        return "score_threshold_error"
    if topk_iou < iou_threshold:
        return "ranking_topk_error"
    if nms_iou < iou_threshold:
        return "nms_error"
    return "retained"


def per_gt_assignment_masks(
    case_labels: dict[str, Any],
    gt_items: Sequence[dict[str, Any]],
    shape_metadata: dict[str, Any],
    device: torch.device,
    positive_radius_tokens: int,
) -> tuple[dict[int, torch.Tensor], int]:
    masks: dict[int, torch.Tensor] = {}
    occupancy: torch.Tensor | None = None
    target_config = DetectionTargetConfig(positive_radius_tokens=int(positive_radius_tokens))
    for item in gt_items:
        single_case = {"case_id": case_labels.get("case_id"), "module1_labels": [item["label"]]}
        targets = build_module1_targets([single_case], shape_metadata, device=device, target_config=target_config)
        mask = targets["positive_mask"][0]
        masks[int(item["gt_index"])] = mask
        occupancy = mask.long() if occupancy is None else occupancy + mask.long()
    conflicts = int((occupancy > 1).sum().item()) if occupancy is not None else 0
    return masks, conflicts


def _score_rank(scores: torch.Tensor, token_index: int) -> int:
    order = scores.argsort(descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel(), device=order.device, dtype=order.dtype) + 1
    return int(ranks[int(token_index)].item())


def _safe_logit(scores: torch.Tensor) -> torch.Tensor:
    return torch.logit(scores.detach().float().clamp(1e-6, 1.0 - 1e-6))


def _candidate_score_modes(predictions: dict[str, Any], options: dict[str, Any]) -> dict[str, torch.Tensor]:
    objectness = torch.sigmoid(predictions["objectness_logits"]).detach()
    side_confidence = torch.softmax(predictions["side_logits"], dim=-1).detach().max(dim=-1).values
    scores = {
        "objectness": objectness,
        "pipeline_objectness_side": objectness * side_confidence,
        "learned_final": compute_module1_final_scores(
            predictions,
            final_score_mode=options["final_score_mode"],
            quality_score_alpha=options["quality_score_alpha"],
            quality_score_beta=options["quality_score_beta"],
        ),
    }
    if "bbox_quality_logits" in predictions:
        quality = torch.sigmoid(predictions["bbox_quality_logits"]).detach()
        scores["quality"] = quality
        scores["objectness_quality"] = objectness * quality
    return scores


def _detach_predictions(predictions: dict[str, Any]) -> dict[str, Any]:
    keys = ("objectness_logits", "bbox_quality_logits", "bbox_pred", "side_logits", "abnormality_logits", "plane_ids")
    output = {key: predictions[key].detach().cpu() for key in keys if key in predictions}
    output["shape_metadata"] = predictions["shape_metadata"]
    return output


def _resolve_device(base_config: dict[str, Any], requested: str) -> torch.device:
    if requested not in {"", "auto", None}:
        return torch.device(str(requested))
    configured = str(base_config.get("device", "cuda"))
    return torch.device(configured if configured == "cuda" and torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def collect_case_diagnostics(
    model: torch.nn.Module,
    dataset: MeniscusDetectionDataset,
    case_indices: Sequence[int],
    device: torch.device,
    options: dict[str, Any],
) -> dict[str, Any]:
    primary_iou = float(options["primary_iou_threshold"])
    radius = int(options.get("positive_radius_tokens", 0))
    amp = bool(options.get("amp", True)) and device.type == "cuda"
    device_type = "cuda" if device.type == "cuda" else "cpu"
    gt_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    score_arrays: dict[str, list[torch.Tensor]] = defaultdict(list)
    strict_iou_arrays: list[torch.Tensor] = []
    geometry_iou_arrays: list[torch.Tensor] = []
    relevant_arrays: list[torch.Tensor] = []
    branch_arrays: dict[str, list[torch.Tensor]] = defaultdict(list)

    for ordinal, dataset_index in enumerate(case_indices):
        sample = dataset[int(dataset_index)]
        batch = move_batch_to_device(module1_collate([sample]), device)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp):
            predictions_device = model(
                batch["sagittal_volume"],
                batch["coronal_volume"],
                batch["sagittal_coords"],
                batch["coronal_coords"],
            )
        shape_metadata = predictions_device["shape_metadata"]
        plane_ids, slice_indices = _token_plane_and_slice_ids(shape_metadata, device)
        boxes = predictions_device["bbox_pred"][0].detach().float()
        side_probs = torch.softmax(predictions_device["side_logits"][0], dim=-1).detach().float()
        side_ids = side_probs.argmax(dim=-1)
        gt_items = normalize_gt_items(sample["labels"], shape_metadata)
        iou_targets = token_iou_targets(gt_items, boxes, plane_ids, slice_indices, side_ids)
        score_modes_batch = _candidate_score_modes(predictions_device, options)
        score_modes = {name: values[0].detach().float() for name, values in score_modes_batch.items()}
        learned_scores = score_modes["learned_final"]
        trace = _postprocess_indices_for_scores(
            scores=learned_scores,
            boxes=boxes,
            plane_ids=plane_ids,
            slice_indices=slice_indices,
            side_ids=side_ids,
            score_threshold=float(options["score_threshold"]),
            topk_per_image=options["topk_per_image"],
            nms_threshold=options["nms_threshold"],
        )
        raw_indices = torch.arange(boxes.shape[0], device=device, dtype=torch.long)
        assignment_masks, conflict_tokens = per_gt_assignment_masks(
            sample["labels"],
            gt_items,
            shape_metadata,
            device,
            positive_radius_tokens=radius,
        )

        for item in gt_items:
            raw_geometry = best_gt_iou(item, raw_indices, boxes, plane_ids, slice_indices, side_ids, strict_side=False)
            raw_strict = best_gt_iou(item, raw_indices, boxes, plane_ids, slice_indices, side_ids, strict_side=True)
            after_threshold = best_gt_iou(item, trace["threshold_indices"], boxes, plane_ids, slice_indices, side_ids, strict_side=True)
            after_topk = best_gt_iou(item, trace["topk_indices"], boxes, plane_ids, slice_indices, side_ids, strict_side=True)
            after_nms = best_gt_iou(item, trace["final_indices"], boxes, plane_ids, slice_indices, side_ids, strict_side=True)
            failure = classify_failure(raw_geometry, raw_strict, after_threshold, after_topk, after_nms, primary_iou)
            failure_counts[failure] += 1

            geometry_mask = (plane_ids == int(item["plane_id"])) & (slice_indices == int(item["processed_slice"]))
            geometry_indices = torch.nonzero(geometry_mask, as_tuple=False).flatten()
            strict_indices = geometry_indices[side_ids[geometry_indices] == int(item["side_id"])]
            if geometry_indices.numel() > 0:
                geometry_ious = box_iou(boxes[geometry_indices], item["box"].to(device).view(1, 4)).flatten()
                best_geometry_idx = int(geometry_indices[int(geometry_ious.argmax().item())].item())
            else:
                best_geometry_idx = -1
            if strict_indices.numel() > 0:
                strict_ious = box_iou(boxes[strict_indices], item["box"].to(device).view(1, 4)).flatten()
                best_strict_idx = int(strict_indices[int(strict_ious.argmax().item())].item())
                best_strict_score = float(learned_scores[best_strict_idx].item())
                best_strict_rank: int | str = _score_rank(learned_scores, best_strict_idx)
            else:
                best_strict_idx = -1
                best_strict_score = 0.0
                best_strict_rank = ""

            metadata = sample[f"{item['plane']}_metadata"]
            sampled_sources = metadata.get("sampled_source_indices", [])
            processed_slice = int(item["processed_slice"])
            sampled_source = int(sampled_sources[processed_slice]) if 0 <= processed_slice < len(sampled_sources) else -1
            source_slice = int(item["source_slice"])
            assignment_count = int(assignment_masks[int(item["gt_index"])].sum().item())
            coverage_counts = {}
            if strict_indices.numel() > 0:
                strict_gt_ious = box_iou(boxes[strict_indices], item["box"].to(device).view(1, 4)).flatten()
            else:
                strict_gt_ious = boxes.new_empty((0,))
            for threshold in options["iou_thresholds"]:
                coverage_counts[f"raw_strict_candidates_iou_ge_{float(threshold):g}"] = int((strict_gt_ious >= float(threshold)).sum().item())

            gt_rows.append(
                {
                    "case_id": str(sample["case_id"]),
                    "dataset_index": int(dataset_index),
                    "gt_index": int(item["gt_index"]),
                    "plane": item["plane"],
                    "processed_slice": processed_slice,
                    "source_slice": source_slice,
                    "sampled_source_slice": sampled_source,
                    "slice_delta": sampled_source - source_slice if sampled_source >= 0 else "",
                    "exact_slice_mapping": sampled_source == source_slice,
                    "side": item["side"],
                    "box_area_normalized": float(item["box_area"]),
                    "positive_token_count": assignment_count,
                    "case_assignment_conflict_tokens": conflict_tokens,
                    "best_raw_geometry_iou": raw_geometry,
                    "best_raw_strict_iou": raw_strict,
                    "best_geometry_token": best_geometry_idx,
                    "best_strict_token": best_strict_idx,
                    "best_strict_learned_score": best_strict_score,
                    "best_strict_learned_rank": best_strict_rank,
                    "best_after_threshold_iou": after_threshold,
                    "best_after_topk_iou": after_topk,
                    "best_after_nms_iou": after_nms,
                    "failure_taxonomy": failure,
                    **coverage_counts,
                }
            )

        high_iou = float(options.get("high_iou_threshold", 0.3))
        low_iou = float(options.get("low_iou_threshold", 0.1))
        low_score = float(options.get("low_score_threshold", 0.05))
        high_score = float(options.get("high_score_threshold", 0.5))
        # Candidate lists use geometry IoU so a good box with the wrong side is
        # not silently reclassified as a localization failure. ``strict_iou``
        # remains in the export to expose that classification error explicitly.
        high_low_mask = (iou_targets["geometry"] >= high_iou) & (learned_scores <= low_score)
        low_high_mask = (iou_targets["geometry"] <= low_iou) & (learned_scores >= high_score)
        export_indices = torch.nonzero(high_low_mask | low_high_mask, as_tuple=False).flatten()
        objectness = score_modes["objectness"]
        quality = score_modes.get("quality", torch.full_like(objectness, float("nan")))
        for token_tensor in export_indices:
            token_idx = int(token_tensor.item())
            category = "high_iou_low_score" if bool(high_low_mask[token_idx].item()) else "low_iou_high_score"
            candidate_rows.append(
                {
                    "category": category,
                    "case_id": str(sample["case_id"]),
                    "token_index": token_idx,
                    "plane": ID_TO_PLANE[int(plane_ids[token_idx].item())],
                    "processed_slice": int(slice_indices[token_idx].item()),
                    "predicted_side": ID_TO_SIDE[int(side_ids[token_idx].item())],
                    "geometry_iou": float(iou_targets["geometry"][token_idx].item()),
                    "strict_iou": float(iou_targets["strict"][token_idx].item()),
                    "objectness": float(objectness[token_idx].item()),
                    "quality": float(quality[token_idx].item()),
                    "learned_final_score": float(learned_scores[token_idx].item()),
                    "pipeline_objectness_side_score": float(score_modes["pipeline_objectness_side"][token_idx].item()),
                    "box_xyxy": json.dumps([float(value) for value in boxes[token_idx].cpu().tolist()]),
                }
            )

        predictions_cpu = _detach_predictions(predictions_device)
        cpu_scores = {name: values.detach().cpu() for name, values in score_modes.items()}
        generator = torch.Generator(device="cpu").manual_seed(int(options["seed"]) + ordinal * 1009)
        for trial in range(int(options.get("random_score_trials", 5))):
            cpu_scores[f"random_{trial}"] = torch.rand(boxes.shape[0], generator=generator)
        cpu_scores["oracle_strict_iou"] = iou_targets["strict"].detach().cpu()
        cpu_scores["oracle_geometry_iou"] = iou_targets["geometry"].detach().cpu()
        case_cache = {
            "case_id": str(sample["case_id"]),
            "dataset_index": int(dataset_index),
            "labels": sample["labels"],
            "gt_items": [{**item, "box": item["box"].detach().cpu()} for item in gt_items],
            "predictions": predictions_cpu,
            "scores": cpu_scores,
            "strict_iou": iou_targets["strict"].detach().cpu(),
            "geometry_iou": iou_targets["geometry"].detach().cpu(),
            "relevant": iou_targets["relevant"].detach().cpu(),
            "sagittal_volume": sample["sagittal_volume"].detach().cpu() if options.get("save_visualizations", True) else None,
            "coronal_volume": sample["coronal_volume"].detach().cpu() if options.get("save_visualizations", True) else None,
        }
        cases.append(case_cache)

        for name, values in cpu_scores.items():
            if not name.startswith("random_") and not name.startswith("oracle_"):
                score_arrays[name].append(values.flatten())
        strict_iou_arrays.append(case_cache["strict_iou"].flatten())
        geometry_iou_arrays.append(case_cache["geometry_iou"].flatten())
        relevant_arrays.append(case_cache["relevant"].flatten())
        branch_arrays["objectness_logits"].append(predictions_cpu["objectness_logits"].flatten())
        branch_arrays["objectness_prob"].append(cpu_scores["objectness"].flatten())
        branch_arrays["side_max_prob"].append(side_probs.max(dim=-1).values.detach().cpu())
        abnormal_probs = torch.softmax(predictions_device["abnormality_logits"][0], dim=-1).detach().float()
        branch_arrays["abnormality_max_prob"].append(abnormal_probs.max(dim=-1).values.cpu())
        widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(0).cpu()
        heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(0).cpu()
        branch_arrays["bbox_width"].append(widths)
        branch_arrays["bbox_height"].append(heights)
        branch_arrays["bbox_area"].append(widths * heights)
        if "bbox_quality_logits" in predictions_cpu:
            branch_arrays["quality_logits"].append(predictions_cpu["bbox_quality_logits"].flatten())
            branch_arrays["quality_prob"].append(cpu_scores["quality"].flatten())

    strict_all = torch.cat(strict_iou_arrays) if strict_iou_arrays else torch.empty(0)
    geometry_all = torch.cat(geometry_iou_arrays) if geometry_iou_arrays else torch.empty(0)
    relevant_all = torch.cat(relevant_arrays).bool() if relevant_arrays else torch.empty(0, dtype=torch.bool)
    correlations: dict[str, dict[str, float]] = {}
    for name, chunks in score_arrays.items():
        values = torch.cat(chunks) if chunks else torch.empty(0)
        correlations[name] = correlation_report(values, strict_all, geometry_all, relevant_all)
    branch_distributions = {name: distribution(torch.cat(chunks).tolist()) for name, chunks in branch_arrays.items() if chunks}

    max_rows = int(options.get("max_candidate_rows", 200))
    high_rows = sorted(
        (row for row in candidate_rows if row["category"] == "high_iou_low_score"),
        key=lambda row: (float(row["learned_final_score"]), -float(row["strict_iou"])),
    )[:max_rows]
    low_rows = sorted(
        (row for row in candidate_rows if row["category"] == "low_iou_high_score"),
        key=lambda row: (-float(row["learned_final_score"]), float(row["strict_iou"])),
    )[:max_rows]
    return {
        "cases": cases,
        "gt_rows": gt_rows,
        "high_iou_low_score_rows": high_rows,
        "low_iou_high_score_rows": low_rows,
        "failure_counts": dict(failure_counts),
        "correlations": correlations,
        "branch_distributions": branch_distributions,
        "plot_data": {
            "strict_iou": strict_all,
            "geometry_iou": geometry_all,
            "relevant": relevant_all,
            "scores": {name: torch.cat(chunks) for name, chunks in score_arrays.items() if chunks},
            "branches": {name: torch.cat(chunks) for name, chunks in branch_arrays.items() if chunks},
        },
    }


def correlation_report(
    scores: torch.Tensor,
    strict_iou: torch.Tensor,
    geometry_iou: torch.Tensor,
    relevant: torch.Tensor,
) -> dict[str, float]:
    """Report pooled all-token and GT-key-only correlations separately."""

    scores_list = scores.float().tolist()
    strict_list = strict_iou.float().tolist()
    output = {
        "all_token_strict_spearman": safe_spearman_corr(scores_list, strict_list),
        "all_token_strict_pearson": safe_pearson_corr(scores_list, strict_list),
        "all_token_geometry_spearman": safe_spearman_corr(scores_list, geometry_iou.float().tolist()),
        "all_token_geometry_pearson": safe_pearson_corr(scores_list, geometry_iou.float().tolist()),
    }
    if relevant.numel() > 0 and relevant.any():
        relevant_scores = scores[relevant].float().tolist()
        output.update(
            {
                "gt_plane_slice_geometry_spearman": safe_spearman_corr(relevant_scores, geometry_iou[relevant].float().tolist()),
                "gt_plane_slice_geometry_pearson": safe_pearson_corr(relevant_scores, geometry_iou[relevant].float().tolist()),
                "gt_plane_slice_strict_spearman": safe_spearman_corr(relevant_scores, strict_iou[relevant].float().tolist()),
                "gt_plane_slice_strict_pearson": safe_pearson_corr(relevant_scores, strict_iou[relevant].float().tolist()),
            }
        )
    return output


def _score_predictions(predictions: dict[str, Any], scores: torch.Tensor) -> dict[str, Any]:
    output = dict(predictions)
    output["objectness_logits"] = _safe_logit(scores).view_as(predictions["objectness_logits"])
    return output


def evaluate_score_setting(
    cases: Sequence[dict[str, Any]],
    score_name: str,
    score_threshold: float,
    topk_per_image: int | None,
    nms_threshold: float | None,
    iou_threshold: float,
) -> dict[str, float]:
    """Evaluate one cached score without re-running the network."""

    tp = fp = fn = 0
    stage_hits = Counter({"raw": 0, "threshold": 0, "topk": 0, "nms": 0})
    total_gt = 0
    final_candidates = 0
    for case in cases:
        predictions = case["predictions"]
        scores = case["scores"][score_name].float()
        scored_predictions = _score_predictions(predictions, scores)
        final = filter_module1_predictions(
            scored_predictions,
            score_threshold=float(score_threshold),
            topk_per_image=topk_per_image,
            nms_threshold=nms_threshold,
            final_score_mode="objectness",
        )
        counts = match_final_predictions(final, [case["labels"]], predictions["shape_metadata"], iou_threshold=float(iou_threshold))
        tp += int(counts["tp"])
        fp += int(counts["fp"])
        fn += int(counts["fn"])
        final_candidates += int(final[0]["counts"]["num_final_predictions_for_matching"])
        plane_ids, slice_indices = _token_plane_and_slice_ids(predictions["shape_metadata"], torch.device("cpu"))
        boxes = predictions["bbox_pred"][0].float()
        side_ids = predictions["side_logits"][0].argmax(dim=-1)
        trace = _postprocess_indices_for_scores(
            scores=scores,
            boxes=boxes,
            plane_ids=plane_ids,
            slice_indices=slice_indices,
            side_ids=side_ids,
            score_threshold=float(score_threshold),
            topk_per_image=topk_per_image,
            nms_threshold=nms_threshold,
        )
        raw_indices = torch.arange(scores.numel(), dtype=torch.long)
        for item in case["gt_items"]:
            total_gt += 1
            for stage_name, indices in (
                ("raw", raw_indices),
                ("threshold", trace["threshold_indices"]),
                ("topk", trace["topk_indices"]),
                ("nms", trace["final_indices"]),
            ):
                hit = best_gt_iou(item, indices, boxes, plane_ids, slice_indices, side_ids, strict_side=True) >= float(iou_threshold)
                stage_hits[stage_name] += int(hit)
    rates = precision_recall_specificity_f1(tp=tp, fp=fp, tn=0, fn=fn)
    return {
        "score_mode": score_name,
        "score_threshold": float(score_threshold),
        "topk_per_image": "all" if topk_per_image is None else int(topk_per_image),
        "nms_threshold": "none" if nms_threshold is None else float(nms_threshold),
        "num_gt": total_gt,
        "num_final_candidates": final_candidates,
        "raw_coverage_recall": float(stage_hits["raw"]) / max(total_gt, 1),
        "after_threshold_coverage_recall": float(stage_hits["threshold"]) / max(total_gt, 1),
        "after_topk_coverage_recall": float(stage_hits["topk"]) / max(total_gt, 1),
        "after_nms_coverage_recall": float(stage_hits["nms"]) / max(total_gt, 1),
        "final_precision": rates["precision"],
        "final_recall": rates["recall"],
        "final_f1": rates["f1"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def run_score_controls_and_sweeps(cases: Sequence[dict[str, Any]], options: dict[str, Any]) -> list[dict[str, Any]]:
    if not cases:
        return []
    rows: list[dict[str, Any]] = []
    primary_iou = float(options["primary_iou_threshold"])
    comparison_threshold = float(options.get("oracle_comparison_threshold", 0.0))
    common_modes = ["learned_final", "objectness", "pipeline_objectness_side"]
    if "quality" in cases[0]["scores"]:
        common_modes.extend(["quality", "objectness_quality"])
    common_modes.extend(["oracle_strict_iou", "oracle_geometry_iou"])
    common_modes.extend([f"random_{trial}" for trial in range(int(options.get("random_score_trials", 5)))])
    for mode in common_modes:
        row = evaluate_score_setting(
            cases,
            score_name=mode,
            score_threshold=comparison_threshold,
            topk_per_image=options["topk_per_image"],
            nms_threshold=options["nms_threshold"],
            iou_threshold=primary_iou,
        )
        row["sweep"] = "score_control"
        rows.append(row)
    if not bool(options.get("run_sweeps", True)):
        return rows
    for threshold in options.get("threshold_sweep", []):
        row = evaluate_score_setting(cases, "learned_final", float(threshold), options["topk_per_image"], options["nms_threshold"], primary_iou)
        row["sweep"] = "threshold"
        rows.append(row)
    for topk in options.get("topk_sweep", []):
        parsed_topk = None if topk is None else int(topk)
        row = evaluate_score_setting(cases, "learned_final", options["score_threshold"], parsed_topk, options["nms_threshold"], primary_iou)
        row["sweep"] = "topk"
        rows.append(row)
    for nms in options.get("nms_sweep", []):
        parsed_nms = None if nms is None else float(nms)
        row = evaluate_score_setting(cases, "learned_final", options["score_threshold"], options["topk_per_image"], parsed_nms, primary_iou)
        row["sweep"] = "nms"
        rows.append(row)
    return rows


def grouped_recall_rows(gt_rows: Sequence[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    def size_bin(area: float) -> str:
        if area < 0.01:
            return "small_lt_0.01"
        if area < 0.04:
            return "medium_0.01_0.04"
        return "large_ge_0.04"

    expanded: list[dict[str, Any]] = []
    for row in gt_rows:
        groups = {
            "plane": row["plane"],
            "side": row["side"],
            "processed_slice": str(row["processed_slice"]),
            "box_size": size_bin(float(row["box_area_normalized"])),
            "slice_mapping": "exact" if bool(row["exact_slice_mapping"]) else "nearest",
        }
        for dimension, value in groups.items():
            expanded.append({"dimension": dimension, "value": value, "row": row})
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in expanded:
        grouped[(item["dimension"], item["value"])].append(item["row"])
    for (dimension, value), rows in sorted(grouped.items()):
        count = len(rows)
        output.append(
            {
                "dimension": dimension,
                "value": value,
                "num_gt": count,
                "raw_geometry_recall": sum(float(row["best_raw_geometry_iou"]) >= iou_threshold for row in rows) / max(count, 1),
                "raw_strict_recall": sum(float(row["best_raw_strict_iou"]) >= iou_threshold for row in rows) / max(count, 1),
                "after_threshold_recall": sum(float(row["best_after_threshold_iou"]) >= iou_threshold for row in rows) / max(count, 1),
                "after_topk_recall": sum(float(row["best_after_topk_iou"]) >= iou_threshold for row in rows) / max(count, 1),
                "after_nms_recall": sum(float(row["best_after_nms_iou"]) >= iou_threshold for row in rows) / max(count, 1),
                "mean_positive_tokens": float(np.mean([float(row["positive_token_count"]) for row in rows])),
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_diagnostic_plots(plot_data: dict[str, Any], output_dir: Path, options: dict[str, Any]) -> list[str]:
    if not bool(options.get("save_plots", True)):
        return []
    mpl_dir = output_dir / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved: list[str] = []
    strict_iou = plot_data["strict_iou"].float().numpy()
    geometry_iou = plot_data["geometry_iou"].float().numpy()
    rng = np.random.default_rng(int(options["seed"]))
    max_points = int(options.get("scatter_max_points", 30000))
    if strict_iou.size > max_points:
        indices = np.sort(rng.choice(strict_iou.size, size=max_points, replace=False))
    else:
        indices = np.arange(strict_iou.size)
    preferred = [name for name in ("objectness", "quality", "learned_final", "pipeline_objectness_side") if name in plot_data["scores"]]
    if preferred:
        fig, axes = plt.subplots(1, len(preferred), figsize=(5 * len(preferred), 4), squeeze=False)
        for axis, name in zip(axes[0], preferred):
            values = plot_data["scores"][name].float().numpy()
            axis.scatter(strict_iou[indices], values[indices], s=4, alpha=0.18, rasterized=True)
            axis.set_xlabel("strict IoU (includes predicted side)")
            axis.set_ylabel(name)
            corr = safe_spearman_corr(values.tolist(), strict_iou.tolist())
            axis.set_title(f"{name}\nSpearman={corr:.4f}")
            axis.grid(alpha=0.2)
        fig.tight_layout()
        path = output_dir / "plots" / "score_vs_strict_iou.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(str(path))

        fig, axes = plt.subplots(1, len(preferred), figsize=(5 * len(preferred), 4), squeeze=False)
        for axis, name in zip(axes[0], preferred):
            values = plot_data["scores"][name].float().numpy()
            axis.scatter(geometry_iou[indices], values[indices], s=4, alpha=0.18, rasterized=True)
            axis.set_xlabel("geometry IoU (same plane/slice, side-agnostic)")
            axis.set_ylabel(name)
            corr = safe_spearman_corr(values.tolist(), geometry_iou.tolist())
            axis.set_title(f"{name}\nSpearman={corr:.4f}")
            axis.grid(alpha=0.2)
        fig.tight_layout()
        path = output_dir / "plots" / "score_vs_geometry_iou.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(str(path))

        good = geometry_iou >= float(options.get("high_iou_threshold", 0.3))
        bad = geometry_iou <= float(options.get("low_iou_threshold", 0.1))
        fig, axes = plt.subplots(1, len(preferred), figsize=(5 * len(preferred), 4), squeeze=False)
        for axis, name in zip(axes[0], preferred):
            values = plot_data["scores"][name].float().numpy()
            if bad.any():
                axis.hist(values[bad], bins=40, alpha=0.55, density=True, label=f"IoU<=low (n={int(bad.sum())})")
            if good.any():
                axis.hist(values[good], bins=40, alpha=0.65, density=True, label=f"IoU>=high (n={int(good.sum())})")
            axis.set_xlabel(name)
            axis.set_ylabel("density")
            axis.set_title(f"{name} by candidate quality")
            axis.legend(fontsize=8)
        fig.tight_layout()
        path = output_dir / "plots" / "score_histograms_by_iou.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(str(path))

    branch_names = [name for name in ("objectness_logits", "quality_logits", "bbox_width", "bbox_height", "bbox_area") if name in plot_data["branches"]]
    if branch_names:
        fig, axes = plt.subplots(1, len(branch_names), figsize=(4.5 * len(branch_names), 4), squeeze=False)
        for axis, name in zip(axes[0], branch_names):
            values = plot_data["branches"][name].float().numpy()
            axis.hist(values[np.isfinite(values)], bins=50, alpha=0.8)
            axis.set_title(name)
            axis.grid(alpha=0.2)
        fig.tight_layout()
        path = output_dir / "plots" / "head_output_histograms.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(str(path))
    return saved


def save_case_overlays(cases: Sequence[dict[str, Any]], output_dir: Path, options: dict[str, Any]) -> list[str]:
    if not bool(options.get("save_visualizations", True)):
        return []
    mpl_dir = output_dir / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    saved: list[str] = []
    case_limit = int(options.get("visualization_cases", 3))
    top_count = int(options.get("predictions_per_overlay", 5))
    for case in list(cases)[:case_limit]:
        predictions = case["predictions"]
        shape = predictions["shape_metadata"]
        plane_ids, slice_indices = _token_plane_and_slice_ids(shape, torch.device("cpu"))
        side_ids = predictions["side_logits"][0].argmax(dim=-1)
        boxes = predictions["bbox_pred"][0].float()
        scores = case["scores"]["learned_final"].float()
        for item in case["gt_items"]:
            plane = item["plane"]
            processed_slice = int(item["processed_slice"])
            volume = case[f"{plane}_volume"]
            if volume is None:
                continue
            image = volume[0, processed_slice].numpy()
            height, width = image.shape
            group = torch.nonzero((plane_ids == int(item["plane_id"])) & (slice_indices == processed_slice), as_tuple=False).flatten()
            group = group[scores[group].argsort(descending=True)[: max(0, top_count)]]
            fig, axis = plt.subplots(figsize=(6, 6))
            axis.imshow(image, cmap="gray")
            gt = item["box"].float().tolist()
            axis.add_patch(Rectangle((gt[0] * width, gt[1] * height), (gt[2] - gt[0]) * width, (gt[3] - gt[1]) * height, fill=False, edgecolor="lime", linewidth=2.5))
            axis.text(gt[0] * width, max(0, gt[1] * height - 3), f"GT {item['side']}", color="lime", fontsize=8, weight="bold")
            for rank, token_tensor in enumerate(group, start=1):
                token_idx = int(token_tensor.item())
                box = boxes[token_idx].tolist()
                color = "tab:blue" if int(side_ids[token_idx].item()) == 0 else "tab:orange"
                axis.add_patch(Rectangle((box[0] * width, box[1] * height), (box[2] - box[0]) * width, (box[3] - box[1]) * height, fill=False, edgecolor=color, linewidth=max(0.8, 2.0 - 0.2 * rank), alpha=0.85))
                axis.text(box[0] * width, min(height - 1, box[1] * height + 8), f"{rank}:{scores[token_idx]:.3f}", color=color, fontsize=6)
            axis.set_title(f"{case['case_id']} {plane} slice={processed_slice} | top {len(group)} learned-score candidates")
            axis.axis("off")
            fig.tight_layout()
            path = output_dir / "overlays" / f"{case['case_id']}_{plane}_slice{processed_slice:02d}_{item['side']}_gt{item['gt_index']}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(str(path))
    return saved


def _parameter_gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    squared = defaultdict(float)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().pow(2).sum().cpu())
        squared["total"] += value
        if name.startswith("backbone."):
            squared["backbone"] += value
        elif name.startswith("heads.tower."):
            squared["head_shared_tower"] += value
        elif name.startswith("heads.objectness."):
            squared["head_objectness"] += value
        elif name.startswith("heads.bbox_quality."):
            squared["head_quality"] += value
        elif name.startswith("heads.bbox."):
            squared["head_bbox"] += value
        elif name.startswith("heads.side."):
            squared["head_side"] += value
        elif name.startswith("heads.abnormality."):
            squared["head_abnormality"] += value
    return {name: math.sqrt(value) for name, value in squared.items()}


def run_gradient_diagnostics(
    model: torch.nn.Module,
    sample: dict[str, Any],
    base_config: dict[str, Any],
    checkpoint_epoch: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Measure weighted per-loss gradient norms without an optimizer step."""

    model.eval()
    batch = move_batch_to_device(module1_collate([sample]), device)
    model.zero_grad(set_to_none=True)
    predictions = model(batch["sagittal_volume"], batch["coronal_volume"], batch["sagittal_coords"], batch["coronal_coords"])
    full_criterion = build_loss(base_config).to(device)
    full_criterion.set_epoch(int(checkpoint_epoch))
    set_diagnostic_seed(seed)
    full_losses = full_criterion(predictions, batch["labels"])
    scalar_losses = {
        key: float(value.detach().float().cpu().item())
        for key, value in full_losses.items()
        if isinstance(value, torch.Tensor) and value.numel() == 1 and key.startswith("loss")
    }
    targets = build_module1_targets(
        batch["labels"],
        predictions["shape_metadata"],
        device=device,
        target_config=full_criterion.target_config,
    )
    positive = targets["positive_mask"]
    if positive.any():
        pred_boxes = predictions["bbox_pred"][positive]
        target_boxes = targets["bbox_targets"][positive]
        scalar_losses["loss_bbox_smooth_l1_unweighted"] = float(F.smooth_l1_loss(pred_boxes, target_boxes).detach().cpu())
        scalar_losses["loss_bbox_giou_unweighted"] = float(generalized_box_iou_loss(pred_boxes, target_boxes).detach().cpu())

    original = {
        "objectness": float(full_criterion.objectness_weight),
        "bbox": float(full_criterion.bbox_weight),
        "side": float(full_criterion.side_weight),
        "abnormality": float(full_criterion.abnormality_weight),
        "bbox_quality": float(full_criterion.bbox_quality_loss_weight),
        "ranking": float(full_criterion.quality_ranking_loss_weight),
    }
    branches = ["objectness", "bbox", "side", "abnormality"]
    if "bbox_quality_logits" in predictions and original["bbox_quality"] != 0.0:
        branches.append("bbox_quality")
    ranking_requested = full_criterion.use_quality_ranking_loss or full_criterion.bbox_quality_loss_type in {"pairwise_ranking", "varifocal_plus_ranking"}
    if "bbox_quality_logits" in predictions and ranking_requested and original["ranking"] != 0.0:
        branches.append("ranking")
    reports: dict[str, Any] = {}
    for branch_index, branch in enumerate(branches):
        criterion = build_loss(base_config).to(device)
        criterion.set_epoch(int(checkpoint_epoch))
        criterion.objectness_weight = 0.0
        criterion.bbox_weight = 0.0
        criterion.side_weight = 0.0
        criterion.abnormality_weight = 0.0
        criterion.bbox_quality_loss_weight = 0.0
        criterion.quality_ranking_loss_weight = 0.0
        criterion.use_quality_ranking_loss = False
        if branch == "objectness":
            criterion.objectness_weight = original["objectness"]
        elif branch == "bbox":
            criterion.bbox_weight = original["bbox"]
        elif branch == "side":
            criterion.side_weight = original["side"]
        elif branch == "abnormality":
            criterion.abnormality_weight = original["abnormality"]
        elif branch == "bbox_quality":
            criterion.bbox_quality_loss_weight = original["bbox_quality"]
        else:
            criterion.use_quality_ranking_loss = True
            criterion.quality_ranking_loss_weight = original["ranking"]
        model.zero_grad(set_to_none=True)
        set_diagnostic_seed(seed + branch_index)
        isolated = criterion(predictions, batch["labels"])["loss"]
        isolated.backward(retain_graph=True)
        reports[branch] = {
            "weighted_loss": float(isolated.detach().float().cpu()),
            "gradient_norms": _parameter_gradient_norms(model),
        }
    model.zero_grad(set_to_none=True)
    return {
        "case_id": sample["case_id"],
        "checkpoint_epoch": int(checkpoint_epoch),
        "loss_values": scalar_losses,
        "configured_weights": original,
        "per_weighted_loss_gradient_norms": reports,
    }


def render_run_markdown(summary: dict[str, Any]) -> str:
    stage = summary["stage_recall"]
    lines = [
        "# Module 1 Detection Diagnostic Run",
        "",
        f"- cases: {summary['num_cases']}",
        f"- GT boxes: {summary['num_gt']}",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- primary IoU: {summary['primary_iou_threshold']}",
        "",
        "| stage | GT coverage recall |",
        "|---|---:|",
        f"| raw geometry (side-agnostic) | {stage['raw_geometry']:.4f} |",
        f"| raw strict side | {stage['raw_strict']:.4f} |",
        f"| after threshold | {stage['after_threshold']:.4f} |",
        f"| after global top-k | {stage['after_topk']:.4f} |",
        f"| after grouped NMS | {stage['after_nms']:.4f} |",
        "",
        "## Failure taxonomy",
        "",
    ]
    for name, count in sorted(summary["failure_taxonomy"].items(), key=lambda item: (-int(item[1]), item[0])):
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Score-IoU correlation", ""])
    for name, values in summary["correlations"].items():
        lines.append(f"- {name}: `{values}`")
    lines.extend(
        [
            "",
            "The raw geometry and strict-side rows intentionally use different contracts. "
            "Their gap is side classification error, not bbox localization error.",
            "",
        ]
    )
    return "\n".join(lines)


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    base_config = load_yaml(args.config)
    diagnostic_config = load_yaml(args.diagnostic_config) if args.diagnostic_config else {}
    options = resolve_diagnostic_options(
        base_config,
        diagnostic_config,
        {
            "output_dir": args.output_dir,
            "max_cases": args.max_cases,
            "seed": args.seed,
            "device": args.device,
            "score_threshold": args.score_threshold,
            "topk_per_image": args.topk_per_image,
            "nms_threshold": args.nms_threshold,
        },
    )
    if not bool(options.get("enabled", True)):
        return {"enabled": False}
    output_dir = Path(options.get("output_dir", "outputs/module1_detection_audit"))
    output_dir.mkdir(parents=True, exist_ok=True)
    set_diagnostic_seed(int(options["seed"]))
    data_cfg = base_config["data"]
    usage = str(options.get("split_usage", "val"))
    if usage == "train":
        index_jsonl = args.index_jsonl or data_cfg["train_index"]
        slice_selection = data_cfg.get("train_slice_selection", data_cfg.get("slice_selection"))
    else:
        index_jsonl = args.index_jsonl or data_cfg["val_index"]
        slice_selection = data_cfg.get("val_slice_selection", data_cfg.get("slice_selection"))
    dataset = MeniscusDetectionDataset(
        index_jsonl=index_jsonl,
        output_size=data_cfg["output_size"],
        target_slices=data_cfg["num_slices"],
        patch_size=data_cfg["patch_size"],
        slice_selection=slice_selection,
        transform=None,
    )
    case_indices = select_case_indices(len(dataset), options["max_cases"], int(options["seed"]))
    options["positive_radius_tokens"] = int(base_config.get("loss", {}).get("positive_radius_tokens", 0))
    device = _resolve_device(base_config, str(options.get("device", "auto")))
    model = build_module1_model(base_config).to(device)
    checkpoint_metadata = load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    resolved = {
        "base_config_path": str(args.config),
        "diagnostic_config_path": None if args.diagnostic_config is None else str(args.diagnostic_config),
        "checkpoint": str(args.checkpoint),
        "index_jsonl": str(index_jsonl),
        "slice_selection": slice_selection,
        "case_indices": case_indices,
        "device": str(device),
        "options": options,
    }
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")

    collected = collect_case_diagnostics(model, dataset, case_indices, device, options)
    gt_rows = collected["gt_rows"]
    primary_iou = float(options["primary_iou_threshold"])
    total_gt = len(gt_rows)
    stage_recall = {
        "raw_geometry": sum(float(row["best_raw_geometry_iou"]) >= primary_iou for row in gt_rows) / max(total_gt, 1),
        "raw_strict": sum(float(row["best_raw_strict_iou"]) >= primary_iou for row in gt_rows) / max(total_gt, 1),
        "after_threshold": sum(float(row["best_after_threshold_iou"]) >= primary_iou for row in gt_rows) / max(total_gt, 1),
        "after_topk": sum(float(row["best_after_topk_iou"]) >= primary_iou for row in gt_rows) / max(total_gt, 1),
        "after_nms": sum(float(row["best_after_nms_iou"]) >= primary_iou for row in gt_rows) / max(total_gt, 1),
    }
    sweep_rows = run_score_controls_and_sweeps(collected["cases"], options)
    group_rows = grouped_recall_rows(gt_rows, primary_iou)
    plot_paths = save_diagnostic_plots(collected["plot_data"], output_dir, options)
    overlay_paths = save_case_overlays(collected["cases"], output_dir, options)
    gradient_report = None
    if bool(options.get("run_gradient_diagnostics", False)) and case_indices:
        sample = dataset[int(case_indices[0])]
        gradient_report = run_gradient_diagnostics(
            model,
            sample,
            base_config,
            checkpoint_epoch=int(checkpoint_metadata.get("epoch", 0)),
            device=device,
            seed=int(options["seed"]),
        )
        (output_dir / "gradient_diagnostics.json").write_text(json.dumps(gradient_report, indent=2, ensure_ascii=False), encoding="utf-8")

    write_csv(output_dir / "per_gt_diagnostics.csv", gt_rows)
    write_csv(output_dir / "high_iou_low_score_candidates.csv", collected["high_iou_low_score_rows"])
    write_csv(output_dir / "low_iou_high_score_candidates.csv", collected["low_iou_high_score_rows"])
    write_csv(output_dir / "grouped_recall.csv", group_rows)
    write_csv(output_dir / "score_controls_and_sweeps.csv", sweep_rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint_metadata.get("epoch"),
        "config": str(args.config),
        "diagnostic_config": None if args.diagnostic_config is None else str(args.diagnostic_config),
        "index_jsonl": str(index_jsonl),
        "seed": int(options["seed"]),
        "num_cases": len(collected["cases"]),
        "num_gt": total_gt,
        "primary_iou_threshold": primary_iou,
        "postprocess": {
            "score_mode": options["final_score_mode"],
            "score_threshold": options["score_threshold"],
            "topk_per_image": options["topk_per_image"],
            "nms_threshold": options["nms_threshold"],
        },
        "stage_recall": stage_recall,
        "stage_absolute_losses": {
            "side_classification": stage_recall["raw_geometry"] - stage_recall["raw_strict"],
            "score_threshold": stage_recall["raw_strict"] - stage_recall["after_threshold"],
            "global_topk": stage_recall["after_threshold"] - stage_recall["after_topk"],
            "nms": stage_recall["after_topk"] - stage_recall["after_nms"],
        },
        "failure_taxonomy": collected["failure_counts"],
        "positive_token_count": distribution([float(row["positive_token_count"]) for row in gt_rows]),
        "best_raw_geometry_iou": distribution([float(row["best_raw_geometry_iou"]) for row in gt_rows]),
        "best_raw_strict_iou": distribution([float(row["best_raw_strict_iou"]) for row in gt_rows]),
        "correlations": collected["correlations"],
        "branch_distributions": collected["branch_distributions"],
        "high_iou_low_score_candidates_exported": len(collected["high_iou_low_score_rows"]),
        "low_iou_high_score_candidates_exported": len(collected["low_iou_high_score_rows"]),
        "plots": plot_paths,
        "overlays": overlay_paths,
        "gradient_diagnostics": gradient_report,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_run_markdown(summary), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("num_cases", "num_gt", "stage_recall", "failure_taxonomy")}, indent=2, ensure_ascii=False))
    print(f"saved diagnostics to {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/module1_meniscovit_cop.yaml")
    parser.add_argument(
        "--diagnostic-config",
        default=None,
        help="Optional YAML overrides; omitted values inherit the main config and built-in defaults.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index-jsonl", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--topk-per-image", type=int, default=None)
    parser.add_argument("--nms-threshold", type=float, default=None)
    args = parser.parse_args()
    run_diagnostics(args)


if __name__ == "__main__":
    main()
