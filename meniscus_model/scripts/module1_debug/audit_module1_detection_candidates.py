"""Audit where Module 1 detection candidates are lost before final matching."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Any
import warnings

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.meniscus_detection_dataset import MeniscusDetectionDataset
from engine.validate_module1 import build_module1_model, module1_collate, move_batch_to_device
from losses.detection_loss import box_iou
from scripts.module1_debug.runtime_utils import limit_dataset, set_diagnostic_seed, write_resolved_run_config
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_module1_final_scores, _nms_boxes, _token_plane_and_slice_ids, precision_recall_specificity_f1


warnings.filterwarnings("ignore", message="Invalid value for VR UI:*")

PLANE_TO_ID = {"sagittal": 0, "coronal": 1}
ID_TO_PLANE = {0: "sagittal", 1: "coronal"}
SIDE_TO_ID = {"medial": 0, "lateral": 1}
ID_TO_SIDE = {0: "medial", 1: "lateral"}
CSV_COLUMNS = [
    "experiment",
    "case_id",
    "plane",
    "processed_slice",
    "side",
    "gt_index",
    "gt_bbox",
    "iou_threshold",
    "best_raw_iou",
    "best_raw_score",
    "best_raw_rank",
    "passed_score_threshold",
    "passed_topk",
    "survived_nms",
    "entered_final_predictions",
    "matched_as_tp",
    "failure_reason",
    "best_final_iou",
    "best_after_threshold_iou",
    "best_after_topk_iou",
    "best_after_nms_iou",
    "best_any_raw_iou",
    "best_any_raw_plane",
    "best_any_raw_processed_slice",
    "best_any_raw_side",
]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_thresholds(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def make_loader(config: dict[str, Any], index_jsonl: str | Path, max_cases: int | None = None) -> DataLoader:
    data_cfg = config["data"]
    dataset = MeniscusDetectionDataset(
        index_jsonl=index_jsonl,
        output_size=data_cfg["output_size"],
        target_slices=data_cfg["num_slices"],
        patch_size=data_cfg["patch_size"],
        slice_selection=data_cfg.get("val_slice_selection", data_cfg.get("slice_selection")),
        transform=None,
    )
    dataset = limit_dataset(dataset, max_cases)
    return DataLoader(
        dataset,
        batch_size=int(config.get("validate", {}).get("batch_size", 1)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=module1_collate,
        drop_last=False,
    )


def resolve_audit_options(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    """Resolve CLI overrides against the validation configuration."""

    val_cfg = config.get("validate", {})
    if not hasattr(args, "score_threshold") or args.score_threshold is None:
        args.score_threshold = float(val_cfg.get("score_threshold", 0.5))
    if not hasattr(args, "topk_per_image") or args.topk_per_image is None:
        configured_topk = val_cfg.get("topk_per_image")
        args.topk_per_image = None if configured_topk is None else int(configured_topk)
    if not hasattr(args, "nms_threshold") or args.nms_threshold is None:
        configured_nms = val_cfg.get("nms_threshold")
        args.nms_threshold = None if configured_nms is None else float(configured_nms)
    if not hasattr(args, "final_score_mode") or args.final_score_mode is None:
        args.final_score_mode = val_cfg.get("final_score_mode", "objectness")
    if not hasattr(args, "quality_score_alpha") or args.quality_score_alpha is None:
        args.quality_score_alpha = float(val_cfg.get("quality_score_alpha", 1.0))
    if not hasattr(args, "quality_score_beta") or args.quality_score_beta is None:
        args.quality_score_beta = float(val_cfg.get("quality_score_beta", 1.0))
    if not hasattr(args, "seed") or args.seed is None:
        args.seed = int(config.get("seed", 42))
    if not hasattr(args, "max_cases"):
        args.max_cases = None
    data_cfg = config["data"]
    args.resolved_slice_selection = data_cfg.get("val_slice_selection", data_cfg.get("slice_selection"))
    return args


def normalize_gt_items(case_labels: dict[str, Any], shape_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for gt_index, label in enumerate(case_labels.get("module1_labels", [])):
        plane = label.get("plane")
        if plane not in PLANE_TO_ID:
            continue
        shape = shape_metadata[plane]
        x1, y1, x2, y2 = [float(value) for value in label["bbox"]]
        box = torch.tensor(
            [
                x1 / float(shape["width"]),
                y1 / float(shape["height"]),
                x2 / float(shape["width"]),
                y2 / float(shape["height"]),
            ],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        x1y1 = torch.minimum(box[:2], box[2:])
        x2y2 = torch.maximum(box[:2], box[2:])
        side = label.get("meniscus_side")
        items.append(
            {
                "gt_index": gt_index,
                "plane_id": PLANE_TO_ID[str(plane)],
                "plane": str(plane),
                "slice_index": int(label["slice_index"]),
                "side_id": SIDE_TO_ID[str(side)],
                "side": str(side),
                "box": torch.cat([x1y1, x2y2]),
            }
        )
    return items


def build_postprocess_trace(
    scores: torch.Tensor,
    boxes: torch.Tensor,
    plane_ids: torch.Tensor,
    slice_indices: torch.Tensor,
    side_ids: torch.Tensor,
    score_threshold: float,
    topk_per_image: int | None,
    nms_threshold: float | None,
) -> dict[str, Any]:
    order = scores.argsort(descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel(), device=order.device, dtype=order.dtype) + 1

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
        "raw_order": order,
        "raw_ranks": ranks,
        "threshold_indices": threshold_indices,
        "topk_indices": topk_indices,
        "final_indices": final_indices,
        "threshold_set": set(int(value) for value in threshold_indices.detach().cpu().tolist()),
        "topk_set": set(int(value) for value in topk_indices.detach().cpu().tolist()),
        "final_set": set(int(value) for value in final_indices.detach().cpu().tolist()),
    }


def match_final_gt_ids(
    gt_items: list[dict[str, Any]],
    final_indices: torch.Tensor,
    scores: torch.Tensor,
    boxes: torch.Tensor,
    plane_ids: torch.Tensor,
    slice_indices: torch.Tensor,
    side_ids: torch.Tensor,
    iou_threshold: float,
) -> tuple[set[int], dict[int, float], dict[int, bool], dict[str, int]]:
    matched: set[int] = set()
    best_final_iou: dict[int, float] = {int(item["gt_index"]): 0.0 for item in gt_items}
    has_final_above_threshold: dict[int, bool] = {int(item["gt_index"]): False for item in gt_items}
    gt_by_key: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    pred_by_key: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for item in gt_items:
        gt_by_key[(int(item["plane_id"]), int(item["slice_index"]), int(item["side_id"]))].append(item)
    for idx_tensor in final_indices:
        idx = int(idx_tensor.item())
        key = (int(plane_ids[idx].item()), int(slice_indices[idx].item()), int(side_ids[idx].item()))
        pred_by_key[key].append(idx)

    tp = fp = fn = 0
    for key in sorted(set(gt_by_key) | set(pred_by_key)):
        key_gt = gt_by_key.get(key, [])
        key_pred = pred_by_key.get(key, [])
        if not key_pred:
            fn += len(key_gt)
            continue
        if not key_gt:
            fp += len(key_pred)
            continue
        gt_boxes = torch.stack([item["box"].to(boxes.device) for item in key_gt], dim=0)
        pred_tensor = torch.tensor(key_pred, device=boxes.device, dtype=torch.long)
        ious = box_iou(boxes[pred_tensor], gt_boxes)
        for gt_local, item in enumerate(key_gt):
            gt_id = int(item["gt_index"])
            best = float(ious[:, gt_local].max().item()) if ious.numel() else 0.0
            best_final_iou[gt_id] = max(best_final_iou[gt_id], best)
            has_final_above_threshold[gt_id] = bool(best >= float(iou_threshold))

        used_gt_local: set[int] = set()
        order = scores[pred_tensor].argsort(descending=True)
        for pred_order_idx in order:
            pred_row = int(pred_order_idx.item())
            best_iou, best_gt_local = ious[pred_row].max(dim=0)
            best_gt_local_int = int(best_gt_local.item())
            if float(best_iou.item()) >= float(iou_threshold) and best_gt_local_int not in used_gt_local:
                tp += 1
                used_gt_local.add(best_gt_local_int)
                matched.add(int(key_gt[best_gt_local_int]["gt_index"]))
            else:
                fp += 1
        fn += max(len(key_gt) - len(used_gt_local), 0)

    return matched, best_final_iou, has_final_above_threshold, {"tp": tp, "fp": fp, "fn": fn}


def best_candidate_for_gt(
    gt_item: dict[str, Any],
    scores: torch.Tensor,
    boxes: torch.Tensor,
    plane_ids: torch.Tensor,
    slice_indices: torch.Tensor,
    side_ids: torch.Tensor,
) -> dict[str, Any]:
    gt_box = gt_item["box"].to(boxes.device)
    all_ious = box_iou(boxes, gt_box.view(1, 4)).flatten()
    best_any_iou, best_any_idx = all_ious.max(dim=0)
    strict_mask = (
        (plane_ids == int(gt_item["plane_id"]))
        & (slice_indices == int(gt_item["slice_index"]))
        & (side_ids == int(gt_item["side_id"]))
    )
    strict_indices = torch.nonzero(strict_mask, as_tuple=False).flatten()
    if strict_indices.numel() == 0:
        return {
            "strict_index": None,
            "strict_iou": 0.0,
            "strict_score": 0.0,
            "best_any_index": int(best_any_idx.item()),
            "best_any_iou": float(best_any_iou.item()),
        }
    strict_ious = all_ious[strict_indices]
    strict_best_iou, strict_best_local = strict_ious.max(dim=0)
    strict_best_idx = int(strict_indices[int(strict_best_local.item())].item())
    return {
        "strict_index": strict_best_idx,
        "strict_iou": float(strict_best_iou.item()),
        "strict_score": float(scores[strict_best_idx].item()),
        "best_any_index": int(best_any_idx.item()),
        "best_any_iou": float(best_any_iou.item()),
    }


def infer_failure_reason(
    gt_item: dict[str, Any],
    candidate: dict[str, Any],
    trace: dict[str, Any],
    matched_as_tp: bool,
    has_final_above_threshold: bool,
    iou_threshold: float,
    plane_ids: torch.Tensor,
    slice_indices: torch.Tensor,
    side_ids: torch.Tensor,
) -> str:
    if matched_as_tp:
        return "matched"
    strict_index = candidate["strict_index"]
    if float(candidate["strict_iou"]) < float(iou_threshold) or strict_index is None:
        if float(candidate["best_any_iou"]) >= float(iou_threshold):
            any_idx = int(candidate["best_any_index"])
            if int(plane_ids[any_idx].item()) != int(gt_item["plane_id"]):
                return "wrong_plane"
            if int(slice_indices[any_idx].item()) != int(gt_item["slice_index"]):
                return "wrong_slice"
            if int(side_ids[any_idx].item()) != int(gt_item["side_id"]):
                return "wrong_side"
        return "no_candidate_iou_above_threshold"
    if float(candidate.get("best_after_threshold_iou", 0.0)) < float(iou_threshold):
        return "score_threshold_filtered"
    if float(candidate.get("best_after_topk_iou", 0.0)) < float(iou_threshold):
        return "topk_filtered"
    if float(candidate.get("best_after_nms_iou", 0.0)) < float(iou_threshold):
        return "nms_suppressed"
    if has_final_above_threshold:
        return "final_matching_conflict"
    return "final_iou_below_threshold"


def best_stage_iou_for_gt(
    gt_item: dict[str, Any],
    stage_indices: torch.Tensor,
    boxes: torch.Tensor,
    plane_ids: torch.Tensor,
    slice_indices: torch.Tensor,
    side_ids: torch.Tensor,
) -> float:
    if stage_indices.numel() == 0:
        return 0.0
    strict_mask = (
        (plane_ids[stage_indices] == int(gt_item["plane_id"]))
        & (slice_indices[stage_indices] == int(gt_item["slice_index"]))
        & (side_ids[stage_indices] == int(gt_item["side_id"]))
    )
    strict_indices = stage_indices[strict_mask]
    if strict_indices.numel() == 0:
        return 0.0
    ious = box_iou(boxes[strict_indices], gt_item["box"].to(boxes.device).view(1, 4)).flatten()
    return float(ious.max().item()) if ious.numel() else 0.0


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def collect_token_quality_scores(
    gt_items: list[dict[str, Any]],
    scores: torch.Tensor,
    boxes: torch.Tensor,
    plane_ids: torch.Tensor,
    slice_indices: torch.Tensor,
    side_ids: torch.Tensor,
) -> tuple[list[float], list[float]]:
    gt_by_key: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in gt_items:
        gt_by_key[(int(item["plane_id"]), int(item["slice_index"]), int(item["side_id"]))].append(item)
    good_scores: list[float] = []
    bad_scores: list[float] = []
    for idx in range(int(scores.numel())):
        key = (int(plane_ids[idx].item()), int(slice_indices[idx].item()), int(side_ids[idx].item()))
        key_gt = gt_by_key.get(key)
        if not key_gt:
            best_iou = 0.0
        else:
            gt_boxes = torch.stack([item["box"].to(boxes.device) for item in key_gt], dim=0)
            best_iou = float(box_iou(boxes[idx].view(1, 4), gt_boxes).max().item())
        score = float(scores[idx].item())
        if best_iou > 0.3:
            good_scores.append(score)
        if best_iou < 0.1:
            bad_scores.append(score)
    return good_scores, bad_scores


@torch.inference_mode()
def audit_candidates(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    args = resolve_audit_options(args, config)
    set_diagnostic_seed(int(args.seed))
    index_jsonl = Path(args.index_jsonl or config["data"]["val_index"])
    args.index_jsonl = str(index_jsonl)
    thresholds = parse_thresholds(args.iou_thresholds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_path) if args.csv_path else output_dir / "candidate_audit.csv"
    summary_json_path = Path(args.summary_json) if args.summary_json else output_dir / "candidate_audit_summary.json"
    summary_md_path = Path(args.summary_md) if args.summary_md else output_dir / "candidate_audit_summary.md"

    device_name = config.get("device", "cuda")
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_module1_model(config).to(device)
    metadata = load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    loader = make_loader(config, index_jsonl, max_cases=args.max_cases)
    args.num_cases = len(loader.dataset)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    amp = bool(config.get("train", {}).get("amp", True)) and device.type == "cuda"
    write_resolved_run_config(
        output_dir / "candidate_audit_run_config.json",
        {
            "diagnostic": "candidate_audit",
            "config_path": str(args.config),
            "checkpoint": str(args.checkpoint),
            "index_jsonl": str(index_jsonl),
            "seed": int(args.seed),
            "max_cases": args.max_cases,
            "num_cases": args.num_cases,
            "slice_selection": args.resolved_slice_selection,
            "score_threshold": float(args.score_threshold),
            "topk_per_image": args.topk_per_image,
            "nms_threshold": args.nms_threshold,
            "final_score_mode": str(args.final_score_mode),
            "quality_score_alpha": float(args.quality_score_alpha),
            "quality_score_beta": float(args.quality_score_beta),
            "iou_thresholds": thresholds,
            "device": str(device),
            "amp": amp,
            "source_config": config,
        },
    )

    rows: list[dict[str, Any]] = []
    gt_stage_records: list[dict[str, Any]] = []
    final_counts_by_threshold = {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in thresholds}
    good_candidate_scores: list[float] = []
    bad_candidate_scores: list[float] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp):
            predictions = model(
                batch["sagittal_volume"],
                batch["coronal_volume"],
                batch["sagittal_coords"],
                batch["coronal_coords"],
            )
        shape_metadata = predictions["shape_metadata"]
        plane_ids, slice_indices = _token_plane_and_slice_ids(shape_metadata, device)
        batch_scores = compute_module1_final_scores(
            predictions,
            final_score_mode=str(args.final_score_mode),
            quality_score_alpha=float(args.quality_score_alpha),
            quality_score_beta=float(args.quality_score_beta),
        )
        batch_boxes = predictions["bbox_pred"].detach()
        batch_side_ids = predictions["side_logits"].detach().argmax(dim=-1)

        for batch_idx, labels in enumerate(batch["labels"]):
            case_id = str(labels.get("case_id", batch["case_id"][batch_idx]))
            scores = batch_scores[batch_idx]
            boxes = batch_boxes[batch_idx]
            side_ids = batch_side_ids[batch_idx]
            trace = build_postprocess_trace(
                scores=scores,
                boxes=boxes,
                plane_ids=plane_ids,
                slice_indices=slice_indices,
                side_ids=side_ids,
                score_threshold=float(args.score_threshold),
                topk_per_image=args.topk_per_image,
                nms_threshold=args.nms_threshold,
            )
            gt_items = normalize_gt_items(labels, shape_metadata)
            good_scores, bad_scores = collect_token_quality_scores(gt_items, scores, boxes, plane_ids, slice_indices, side_ids)
            good_candidate_scores.extend(good_scores)
            bad_candidate_scores.extend(bad_scores)
            candidates = {
                int(item["gt_index"]): best_candidate_for_gt(item, scores, boxes, plane_ids, slice_indices, side_ids)
                for item in gt_items
            }
            for threshold in thresholds:
                matched, best_final_iou, has_final_above, counts = match_final_gt_ids(
                    gt_items=gt_items,
                    final_indices=trace["final_indices"],
                    scores=scores,
                    boxes=boxes,
                    plane_ids=plane_ids,
                    slice_indices=slice_indices,
                    side_ids=side_ids,
                    iou_threshold=float(threshold),
                )
                for key in final_counts_by_threshold[threshold]:
                    final_counts_by_threshold[threshold][key] += int(counts[key])
                for item in gt_items:
                    gt_id = int(item["gt_index"])
                    candidate = candidates[gt_id]
                    candidate["best_after_threshold_iou"] = best_stage_iou_for_gt(
                        item,
                        trace["threshold_indices"],
                        boxes,
                        plane_ids,
                        slice_indices,
                        side_ids,
                    )
                    candidate["best_after_topk_iou"] = best_stage_iou_for_gt(
                        item,
                        trace["topk_indices"],
                        boxes,
                        plane_ids,
                        slice_indices,
                        side_ids,
                    )
                    candidate["best_after_nms_iou"] = best_stage_iou_for_gt(
                        item,
                        trace["final_indices"],
                        boxes,
                        plane_ids,
                        slice_indices,
                        side_ids,
                    )
                    token_idx = candidate["strict_index"]
                    passed_score = token_idx is not None and int(token_idx) in trace["threshold_set"]
                    passed_topk = token_idx is not None and int(token_idx) in trace["topk_set"]
                    survived_nms = token_idx is not None and int(token_idx) in trace["final_set"]
                    matched_as_tp = gt_id in matched
                    reason = infer_failure_reason(
                        gt_item=item,
                        candidate=candidate,
                        trace=trace,
                        matched_as_tp=matched_as_tp,
                        has_final_above_threshold=bool(has_final_above[gt_id]),
                        iou_threshold=float(threshold),
                        plane_ids=plane_ids,
                        slice_indices=slice_indices,
                        side_ids=side_ids,
                    )
                    best_any_idx = int(candidate["best_any_index"])
                    best_raw_rank = int(trace["raw_ranks"][int(token_idx)].item()) if token_idx is not None else ""
                    row = {
                        "experiment": args.experiment,
                        "case_id": case_id,
                        "plane": item["plane"],
                        "processed_slice": int(item["slice_index"]),
                        "side": item["side"],
                        "gt_index": gt_id,
                        "gt_bbox": json.dumps([float(value) for value in item["box"].detach().cpu().tolist()]),
                        "iou_threshold": float(threshold),
                        "best_raw_iou": float(candidate["strict_iou"]),
                        "best_raw_score": float(candidate["strict_score"]),
                        "best_raw_rank": best_raw_rank,
                        "passed_score_threshold": bool(passed_score),
                        "passed_topk": bool(passed_topk),
                        "survived_nms": bool(survived_nms),
                        "entered_final_predictions": bool(survived_nms),
                        "matched_as_tp": bool(matched_as_tp),
                        "failure_reason": reason,
                        "best_final_iou": float(best_final_iou[gt_id]),
                        "best_after_threshold_iou": float(candidate["best_after_threshold_iou"]),
                        "best_after_topk_iou": float(candidate["best_after_topk_iou"]),
                        "best_after_nms_iou": float(candidate["best_after_nms_iou"]),
                        "best_any_raw_iou": float(candidate["best_any_iou"]),
                        "best_any_raw_plane": ID_TO_PLANE[int(plane_ids[best_any_idx].item())],
                        "best_any_raw_processed_slice": int(slice_indices[best_any_idx].item()),
                        "best_any_raw_side": ID_TO_SIDE[int(side_ids[best_any_idx].item())],
                    }
                    rows.append(row)
                    gt_stage_records.append(row)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = build_summary(
        rows=gt_stage_records,
        thresholds=thresholds,
        final_counts_by_threshold=final_counts_by_threshold,
        good_candidate_scores=good_candidate_scores,
        bad_candidate_scores=bad_candidate_scores,
        metadata=metadata,
        args=args,
    )
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_md_path.write_text(render_summary_markdown(summary), encoding="utf-8")
    print(f"saved candidate audit CSV: {csv_path}")
    print(f"saved candidate audit summary JSON: {summary_json_path}")
    print(f"saved candidate audit summary Markdown: {summary_md_path}")
    return summary


def build_summary(
    rows: list[dict[str, Any]],
    thresholds: list[float],
    final_counts_by_threshold: dict[float, dict[str, int]],
    good_candidate_scores: list[float],
    bad_candidate_scores: list[float],
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    unique_gt = {(row["case_id"], row["plane"], row["processed_slice"], row["side"], row["gt_index"]) for row in rows}
    total_gt = len(unique_gt)
    summary: dict[str, Any] = {
        "experiment": args.experiment,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "index_jsonl": str(args.index_jsonl),
        "checkpoint_epoch": metadata.get("epoch"),
        "score_threshold": float(args.score_threshold),
        "topk_per_image": args.topk_per_image,
        "nms_threshold": args.nms_threshold,
        "final_score_mode": str(args.final_score_mode),
        "seed": int(args.seed),
        "max_cases": args.max_cases,
        "num_cases": int(args.num_cases),
        "slice_selection": args.resolved_slice_selection,
        "total_gt": total_gt,
        "failure_reason_counts": {},
        "mean_best_raw_iou": 0.0,
        "mean_best_final_iou": 0.0,
        "score_distribution_good_candidates_iou_gt_0.3": distribution(good_candidate_scores),
        "score_distribution_bad_candidates_iou_lt_0.1": distribution(bad_candidate_scores),
    }
    if total_gt > 0:
        one_per_gt: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in rows:
            key = (row["case_id"], row["plane"], row["processed_slice"], row["side"], row["gt_index"])
            one_per_gt.setdefault(key, row)
        summary["mean_best_raw_iou"] = float(np.mean([float(row["best_raw_iou"]) for row in one_per_gt.values()]))
        summary["mean_best_final_iou"] = float(np.mean([float(row["best_final_iou"]) for row in one_per_gt.values()]))

    for threshold in thresholds:
        suffix = f"{float(threshold):g}"
        threshold_rows = [row for row in rows if float(row["iou_threshold"]) == float(threshold)]
        summary[f"recall_upper_bound_raw_candidates@{suffix}"] = _ratio(
            sum(float(row["best_raw_iou"]) >= float(threshold) for row in threshold_rows),
            total_gt,
        )
        summary[f"recall_after_threshold@{suffix}"] = _ratio(
            sum(float(row["best_after_threshold_iou"]) >= float(threshold) for row in threshold_rows),
            total_gt,
        )
        summary[f"recall_after_topk@{suffix}"] = _ratio(
            sum(float(row["best_after_topk_iou"]) >= float(threshold) for row in threshold_rows),
            total_gt,
        )
        summary[f"recall_after_nms@{suffix}"] = _ratio(
            sum(float(row["best_after_nms_iou"]) >= float(threshold) for row in threshold_rows),
            total_gt,
        )
        final_counts = final_counts_by_threshold[threshold]
        rates = precision_recall_specificity_f1(
            tp=int(final_counts["tp"]),
            fp=int(final_counts["fp"]),
            tn=0,
            fn=int(final_counts["fn"]),
        )
        summary[f"final_recall@{suffix}"] = rates["recall"]
        summary[f"final_precision@{suffix}"] = rates["precision"]
        summary[f"final_f1@{suffix}"] = rates["f1"]
        summary["failure_reason_counts"][suffix] = dict(Counter(str(row["failure_reason"]) for row in threshold_rows))
    return summary


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(max(denominator, 1))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes"}


def render_summary_markdown(summary: dict[str, Any]) -> str:
    thresholds = sorted(
        float(key.split("@", 1)[1])
        for key in summary
        if key.startswith("final_recall@")
    )
    lines = [
        "# Module 1 Candidate Audit",
        "",
        f"- experiment: `{summary['experiment']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- total_gt: {summary['total_gt']}",
        f"- score_threshold: {summary['score_threshold']}",
        f"- topk_per_image: {summary['topk_per_image']}",
        f"- nms_threshold: {summary['nms_threshold']}",
        f"- final_score_mode: `{summary.get('final_score_mode', 'objectness')}`",
        "",
        "| IoU | raw upper | after threshold | after topk | after nms | final recall | final precision |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold in thresholds:
        suffix = f"{threshold:g}"
        lines.append(
            "| "
            f"{suffix} | "
            f"{summary[f'recall_upper_bound_raw_candidates@{suffix}']:.4f} | "
            f"{summary[f'recall_after_threshold@{suffix}']:.4f} | "
            f"{summary[f'recall_after_topk@{suffix}']:.4f} | "
            f"{summary[f'recall_after_nms@{suffix}']:.4f} | "
            f"{summary[f'final_recall@{suffix}']:.4f} | "
            f"{summary[f'final_precision@{suffix}']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- mean_best_raw_iou: {summary['mean_best_raw_iou']:.4f}",
            f"- mean_best_final_iou: {summary['mean_best_final_iou']:.4f}",
            "",
            "## Failure Reasons",
            "",
        ]
    )
    for suffix, counts in summary["failure_reason_counts"].items():
        lines.append(f"### IoU {suffix}")
        for reason, count in sorted(counts.items(), key=lambda item: (-int(item[1]), item[0])):
            lines.append(f"- {reason}: {count}")
        lines.append("")
    lines.extend(
        [
            "## Score Distributions",
            "",
            f"- good candidates IoU>0.3: `{summary['score_distribution_good_candidates_iou_gt_0.3']}`",
            f"- bad candidates IoU<0.1: `{summary['score_distribution_bad_candidates_iou_lt_0.1']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--index-jsonl", default=None, help="Validation split JSONL. Defaults to config data.val_index.")
    parser.add_argument("--output-dir", default="outputs/module1_debug")
    parser.add_argument("--experiment", default="module1_detection")
    parser.add_argument("--iou-thresholds", default="0.1,0.3,0.45")
    parser.add_argument("--score-threshold", type=float, default=None, help="Override validate.score_threshold.")
    parser.add_argument("--topk-per-image", type=int, default=None, help="Override validate.topk_per_image.")
    parser.add_argument("--nms-threshold", type=float, default=None, help="Override validate.nms_threshold.")
    parser.add_argument("--final-score-mode", default=None)
    parser.add_argument("--quality-score-alpha", type=float, default=None)
    parser.add_argument("--quality-score-beta", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Defaults to config seed, then 42.")
    parser.add_argument("--max-cases", type=int, default=None, help="Only evaluate the first N cases.")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-md", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    args = resolve_audit_options(args, config)
    audit_candidates(args)


if __name__ == "__main__":
    main()
