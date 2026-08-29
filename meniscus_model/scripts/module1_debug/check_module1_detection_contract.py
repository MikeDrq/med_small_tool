"""Check Module 1 bbox, slice, threshold, and positive-target contracts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from datasets.meniscus_detection_dataset import MeniscusDetectionDataset, read_jsonl
from engine.validate_module1 import build_module1_model, module1_collate, move_batch_to_device
from losses.detection_loss import DetectionTargetConfig, build_module1_targets
from utils.checkpoint import load_checkpoint
from utils.metrics import detection_filter_count_sums, detection_match_counts, filter_module1_predictions


warnings.filterwarnings("ignore", message="Invalid value for VR UI:*")


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _find_record(records: list[dict[str, Any]], case_id: str | None) -> tuple[int, dict[str, Any]]:
    if case_id is None:
        return 0, records[0]
    for idx, record in enumerate(records):
        if str(record.get("case_id")) == str(case_id):
            return idx, record
    raise ValueError(f"case_id={case_id} not found")


def _expected_processed_slices(
    source_slice: int,
    sampled_source_indices: list[int],
    unmatched_label_policy: str = "drop",
) -> list[int]:
    exact = [idx for idx, value in enumerate(sampled_source_indices) if int(value) == int(source_slice)]
    if exact:
        return exact
    policy = str(unmatched_label_policy).lower()
    if policy == "drop":
        return []
    if policy == "error":
        raise ValueError(f"source label slice {source_slice} is not sampled: {sampled_source_indices}")
    if policy != "nearest":
        raise ValueError(f"unsupported unmatched_label_policy={unmatched_label_policy!r}")
    if not sampled_source_indices:
        raise ValueError("cannot use unmatched_label_policy='nearest' with no sampled source slices")
    distances = [abs(int(value) - int(source_slice)) for value in sampled_source_indices]
    return [int(min(range(len(distances)), key=lambda idx: distances[idx]))]


def _box_stats(boxes: list[list[float]] | torch.Tensor) -> dict[str, Any]:
    if isinstance(boxes, torch.Tensor):
        tensor = boxes.detach().float().cpu().reshape(-1, 4)
    elif boxes:
        tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    else:
        return {"count": 0, "min": None, "max": None, "invalid_order": 0, "outside_unit": 0}
    invalid_order = int(((tensor[:, 2] < tensor[:, 0]) | (tensor[:, 3] < tensor[:, 1])).sum().item())
    outside_unit = int(((tensor < 0.0) | (tensor > 1.0)).any(dim=1).sum().item())
    return {
        "count": int(tensor.shape[0]),
        "min": [float(value) for value in tensor.min(dim=0).values.tolist()],
        "max": [float(value) for value in tensor.max(dim=0).values.tolist()],
        "invalid_order": invalid_order,
        "outside_unit": outside_unit,
    }


def _label_bbox_report(labels: list[dict[str, Any]], sample: dict[str, Any]) -> dict[str, Any]:
    plane_side_counts = Counter(f"{label['plane']}:{label['meniscus_side']}" for label in labels)
    report: dict[str, Any] = {"count": len(labels), "invalid": [], "by_plane_side": dict(plane_side_counts)}
    for idx, label in enumerate(labels):
        plane = label["plane"]
        _, _, height, width = sample[f"{plane}_volume"].shape
        x1, y1, x2, y2 = [float(value) for value in label["bbox"]]
        issues = []
        if x2 < x1 or y2 < y1:
            issues.append("not_xyxy_ordered")
        if min(x1, y1, x2, y2) < 0 or x2 > width or y2 > height:
            issues.append("outside_processed_image")
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
            issues.append("looks_normalized_but_dataset_should_be_pixel_xyxy")
        if issues:
            report["invalid"].append({"index": idx, "plane": plane, "slice_index": int(label["slice_index"]), "bbox": label["bbox"], "issues": issues})
    return report


def _slice_mapping_report(raw_record: dict[str, Any], processed_labels: list[dict[str, Any]], sample: dict[str, Any]) -> dict[str, Any]:
    mapping_summary = sample.get("labels", {}).get("module1_label_mapping", {})
    unmatched_label_policy = str(mapping_summary.get("unmatched_label_policy", "drop"))
    active_source_label_indices = mapping_summary.get("active_source_label_indices")
    active_index_set = None if active_source_label_indices is None else set(active_source_label_indices)
    actual: dict[tuple[str, int, str, tuple[float, ...]], set[int]] = defaultdict(set)
    for label in processed_labels:
        rounded_box = tuple(round(float(value), 4) for value in label["bbox"])
        key = (label["plane"], int(label.get("source_slice_index", label["slice_index"])), label["meniscus_side"], rounded_box)
        actual[key].add(int(label["slice_index"]))

    mismatches = []
    expanded_sources = []
    for raw_label_index, raw_label in enumerate(raw_record.get("module1_labels", [])):
        if active_index_set is not None and raw_label_index not in active_index_set:
            continue
        plane = raw_label["plane"]
        source_slice = int(raw_label["slice_index"])
        side = raw_label["meniscus_side"]
        sampled = [int(value) for value in sample[f"{plane}_metadata"]["sampled_source_indices"]]
        expected = _expected_processed_slices(
            source_slice,
            sampled,
            unmatched_label_policy=unmatched_label_policy,
        )
        if len(expected) > 1:
            expanded_sources.append({"plane": plane, "side": side, "source_slice": source_slice, "processed_slices": expected})
        candidates = [
            sorted(values)
            for (actual_plane, actual_source, actual_side, _), values in actual.items()
            if actual_plane == plane and actual_source == source_slice and actual_side == side
        ]
        mapping_matches = not candidates if not expected else sorted(expected) in candidates
        if not mapping_matches:
            mismatches.append({"plane": plane, "side": side, "source_slice": source_slice, "expected": expected, "actual_candidates": candidates})
    return {
        "unmatched_label_policy": unmatched_label_policy,
        "label_mapping_summary": mapping_summary,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "expanded_source_slice_count": len(expanded_sources),
        "expanded_source_slices": expanded_sources[:20],
    }


def _token_plane_side_counts(targets: dict[str, torch.Tensor], shape_metadata: dict[str, Any]) -> dict[str, int]:
    sagittal_tokens = int(shape_metadata["sagittal"]["slices"] * shape_metadata["sagittal"]["patches_per_slice"])
    side_names = {0: "medial", 1: "lateral"}
    counts: Counter[str] = Counter()
    for global_idx in torch.nonzero(targets["positive_mask"][0], as_tuple=False).flatten().cpu().tolist():
        plane = "sagittal" if global_idx < sagittal_tokens else "coronal"
        side_id = int(targets["side_targets"][0, global_idx].item())
        counts[f"{plane}:{side_names.get(side_id, str(side_id))}"] += 1
    return dict(counts)


def _threshold_report(
    predictions: dict[str, torch.Tensor | dict[str, Any]],
    labels: list[dict[str, Any]],
    thresholds: list[float],
    target_config: DetectionTargetConfig,
    topk_per_image: int | None,
    nms_threshold: float | None,
) -> list[dict[str, Any]]:
    scores = torch.sigmoid(predictions["objectness_logits"][0]).detach().float().cpu()
    side_prob = torch.softmax(predictions["side_logits"][0], dim=-1).detach().float().cpu()
    max_side_prob = side_prob.max(dim=-1).values
    combined_scores = scores * max_side_prob
    rows = []
    for threshold in thresholds:
        counts = detection_match_counts(
            predictions,
            labels,
            score_threshold=float(threshold),
            iou_threshold=0.3,
            topk_per_image=topk_per_image,
            nms_threshold=nms_threshold,
            target_config=target_config,
        )
        final_predictions = filter_module1_predictions(
            predictions,
            score_threshold=float(threshold),
            topk_per_image=topk_per_image,
            nms_threshold=nms_threshold,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "objectness_tokens_ge": int((scores >= threshold).sum().item()),
                "roi_combined_tokens_ge": int((combined_scores >= threshold).sum().item()),
                "score_max": float(scores.max().item()),
                "roi_combined_max": float(combined_scores.max().item()),
                "final_filter_counts": detection_filter_count_sums(final_predictions),
                "match_counts_iou_0.3": counts,
            }
        )
    return rows


@torch.inference_mode()
def check_contract(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    data_cfg = config["data"]
    index_jsonl = Path(args.index_jsonl or data_cfg["train_index"])
    raw_records = read_jsonl(index_jsonl)
    record_idx, raw_record = _find_record(raw_records, args.case_id)
    dataset = MeniscusDetectionDataset(
        index_jsonl=index_jsonl,
        output_size=data_cfg["output_size"],
        target_slices=data_cfg["num_slices"],
        patch_size=data_cfg["patch_size"],
        slice_selection=data_cfg.get("train_slice_selection", data_cfg.get("slice_selection")),
        transform=None,
    )
    dataset_indices = dataset.dataset_indices_for_record(record_idx)
    if not dataset_indices:
        raise RuntimeError(f"record_index={record_idx} has no Dataset sample plan")
    sample = dataset[dataset_indices[0]]
    batch = module1_collate([sample])
    device_name = config.get("device", "cuda")
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_module1_model(config).to(device)
    checkpoint = args.checkpoint
    if checkpoint:
        metadata = load_checkpoint(checkpoint, model, map_location=device)
    else:
        metadata = {}
    model.eval()
    batch = move_batch_to_device(batch, device)
    predictions = model(batch["sagittal_volume"], batch["coronal_volume"], batch["sagittal_coords"], batch["coronal_coords"])

    radius = int(config.get("loss", {}).get("positive_radius_tokens", 0))
    target_config = DetectionTargetConfig(positive_radius_tokens=radius)
    targets = build_module1_targets(batch["labels"], predictions["shape_metadata"], device=device, target_config=target_config)
    targets_radius_0 = build_module1_targets(batch["labels"], predictions["shape_metadata"], device=device, target_config=DetectionTargetConfig(positive_radius_tokens=0))
    positive_count = int(targets["positive_mask"].sum().item())
    positive_count_radius_0 = int(targets_radius_0["positive_mask"].sum().item())
    negative_count = int((~targets["positive_mask"]).sum().item())
    positive_boxes = targets["bbox_targets"][targets["positive_mask"]]

    validate_threshold = float(config.get("validate", {}).get("score_threshold", 0.5))
    thresholds = sorted(set([validate_threshold, 0.1, 0.3, 0.5, float(args.score_threshold)]))

    report = {
        "case_id": sample["case_id"],
        "sample_id": sample.get("sample_id"),
        "record_index": sample.get("record_index"),
        "chunk_index": sample.get("chunk_index"),
        "chunk_count": sample.get("chunk_count"),
        "index_jsonl": str(index_jsonl),
        "config": str(args.config),
        "checkpoint": checkpoint,
        "checkpoint_metadata": metadata,
        "dataset_bbox_contract": _label_bbox_report(sample["labels"]["module1_labels"], sample),
        "slice_mapping_contract": _slice_mapping_report(raw_record, sample["labels"]["module1_labels"], sample),
        "target_contract": {
            "positive_radius_tokens": radius,
            "positive_count_radius_0": positive_count_radius_0,
            "positive_count_radius_config": positive_count,
            "negative_count_radius_config": negative_count,
            "positive_by_plane_side": _token_plane_side_counts(targets, predictions["shape_metadata"]),
            "positive_bbox_norm_stats": _box_stats(positive_boxes),
        },
        "model_prediction_contract": {
            "bbox_pred_norm_stats": _box_stats(predictions["bbox_pred"]),
            "objectness_logit_min": float(predictions["objectness_logits"].detach().float().min().cpu().item()),
            "objectness_logit_max": float(predictions["objectness_logits"].detach().float().max().cpu().item()),
        },
        "threshold_contract": _threshold_report(
            predictions,
            batch["labels"],
            thresholds,
            target_config,
            topk_per_image=config.get("validate", {}).get("topk_per_image"),
            nms_threshold=config.get("validate", {}).get("nms_threshold"),
        ),
    }

    print(f"case_id={report['case_id']} checkpoint={checkpoint or '<random init>'}")
    dataset_invalid = len(report["dataset_bbox_contract"]["invalid"])
    print(f"bbox dataset pixel xyxy invalid={dataset_invalid} labels={report['dataset_bbox_contract']['count']}")
    mapping_report = report["slice_mapping_contract"]
    print(
        f"slice mapping policy={mapping_report['unmatched_label_policy']} "
        f"mismatches={mapping_report['mismatch_count']} "
        f"expanded_source_slices={mapping_report['expanded_source_slice_count']} "
        f"dropped={mapping_report['label_mapping_summary'].get('dropped_source_label_count', 0)}"
    )
    print(
        "target positives "
        f"radius_0={positive_count_radius_0} radius_{radius}={positive_count} negatives={negative_count} "
        f"by_plane_side={report['target_contract']['positive_by_plane_side']}"
    )
    target_stats = report["target_contract"]["positive_bbox_norm_stats"]
    pred_stats = report["model_prediction_contract"]["bbox_pred_norm_stats"]
    print(f"target bbox normalized outside_unit={target_stats['outside_unit']} invalid_order={target_stats['invalid_order']}")
    print(f"model bbox normalized outside_unit={pred_stats['outside_unit']} invalid_order={pred_stats['invalid_order']}")
    for row in report["threshold_contract"]:
        counts = row["match_counts_iou_0.3"]
        final_counts = row["final_filter_counts"]
        print(
            f"threshold={row['threshold']:.3f} objectness_ge={row['objectness_tokens_ge']} "
            f"roi_combined_ge={row['roi_combined_tokens_ge']} max_obj={row['score_max']:.4f} "
            f"max_roi={row['roi_combined_max']:.4f} final={final_counts['num_final_predictions_for_matching']} "
            f"tp={counts['tp']} fp={counts['fp']} fn={counts['fn']}"
        )
    if positive_count == 0:
        print("WARN: positive assignment is zero")
    if all(row["roi_combined_tokens_ge"] == 0 for row in report["threshold_contract"] if row["threshold"] >= validate_threshold):
        print("WARN: configured/greater score thresholds produce no ROI candidates")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "detection_contract.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved JSON report: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/module1_meniscovit_cop.yaml")
    parser.add_argument("--index-jsonl", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/module1/best.pt")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Extra threshold to include in the report.")
    parser.add_argument("--output-dir", default="outputs/module1_detection_contract")
    args = parser.parse_args()
    check_contract(args)


if __name__ == "__main__":
    main()
