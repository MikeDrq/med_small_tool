"""Evaluate Module 1 post-processing with oracle/quality ranking scores.

This is a diagnostic-only script. It does not change the model or loss. The
predicted boxes and side logits stay fixed; only the score used by thresholding,
top-k, NMS ordering, and final matching is replaced.
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
import sys
from typing import Any
import warnings

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.meniscus_detection_dataset import MeniscusDetectionDataset
from engine.validate_module1 import build_module1_model, module1_collate, move_batch_to_device
from losses.detection_loss import (
    DetectionTargetConfig,
    build_module1_targets,
    build_pred_iou_objectness_targets,
    build_quality_iou_objectness_targets,
    build_strict_pred_iou_targets,
)
from utils.checkpoint import load_checkpoint
from scripts.module1_debug.runtime_utils import limit_dataset, set_diagnostic_seed, write_resolved_run_config
from utils.metrics import (
    detection_filter_count_sums,
    filter_module1_predictions,
    match_final_predictions,
    precision_recall_specificity_f1,
    safe_pearson_corr,
    safe_spearman_corr,
)


warnings.filterwarnings("ignore", message="Invalid value for VR UI:*")

DEFAULT_EXPERIMENTS = [
    "module1=configs/module1_meniscovit_cop.yaml:checkpoints/module1/best.pt",
]
DEFAULT_SCORE_MODES = [
    "objectness",
    "oracle_iou_strict",
    "oracle_iou_plane_slice",
    "quality_iou_target",
    "task_alignment_detached",
]
IOU_THRESHOLDS = [0.1, 0.3, 0.45]
CSV_COLUMNS = [
    "experiment",
    "checkpoint",
    "score_mode",
    "score_threshold",
    "topk_per_image",
    "nms_threshold",
    "num_final_predictions",
    "score_iou_spearman_corr",
    "score_iou_pearson_corr",
    "good_iou_gt_0.3_score_mean",
    "bad_iou_lt_0.1_score_mean",
    "precision@0.1",
    "precision@0.3",
    "precision@0.45",
    "recall@0.1",
    "recall@0.3",
    "recall@0.45",
    "f1@0.1",
    "f1@0.3",
    "f1@0.45",
]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_float_list(value: str | list[float]) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_topk_list(value: str | list[int | str]) -> list[int | None]:
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = [item.strip() for item in str(value).split(",") if item.strip()]
    parsed: list[int | None] = []
    for item in raw_items:
        if str(item).lower() in {"all", "none", "null"}:
            parsed.append(None)
        else:
            parsed.append(int(item))
    return parsed


def parse_experiment(spec: str) -> dict[str, str]:
    if "=" not in spec or ":" not in spec:
        raise ValueError("Experiment spec must be name=config:checkpoint")
    name, remainder = spec.split("=", 1)
    config, checkpoint = remainder.split(":", 1)
    return {"name": name, "config": config, "checkpoint": checkpoint}


def make_loader(config: dict[str, Any], index_jsonl: str | None, max_cases: int | None = None) -> DataLoader:
    data_cfg = config["data"]
    dataset = MeniscusDetectionDataset(
        index_jsonl=index_jsonl or data_cfg["val_index"],
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


def detach_predictions(predictions: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach() if isinstance(value, torch.Tensor) else value
        for key, value in predictions.items()
    }


def replace_scores_with_logits(predictions: dict[str, Any], scores: torch.Tensor) -> dict[str, Any]:
    clamped = scores.detach().float().clamp(1e-6, 1.0 - 1e-6)
    scored = dict(predictions)
    scored["objectness_logits"] = torch.logit(clamped).to(device=predictions["objectness_logits"].device)
    return scored


def score_mode_tensors(
    predictions: dict[str, Any],
    labels: list[dict[str, Any]],
    config: dict[str, Any],
    score_modes: list[str],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, float]]]:
    device = predictions["objectness_logits"].device
    loss_cfg = config.get("loss", {})
    target_config = DetectionTargetConfig(positive_radius_tokens=int(loss_cfg.get("positive_radius_tokens", 0)))
    assigned = build_module1_targets(labels, predictions["shape_metadata"], device=device, target_config=target_config)
    objectness_scores = torch.sigmoid(predictions["objectness_logits"]).detach()
    strict_iou = build_strict_pred_iou_targets(
        labels,
        predictions["shape_metadata"],
        predictions["bbox_pred"],
        predictions["side_logits"],
        device=device,
    ).detach()
    plane_slice_iou = build_pred_iou_objectness_targets(
        labels,
        predictions["shape_metadata"],
        predictions["bbox_pred"],
        device=device,
    ).detach()
    quality = build_quality_iou_objectness_targets(
        labels,
        predictions["shape_metadata"],
        predictions["bbox_pred"],
        predictions["side_logits"],
        assigned["objectness"],
        assigned["positive_mask"],
        device=device,
        quality_iou_threshold=float(loss_cfg.get("quality_iou_threshold", 0.3)),
    )["objectness_targets"].detach()
    alpha = float(loss_cfg.get("task_aligned_alpha", 1.0))
    beta = float(loss_cfg.get("task_aligned_beta", 6.0))
    task_alignment = objectness_scores.pow(alpha) * strict_iou.pow(beta)

    available = {
        "objectness": objectness_scores,
        "oracle_iou_strict": strict_iou,
        "oracle_iou_plane_slice": plane_slice_iou,
        "quality_iou_target": quality,
        "task_alignment_detached": task_alignment.detach(),
    }
    if "bbox_quality_logits" in predictions:
        bbox_quality_scores = torch.sigmoid(predictions["bbox_quality_logits"]).detach()
        available["quality"] = bbox_quality_scores
        available["objectness_quality"] = objectness_scores * bbox_quality_scores
        available["task_aligned_score"] = objectness_scores.pow(alpha) * bbox_quality_scores.pow(beta)
    scores: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    strict_flat = strict_iou.detach().cpu().flatten().tolist()
    good_mask = strict_iou >= 0.3
    bad_mask = strict_iou < 0.1
    for mode in score_modes:
        if mode not in available:
            raise ValueError(f"Unknown score mode: {mode}")
        mode_scores = available[mode].detach()
        scores[mode] = mode_scores
        mode_flat = mode_scores.cpu().flatten().tolist()
        diagnostics[mode] = {
            "score_iou_spearman_corr": safe_spearman_corr(mode_flat, strict_flat),
            "score_iou_pearson_corr": safe_pearson_corr(mode_flat, strict_flat),
            "good_iou_gt_0.3_score_mean": float(mode_scores[good_mask].mean().item()) if good_mask.any() else 0.0,
            "bad_iou_lt_0.1_score_mean": float(mode_scores[bad_mask].mean().item()) if bad_mask.any() else 0.0,
        }
    return scores, diagnostics


@torch.inference_mode()
def collect_scored_batches(
    config: dict[str, Any],
    checkpoint: str,
    index_jsonl: str | None,
    score_modes: list[str],
    max_cases: int | None = None,
) -> list[dict[str, Any]]:
    device_name = config.get("device", "cuda")
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_module1_model(config).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    loader = make_loader(config, index_jsonl, max_cases=max_cases)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    amp = bool(config.get("train", {}).get("amp", True)) and device.type == "cuda"
    cached: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp):
            predictions = model(
                batch["sagittal_volume"],
                batch["coronal_volume"],
                batch["sagittal_coords"],
                batch["coronal_coords"],
            )
        scores, diagnostics = score_mode_tensors(predictions, batch["labels"], config, score_modes)
        cached.append(
            {
                "predictions": detach_predictions(predictions),
                "labels": batch["labels"],
                "scores": {mode: value.detach() for mode, value in scores.items()},
                "diagnostics": diagnostics,
            }
        )
    return cached


def aggregate_score_diagnostics(cached_batches: list[dict[str, Any]], score_mode: str) -> dict[str, float]:
    keys = (
        "score_iou_spearman_corr",
        "score_iou_pearson_corr",
        "good_iou_gt_0.3_score_mean",
        "bad_iou_lt_0.1_score_mean",
    )
    values = {key: [] for key in keys}
    for item in cached_batches:
        diag = item["diagnostics"][score_mode]
        for key in keys:
            values[key].append(float(diag[key]))
    return {key: sum(vals) / max(len(vals), 1) for key, vals in values.items()}


def evaluate_cached(
    cached_batches: list[dict[str, Any]],
    score_mode: str,
    score_threshold: float,
    topk_per_image: int | None,
    nms_threshold: float,
) -> dict[str, float]:
    counts_by_threshold = {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in IOU_THRESHOLDS}
    num_final_predictions = 0
    for item in cached_batches:
        predictions = replace_scores_with_logits(item["predictions"], item["scores"][score_mode])
        final_predictions = filter_module1_predictions(
            predictions,
            score_threshold=float(score_threshold),
            topk_per_image=topk_per_image,
            nms_threshold=float(nms_threshold),
        )
        num_final_predictions += int(detection_filter_count_sums(final_predictions)["num_final_predictions_for_matching"])
        for threshold in IOU_THRESHOLDS:
            counts = match_final_predictions(
                final_predictions,
                item["labels"],
                predictions["shape_metadata"],
                iou_threshold=float(threshold),
            )
            for key in counts_by_threshold[threshold]:
                counts_by_threshold[threshold][key] += int(counts[key])
    metrics: dict[str, float] = {"num_final_predictions": float(num_final_predictions)}
    for threshold in IOU_THRESHOLDS:
        suffix = f"{threshold:g}"
        counts = counts_by_threshold[threshold]
        rates = precision_recall_specificity_f1(
            tp=int(counts["tp"]),
            fp=int(counts["fp"]),
            tn=0,
            fn=int(counts["fn"]),
        )
        metrics[f"precision@{suffix}"] = rates["precision"]
        metrics[f"recall@{suffix}"] = rates["recall"]
        metrics[f"f1@{suffix}"] = rates["f1"]
    return metrics


def run_evaluation(args: argparse.Namespace) -> list[dict[str, Any]]:
    seed = int(getattr(args, "seed", 42))
    max_cases = getattr(args, "max_cases", None)
    set_diagnostic_seed(seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_path) if args.csv_path else output_dir / "oracle_ranking_eval.csv"
    md_path = Path(args.markdown_path) if args.markdown_path else output_dir / "oracle_ranking_eval.md"
    score_modes = [item.strip() for item in str(args.score_modes).split(",") if item.strip()]
    score_thresholds = parse_float_list(args.score_thresholds)
    topk_values = parse_topk_list(args.topk_per_image)
    nms_values = parse_float_list(args.nms_thresholds)
    rows: list[dict[str, Any]] = []
    experiments = [parse_experiment(spec) for spec in args.experiment]
    resolved_experiments: list[dict[str, Any]] = []

    for experiment in experiments:
        set_diagnostic_seed(seed)
        print(f"collecting predictions for {experiment['name']} from {experiment['checkpoint']}")
        config = load_config(experiment["config"])
        data_cfg = config["data"]
        resolved_experiments.append(
            {
                **experiment,
                "index_jsonl": str(args.index_jsonl or data_cfg["val_index"]),
                "slice_selection": data_cfg.get("val_slice_selection", data_cfg.get("slice_selection")),
                "source_config": config,
            }
        )
        cached_batches = collect_scored_batches(
            config,
            experiment["checkpoint"],
            args.index_jsonl,
            score_modes,
            max_cases=max_cases,
        )
        diagnostics_by_mode = {
            mode: aggregate_score_diagnostics(cached_batches, mode)
            for mode in score_modes
        }
        for score_mode, score_threshold, topk_per_image, nms_threshold in product(
            score_modes,
            score_thresholds,
            topk_values,
            nms_values,
        ):
            metrics = evaluate_cached(
                cached_batches,
                score_mode=score_mode,
                score_threshold=float(score_threshold),
                topk_per_image=topk_per_image,
                nms_threshold=float(nms_threshold),
            )
            row: dict[str, Any] = {
                "experiment": experiment["name"],
                "checkpoint": experiment["checkpoint"],
                "score_mode": score_mode,
                "score_threshold": float(score_threshold),
                "topk_per_image": "all" if topk_per_image is None else int(topk_per_image),
                "nms_threshold": float(nms_threshold),
                "num_final_predictions": int(metrics["num_final_predictions"]),
                **diagnostics_by_mode[score_mode],
            }
            for threshold in IOU_THRESHOLDS:
                suffix = f"{threshold:g}"
                row[f"precision@{suffix}"] = float(metrics[f"precision@{suffix}"])
                row[f"recall@{suffix}"] = float(metrics[f"recall@{suffix}"])
                row[f"f1@{suffix}"] = float(metrics[f"f1@{suffix}"])
            rows.append(row)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(render_markdown(rows), encoding="utf-8")
    write_resolved_run_config(
        output_dir / "oracle_ranking_run_config.json",
        {
            "diagnostic": "oracle_ranking",
            "seed": seed,
            "max_cases": max_cases,
            "index_jsonl_override": args.index_jsonl,
            "score_modes": score_modes,
            "score_thresholds": score_thresholds,
            "topk_per_image": ["all" if value is None else value for value in topk_values],
            "nms_thresholds": nms_values,
            "experiments": resolved_experiments,
        },
    )
    print(f"saved oracle ranking CSV: {csv_path}")
    print(f"saved oracle ranking Markdown: {md_path}")
    return rows


def render_markdown(rows: list[dict[str, Any]], limit: int = 40) -> str:
    best_by_mode: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["experiment"]), str(row["score_mode"]))
        current = best_by_mode.get(key)
        if current is None or _ranking_key(row) > _ranking_key(current):
            best_by_mode[key] = row
    ranked = sorted(rows, key=_ranking_key, reverse=True)

    lines = [
        "# Module 1 Oracle Ranking Evaluation",
        "",
        "Diagnostic-only evaluation: predicted boxes and side logits are fixed; only the score used by threshold/top-k/NMS/final matching is replaced.",
        "",
        "## Best By Score Mode",
        "",
        "| experiment | score mode | best recall@0.3 | precision@0.3 | f1@0.3 | threshold | topk | nms | final preds | score-IoU spearman | good score | bad score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(best_by_mode.values(), key=_ranking_key, reverse=True):
        lines.append(
            "| "
            f"{row['experiment']} | "
            f"{row['score_mode']} | "
            f"{float(row['recall@0.3']):.4f} | "
            f"{float(row['precision@0.3']):.4f} | "
            f"{float(row['f1@0.3']):.4f} | "
            f"{float(row['score_threshold']):g} | "
            f"{row['topk_per_image']} | "
            f"{float(row['nms_threshold']):g} | "
            f"{int(row['num_final_predictions'])} | "
            f"{float(row['score_iou_spearman_corr']):.4f} | "
            f"{float(row['good_iou_gt_0.3_score_mean']):.4g} | "
            f"{float(row['bad_iou_lt_0.1_score_mean']):.4g} |"
        )

    lines.extend(
        [
            "",
            "## Top Rows",
            "",
            "| rank | experiment | score mode | threshold | topk | nms | recall@0.3 | precision@0.3 | f1@0.3 | final preds | recall@0.45 |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(ranked[: int(limit)], start=1):
        lines.append(
            "| "
            f"{rank} | "
            f"{row['experiment']} | "
            f"{row['score_mode']} | "
            f"{float(row['score_threshold']):g} | "
            f"{row['topk_per_image']} | "
            f"{float(row['nms_threshold']):g} | "
            f"{float(row['recall@0.3']):.4f} | "
            f"{float(row['precision@0.3']):.4f} | "
            f"{float(row['f1@0.3']):.4f} | "
            f"{int(row['num_final_predictions'])} | "
            f"{float(row['recall@0.45']):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _ranking_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["recall@0.3"]),
        float(row["precision@0.3"]),
        float(row["f1@0.3"]),
        -float(row["num_final_predictions"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        help="Experiment spec name=config:checkpoint. Defaults to task_aligned_5cases.",
    )
    parser.add_argument("--index-jsonl", default=None, help="Optional JSONL split override for all experiments.")
    parser.add_argument("--output-dir", default="outputs/module1_debug")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--markdown-path", default=None)
    parser.add_argument("--score-modes", default=",".join(DEFAULT_SCORE_MODES))
    parser.add_argument("--score-thresholds", default="0.0,0.001,0.005,0.01,0.03,0.1,0.3,0.45")
    parser.add_argument("--topk-per-image", default="100,500,1000")
    parser.add_argument("--nms-thresholds", default="0.3,0.5,0.7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cases", type=int, default=None, help="Only evaluate the first N cases.")
    args = parser.parse_args()
    if args.experiment is None:
        args.experiment = DEFAULT_EXPERIMENTS
    run_evaluation(args)


if __name__ == "__main__":
    main()
