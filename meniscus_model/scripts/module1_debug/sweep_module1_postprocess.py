"""Sweep Module 1 detection post-processing thresholds on cached predictions."""

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
from scripts.module1_debug.runtime_utils import limit_dataset, set_diagnostic_seed, write_resolved_run_config
from utils.checkpoint import load_checkpoint
from utils.metrics import (
    detection_filter_count_sums,
    filter_module1_predictions,
    match_final_predictions,
    precision_recall_specificity_f1,
)


warnings.filterwarnings("ignore", message="Invalid value for VR UI:*")

DEFAULT_EXPERIMENTS = [
    "paper_reproduction=configs/module1_meniscovit_cop.yaml:checkpoints/paper_reproduction/module1/best.pt",
]
DEFAULT_SCORE_THRESHOLDS = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
DEFAULT_TOPK: list[int | None] = [None]
DEFAULT_NMS = [0.3]
IOU_THRESHOLDS = [0.1, 0.3, 0.45]
CSV_COLUMNS = [
    "experiment",
    "checkpoint",
    "score_threshold",
    "topk_per_image",
    "nms_threshold",
    "num_final_predictions",
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


def parse_topk_list(value: str | list[int | None]) -> list[int | None]:
    if isinstance(value, list):
        return [None if item is None else int(item) for item in value]
    parsed: list[int | None] = []
    for item in str(value).split(","):
        normalized = item.strip().lower()
        if not normalized:
            continue
        parsed.append(None if normalized in {"all", "none", "null"} else int(normalized))
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


@torch.inference_mode()
def collect_prediction_batches(
    config: dict[str, Any],
    checkpoint: str,
    index_jsonl: str | None,
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
        cached.append({"predictions": detach_predictions(predictions), "labels": batch["labels"]})
    return cached


def evaluate_cached(
    cached_batches: list[dict[str, Any]],
    score_threshold: float,
    topk_per_image: int | None,
    nms_threshold: float,
    final_score_mode: str = "objectness",
    quality_score_alpha: float = 1.0,
    quality_score_beta: float = 1.0,
) -> dict[str, float]:
    counts_by_threshold = {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in IOU_THRESHOLDS}
    num_final_predictions = 0
    for item in cached_batches:
        predictions = item["predictions"]
        labels = item["labels"]
        final_predictions = filter_module1_predictions(
            predictions,
            score_threshold=float(score_threshold),
            topk_per_image=topk_per_image,
            nms_threshold=float(nms_threshold),
            final_score_mode=final_score_mode,
            quality_score_alpha=quality_score_alpha,
            quality_score_beta=quality_score_beta,
        )
        num_final_predictions += int(detection_filter_count_sums(final_predictions)["num_final_predictions_for_matching"])
        for threshold in IOU_THRESHOLDS:
            counts = match_final_predictions(
                final_predictions,
                labels,
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


def run_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    seed = int(getattr(args, "seed", 42))
    max_cases = getattr(args, "max_cases", None)
    set_diagnostic_seed(seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_path) if args.csv_path else output_dir / "postprocess_sweep.csv"
    md_path = Path(args.markdown_path) if args.markdown_path else output_dir / "postprocess_sweep.md"
    score_thresholds = parse_float_list(args.score_thresholds)
    topk_values = parse_topk_list(args.topk_per_image)
    nms_values = parse_float_list(args.nms_thresholds)
    experiments = [parse_experiment(spec) for spec in args.experiment]
    resolved_experiments: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        set_diagnostic_seed(seed)
        print(f"collecting predictions for {experiment['name']} from {experiment['checkpoint']}")
        config = load_config(experiment["config"])
        val_cfg = config.get("validate", {})
        data_cfg = config["data"]
        resolved_experiments.append(
            {
                **experiment,
                "index_jsonl": str(args.index_jsonl or data_cfg["val_index"]),
                "slice_selection": data_cfg.get("val_slice_selection", data_cfg.get("slice_selection")),
                "final_score_mode": val_cfg.get("final_score_mode", "objectness"),
                "quality_score_alpha": val_cfg.get("quality_score_alpha", 1.0),
                "quality_score_beta": val_cfg.get("quality_score_beta", 1.0),
                "source_config": config,
            }
        )
        cached_batches = collect_prediction_batches(
            config,
            experiment["checkpoint"],
            args.index_jsonl,
            max_cases=max_cases,
        )
        for score_threshold, topk_per_image, nms_threshold in product(score_thresholds, topk_values, nms_values):
            metrics = evaluate_cached(
                cached_batches,
                score_threshold=float(score_threshold),
                topk_per_image=topk_per_image,
                nms_threshold=float(nms_threshold),
                final_score_mode=val_cfg.get("final_score_mode", "objectness"),
                quality_score_alpha=val_cfg.get("quality_score_alpha", 1.0),
                quality_score_beta=val_cfg.get("quality_score_beta", 1.0),
            )
            row: dict[str, Any] = {
                "experiment": experiment["name"],
                "checkpoint": experiment["checkpoint"],
                "score_threshold": float(score_threshold),
                "topk_per_image": "all" if topk_per_image is None else int(topk_per_image),
                "nms_threshold": float(nms_threshold),
                "num_final_predictions": int(metrics["num_final_predictions"]),
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
    md_path.write_text(render_ranking_markdown(rows), encoding="utf-8")
    write_resolved_run_config(
        output_dir / "postprocess_sweep_run_config.json",
        {
            "diagnostic": "postprocess_sweep",
            "seed": seed,
            "max_cases": max_cases,
            "index_jsonl_override": args.index_jsonl,
            "score_thresholds": score_thresholds,
            "topk_per_image": topk_values,
            "nms_thresholds": nms_values,
            "experiments": resolved_experiments,
        },
    )
    print(f"saved postprocess sweep CSV: {csv_path}")
    print(f"saved postprocess sweep Markdown: {md_path}")
    return rows


def render_ranking_markdown(rows: list[dict[str, Any]], limit: int = 30) -> str:
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["f1@0.45"]),
            -float(row["precision@0.45"]),
            -float(row["recall@0.45"]),
            int(row["num_final_predictions"]),
        ),
    )
    lines = [
        "# Module 1 Postprocess Sweep",
        "",
        "Sorted by `f1@0.45`, matching the manuscript's max-F1 rule and IoU>0.45 TP definition.",
        "",
        "| rank | experiment | score_threshold | topk | nms | precision@0.45 | recall@0.45 | f1@0.45 | final preds | f1@0.3 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked[: int(limit)], start=1):
        lines.append(
            "| "
            f"{rank} | "
            f"{row['experiment']} | "
            f"{float(row['score_threshold']):g} | "
            f"{row['topk_per_image']} | "
            f"{float(row['nms_threshold']):g} | "
            f"{float(row['precision@0.45']):.4f} | "
            f"{float(row['recall@0.45']):.4f} | "
            f"{float(row['f1@0.45']):.4f} | "
            f"{int(row['num_final_predictions'])} | "
            f"{float(row['f1@0.3']):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        help="Experiment spec name=config:checkpoint. Defaults to task_aligned and two_stage_quality_iou.",
    )
    parser.add_argument("--index-jsonl", default=None, help="Optional JSONL split override for all experiments.")
    parser.add_argument("--output-dir", default="outputs/module1_debug")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--markdown-path", default=None)
    parser.add_argument("--score-thresholds", default=",".join(str(value) for value in DEFAULT_SCORE_THRESHOLDS))
    parser.add_argument("--topk-per-image", default="all", help="Comma-separated integers or 'all'.")
    parser.add_argument("--nms-thresholds", default=",".join(str(value) for value in DEFAULT_NMS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cases", type=int, default=None, help="Only evaluate the first N cases.")
    args = parser.parse_args()
    if args.experiment is None:
        args.experiment = DEFAULT_EXPERIMENTS
    run_sweep(args)


if __name__ == "__main__":
    main()
