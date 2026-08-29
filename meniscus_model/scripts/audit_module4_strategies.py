"""Audit Module 4 slice aggregation with GT and Module 1 boxes in one U2Net pass."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from datasets.dicom_reader import read_dicom_series
from engine.validate_module4 import load_config, read_jsonl
from models.module4_rule_based import (
    PAIRED_WIDTH_AGGREGATION_STRATEGIES,
    PairedWidthMeasurement,
    WidthMeasurement,
    paired_width_candidates_from_mask_and_roi_cache,
    select_paired_width_candidate,
)
from models.tibia_segmentation import CoronalTibiaU2Net
from utils.logging import format_metrics, get_logger
from utils.metrics import binary_task_metrics, binary_task_metrics_with_ci


def _load_roi_payload(cache_dir: str | Path, case_id: str) -> dict[str, Any] | None:
    path = Path(cache_dir) / f"{case_id}.pt"
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _candidate_to_dict(candidate: PairedWidthMeasurement) -> dict[str, Any]:
    return {
        "slice_index": candidate.meniscus.slice_index,
        "meniscus_width": candidate.meniscus.width,
        "tibia_width": candidate.tibia.width,
        "beta": candidate.beta,
        "meniscus_source": candidate.meniscus.source,
        "tibia_source": candidate.tibia.source,
        "meniscus_bbox_xyxy": candidate.meniscus.bbox_xyxy,
        "tibia_bbox_xyxy": candidate.tibia.bbox_xyxy,
        "unit": candidate.tibia.unit,
    }


def _candidate_from_dict(item: dict[str, Any]) -> PairedWidthMeasurement:
    return PairedWidthMeasurement(
        meniscus=WidthMeasurement(
            width=float(item["meniscus_width"]),
            source=str(item["meniscus_source"]),
            unit=str(item.get("unit", "mm")),
            slice_index=int(item["slice_index"]),
            bbox_xyxy=None
            if item.get("meniscus_bbox_xyxy") is None
            else tuple(float(value) for value in item["meniscus_bbox_xyxy"]),
        ),
        tibia=WidthMeasurement(
            width=float(item["tibia_width"]),
            source=str(item["tibia_source"]),
            unit=str(item.get("unit", "mm")),
            slice_index=int(item["slice_index"]),
            bbox_xyxy=None
            if item.get("tibia_bbox_xyxy") is None
            else tuple(float(value) for value in item["tibia_bbox_xyxy"]),
        ),
    )


def extract_case_widths(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    gt_roi_cache_dir: str | Path,
    pred_roi_cache_dir: str | Path,
    output_path: Path,
    max_cases: int | None = None,
) -> list[dict[str, Any]]:
    """Run U2Net once per case and save all same-slice width candidates."""

    logger = get_logger("audit_module4_strategies")
    segmenter_cfg = config["module4"]["tibia_segmenter"]
    requested_device = str(config.get("pipeline", {}).get("device", "cuda"))
    device = torch.device(
        requested_device
        if requested_device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    segmenter = CoronalTibiaU2Net.from_config(segmenter_cfg, device=device)
    logger.info(f"device={device} tibia_source={segmenter.source}")

    pipeline_cfg = config.get("pipeline", {})
    module4_cfg = config["module4"]
    image_width = int(module4_cfg.get("image_width") or pipeline_cfg.get("input_size", 512))
    limit = len(records) if max_cases is None else min(len(records), int(max_cases))
    case_rows: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records[:limit], start=1):
            label = record.get("module4_labels", {}).get("lateral_discoid")
            if label is None:
                continue
            series = read_dicom_series(
                record["coronal_dicom_dir"],
                output_size=image_width,
                target_slices=int(pipeline_cfg.get("num_slices", 32)),
                slice_selection=pipeline_cfg.get("slice_selection"),
            )
            spacing_col = float(series.metadata["slices"][0].get("PixelSpacing", [1.0, 1.0])[1])
            volume = torch.from_numpy(series.volume).unsqueeze(0).float()
            mask = segmenter.predict_mask(volume)
            row: dict[str, Any] = {
                "case_id": str(record["case_id"]),
                "group_id": str(record.get("patient_group_id") or record["case_id"]),
                "label": int(label),
                "sampled_source_indices": series.metadata.get("sampled_source_indices", []),
                "sources": {},
            }
            for source_name, cache_dir in (
                ("gt", gt_roi_cache_dir),
                ("pred", pred_roi_cache_dir),
            ):
                payload = _load_roi_payload(cache_dir, row["case_id"])
                candidates = [] if payload is None else paired_width_candidates_from_mask_and_roi_cache(
                    mask,
                    payload,
                    image_width=image_width,
                    pixel_spacing_col=spacing_col,
                    tibia_source=segmenter.source,
                    min_slice_pixels=int(module4_cfg.get("min_tibia_slice_pixels", 1)),
                )
                row["sources"][source_name] = [_candidate_to_dict(item) for item in candidates]
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            case_rows.append(row)
            if index % 25 == 0 or index == limit:
                logger.info(f"extracted={index}/{limit} latest_case={row['case_id']}")
    return case_rows


def load_case_widths(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict) and "low" in value and "high" in value:
            output[f"{key}_low"] = value["low"]
            output[f"{key}_high"] = value["high"]
        else:
            output[key] = value
    return output


def evaluate_strategies(
    case_rows: list[dict[str, Any]],
    strategies: list[str],
    plateau_fractions: list[float],
    threshold: float,
    min_candidate_slices: int,
    central_window_radius: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    total_cases = len(case_rows)
    for source_name in ("gt", "pred"):
        for fraction in plateau_fractions:
            for strategy in strategies:
                labels: list[int] = []
                scores: list[float] = []
                predictions: list[int] = []
                group_ids: list[str] = []
                selected_rows: list[dict[str, Any]] = []
                for case in case_rows:
                    candidates = [
                        _candidate_from_dict(item)
                        for item in case["sources"].get(source_name, [])
                    ]
                    selected = select_paired_width_candidate(
                        candidates,
                        aggregation_strategy=strategy,
                        min_plateau_width_fraction=fraction,
                        min_candidate_slices=min_candidate_slices,
                        central_window_radius=central_window_radius,
                    )
                    if selected is None:
                        continue
                    label = int(case["label"])
                    beta = float(selected.beta)
                    prediction = int(beta > threshold)
                    labels.append(label)
                    scores.append(beta)
                    predictions.append(prediction)
                    group_ids.append(str(case["group_id"]))
                    selected_rows.append(
                        {
                            "case_id": case["case_id"],
                            "group_id": case["group_id"],
                            "label": label,
                            "box_source": source_name,
                            "strategy": strategy,
                            "plateau_fraction": fraction,
                            "threshold": threshold,
                            "beta": beta,
                            "predicted_label": prediction,
                            "measurement_slice_index": selected.meniscus.slice_index,
                            "meniscus_width": selected.meniscus.width,
                            "tibia_width": selected.tibia.width,
                            "candidate_count": selected.candidate_count,
                            "eligible_candidate_count": selected.eligible_candidate_count,
                        }
                    )
                if bootstrap_samples > 0:
                    metrics = binary_task_metrics_with_ci(
                        labels,
                        scores,
                        threshold=threshold,
                        n_boot=bootstrap_samples,
                        seed=seed,
                        group_ids=group_ids,
                        predictions=predictions,
                    )
                else:
                    metrics = binary_task_metrics(
                        labels,
                        scores,
                        threshold=threshold,
                        predictions=predictions,
                    )
                row = {
                    "box_source": source_name,
                    "strategy": strategy,
                    "plateau_fraction": float(fraction),
                    "threshold": threshold,
                    "total_cases": total_cases,
                    "evaluated_cases": len(labels),
                    "skipped_cases": total_cases - len(labels),
                    "coverage": len(labels) / max(total_cases, 1),
                    **_flatten_metrics(metrics),
                }
                metric_rows.append(row)
                prediction_rows.extend(selected_rows)
    return metric_rows, prediction_rows


def save_strategy_outputs(
    output_dir: str | Path,
    metrics: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "strategy_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    fieldnames = list(metrics[0]) if metrics else []
    with (output_dir / "strategy_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    with (output_dir / "strategy_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item) + "\n")

    pred_metrics = [
        item
        for item in metrics
        if item["box_source"] == "pred" and not math.isnan(float(item["roc_auc"]))
    ]
    if pred_metrics:
        selected = max(
            pred_metrics,
            key=lambda item: (float(item["roc_auc"]), float(item["coverage"]), float(item["f1"])),
        )
        recommendation = {
            "selection_split": "validation",
            "selection_box_source": "pred",
            "selection_metric": "roc_auc",
            "strategy": selected["strategy"],
            "plateau_fraction": selected["plateau_fraction"],
            "metrics": selected,
        }
        (output_dir / "selected_strategy.json").write_text(
            json.dumps(recommendation, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline_full.yaml")
    parser.add_argument("--index", default=None)
    parser.add_argument("--gt-roi-cache-dir", default=None)
    parser.add_argument("--pred-roi-cache-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--width-cache",
        default=None,
        help="Optional per_slice_widths.jsonl path; useful with --reuse-width-cache.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=sorted(PAIRED_WIDTH_AGGREGATION_STRATEGIES),
        default=["max_w2", "max_w1", "median_beta", "q25_beta", "central3_median"],
    )
    parser.add_argument("--plateau-fractions", nargs="+", type=float, default=[0.90, 0.95, 1.0])
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold; defaults to module4.threshold in the config.",
    )
    parser.add_argument("--min-candidate-slices", type=int, default=1)
    parser.add_argument("--central-window-radius", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--reuse-width-cache",
        action="store_true",
        help="Skip DICOM/U2Net and recompute metrics from per_slice_widths.jsonl.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    width_cache = Path(args.width_cache) if args.width_cache else output_dir / "per_slice_widths.jsonl"
    if args.reuse_width_cache:
        case_rows = load_case_widths(width_cache)
    else:
        if not args.index or not args.gt_roi_cache_dir or not args.pred_roi_cache_dir:
            parser.error(
                "--index, --gt-roi-cache-dir and --pred-roi-cache-dir are required "
                "unless --reuse-width-cache is used"
            )
        records = read_jsonl(args.index)
        case_rows = extract_case_widths(
            records,
            config,
            gt_roi_cache_dir=args.gt_roi_cache_dir,
            pred_roi_cache_dir=args.pred_roi_cache_dir,
            output_path=width_cache,
            max_cases=args.max_cases,
        )
    metrics, predictions = evaluate_strategies(
        case_rows,
        strategies=args.strategies,
        plateau_fractions=args.plateau_fractions,
        threshold=float(
            config["module4"].get("threshold", 0.4)
            if args.threshold is None
            else args.threshold
        ),
        min_candidate_slices=args.min_candidate_slices,
        central_window_radius=args.central_window_radius,
        bootstrap_samples=args.bootstrap,
        seed=args.seed,
    )
    save_strategy_outputs(output_dir, metrics, predictions)
    selected_path = output_dir / "selected_strategy.json"
    if selected_path.exists():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        get_logger("audit_module4_strategies").info(
            "selected "
            + format_metrics(
                {
                    "plateau_fraction": float(selected["plateau_fraction"]),
                    "roc_auc": float(selected["metrics"]["roc_auc"]),
                    "f1": float(selected["metrics"]["f1"]),
                    "coverage": float(selected["metrics"]["coverage"]),
                }
            )
            + f" strategy={selected['strategy']}"
        )


if __name__ == "__main__":
    main()
