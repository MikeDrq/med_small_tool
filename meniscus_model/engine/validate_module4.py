"""Validation entrypoint for Module 4 rule-based discoid meniscus detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
import torch

from datasets.dicom_reader import read_dicom_series
from models.module4_rule_based import (
    DiscoidMeasurement,
    PAIRED_WIDTH_AGGREGATION_STRATEGIES,
    detect_discoid_meniscus,
    lateral_coronal_gt_meniscus_width,
    lateral_coronal_predicted_meniscus_width_from_roi_cache,
    meniscus_width_from_bbox,
    paired_widths_from_unet_and_roi_cache,
    tibia_width_from_annotation,
    tibia_width_from_mask,
    tibia_width_from_unet,
)
from models.tibia_segmentation import CoronalTibiaU2Net
from utils.bootstrap import grouped_bootstrap_ci
from utils.logging import format_metrics, get_logger
from utils.metrics import precision_recall_specificity_f1, safe_roc_auc
from utils.visualization import save_module4_measurement_plot


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc
    return records


def coronal_pixel_geometry(record: dict[str, Any]) -> tuple[float, int | None]:
    """Read original coronal column spacing and image width."""

    module4 = record.get("module4_labels", {})
    spacing = module4.get("pixel_spacing_col")
    columns = module4.get("image_columns")
    dicom_dir = record.get("coronal_dicom_dir")
    if not dicom_dir:
        return (float(spacing) if spacing is not None else 1.0, None if columns is None else int(columns))
    try:
        import pydicom

        for path in Path(dicom_dir).rglob("*"):
            if not path.is_file():
                continue
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            dicom_spacing = getattr(ds, "PixelSpacing", None)
            if spacing is None and dicom_spacing is not None and len(dicom_spacing) >= 2:
                spacing = float(dicom_spacing[1])
            if columns is None and getattr(ds, "Columns", None) is not None:
                columns = int(ds.Columns)
            if spacing is not None and columns is not None:
                break
    except Exception:
        pass
    return (float(spacing) if spacing is not None else 1.0, None if columns is None else int(columns))


def coronal_pixel_spacing_col(record: dict[str, Any]) -> float:
    """Read original coronal column spacing from JSONL metadata or DICOM tags."""

    return coronal_pixel_geometry(record)[0]


def roi_cache_payload(record: dict[str, Any], roi_cache_dir: str | None) -> dict[str, Any] | None:
    if not roi_cache_dir:
        return None
    path = Path(roi_cache_dir) / f"{record['case_id']}.pt"
    if not path.exists():
        return None
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def select_tibia_width(
    record: dict[str, Any],
    config: dict[str, Any],
    pixel_spacing_col: float,
    tibia_segmenter: CoronalTibiaU2Net | None = None,
):
    module4_cfg = config["module4"]
    for source in module4_cfg.get("tibia_width_priority", ["annotation", "mask", "unet"]):
        if source == "annotation":
            measurement = tibia_width_from_annotation(record)
            if measurement is not None:
                return measurement
        elif source == "mask":
            mask_path = record.get("module4_labels", {}).get("tibia_mask_path")
            if mask_path:
                return tibia_width_from_mask(mask_path, pixel_spacing_col=pixel_spacing_col)
        elif source == "unet":
            if tibia_segmenter is None or not record.get("coronal_dicom_dir"):
                continue
            series = read_dicom_series(record["coronal_dicom_dir"])
            volume = torch.from_numpy(series.volume).unsqueeze(0).float()
            return tibia_width_from_unet(
                volume,
                model=tibia_segmenter,
                pixel_spacing_col=pixel_spacing_col,
                min_slice_pixels=int(module4_cfg.get("min_tibia_slice_pixels", 1)),
            )
        else:
            raise ValueError(f"Unknown tibia width source: {source}")
    return None


def select_meniscus_width(
    record: dict[str, Any],
    config: dict[str, Any],
    pixel_spacing_col: float,
    original_columns: int | None = None,
):
    module4_cfg = config["module4"]
    roi_payload = roi_cache_payload(record, module4_cfg.get("roi_cache_dir"))
    for source in module4_cfg.get("meniscus_width_priority", ["roi_cache", "gt_bbox", "module1_prediction"]):
        if source == "roi_cache" and roi_payload is not None:
            cache_image_width = module4_cfg.get("image_width")
            cache_spacing_col = pixel_spacing_col
            if original_columns is not None and cache_image_width is not None:
                cache_spacing_col *= float(original_columns) / float(cache_image_width)
            measurement = lateral_coronal_predicted_meniscus_width_from_roi_cache(
                roi_payload,
                image_width=cache_image_width,
                pixel_spacing_col=cache_spacing_col,
            )
            if measurement is not None:
                return measurement
        if source == "gt_bbox":
            measurement = lateral_coronal_gt_meniscus_width(record, pixel_spacing_col=pixel_spacing_col)
            if measurement is not None:
                return measurement
        if source == "module1_prediction":
            # Future direct Module 1 inference path. In the current pipeline,
            # predicted boxes are expected to be read from ROI cache.
            continue
        if source == "manual_meniscus_width":
            value = record.get("module4_labels", {}).get("meniscus_width")
            if value is not None:
                return meniscus_width_from_bbox([0.0, 0.0, float(value), 1.0], pixel_spacing_col=1.0, source="manual_meniscus_width")
    return None


def select_paired_unet_roi_widths(
    record: dict[str, Any],
    config: dict[str, Any],
    tibia_segmenter: CoronalTibiaU2Net | None,
):
    """Measure ``w1`` and ``w2`` on the same sampled coronal slice."""

    if tibia_segmenter is None or not record.get("coronal_dicom_dir"):
        return None
    module4_cfg = config["module4"]
    roi_payload = roi_cache_payload(record, module4_cfg.get("roi_cache_dir"))
    if roi_payload is None:
        return None
    pipeline_cfg = config.get("pipeline", {})
    image_width = int(module4_cfg.get("image_width") or pipeline_cfg.get("input_size", 512))
    series = read_dicom_series(
        record["coronal_dicom_dir"],
        output_size=image_width,
        target_slices=int(pipeline_cfg.get("num_slices", 32)),
        slice_selection=pipeline_cfg.get("slice_selection"),
    )
    spacing_col = float(series.metadata["slices"][0].get("PixelSpacing", [1.0, 1.0])[1])
    volume = torch.from_numpy(series.volume).unsqueeze(0).float()
    return paired_widths_from_unet_and_roi_cache(
        volume,
        tibia_segmenter,
        roi_payload,
        image_width=image_width,
        pixel_spacing_col=spacing_col,
        min_slice_pixels=int(module4_cfg.get("min_tibia_slice_pixels", 1)),
        min_plateau_width_fraction=float(module4_cfg.get("min_plateau_width_fraction", 0.95)),
        aggregation_strategy=str(module4_cfg.get("beta_aggregation", "max_w2")),
        min_candidate_slices=int(module4_cfg.get("min_candidate_slices", 1)),
        central_window_radius=int(module4_cfg.get("central_window_radius", 1)),
    )


def evaluate_records(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    bootstrap_samples: int | None = None,
    tibia_segmenter: CoronalTibiaU2Net | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    module4_cfg = config["module4"]
    threshold = float(module4_cfg.get("threshold", 0.4))
    temperature = float(module4_cfg.get("probability_temperature", 0.05))
    results = []
    labels = []
    scores = []
    preds = []
    case_ids: list[str] = []
    group_ids: list[str] = []

    for record in records:
        label = record.get("module4_labels", {}).get("lateral_discoid")
        if label is None:
            continue
        labels_cfg = record.get("module4_labels", {})
        use_paired_unet = (
            labels_cfg.get("tibia_width") is None
            and not labels_cfg.get("tibia_mask_path")
            and "unet" in module4_cfg.get("tibia_width_priority", [])
            and "roi_cache" in module4_cfg.get("meniscus_width_priority", [])
        )
        paired = select_paired_unet_roi_widths(record, config, tibia_segmenter) if use_paired_unet else None
        if paired is not None:
            tibia, meniscus = paired.tibia, paired.meniscus
        else:
            spacing_col, original_columns = coronal_pixel_geometry(record)
            tibia = select_tibia_width(record, config, spacing_col, tibia_segmenter=None if use_paired_unet else tibia_segmenter)
            meniscus = select_meniscus_width(
                record,
                config,
                spacing_col,
                original_columns=original_columns,
            )
        if tibia is None or meniscus is None:
            results.append({"case_id": record.get("case_id"), "status": "skipped_missing_width"})
            continue
        measurement = detect_discoid_meniscus(
            meniscus_width=meniscus.width,
            tibia_width=tibia.width,
            threshold=threshold,
            probability_temperature=temperature,
            meniscus_source=meniscus.source,
            tibia_source=tibia.source,
            unit=tibia.unit,
        )
        item = {
            "case_id": record["case_id"],
            "label": int(label),
            "beta": measurement.beta,
            "probability": measurement.probability,
            "predicted_label": measurement.predicted_label,
            "meniscus_width": measurement.meniscus_width,
            "tibia_width": measurement.tibia_width,
            "meniscus_source": measurement.meniscus_source,
            "tibia_source": measurement.tibia_source,
            "measurement_slice_index": meniscus.slice_index,
            "aggregation_strategy": paired.aggregation_strategy if paired is not None else "legacy",
            "candidate_count": paired.candidate_count if paired is not None else 1,
            "eligible_candidate_count": paired.eligible_candidate_count if paired is not None else 1,
            "unit": measurement.unit,
        }
        results.append(item)
        labels.append(int(label))
        scores.append(float(measurement.beta))
        preds.append(int(measurement.predicted_label))
        case_ids.append(str(record["case_id"]))
        group_ids.append(str(record.get("patient_group_id") or record["case_id"]))

    tp = sum(int(pred == 1 and target == 1) for pred, target in zip(preds, labels))
    tn = sum(int(pred == 0 and target == 0) for pred, target in zip(preds, labels))
    fp = sum(int(pred == 1 and target == 0) for pred, target in zip(preds, labels))
    fn = sum(int(pred == 0 and target == 1) for pred, target in zip(preds, labels))
    rates = precision_recall_specificity_f1(tp=tp, fp=fp, tn=tn, fn=fn)
    no_cases = not labels
    metrics = {
        "threshold": threshold,
        "accuracy": float("nan") if no_cases else (tp + tn) / len(labels),
        "roc_auc": safe_roc_auc(labels, scores),
        "precision": float("nan") if no_cases else rates["precision"],
        "recall": float("nan") if no_cases else rates["recall"],
        "specificity": float("nan") if no_cases else rates["specificity"],
        "f1": float("nan") if no_cases else rates["f1"],
        "evaluated_cases": float(len(labels)),
        "skipped_cases": float(len([item for item in results if item.get("status") == "skipped_missing_width"])),
    }

    n_boot = int(
        module4_cfg.get("bootstrap_samples", 1000)
        if bootstrap_samples is None
        else bootstrap_samples
    )
    pairs = list(zip(labels, scores, preds))

    def task_metric(items: list[tuple[int, float, int]], name: str) -> float:
        item_labels = [item[0] for item in items]
        item_scores = [item[1] for item in items]
        item_preds = [item[2] for item in items]
        if name == "roc_auc":
            return safe_roc_auc(item_labels, item_scores)
        item_tp = sum(int(pred == 1 and target == 1) for pred, target in zip(item_preds, item_labels))
        item_tn = sum(int(pred == 0 and target == 0) for pred, target in zip(item_preds, item_labels))
        item_fp = sum(int(pred == 1 and target == 0) for pred, target in zip(item_preds, item_labels))
        item_fn = sum(int(pred == 0 and target == 1) for pred, target in zip(item_preds, item_labels))
        if name == "accuracy":
            return (item_tp + item_tn) / max(len(item_labels), 1)
        return precision_recall_specificity_f1(item_tp, item_fp, item_tn, item_fn)[name]

    if n_boot > 0:
        for metric_name in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc"):
            _, lo, hi = grouped_bootstrap_ci(
                pairs,
                group_ids,
                lambda items, name=metric_name: task_metric(items, name),
                n_boot=n_boot,
            )
            metrics[f"{metric_name}_ci_low"] = lo
            metrics[f"{metric_name}_ci_high"] = hi
        metrics["bootstrap_case_count"] = float(len(set(case_ids)))
        metrics["bootstrap_group_count"] = float(len(set(group_ids)))
    return metrics, results


def save_outputs(config: dict[str, Any], metrics: dict[str, float], results: list[dict[str, Any]]) -> None:
    output_dir = Path(config.get("outputs", {}).get("output_dir", "outputs")) / "module4"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item) + "\n")
    if config.get("outputs", {}).get("save_visualizations", True):
        for item in results:
            if item.get("status"):
                continue
            save_module4_measurement_plot(
                output_dir / "visualizations" / f"{item['case_id']}_module4.png",
                tibia_width=float(item["tibia_width"]),
                meniscus_width=float(item["meniscus_width"]),
                beta=float(item["beta"]),
                threshold=float(config["module4"].get("threshold", 0.4)),
                title=f"{item['case_id']} Module 4",
            )


def validate_module4(
    config_path: str,
    index: str,
    roi_cache_dir: str | None = None,
    bootstrap_samples: int | None = None,
    aggregation_strategy: str | None = None,
    plateau_fraction: float | None = None,
    threshold: float | None = None,
    output_dir: str | None = None,
) -> dict[str, float]:
    config = load_config(config_path)
    if roi_cache_dir is not None:
        config["module4"]["roi_cache_dir"] = roi_cache_dir
    if aggregation_strategy is not None:
        config["module4"]["beta_aggregation"] = aggregation_strategy
    if plateau_fraction is not None:
        config["module4"]["min_plateau_width_fraction"] = plateau_fraction
    if threshold is not None:
        config["module4"]["threshold"] = threshold
    if output_dir is not None:
        config.setdefault("outputs", {})["output_dir"] = output_dir
    logger = get_logger("validate_module4")
    records = read_jsonl(index)
    tibia_segmenter = None
    segmenter_cfg = config.get("module4", {}).get("tibia_segmenter", {})
    if bool(segmenter_cfg.get("enabled", False)):
        requested_device = str(config.get("pipeline", {}).get("device", "cuda"))
        device = torch.device(requested_device if requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        tibia_segmenter = CoronalTibiaU2Net.from_config(segmenter_cfg, device=device)
    metrics, results = evaluate_records(
        config,
        records,
        bootstrap_samples=bootstrap_samples,
        tibia_segmenter=tibia_segmenter,
    )
    save_outputs(config, metrics, results)
    if metrics["evaluated_cases"] == 0:
        logger.warning(
            "Module 4 was not evaluated: no case has both a tibia-width source "
            "and a meniscus-width source. Missing measurements are not counted as negatives."
        )
    logger.info(format_metrics(metrics))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--index", required=True)
    parser.add_argument("--roi-cache-dir", default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument(
        "--aggregation-strategy",
        choices=sorted(PAIRED_WIDTH_AGGREGATION_STRATEGIES),
        default=None,
    )
    parser.add_argument("--plateau-fraction", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    validate_module4(
        args.config,
        args.index,
        roi_cache_dir=args.roi_cache_dir,
        bootstrap_samples=args.bootstrap,
        aggregation_strategy=args.aggregation_strategy,
        plateau_fraction=args.plateau_fraction,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
