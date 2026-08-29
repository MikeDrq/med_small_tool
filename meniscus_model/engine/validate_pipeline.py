"""Validate the full four-module pipeline with paper-style task metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from engine.infer_pipeline import build_pipeline, load_yaml
from models.pipeline import MeniscusDiagnosticPipeline
from utils.logging import get_logger
from utils.metrics import binary_task_metrics_with_ci


TASKS = [
    {"key": "medial_tear", "name": "Medial meniscus tear", "side": "medial", "type": "status", "positive": "tear"},
    {"key": "lateral_tear", "name": "Lateral meniscus tear", "side": "lateral", "type": "status", "positive": "tear"},
    {"key": "medial_degeneration", "name": "Medial meniscus degeneration", "side": "medial", "type": "status", "positive": "degeneration"},
    {"key": "lateral_degeneration", "name": "Lateral meniscus degeneration", "side": "lateral", "type": "status", "positive": "degeneration"},
    {"key": "medial_bucket_handle", "name": "Medial meniscus bucket-handle tear", "side": "medial", "type": "bucket_handle"},
    {"key": "lateral_bucket_handle", "name": "Lateral meniscus bucket-handle tear", "side": "lateral", "type": "bucket_handle"},
    {"key": "lateral_discoid", "name": "Lateral discoid meniscus", "side": "lateral", "type": "discoid"},
]


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


def _status_label(record: dict[str, Any], side: str, positive: str) -> int | None:
    value = record.get("module2_labels", {}).get(f"{side}_status")
    if value is None:
        return None
    return int(value == positive)


def _bucket_label(record: dict[str, Any], side: str) -> int | None:
    value = record.get("module3_labels", {}).get(f"{side}_bucket_handle")
    return None if value is None else int(value)


def _discoid_label(record: dict[str, Any]) -> int | None:
    value = record.get("module4_labels", {}).get("lateral_discoid")
    return None if value is None else int(value)


def task_label(record: dict[str, Any], task: dict[str, Any]) -> int | None:
    if task["type"] == "status":
        return _status_label(record, task["side"], task["positive"])
    if task["type"] == "bucket_handle":
        return _bucket_label(record, task["side"])
    if task["type"] == "discoid":
        return _discoid_label(record)
    raise ValueError(f"Unknown task type: {task['type']}")


def task_score_and_pred(result: dict[str, Any], task: dict[str, Any]) -> tuple[float | None, int | None]:
    side_payload = result.get("menisci", {}).get(task["side"], {})
    if not side_payload:
        return None, None
    if task["type"] == "status":
        probs = side_payload.get("confidence", {}).get("status_probabilities", {})
        score = probs.get(task["positive"])
        if score is None:
            pred_status = side_payload.get("status")
            confidence = float(side_payload.get("confidence", {}).get("status", 0.0))
            score = confidence if pred_status == task["positive"] else 1.0 - confidence
        return float(score), int(side_payload.get("status") == task["positive"])
    if task["type"] == "bucket_handle":
        return float(side_payload.get("confidence", {}).get("bucket_handle", 0.0)), int(bool(side_payload.get("bucket_handle", False)))
    if task["type"] == "discoid":
        beta = side_payload.get("discoid_beta")
        available = side_payload.get("discoid_available", beta is not None)
        if not available:
            return None, None
        score = float(beta) if beta is not None else side_payload.get("confidence", {}).get("discoid")
        if score is None:
            return None, None
        return float(score), int(bool(side_payload.get("discoid", False)))
    raise ValueError(f"Unknown task type: {task['type']}")


def _case_tibia_args(record: dict[str, Any]) -> tuple[float | None, str | None]:
    module4 = record.get("module4_labels", {})
    tibia_width = module4.get("tibia_width")
    tibia_mask_path = module4.get("tibia_mask_path")
    return (None if tibia_width is None else float(tibia_width), tibia_mask_path)


def run_or_load_case(
    pipeline: MeniscusDiagnosticPipeline | None,
    config: dict[str, Any],
    record: dict[str, Any],
    case_output_dir: Path,
    predictions_dir: str | None = None,
) -> dict[str, Any]:
    """Load an existing pipeline JSON or run inference for one case."""

    case_id = record["case_id"]
    if predictions_dir is not None:
        path = Path(predictions_dir) / case_id / "result.json"
        if not path.exists():
            path = Path(predictions_dir) / f"{case_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    if pipeline is None:
        raise FileNotFoundError(f"No existing prediction found for case {case_id} in {predictions_dir}")

    pipe_cfg = config["pipeline"]
    case = MeniscusDiagnosticPipeline.load_case(
        case_id=case_id,
        sagittal_dicom_dir=record["sagittal_dicom_dir"],
        coronal_dicom_dir=record["coronal_dicom_dir"],
        output_size=int(pipe_cfg.get("input_size", 224)),
        num_slices=int(pipe_cfg.get("num_slices", 16)),
        patch_size=int(pipe_cfg.get("patch_size", 16)),
        slice_selection=pipe_cfg.get("slice_selection"),
    )
    tibia_width, tibia_mask_path = _case_tibia_args(record)
    result = pipeline.infer_case(case, output_dir=case_output_dir, tibia_width=tibia_width or pipe_cfg.get("tibia_width"), tibia_mask_path=tibia_mask_path)
    case_output_dir.mkdir(parents=True, exist_ok=True)
    (case_output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def collect_task_samples(records: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    samples = {task["key"]: [] for task in TASKS}
    for record in records:
        result = results.get(record["case_id"])
        if result is None:
            continue
        for task in TASKS:
            label = task_label(record, task)
            if label is None:
                continue
            score, pred = task_score_and_pred(result, task)
            if score is None or pred is None:
                continue
            samples[task["key"]].append(
                {
                    "case_id": record["case_id"],
                    "group_id": record.get("patient_group_id") or record["case_id"],
                    "label": int(label),
                    "score": float(score),
                    "pred": int(pred),
                }
            )
    return samples


def compute_task_metrics(samples: dict[str, list[dict[str, Any]]], bootstrap: int, seed: int, thresholds: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
    output = {}
    thresholds = thresholds or {}
    for task in TASKS:
        items = samples[task["key"]]
        if not items:
            output[task["key"]] = {"task": task["name"], "n": 0, "available": False}
            continue
        labels = [item["label"] for item in items]
        scores = [item["score"] for item in items]
        predictions = [item["pred"] for item in items]
        group_ids = [item["group_id"] for item in items]
        metrics = binary_task_metrics_with_ci(
            labels,
            scores,
            threshold=thresholds.get(task["key"], 0.5),
            n_boot=bootstrap,
            seed=seed,
            group_ids=group_ids,
            predictions=predictions,
        )
        metrics["bootstrap_case_count"] = len(set(item["case_id"] for item in items))
        metrics["bootstrap_group_count"] = len(set(group_ids))
        metrics["task"] = task["name"]
        metrics["available"] = True
        output[task["key"]] = metrics
    return output


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if value != value:
            return "NA"
        return f"{value:.3f}"
    return str(value)


def _fmt_ci(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    ci = metrics.get(f"{key}_ci")
    if value is None or not isinstance(value, float) or value != value:
        return "NA"
    if isinstance(ci, dict) and ci.get("low") == ci.get("low") and ci.get("high") == ci.get("high"):
        return f"{value:.3f} ({ci['low']:.3f}, {ci['high']:.3f})"
    return f"{value:.3f}"


def print_table(metrics: dict[str, dict[str, Any]]) -> None:
    columns = ["Task", "n", "Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC", "PR-AUC"]
    rows = []
    for task in TASKS:
        item = metrics[task["key"]]
        rows.append(
            [
                task["name"],
                str(item.get("n", 0)),
                _fmt_ci(item, "accuracy"),
                _fmt_ci(item, "precision"),
                _fmt_ci(item, "recall"),
                _fmt_ci(item, "specificity"),
                _fmt_ci(item, "f1"),
                _fmt_ci(item, "roc_auc"),
                _fmt_ci(item, "pr_auc"),
            ]
        )
    widths = [max(len(columns[i]), *(len(row[i]) for row in rows)) for i in range(len(columns))]
    line = " | ".join(columns[i].ljust(widths[i]) for i in range(len(columns)))
    print(line)
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(columns))))


def save_metrics_csv(metrics: dict[str, dict[str, Any]], output_path: Path) -> None:
    fields = ["task_key", "task", "n", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, item in metrics.items():
            row = {"task_key": key, **{field: item.get(field) for field in fields if field != "task_key"}}
            writer.writerow(row)


def save_confusion_matrix(task_key: str, task_name: str, items: list[dict[str, Any]], output_dir: Path) -> None:
    matrix = np.zeros((2, 2), dtype=int)
    for item in items:
        matrix[int(item["label"]), int(item["pred"])] += 1
    path = output_dir / f"{task_key}_confusion_matrix.csv"
    np.savetxt(path, matrix, delimiter=",", fmt="%d")

    fig, ax = plt.subplots(figsize=(3.5, 3))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(task_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], ["Negative", "Positive"])
    ax.set_yticks([0, 1], ["Negative", "Positive"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / f"{task_key}_confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_curves(task_key: str, task_name: str, items: list[dict[str, Any]], output_dir: Path) -> None:
    labels = np.asarray([item["label"] for item in items], dtype=int)
    scores = np.asarray([item["score"] for item in items], dtype=float)
    if labels.size == 0 or np.unique(labels).size < 2:
        return
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    with (output_dir / f"{task_key}_roc_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fpr", "tpr"])
        writer.writerows(zip(fpr, tpr))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC: {task_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / f"{task_key}_roc_curve.png", dpi=150)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    with (output_dir / f"{task_key}_pr_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["recall", "precision"])
        writer.writerows(zip(recall, precision))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision, label=f"AP={pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR: {task_name}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_dir / f"{task_key}_pr_curve.png", dpi=150)
    plt.close(fig)


def save_artifacts(metrics: dict[str, dict[str, Any]], samples: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")
    save_metrics_csv(metrics, output_dir / "metrics.csv")
    confusion_dir = output_dir / "confusion_matrices"
    curves_dir = output_dir / "curves"
    confusion_dir.mkdir(exist_ok=True)
    curves_dir.mkdir(exist_ok=True)
    for task in TASKS:
        items = samples[task["key"]]
        if not items:
            continue
        save_confusion_matrix(task["key"], task["name"], items, confusion_dir)
        save_curves(task["key"], task["name"], items, curves_dir)


def validate_pipeline(config_path: str, index: str, output_dir: str, bootstrap: int, seed: int, predictions_dir: str | None = None) -> dict[str, dict[str, Any]]:
    config = load_yaml(config_path)
    logger = get_logger("validate_pipeline")
    records = read_jsonl(index)
    output_path = Path(output_dir)
    case_output_root = output_path / "cases"
    pipeline = None if predictions_dir is not None else build_pipeline(config)
    results: dict[str, dict[str, Any]] = {}
    for idx, record in enumerate(records, start=1):
        case_id = record["case_id"]
        logger.info(f"[{idx}/{len(records)}] validating case {case_id}")
        results[case_id] = run_or_load_case(pipeline, config, record, case_output_root / case_id, predictions_dir=predictions_dir)
    samples = collect_task_samples(records, results)
    thresholds = {"lateral_discoid": float(config.get("module4", {}).get("threshold", config.get("thresholds", {}).get("discoid_beta", 0.4)))}
    metrics = compute_task_metrics(samples, bootstrap=bootstrap, seed=seed, thresholds=thresholds)
    if not metrics["lateral_discoid"].get("available", False):
        logger.warning(
            "Discoid-meniscus metrics are unavailable because no valid tibia-width "
            "measurement was produced; these cases were not treated as negatives."
        )
    print_table(metrics)
    save_artifacts(metrics, samples, output_path)
    logger.info(f"saved validation artifacts to {output_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predictions-dir", default=None, help="Optional directory of existing per-case result.json files.")
    args = parser.parse_args()
    validate_pipeline(args.config, args.index, args.output_dir, args.bootstrap, args.seed, predictions_dir=args.predictions_dir)


if __name__ == "__main__":
    main()
