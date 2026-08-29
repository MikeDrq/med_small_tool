"""Compare Module 1 quality/ranking experiments.

The script is intentionally post-processing only: it reads experiment output
directories, extracts the best checkpoint metadata plus train-log metrics, and
optionally renders candidate-quality visual diagnostics from ``best.pt``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any
import warnings

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.validate_module1 import build_module1_model, create_val_loader, move_batch_to_device
from losses.detection_loss import box_iou, build_strict_pred_iou_targets
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_module1_final_scores, _token_plane_and_slice_ids


warnings.filterwarnings("ignore", message="Invalid value for VR UI:*")

BASELINES = {
    "good_bad_score_gap": 0.0086,
    "quality_score_spearman": 0.0355,
    "after_topk_recall@0.3": 0.2293,
    "final_F1@0.3": 0.1070,
    "final_R@0.3": 0.1997,
}

CSV_COLUMNS = [
    "experiment",
    "output_dir",
    "best_epoch",
    "total_loss",
    "loss_bbox",
    "loss_objectness",
    "loss_bbox_quality",
    "loss_quality_ranking",
    "good_quality_score_mean",
    "bad_quality_score_mean",
    "good_bad_score_gap",
    "quality_score_spearman",
    "quality_score_pearson",
    "objectness_score_spearman",
    "objectness_quality_spearman",
    "after_topk_recall@0.3",
    "final_R@0.3",
    "final_P@0.3",
    "final_F1@0.3",
    "final_R@0.45",
    "final_P@0.45",
    "final_F1@0.45",
    "oracle_iou_strict_R@0.3",
    "oracle_iou_strict_P@0.3",
    "oracle_iou_strict_F1@0.3",
    "objectness_R@0.3",
    "objectness_P@0.3",
    "objectness_F1@0.3",
    "quality_R@0.3",
    "quality_P@0.3",
    "quality_F1@0.3",
    "objectness_quality_R@0.3",
    "objectness_quality_P@0.3",
    "objectness_quality_F1@0.3",
    "task_aligned_score_R@0.3",
    "task_aligned_score_P@0.3",
    "task_aligned_score_F1@0.3",
    "oracle_gap",
]

METRIC_RE = re.compile(r"([A-Za-z0-9_@.\-]+)=(-?nan|[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)")
EPOCH_RE = re.compile(r"epoch=(\d+)\s+(train|validation)\b")


def parse_experiment_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, output_dir = spec.split("=", 1)
        return name.strip(), Path(output_dir.strip())
    output_dir = Path(spec)
    return output_dir.name, output_dir


def parse_log_metrics(log_path: Path) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    train_by_epoch: dict[int, dict[str, float]] = {}
    val_by_epoch: dict[int, dict[str, float]] = {}
    if not log_path.exists():
        return train_by_epoch, val_by_epoch
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EPOCH_RE.search(line)
        if not match:
            continue
        epoch = int(match.group(1))
        kind = match.group(2)
        parsed = {key: _float_or_nan(value) for key, value in METRIC_RE.findall(line)}
        parsed.pop("epoch", None)
        if kind == "train":
            train_by_epoch[epoch] = parsed
        else:
            val_by_epoch[epoch] = parsed
    return train_by_epoch, val_by_epoch


def load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return dict(payload.get("metadata", {}))


def collect_experiment(name: str, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    metadata = load_checkpoint_metadata(output_dir / "best.pt")
    train_by_epoch, val_by_epoch = parse_log_metrics(output_dir / "train.log")
    best_epoch = int(metadata.get("epoch", _pick_best_epoch(val_by_epoch)))
    val_metrics = dict(val_by_epoch.get(best_epoch, {}))
    val_metrics.update({k: float(v) for k, v in metadata.get("metrics", {}).items() if _is_number(v)})
    train_metrics = train_by_epoch.get(best_epoch, {})

    row: dict[str, Any] = {"experiment": name, "output_dir": str(output_dir), "best_epoch": best_epoch}
    row["total_loss"] = _first(train_metrics, ["loss"], _first(val_metrics, ["train_loss"]))
    row["loss_bbox"] = _first(train_metrics, ["loss_bbox"], _first(val_metrics, ["train_bbox_loss"]))
    row["loss_objectness"] = _first(train_metrics, ["loss_objectness"])
    row["loss_bbox_quality"] = _first(train_metrics, ["loss_bbox_quality"])
    row["loss_quality_ranking"] = _first(train_metrics, ["loss_quality_ranking"])

    row["good_quality_score_mean"] = _first(
        val_metrics,
        [
            "good_quality_score_mean",
            "score_mode_quality_good_candidate_quality_score_mean",
            "score_mode_quality_good_candidate_score_mean_iou_gt_0.3",
        ],
    )
    row["bad_quality_score_mean"] = _first(
        val_metrics,
        [
            "bad_quality_score_mean",
            "score_mode_quality_bad_candidate_quality_score_mean",
            "score_mode_quality_bad_candidate_score_mean_iou_lt_0.1",
        ],
    )
    row["good_bad_score_gap"] = _first(
        val_metrics,
        ["good_bad_score_gap", "score_mode_quality_good_bad_quality_score_gap"],
        _nan_sub(row["good_quality_score_mean"], row["bad_quality_score_mean"]),
    )
    row["quality_score_spearman"] = _first(
        val_metrics,
        ["quality_score_spearman", "score_mode_quality_quality_score_spearman_iou", "score_mode_quality_score_iou_spearman_corr"],
    )
    row["quality_score_pearson"] = _first(
        val_metrics,
        ["quality_score_pearson", "score_mode_quality_quality_score_pearson_iou", "score_mode_quality_score_iou_pearson_corr"],
    )
    row["objectness_score_spearman"] = _first(
        val_metrics,
        ["score_mode_objectness_score_iou_spearman_corr", "objectness_iou_spearman_corr"],
    )
    row["objectness_quality_spearman"] = _first(
        val_metrics,
        ["score_mode_objectness_quality_score_iou_spearman_corr", "score_mode_task_aligned_score_score_iou_spearman_corr"],
    )
    row["after_topk_recall@0.3"] = _first(
        val_metrics,
        ["after_topk_recall@0.3", "score_mode_objectness_quality_after_topk_recall@0.3"],
    )

    _copy_metric(row, val_metrics, "final_R@0.3", ["final_R@0.3", "iou_detection_recall@0.3", "iou_detection_recall"])
    _copy_metric(row, val_metrics, "final_P@0.3", ["final_P@0.3", "iou_detection_precision@0.3", "iou_detection_precision"])
    _copy_metric(row, val_metrics, "final_F1@0.3", ["final_F1@0.3", "iou_detection_f1@0.3", "iou_detection_f1"])
    _copy_metric(row, val_metrics, "final_R@0.45", ["iou_detection_recall@0.45"])
    _copy_metric(row, val_metrics, "final_P@0.45", ["iou_detection_precision@0.45"])
    _copy_metric(row, val_metrics, "final_F1@0.45", ["iou_detection_f1@0.45"])
    _copy_metric(row, val_metrics, "oracle_iou_strict_R@0.3", ["oracle_iou_strict_R@0.3", "oracle_iou_strict_recall@0.3"])
    _copy_metric(row, val_metrics, "oracle_iou_strict_P@0.3", ["oracle_iou_strict_P@0.3", "oracle_iou_strict_precision@0.3"])
    _copy_metric(row, val_metrics, "oracle_iou_strict_F1@0.3", ["oracle_iou_strict_F1@0.3", "oracle_iou_strict_f1@0.3"])

    for mode, label in [
        ("objectness", "objectness"),
        ("quality", "quality"),
        ("objectness_quality", "objectness_quality"),
        ("task_aligned_score", "task_aligned_score"),
    ]:
        row[f"{label}_R@0.3"] = _first(val_metrics, [f"score_mode_{mode}_recall@0.3"])
        row[f"{label}_P@0.3"] = _first(val_metrics, [f"score_mode_{mode}_precision@0.3"])
        row[f"{label}_F1@0.3"] = _first(val_metrics, [f"score_mode_{mode}_f1@0.3"])

    row["oracle_gap"] = _nan_sub(row["oracle_iou_strict_R@0.3"], row["final_R@0.3"])
    _write_best_metric_files(output_dir, row, train_metrics, val_metrics)
    return row


def _write_best_metric_files(
    output_dir: Path,
    row: dict[str, Any],
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> None:
    payload = {
        "summary": row,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
    }
    (output_dir / "metrics_best.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "metrics_best.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(_csv_row(row))
    (output_dir / "quality_ranking_diagnostic.md").write_text(
        render_experiment_diagnostic_markdown(row),
        encoding="utf-8",
    )


def render_comparison_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Module 1 Quality Ranking Comparison",
        "",
        "| experiment | best epoch | gap | quality spearman | after_topk R@0.3 | final R@0.3 | final P@0.3 | final F1@0.3 | oracle R@0.3 | oracle gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=_ranking_key, reverse=True):
        lines.append(
            "| "
            f"{row['experiment']} | {int(row['best_epoch'])} | "
            f"{_fmt(row['good_bad_score_gap'])} | {_fmt(row['quality_score_spearman'])} | "
            f"{_fmt(row['after_topk_recall@0.3'])} | {_fmt(row['final_R@0.3'])} | "
            f"{_fmt(row['final_P@0.3'])} | {_fmt(row['final_F1@0.3'])} | "
            f"{_fmt(row['oracle_iou_strict_R@0.3'])} | {_fmt(row['oracle_gap'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_experiment_diagnostic_markdown(row: dict[str, Any]) -> str:
    lines = [
        f"# Quality Ranking Diagnostic: {row['experiment']}",
        "",
        f"- best_epoch: {int(row['best_epoch'])}",
        f"- good_quality_score_mean: {_fmt(row['good_quality_score_mean'])}",
        f"- bad_quality_score_mean: {_fmt(row['bad_quality_score_mean'])}",
        f"- good_bad_score_gap: {_fmt(row['good_bad_score_gap'])}",
        f"- quality_score_spearman: {_fmt(row['quality_score_spearman'])}",
        f"- quality_score_pearson: {_fmt(row['quality_score_pearson'])}",
        f"- after_topk_recall@0.3: {_fmt(row['after_topk_recall@0.3'])}",
        f"- final_R/P/F1@0.3: {_fmt(row['final_R@0.3'])} / {_fmt(row['final_P@0.3'])} / {_fmt(row['final_F1@0.3'])}",
        f"- oracle_iou_strict_R/P/F1@0.3: {_fmt(row['oracle_iou_strict_R@0.3'])} / {_fmt(row['oracle_iou_strict_P@0.3'])} / {_fmt(row['oracle_iou_strict_F1@0.3'])}",
        f"- oracle_gap: {_fmt(row['oracle_gap'])}",
        f"- loss_bbox_quality: {_fmt(row['loss_bbox_quality'])}",
        f"- loss_quality_ranking: {_fmt(row['loss_quality_ranking'])}",
        "",
        "## Score Mode Snapshot",
        "",
        "| mode | R@0.3 | P@0.3 | F1@0.3 |",
        "|---|---:|---:|---:|",
    ]
    for mode in ["objectness", "quality", "objectness_quality", "task_aligned_score"]:
        lines.append(
            f"| {mode} | {_fmt(row[f'{mode}_R@0.3'])} | {_fmt(row[f'{mode}_P@0.3'])} | {_fmt(row[f'{mode}_F1@0.3'])} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in _diagnostics_for_row(row)])
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- metrics_best.json",
            "- metrics_best.csv",
            "- quality_ranking_visualizations/",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_best_summary(rows: list[dict[str, Any]]) -> str:
    ranked = sorted(rows, key=_ranking_key, reverse=True)
    best = ranked[0] if ranked else {}
    lines = [
        "# Best Config Summary",
        "",
        f"Recommended next main config: `{best.get('experiment', '<none>')}`",
        "",
        "## Automatic Diagnostics",
        "",
    ]
    for row in ranked:
        lines.append(f"### {row['experiment']}")
        diagnostics = _diagnostics_for_row(row)
        lines.extend([f"- {item}" for item in diagnostics])
        lines.append("")
    if best:
        lines.extend(
            [
                "## Selection Rule",
                "",
                "Priority was final_F1@0.3 first, then good/bad score gap, quality Spearman, after_topk recall@0.3, and precision.",
                "",
            ]
        )
        if best["experiment"].endswith("varifocal_ranking") and _finite(best.get("good_bad_score_gap")):
            lines.append("- If varifocal_ranking has the largest separation but unstable recall, lower ranking_loss_weight or add warmup.")
        if best["experiment"].endswith("qfl_hardneg"):
            lines.append("- If qfl_hardneg has the strongest Spearman but weak precision, tune score_threshold/topk/final_score_mode.")
        if best["experiment"].endswith("varifocal_hardneg"):
            lines.append("- If varifocal_hardneg is most stable, use it as the next 20-case / 100-case scaling baseline.")
    return "\n".join(lines).rstrip() + "\n"


def _diagnostics_for_row(row: dict[str, Any]) -> list[str]:
    items: list[str] = []
    gap = row.get("good_bad_score_gap", float("nan"))
    q_spearman = row.get("quality_score_spearman", float("nan"))
    after_topk = row.get("after_topk_recall@0.3", float("nan"))
    final_f1 = row.get("final_F1@0.3", float("nan"))
    oracle_gap = row.get("oracle_gap", float("nan"))
    final_r = row.get("final_R@0.3", float("nan"))
    loss_bbox_quality = row.get("loss_bbox_quality", float("nan"))

    items.append(_flag(gap, BASELINES["good_bad_score_gap"], 0.010, "quality score separation improved", "quality score separation is still weak"))
    items.append(_flag(q_spearman, BASELINES["quality_score_spearman"], 0.020, "quality-IoU correlation improved", "quality-IoU correlation is still weak"))
    items.append(_flag(after_topk, BASELINES["after_topk_recall@0.3"], 0.020, "ranking improves candidate selection", "candidate selection did not clearly improve"))
    items.append(_flag(final_f1, BASELINES["final_F1@0.3"], 0.000, "final detection improved", "final detection has not beaten baseline F1"))
    if _finite(oracle_gap) and oracle_gap > 0.25:
        items.append("oracle_gap remains large; learned score is still far below oracle ranking and needs more ranking/calibration work")
    if _finite(gap) and _finite(final_r) and gap > BASELINES["good_bad_score_gap"] + 0.010 and final_r <= BASELINES["final_R@0.3"]:
        items.append("good/bad gap improved but final recall did not; check score_threshold, topk, and final_score_mode")
    if _finite(loss_bbox_quality) and loss_bbox_quality > 0.0 and not (_finite(gap) and gap > BASELINES["good_bad_score_gap"] + 0.010):
        items.append("bbox quality loss is active but good/bad gap did not move enough; inspect quality target and sampling mask")
    return items


def generate_visualizations(
    rows: list[dict[str, Any]],
    limit_per_category: int,
    device_name: str | None,
) -> None:
    for row in rows:
        output_dir = Path(row["output_dir"])
        config_path = output_dir / "config.yaml"
        checkpoint = output_dir / "best.pt"
        if not config_path.exists() or not checkpoint.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        device = torch.device(device_name or (config.get("device", "cuda") if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        model = build_module1_model(config).to(device)
        load_checkpoint(checkpoint, model, map_location=device)
        model.eval()
        loader = create_val_loader(config)
        categories: dict[str, list[dict[str, Any]]] = {
            "high_iou_low_quality": [],
            "low_iou_high_quality": [],
            "high_objectness_low_iou": [],
            "high_quality_high_iou": [],
        }
        with torch.inference_mode():
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                predictions = model(
                    batch["sagittal_volume"],
                    batch["coronal_volume"],
                    batch["sagittal_coords"],
                    batch["coronal_coords"],
                )
                _collect_visual_candidates(batch, predictions, config, categories)
                if all(len(values) >= limit_per_category * 4 for values in categories.values()):
                    break
        visual_dir = output_dir / "quality_ranking_visualizations"
        _render_candidate_categories(categories, visual_dir, limit_per_category)


def _collect_visual_candidates(
    batch: dict[str, Any],
    predictions: dict[str, Any],
    config: dict[str, Any],
    categories: dict[str, list[dict[str, Any]]],
) -> None:
    device = predictions["objectness_logits"].device
    shape_metadata = predictions["shape_metadata"]
    plane_ids, slice_indices = _token_plane_and_slice_ids(shape_metadata, device)
    side_ids = predictions["side_logits"].detach().argmax(dim=-1)
    objectness = torch.sigmoid(predictions["objectness_logits"]).detach()
    quality = torch.sigmoid(predictions["bbox_quality_logits"]).detach() if "bbox_quality_logits" in predictions else torch.zeros_like(objectness)
    final_score = compute_module1_final_scores(
        predictions,
        final_score_mode=config.get("validate", {}).get("final_score_mode", "objectness_quality"),
        quality_score_alpha=float(config.get("validate", {}).get("quality_score_alpha", 1.0)),
        quality_score_beta=float(config.get("validate", {}).get("quality_score_beta", 1.0)),
    )
    strict_ious = build_strict_pred_iou_targets(
        batch["labels"],
        shape_metadata,
        predictions["bbox_pred"].detach(),
        predictions["side_logits"],
        device=device,
    )
    boxes = predictions["bbox_pred"].detach()
    for batch_idx, labels in enumerate(batch["labels"]):
        gt_by_key = _case_gt_boxes(labels, shape_metadata, device)
        for token_idx in range(int(objectness.shape[1])):
            key = (
                int(plane_ids[token_idx].item()),
                int(slice_indices[token_idx].item()),
                int(side_ids[batch_idx, token_idx].item()),
            )
            gt_boxes = gt_by_key.get(key)
            if gt_boxes is None or gt_boxes.numel() == 0:
                continue
            pred_box = boxes[batch_idx, token_idx]
            ious = box_iou(pred_box.view(1, 4), gt_boxes).flatten()
            best_iou, best_gt_idx = ious.max(dim=0)
            record = {
                "case_id": labels.get("case_id", str(batch_idx)),
                "batch_idx": batch_idx,
                "token_idx": token_idx,
                "plane_id": key[0],
                "plane": "sagittal" if key[0] == 0 else "coronal",
                "processed_slice": key[1],
                "side": "medial" if key[2] == 0 else "lateral",
                "gt_box": gt_boxes[int(best_gt_idx)].detach().cpu().tolist(),
                "pred_box": pred_box.detach().cpu().tolist(),
                "iou": float(best_iou.item()),
                "objectness": float(objectness[batch_idx, token_idx].item()),
                "quality": float(quality[batch_idx, token_idx].item()),
                "final_score": float(final_score[batch_idx, token_idx].item()),
                "image": batch["sagittal_volume" if key[0] == 0 else "coronal_volume"][batch_idx].detach().cpu(),
            }
            if record["iou"] >= 0.3:
                categories["high_iou_low_quality"].append(record)
                categories["high_quality_high_iou"].append(record)
            if record["iou"] < 0.1:
                categories["low_iou_high_quality"].append(record)
                categories["high_objectness_low_iou"].append(record)


def _render_candidate_categories(
    categories: dict[str, list[dict[str, Any]]],
    visual_dir: Path,
    limit_per_category: int,
) -> None:
    sorters = {
        "high_iou_low_quality": lambda row: (row["quality"], -row["iou"]),
        "low_iou_high_quality": lambda row: (-row["quality"], row["iou"]),
        "high_objectness_low_iou": lambda row: (-row["objectness"], row["iou"]),
        "high_quality_high_iou": lambda row: (-row["quality"], -row["iou"]),
    }
    summary: dict[str, list[dict[str, Any]]] = {}
    for category, items in categories.items():
        selected = sorted(items, key=sorters[category])[:limit_per_category]
        summary[category] = [_record_without_image(row) for row in selected]
        for idx, row in enumerate(selected, start=1):
            path = visual_dir / category / f"{idx:02d}_{row['case_id']}_{row['plane']}_slice{row['processed_slice']:02d}_{row['side']}.png"
            _save_candidate_overlay(row, path)
    visual_dir.mkdir(parents=True, exist_ok=True)
    (visual_dir / "visualization_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_candidate_overlay(row: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image = row["image"][0, int(row["processed_slice"])].numpy()
    height, width = image.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    _add_box(ax, row["gt_box"], width, height, "lime", "GT")
    _add_box(ax, row["pred_box"], width, height, "tab:red", "candidate")
    ax.set_title(
        f"{row['case_id']} {row['plane']} slice={row['processed_slice']} {row['side']}\n"
        f"IoU={row['iou']:.3f} obj={row['objectness']:.3f} quality={row['quality']:.3f} final={row['final_score']:.3f}",
        fontsize=9,
    )
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _add_box(ax: Any, box: list[float], width: int, height: int, color: str, label: str) -> None:
    from matplotlib.patches import Rectangle

    x1, y1, x2, y2 = [float(value) for value in box]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2.0))
    ax.text(x1, max(0.0, y1 - 3.0), label, color=color, fontsize=8, weight="bold")


def _case_gt_boxes(case_labels: dict[str, Any], shape_metadata: dict[str, Any], device: torch.device) -> dict[tuple[int, int, int], torch.Tensor]:
    boxes_by_key: dict[tuple[int, int, int], list[torch.Tensor]] = {}
    for label in case_labels.get("module1_labels", []):
        plane = label.get("plane")
        if plane not in {"sagittal", "coronal"}:
            continue
        shape = shape_metadata[plane]
        plane_id = 0 if plane == "sagittal" else 1
        side_id = 0 if label.get("meniscus_side") == "medial" else 1
        key = (plane_id, int(label["slice_index"]), side_id)
        x1, y1, x2, y2 = [float(value) for value in label["bbox"]]
        box = torch.tensor(
            [x1 / float(shape["width"]), y1 / float(shape["height"]), x2 / float(shape["width"]), y2 / float(shape["height"])],
            device=device,
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        boxes_by_key.setdefault(key, []).append(torch.cat([torch.minimum(box[:2], box[2:]), torch.maximum(box[:2], box[2:])]))
    return {key: torch.stack(items, dim=0) for key, items in boxes_by_key.items()}


def _record_without_image(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "image"}


def _copy_metric(row: dict[str, Any], metrics: dict[str, float], out_key: str, keys: list[str]) -> None:
    row[out_key] = _first(metrics, keys)


def _first(metrics: dict[str, float], keys: list[str], default: float = float("nan")) -> float:
    for key in keys:
        value = metrics.get(key)
        if _finite(value):
            return float(value)
    return default


def _pick_best_epoch(val_by_epoch: dict[int, dict[str, float]]) -> int:
    if not val_by_epoch:
        return 0
    keys = ["objectness_quality_f1@0.3", "score_mode_objectness_quality_f1@0.3", "final_F1@0.3", "iou_detection_f1@0.3", "iou_detection_f1"]

    def score(epoch: int) -> float:
        return _first(val_by_epoch[epoch], keys, -float("inf"))

    return max(val_by_epoch, key=score)


def _ranking_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        _safe(row.get("final_F1@0.3")),
        _safe(row.get("good_bad_score_gap")),
        _safe(row.get("quality_score_spearman")),
        _safe(row.get("after_topk_recall@0.3")),
        _safe(row.get("final_P@0.3")),
    )


def _flag(value: float, baseline: float, margin: float, good: str, bad: str) -> str:
    if _finite(value) and value > baseline + margin:
        return f"{good}: {value:.4f} vs baseline {baseline:.4f}"
    return f"{bad}: {value:.4f} vs baseline {baseline:.4f}" if _finite(value) else f"{bad}: missing metric"


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key, "") for key in CSV_COLUMNS}


def _fmt(value: Any) -> str:
    return f"{float(value):.4f}" if _finite(value) else "nan"


def _float_or_nan(value: str) -> float:
    return float("nan") if value.lower() == "nan" else float(value)


def _nan_sub(a: Any, b: Any) -> float:
    return float(a) - float(b) if _finite(a) and _finite(b) else float("nan")


def _safe(value: Any) -> float:
    return float(value) if _finite(value) else -float("inf")


def _finite(value: Any) -> bool:
    try:
        return value is not None and float(value) == float(value)
    except (TypeError, ValueError):
        return False


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and float(value) == float(value)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [collect_experiment(name, path) for name, path in map(parse_experiment_spec, args.experiment)]
    csv_path = output_dir / "quality_ranking_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(_csv_row(row) for row in rows)
    (output_dir / "quality_ranking_comparison.md").write_text(render_comparison_markdown(rows), encoding="utf-8")
    (output_dir / "best_config_summary.md").write_text(render_best_summary(rows), encoding="utf-8")
    if not args.no_visuals:
        generate_visualizations(rows, limit_per_category=int(args.visual_limit), device_name=args.device)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        nargs="+",
        help="Experiment output directory, optionally named as name=path.",
    )
    parser.add_argument("--output-dir", default="outputs/module1_debug/quality_ranking_comparison")
    parser.add_argument("--visual-limit", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-visuals", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
