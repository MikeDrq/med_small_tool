"""Debug Module 1 GT boxes and token target assignment.

This script is intentionally data-path aware: it loads labels through
``MeniscusDetectionDataset`` so bbox scaling and source-slice remapping match
training, then calls ``build_module1_targets`` to inspect positive tokens.
"""

from __future__ import annotations

import argparse
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
from losses.detection_loss import DetectionTargetConfig, build_module1_targets


SIDE_ID_TO_NAME = {0: "medial", 1: "lateral"}
ABNORMALITY_ID_TO_NAME = {0: "normal", 1: "abnormal"}
SIDE_COLORS = {"medial": "tab:orange", "lateral": "tab:cyan"}
warnings.filterwarnings("ignore", message="Invalid value for VR UI:*")


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _patch_hw(patch_size: int | list[int] | tuple[int, ...]) -> tuple[int, int]:
    if isinstance(patch_size, int):
        return patch_size, patch_size
    if len(patch_size) == 2:
        return int(patch_size[0]), int(patch_size[1])
    if len(patch_size) == 3:
        return int(patch_size[1]), int(patch_size[2])
    raise ValueError(f"Unsupported patch_size={patch_size!r}")


def _plane_shape(volume: torch.Tensor, patch_size: int | list[int] | tuple[int, ...]) -> dict[str, int]:
    _, slices, height, width = volume.shape
    patch_height, patch_width = _patch_hw(patch_size)
    grid_height = int(height) // patch_height
    grid_width = int(width) // patch_width
    return {
        "batch_size": 1,
        "channels": int(volume.shape[0]),
        "slices": int(slices),
        "height": int(height),
        "width": int(width),
        "patch_height": int(patch_height),
        "patch_width": int(patch_width),
        "grid_height": grid_height,
        "grid_width": grid_width,
        "patches_per_slice": grid_height * grid_width,
    }


def _shape_metadata(sample: dict[str, Any], patch_size: int | list[int] | tuple[int, ...]) -> dict[str, Any]:
    sagittal_shape = _plane_shape(sample["sagittal_volume"], patch_size)
    coronal_shape = _plane_shape(sample["coronal_volume"], patch_size)
    sag_tokens = sagittal_shape["slices"] * sagittal_shape["patches_per_slice"]
    cor_tokens = coronal_shape["slices"] * coronal_shape["patches_per_slice"]
    return {
        "sagittal": sagittal_shape,
        "coronal": coronal_shape,
        "tokens_per_plane": sag_tokens,
        "fused_tokens": sag_tokens + cor_tokens,
    }


def _normalize_box(box: list[float], shape: dict[str, Any]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    width = float(shape["width"])
    height = float(shape["height"])
    x1, x2 = sorted((x1 / width, x2 / width))
    y1, y2 = sorted((y1 / height, y2 / height))
    return [max(0.0, min(1.0, value)) for value in (x1, y1, x2, y2)]


def _center_norm(local_index: int, shape: dict[str, Any]) -> tuple[int, int, int, float, float]:
    per_slice = int(shape["patches_per_slice"])
    grid_width = int(shape["grid_width"])
    slice_index = local_index // per_slice
    index_in_slice = local_index - slice_index * per_slice
    row = index_in_slice // grid_width
    col = index_in_slice % grid_width
    x_norm = (float(col) + 0.5) / float(shape["grid_width"])
    y_norm = (float(row) + 0.5) / float(shape["grid_height"])
    return slice_index, row, col, x_norm, y_norm


def _distance_to_box_px(x: float, y: float, box: list[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in box]
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float((dx * dx + dy * dy) ** 0.5)


def _labels_by_key(labels: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for label in labels:
        key = (str(label["plane"]), int(label["slice_index"]))
        grouped.setdefault(key, []).append(label)
    return grouped


def _print_slice_sampling(sample: dict[str, Any], processed_labels: list[dict[str, Any]]) -> dict[str, Any]:
    print("\nSlice sampling metadata:")
    sampling: dict[str, Any] = {}
    for plane in ("sagittal", "coronal"):
        metadata = sample[f"{plane}_metadata"]
        sampled = [int(value) for value in metadata["sampled_source_indices"]]
        original_shape = tuple(int(value) for value in metadata["original_shape"])
        processed_shape = tuple(int(value) for value in metadata["processed_shape"])
        print(f"  {plane:<8} original_shape={original_shape} processed_shape={processed_shape}")
        print(f"  {plane:<8} sampled_source_indices={sampled}")
        sampling[plane] = {
            "original_shape": list(original_shape),
            "processed_shape": list(processed_shape),
            "sampled_source_indices": sampled,
        }

    repeated_label_sources = []
    for label in processed_labels:
        plane = str(label["plane"])
        source_slice = int(label.get("source_slice_index", label["slice_index"]))
        sampled = sampling[plane]["sampled_source_indices"]
        processed_positions = [idx for idx, value in enumerate(sampled) if int(value) == source_slice]
        if len(processed_positions) > 1:
            repeated_label_sources.append(
                {
                    "plane": plane,
                    "side": label["meniscus_side"],
                    "source_slice": source_slice,
                    "chosen_processed_slice": int(label["slice_index"]),
                    "all_processed_positions": processed_positions,
                }
            )

    if repeated_label_sources:
        print("\nRepeated labeled source slices:")
        seen = set()
        for item in repeated_label_sources:
            key = (item["plane"], item["side"], item["source_slice"], tuple(item["all_processed_positions"]))
            if key in seen:
                continue
            seen.add(key)
            print(
                f"  plane={item['plane']:<8} side={item['side']:<7} "
                f"source_slice={item['source_slice']:02d} chosen_processed_slice={item['chosen_processed_slice']:02d} "
                f"all_processed_positions={item['all_processed_positions']}"
            )
    else:
        print("\nRepeated labeled source slices: none")

    sampling["repeated_labeled_source_slices"] = repeated_label_sources
    return sampling


def _print_gt_boxes(raw_record: dict[str, Any], processed_labels: list[dict[str, Any]], shape_metadata: dict[str, Any]) -> None:
    print("\nGT boxes from JSONL before dataset preprocessing:")
    for idx, label in enumerate(
        sorted(
            raw_record.get("module1_labels", []),
            key=lambda item: (item["plane"], item["meniscus_side"], int(item["slice_index"]), item["bbox"]),
        ),
        start=1,
    ):
        print(
            f"  raw#{idx:02d} plane={label['plane']:<8} side={label['meniscus_side']:<7} "
            f"slice={int(label['slice_index']):02d} abnormality={label['abnormality']:<8} "
            f"bbox_px={[round(float(v), 2) for v in label['bbox']]}"
        )

    print("\nGT boxes after MeniscusDetectionDataset preprocessing:")
    for idx, label in enumerate(
        sorted(
            processed_labels,
            key=lambda item: (item["plane"], item["meniscus_side"], int(item["slice_index"]), item["bbox"]),
        ),
        start=1,
    ):
        shape = shape_metadata[label["plane"]]
        box_norm = _normalize_box(label["bbox"], shape)
        print(
            f"  gt#{idx:02d} plane={label['plane']:<8} side={label['meniscus_side']:<7} "
            f"source_slice={int(label.get('source_slice_index', -1)):02d} "
            f"processed_slice={int(label['slice_index']):02d} abnormality={label['abnormality']:<8} "
            f"bbox_px={[round(float(v), 2) for v in label['bbox']]} "
            f"bbox_norm={[round(v, 4) for v in box_norm]}"
        )


def _positive_token_rows(
    targets: dict[str, torch.Tensor],
    shape_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    sagittal_tokens = int(shape_metadata["sagittal"]["slices"] * shape_metadata["sagittal"]["patches_per_slice"])
    rows = []
    positive_indices = torch.nonzero(targets["positive_mask"][0], as_tuple=False).flatten().tolist()
    for global_index in positive_indices:
        if global_index < sagittal_tokens:
            plane = "sagittal"
            local_index = int(global_index)
        else:
            plane = "coronal"
            local_index = int(global_index - sagittal_tokens)
        shape = shape_metadata[plane]
        slice_index, row, col, x_norm, y_norm = _center_norm(local_index, shape)
        center_px = [x_norm * float(shape["width"]), y_norm * float(shape["height"])]
        box_norm = targets["bbox_targets"][0, global_index].detach().cpu().tolist()
        box_px = [
            box_norm[0] * float(shape["width"]),
            box_norm[1] * float(shape["height"]),
            box_norm[2] * float(shape["width"]),
            box_norm[3] * float(shape["height"]),
        ]
        side_id = int(targets["side_targets"][0, global_index].item())
        abnormality_id = int(targets["abnormality_targets"][0, global_index].item())
        rows.append(
            {
                "global_token": int(global_index),
                "local_token": int(local_index),
                "plane": plane,
                "slice_index": int(slice_index),
                "row": int(row),
                "col": int(col),
                "center_norm": [float(x_norm), float(y_norm)],
                "center_px": [float(center_px[0]), float(center_px[1])],
                "side_id": side_id,
                "side": SIDE_ID_TO_NAME.get(side_id, str(side_id)),
                "abnormality_id": abnormality_id,
                "abnormality": ABNORMALITY_ID_TO_NAME.get(abnormality_id, str(abnormality_id)),
                "bbox_target_norm": [float(value) for value in box_norm],
                "bbox_target_px": [float(value) for value in box_px],
                "distance_to_assigned_box_px": _distance_to_box_px(center_px[0], center_px[1], box_px),
            }
        )
    return sorted(rows, key=lambda item: (item["plane"], item["slice_index"], item["side"], item["row"], item["col"]))


def _print_positive_tokens(rows: list[dict[str, Any]], max_rows: int) -> None:
    print(f"\nPositive tokens from target builder: count={len(rows)}")
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        counts[(row["plane"], row["side"])] = counts.get((row["plane"], row["side"]), 0) + 1
    for (plane, side), count in sorted(counts.items()):
        print(f"  {plane:<8} {side:<7}: {count}")

    print("\nPositive token detail:")
    for idx, row in enumerate(rows[:max_rows], start=1):
        print(
            f"  pos#{idx:03d} token={row['global_token']:05d} plane={row['plane']:<8} "
            f"slice={row['slice_index']:02d} grid=({row['row']:02d},{row['col']:02d}) "
            f"center_px=({row['center_px'][0]:6.2f},{row['center_px'][1]:6.2f}) "
            f"center_norm=({row['center_norm'][0]:.4f},{row['center_norm'][1]:.4f}) "
            f"side={row['side']:<7} abnormality={row['abnormality']:<8} "
            f"dist_to_box_px={row['distance_to_assigned_box_px']:.2f}"
        )
    if len(rows) > max_rows:
        print(f"  ... truncated {len(rows) - max_rows} rows; full detail is in target_debug.json")


def _distance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    distances = sorted(float(row["distance_to_assigned_box_px"]) for row in rows)

    def percentile(q: float) -> float:
        if len(distances) == 1:
            return distances[0]
        pos = (len(distances) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(distances) - 1)
        weight = pos - lo
        return distances[lo] * (1.0 - weight) + distances[hi] * weight

    by_group: dict[str, dict[str, float | int]] = {}
    for row in rows:
        key = f"{row['plane']}:{row['side']}"
        group = by_group.setdefault(key, {"count": 0, "inside": 0, "max_px": 0.0})
        distance = float(row["distance_to_assigned_box_px"])
        group["count"] = int(group["count"]) + 1
        group["inside"] = int(group["inside"]) + int(distance <= 1e-5)
        group["max_px"] = max(float(group["max_px"]), distance)
    return {
        "count": len(rows),
        "inside_count": sum(1 for value in distances if value <= 1e-5),
        "outside_count": sum(1 for value in distances if value > 1e-5),
        "max_px": distances[-1],
        "p50_px": percentile(0.50),
        "p90_px": percentile(0.90),
        "by_plane_side": by_group,
    }


def _print_distance_summary(summary: dict[str, Any]) -> None:
    print("\nPositive-center distance to assigned GT box:")
    if not summary.get("count"):
        print("  no positive tokens")
        return
    print(
        f"  inside={summary['inside_count']} outside={summary['outside_count']} "
        f"p50={summary['p50_px']:.2f}px p90={summary['p90_px']:.2f}px max={summary['max_px']:.2f}px"
    )
    for key, values in sorted(summary["by_plane_side"].items()):
        print(
            f"  {key:<17} count={values['count']:3d} inside={values['inside']:3d} "
            f"max={values['max_px']:.2f}px"
        )


def _draw_visualizations(
    sample: dict[str, Any],
    processed_labels: list[dict[str, Any]],
    positive_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_by_key = _labels_by_key(processed_labels)
    positives_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in positive_rows:
        key = (row["plane"], int(row["slice_index"]))
        positives_by_key.setdefault(key, []).append(row)

    saved = []
    for plane, slice_index in sorted(set(labels_by_key) | set(positives_by_key)):
        volume = sample[f"{plane}_volume"][0]
        image = volume[int(slice_index)].detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image, cmap="gray")

        for label in labels_by_key.get((plane, slice_index), []):
            side = str(label["meniscus_side"])
            x1, y1, x2, y2 = [float(value) for value in label["bbox"]]
            color = SIDE_COLORS.get(side, "tab:red")
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2.0))
            ax.text(x1, max(0.0, y1 - 3.0), side, color=color, fontsize=8, weight="bold")

        for row in positives_by_key.get((plane, slice_index), []):
            side = str(row["side"])
            color = SIDE_COLORS.get(side, "tab:green")
            x, y = row["center_px"]
            ax.scatter([x], [y], marker="+", c=color, s=70, linewidths=1.7)
            ax.text(x + 2.0, y + 2.0, f"r{row['row']}c{row['col']}", color=color, fontsize=7)

        ax.set_title(f"{sample['case_id']} | {plane} slice {slice_index} | GT boxes + positive token centers")
        ax.axis("off")
        path = output_dir / f"{sample['case_id']}_{plane}_slice{int(slice_index):02d}_positive_tokens.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))
    return saved


def debug_targets(
    config_path: str | Path,
    index_jsonl: str | Path | None,
    case_id: str | None,
    output_dir: str | Path,
    max_rows: int,
) -> dict[str, Any]:
    config = _load_config(config_path)
    data_cfg = config["data"]
    loss_cfg = config.get("loss", {})
    index_path = Path(index_jsonl or data_cfg["train_index"])
    output_root = Path(output_dir)
    patch_size = data_cfg.get("patch_size", config.get("model", {}).get("patch_size", 16))
    target_slices = data_cfg.get("num_slices", config.get("model", {}).get("num_slices", 16))

    records = read_jsonl(index_path)
    record_idx = 0
    if case_id is not None:
        matches = [idx for idx, record in enumerate(records) if str(record.get("case_id")) == str(case_id)]
        if not matches:
            raise ValueError(f"case_id={case_id} not found in {index_path}")
        record_idx = matches[0]
    raw_record = records[record_idx]

    dataset = MeniscusDetectionDataset(
        index_jsonl=index_path,
        output_size=data_cfg["output_size"],
        target_slices=target_slices,
        patch_size=patch_size,
        slice_selection=data_cfg.get("train_slice_selection", data_cfg.get("slice_selection")),
        transform=None,
    )
    dataset_indices = dataset.dataset_indices_for_record(record_idx)
    if not dataset_indices:
        raise RuntimeError(f"record_index={record_idx} has no Dataset sample plan")
    sample = dataset[dataset_indices[0]]
    shape_metadata = _shape_metadata(sample, patch_size)
    processed_labels = sample["labels"]["module1_labels"]
    radius = int(loss_cfg.get("positive_radius_tokens", 0))

    print(f"case_id={sample['case_id']}")
    print(
        f"chunk={sample.get('chunk_index', 0) + 1}/{sample.get('chunk_count', 1)} "
        f"dataset_index={dataset_indices[0]}"
    )
    print(f"index_jsonl={index_path}")
    print(f"output_size={data_cfg['output_size']} target_slices={target_slices} patch_size={patch_size}")
    print(
        "shape_metadata="
        + json.dumps(
            {
                plane: {
                    key: shape_metadata[plane][key]
                    for key in ("slices", "height", "width", "grid_height", "grid_width", "patches_per_slice")
                }
                for plane in ("sagittal", "coronal")
            },
            ensure_ascii=False,
        )
    )
    print(f"positive_radius_tokens={radius}")
    sampling_summary = _print_slice_sampling(sample, processed_labels)
    _print_gt_boxes(raw_record, processed_labels, shape_metadata)

    base_targets = build_module1_targets(
        [sample["labels"]],
        shape_metadata,
        device=torch.device("cpu"),
        target_config=DetectionTargetConfig(positive_radius_tokens=0),
    )
    targets = build_module1_targets(
        [sample["labels"]],
        shape_metadata,
        device=torch.device("cpu"),
        target_config=DetectionTargetConfig(positive_radius_tokens=radius),
    )
    base_count = int(base_targets["positive_mask"].sum().item())
    radius_count = int(targets["positive_mask"].sum().item())
    print(f"\nPositive token counts: radius_0={base_count}, radius_{radius}={radius_count}")

    positive_rows = _positive_token_rows(targets, shape_metadata)
    _print_positive_tokens(positive_rows, max_rows=max_rows)
    positive_distance_summary = _distance_summary(positive_rows)
    _print_distance_summary(positive_distance_summary)

    visual_dir = output_root / "positive_token_overlays"
    saved_paths = _draw_visualizations(sample, processed_labels, positive_rows, visual_dir)
    print(f"\nSaved positive token overlays: {visual_dir} ({len(saved_paths)} PNGs)")

    summary = {
        "case_id": sample["case_id"],
        "index_jsonl": str(index_path),
        "config": str(config_path),
        "positive_radius_tokens": radius,
        "positive_count_radius_0": base_count,
        "positive_count_radius_config": radius_count,
        "shape_metadata": shape_metadata,
        "slice_sampling": sampling_summary,
        "processed_labels": processed_labels,
        "positive_tokens": positive_rows,
        "positive_distance_summary": positive_distance_summary,
        "visualizations": saved_paths,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "target_debug.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved JSON report: {report_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/module1_meniscovit_cop.yaml")
    parser.add_argument("--index-jsonl", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--output-dir", default="outputs/module1_target_debug")
    parser.add_argument("--max-rows", type=int, default=160)
    args = parser.parse_args()

    debug_targets(
        config_path=args.config,
        index_jsonl=args.index_jsonl,
        case_id=args.case_id,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
