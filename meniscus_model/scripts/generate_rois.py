"""Generate meniscus ROI sequences from Module 1 detections or GT boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.meniscus_detection_dataset import MeniscusDetectionDataset
from engine.validate_module1 import build_module1_model
from utils.checkpoint import load_checkpoint
from utils.logging import get_logger
from utils.metrics import filter_module1_predictions
from utils.visualization import save_roi_preview, save_slice_overlay


STATUS_TO_ID = {"normal": 0, "degeneration": 1, "tear": 2}
SIDE_TO_ID = {"medial": 0, "lateral": 1}
ID_TO_SIDE = {0: "medial", 1: "lateral"}


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y"}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _empty_plane_boxes(num_slices: int) -> list[dict[str, Any] | None]:
    return [None for _ in range(num_slices)]


def _box_clip(box: torch.Tensor) -> torch.Tensor:
    box = box.float().clamp(0.0, 1.0)
    x1y1 = torch.minimum(box[:2], box[2:])
    x2y2 = torch.maximum(box[:2], box[2:])
    return torch.cat([x1y1, x2y2])


def _normalize_label_box(label: dict[str, Any], width: int, height: int) -> torch.Tensor:
    x1, y1, x2, y2 = [float(v) for v in label["bbox"]]
    return _box_clip(torch.tensor([x1 / width, y1 / height, x2 / width, y2 / height]))


def resolve_score_threshold(config: dict[str, Any], score_threshold: float | None) -> float:
    if score_threshold is not None:
        return float(score_threshold)
    return float(config.get("validate", {}).get("score_threshold", 0.5))


def resolve_slice_selection(config: dict[str, Any], usage: str | None, use_gt_boxes: bool) -> dict[str, Any] | None:
    data_cfg = config["data"]
    # GT ROI caches must contain one deterministic row per case.  Using the
    # Module 1 label-chunk training policy here could overwrite the same
    # ``case_id.pt`` multiple times and duplicate index entries.
    resolved_usage = usage or ("uniform" if use_gt_boxes else "val")
    if resolved_usage == "train":
        return data_cfg.get("train_slice_selection", data_cfg.get("slice_selection"))
    if resolved_usage in {"val", "validate", "inference", "infer"}:
        return data_cfg.get("val_slice_selection", data_cfg.get("slice_selection"))
    if resolved_usage == "uniform":
        return {"mode": "uniform"}
    raise ValueError("slice_selection_usage must be one of: train, val, inference, uniform")


def collect_gt_boxes(sample: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any] | None]]]:
    """Collect GT boxes into side -> plane -> per-slice normalized boxes."""

    result = {
        side: {
            "sagittal": _empty_plane_boxes(sample["sagittal_volume"].shape[1]),
            "coronal": _empty_plane_boxes(sample["coronal_volume"].shape[1]),
        }
        for side in ("medial", "lateral")
    }
    plane_shapes = {
        "sagittal": sample["sagittal_volume"].shape[-2:],
        "coronal": sample["coronal_volume"].shape[-2:],
    }
    for label in sample["labels"].get("module1_labels", []):
        plane = label["plane"]
        side = label["meniscus_side"]
        height, width = plane_shapes[plane]
        box = _normalize_label_box(label, width=width, height=height)
        slice_idx = int(label["slice_index"])
        current = result[side][plane][slice_idx]
        score = 1.0
        if current is None or score >= float(current["score"]):
            result[side][plane][slice_idx] = {"box": box, "score": score, "source": "gt"}
    return result


@torch.inference_mode()
def collect_predicted_boxes(
    model: torch.nn.Module,
    sample: dict[str, Any],
    device: torch.device,
    score_threshold: float,
    amp: bool,
    final_score_mode: str = "objectness",
    quality_score_alpha: float = 1.0,
    quality_score_beta: float = 1.0,
    topk_per_image: int | None = None,
    nms_threshold: float | None = None,
) -> dict[str, dict[str, list[dict[str, Any] | None]]]:
    """Collect highest-confidence predicted boxes per side/plane/slice."""

    sagittal = sample["sagittal_volume"].unsqueeze(0).to(device)
    coronal = sample["coronal_volume"].unsqueeze(0).to(device)
    sagittal_coords = sample["sagittal_coords"].unsqueeze(0).to(device)
    coronal_coords = sample["coronal_coords"].unsqueeze(0).to(device)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        pred = model(sagittal, coronal, sagittal_coords, coronal_coords)

    result = {
        side: {
            "sagittal": _empty_plane_boxes(sample["sagittal_volume"].shape[1]),
            "coronal": _empty_plane_boxes(sample["coronal_volume"].shape[1]),
        }
        for side in ("medial", "lateral")
    }
    filtered = filter_module1_predictions(
        pred,
        score_threshold=score_threshold,
        topk_per_image=topk_per_image,
        nms_threshold=nms_threshold,
        final_score_mode=final_score_mode,
        quality_score_alpha=quality_score_alpha,
        quality_score_beta=quality_score_beta,
    )[0]
    for box, score_tensor, plane_id_tensor, slice_idx_tensor, side_id_tensor in zip(
        filtered["boxes"],
        filtered["scores"],
        filtered["plane_ids"],
        filtered["slice_indices"],
        filtered["side_ids"],
    ):
        plane = "sagittal" if int(plane_id_tensor) == 0 else "coronal"
        side = ID_TO_SIDE[int(side_id_tensor)]
        slice_idx = int(slice_idx_tensor)
        score = float(score_tensor)
        current = result[side][plane][slice_idx]
        if current is None or score > float(current["score"]):
            result[side][plane][slice_idx] = {
                "box": _box_clip(box.detach().cpu()),
                "score": score,
                "source": "pred",
            }
    return result


def fill_missing_boxes(boxes: list[dict[str, Any] | None], strategy: str = "nearest") -> list[dict[str, Any]]:
    """Fill missing slice boxes by nearest/interpolation/highest-confidence strategy."""

    available = [(idx, item) for idx, item in enumerate(boxes) if item is not None]
    if not available:
        center_box = torch.tensor([0.25, 0.25, 0.75, 0.75], dtype=torch.float32)
        return [{"box": center_box, "score": 0.0, "source": "fallback_center"} for _ in boxes]
    if strategy == "highest_confidence":
        best_idx, best = max(available, key=lambda pair: float(pair[1]["score"]))
        del best_idx
        return [{"box": best["box"].clone(), "score": float(best["score"]), "source": "filled_highest"} for _ in boxes]
    filled: list[dict[str, Any]] = []
    for idx, item in enumerate(boxes):
        if item is not None:
            filled.append({"box": item["box"].clone(), "score": float(item["score"]), "source": item["source"]})
            continue
        before = [(j, v) for j, v in available if j < idx]
        after = [(j, v) for j, v in available if j > idx]
        if strategy == "interpolate" and before and after:
            left_idx, left = before[-1]
            right_idx, right = after[0]
            alpha = (idx - left_idx) / float(right_idx - left_idx)
            box = left["box"] * (1.0 - alpha) + right["box"] * alpha
            filled.append({"box": _box_clip(box), "score": min(float(left["score"]), float(right["score"])), "source": "filled_interpolate"})
        else:
            nearest_idx, nearest = min(available, key=lambda pair: abs(pair[0] - idx))
            del nearest_idx
            filled.append({"box": nearest["box"].clone(), "score": float(nearest["score"]), "source": "filled_nearest"})
    return filled


def crop_roi_sequence(volume: torch.Tensor, boxes: list[dict[str, Any]], crop_size: int) -> torch.Tensor:
    """Crop and resize a per-slice ROI sequence.

    Args:
        volume: Tensor ``[1,T,H,W]``.
        boxes: List of normalized ``xyxy`` boxes, one per slice.
        crop_size: Output H/W.

    Returns:
        Tensor ``[1,T,crop_size,crop_size]``.
    """

    _, num_slices, height, width = volume.shape
    if len(boxes) != num_slices:
        raise ValueError(f"expected {num_slices} ROI boxes, got {len(boxes)}")

    box_tensor = torch.stack(
        [torch.as_tensor(item["box"], device=volume.device, dtype=volume.dtype) for item in boxes]
    ).clamp(0.0, 1.0)
    x1 = torch.floor(box_tensor[:, 0] * width).clamp(0, width - 1)
    y1 = torch.floor(box_tensor[:, 1] * height).clamp(0, height - 1)
    x2 = torch.ceil(box_tensor[:, 2] * width).clamp(1, width)
    y2 = torch.ceil(box_tensor[:, 3] * height).clamp(1, height)
    x2 = torch.maximum(x2, x1 + 1).clamp_max(width)
    y2 = torch.maximum(y2, y1 + 1).clamp_max(height)

    # Treat MRI slices as a batch and crop all T planes in one grid_sample.
    # Clamp sampling coordinates to each crop's edge pixels to exactly match
    # integer slicing followed by F.interpolate(..., align_corners=False),
    # including one-pixel-wide boxes.
    output_positions = torch.arange(int(crop_size), device=volume.device, dtype=volume.dtype) + 0.5
    source_x = x1[:, None] + output_positions[None, :] * (x2 - x1)[:, None] / float(crop_size) - 0.5
    source_y = y1[:, None] + output_positions[None, :] * (y2 - y1)[:, None] / float(crop_size) - 0.5
    source_x = torch.minimum(torch.maximum(source_x, x1[:, None]), (x2 - 1)[:, None])
    source_y = torch.minimum(torch.maximum(source_y, y1[:, None]), (y2 - 1)[:, None])
    normalized_x = 2.0 * (source_x + 0.5) / float(width) - 1.0
    normalized_y = 2.0 * (source_y + 0.5) / float(height) - 1.0
    grid = volume.new_empty((num_slices, int(crop_size), int(crop_size), 2))
    grid[..., 0] = normalized_x[:, None, :]
    grid[..., 1] = normalized_y[:, :, None]
    slice_batch = volume.squeeze(0).unsqueeze(1)
    crops = F.grid_sample(slice_batch, grid, mode="bilinear", align_corners=False)
    return crops.permute(1, 0, 2, 3).contiguous()


def labels_to_ids(labels: dict[str, Any]) -> dict[str, int]:
    module2 = labels.get("module2_labels", {})
    module3 = labels.get("module3_labels", {})
    module4 = labels.get("module4_labels", {})
    return {
        "medial_status": STATUS_TO_ID.get(module2.get("medial_status", "normal"), 0),
        "lateral_status": STATUS_TO_ID.get(module2.get("lateral_status", "normal"), 0),
        "medial_bucket_handle": int(module3.get("medial_bucket_handle", 0)),
        "lateral_bucket_handle": int(module3.get("lateral_bucket_handle", 0)),
        "lateral_discoid": int(module4.get("lateral_discoid", 0)),
    }


def generate_rois(
    index: str,
    module1_checkpoint: str,
    output_dir: str,
    config_path: str = "configs/module1_meniscovit_cop.yaml",
    use_gt_boxes: bool = False,
    crop_size: int = 112,
    fill_strategy: str = "nearest",
    score_threshold: float | None = None,
    slice_selection_usage: str | None = None,
    save_visualizations: bool = False,
    storage_dtype: str = "float16",
    num_workers: int = 0,
    max_cases: int | None = None,
) -> None:
    """Generate and cache ROI tensors for Module 2/3."""

    config = load_config(config_path)
    resolved_score_threshold = resolve_score_threshold(config, score_threshold)
    logger = get_logger("generate_rois")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vis_dir = output_path / "visualizations"
    slice_selection = resolve_slice_selection(config, slice_selection_usage, use_gt_boxes=use_gt_boxes)
    dataset = MeniscusDetectionDataset(
        index_jsonl=index,
        output_size=config["data"]["output_size"],
        target_slices=config["data"]["num_slices"],
        patch_size=config["data"]["patch_size"],
        slice_selection=slice_selection,
        transform=None,
        return_coordinates=not use_gt_boxes,
    )
    total_cases = len(dataset) if max_cases is None else min(len(dataset), max(0, int(max_cases)))
    loader_dataset = dataset if total_cases == len(dataset) else Subset(dataset, range(total_cases))
    worker_count = max(0, int(num_workers))
    loader_kwargs: dict[str, Any] = {
        "dataset": loader_dataset,
        "batch_size": None,
        "shuffle": False,
        "num_workers": worker_count,
    }
    if worker_count > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(**loader_kwargs)

    model = None
    device = torch.device("cpu")
    if not use_gt_boxes:
        device_name = config.get("device", "cuda")
        device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
        model = build_module1_model(config).to(device)
        load_checkpoint(module1_checkpoint, model, map_location=device)
        model.eval()
        logger.info(f"loaded Module 1 checkpoint from {module1_checkpoint} on {device}")
        logger.info(f"using Module 1 score_threshold={resolved_score_threshold:.4g}")

    normalized_storage_dtype = str(storage_dtype).lower()
    storage_types = {"float16": torch.float16, "float32": torch.float32}
    if normalized_storage_dtype not in storage_types:
        raise ValueError("storage_dtype must be float16 or float32")
    cache_dtype = storage_types[normalized_storage_dtype]
    index_records = []
    for idx, sample in enumerate(loader):
        case_id = sample["case_id"]
        group_id = sample.get("group_id", case_id)
        raw_boxes = collect_gt_boxes(sample) if use_gt_boxes else collect_predicted_boxes(
            model=model,
            sample=sample,
            device=device,
            score_threshold=resolved_score_threshold,
            amp=config.get("train", {}).get("amp", True),
            final_score_mode=config.get("validate", {}).get("final_score_mode", "objectness"),
            quality_score_alpha=float(config.get("validate", {}).get("quality_score_alpha", 1.0)),
            quality_score_beta=float(config.get("validate", {}).get("quality_score_beta", 1.0)),
            topk_per_image=config.get("validate", {}).get("topk_per_image"),
            nms_threshold=config.get("validate", {}).get("nms_threshold"),
        )
        payload: dict[str, Any] = {
            "case_id": case_id,
            "group_id": group_id,
            "labels": labels_to_ids(sample["labels"]),
        }
        for side in ("medial", "lateral"):
            side_payload: dict[str, Any] = {}
            boxes_payload: dict[str, Any] = {}
            for plane in ("sagittal", "coronal"):
                filled = fill_missing_boxes(raw_boxes[side][plane], strategy=fill_strategy)
                volume = sample[f"{plane}_volume"]
                roi = crop_roi_sequence(volume, filled, crop_size=crop_size)
                side_payload[f"{plane}_roi"] = roi.to(dtype=cache_dtype)
                boxes_payload[plane] = [
                    {
                        "box": item["box"].tolist(),
                        "score": float(item["score"]),
                        "source": item["source"],
                    }
                    for item in filled
                ]
                if save_visualizations:
                    mid = volume.shape[1] // 2
                    save_slice_overlay(
                        volume,
                        vis_dir / f"{case_id}_{side}_{plane}_bbox.png",
                        boxes_xyxy=[filled[mid]["box"].tolist()],
                        slice_index=mid,
                        title=f"{case_id} {side} {plane}",
                    )
            side_payload["boxes"] = boxes_payload
            payload[side] = side_payload
            if save_visualizations:
                save_roi_preview(payload[side]["sagittal_roi"], payload[side]["coronal_roi"], vis_dir / f"{case_id}_{side}_roi.png", title=f"{case_id} {side}")

        roi_path = output_path / f"{case_id}.pt"
        torch.save(payload, roi_path)
        index_records.append(
            {
                "case_id": case_id,
                "group_id": group_id,
                "roi_path": str(roi_path),
                "labels": payload["labels"],
            }
        )
        if (idx + 1) % 25 == 0 or idx + 1 == total_cases:
            logger.info(f"saved ROI caches: {idx + 1}/{total_cases} (latest={roi_path})")

    with (output_path / "index.jsonl").open("w", encoding="utf-8") as handle:
        for record in index_records:
            handle.write(json.dumps(record) + "\n")
    logger.info(f"saved ROI index: {output_path / 'index.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/module1_meniscovit_cop.yaml")
    parser.add_argument("--index", required=True)
    parser.add_argument("--module1-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-gt-boxes", default="false")
    parser.add_argument("--crop-size", type=int, default=112)
    parser.add_argument("--fill-strategy", choices=["nearest", "interpolate", "highest_confidence"], default="nearest")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--slice-selection-usage", choices=["train", "val", "validate", "inference", "infer", "uniform"], default=None)
    parser.add_argument("--save-visualizations", default="false")
    parser.add_argument("--storage-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--num-workers", type=int, default=0, help="Parallel DICOM-loading workers.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional smoke-test limit.")
    args = parser.parse_args()
    generate_rois(
        index=args.index,
        module1_checkpoint=args.module1_checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
        use_gt_boxes=str2bool(args.use_gt_boxes),
        crop_size=args.crop_size,
        fill_strategy=args.fill_strategy,
        score_threshold=args.score_threshold,
        slice_selection_usage=args.slice_selection_usage,
        save_visualizations=str2bool(args.save_visualizations),
        storage_dtype=args.storage_dtype,
        num_workers=args.num_workers,
        max_cases=args.max_cases,
    )


if __name__ == "__main__":
    main()
