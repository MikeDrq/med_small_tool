"""Run realistic-shape CUDA smoke tests for the paper reproduction models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.train_module1 import build_loss
from engine.validate_module1 import build_module1_model
from engine.validate_module2 import build_module2_model, roi_collate
from datasets.roi_classification_dataset import ROIClassificationDataset


def _coordinates(depth: int, height: int, width: int, patch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.linspace(-1.0, 1.0, depth, device=device)
    y = torch.linspace(-1.0, 1.0, height // patch, device=device)
    x = torch.linspace(-1.0, 1.0, width // patch, device=device)
    tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
    sagittal = torch.stack([tt, yy, xx], dim=-1).reshape(1, -1, 3)
    coronal = torch.stack([xx, tt, yy], dim=-1).reshape(1, -1, 3)
    return sagittal, coronal


def smoke_module1(config_path: str, backward: bool, batch_size: int) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device("cuda")
    depth = int(config["data"]["num_slices"])
    size = int(config["data"]["output_size"])
    patch = int(config["data"]["patch_size"])
    torch.cuda.reset_peak_memory_stats()
    model = build_module1_model(config).to(device).train(backward)
    criterion = build_loss(config).to(device)
    criterion.set_epoch(2)
    sagittal = torch.rand(batch_size, 1, depth, size, size, device=device)
    coronal = torch.rand_like(sagittal)
    sagittal_coords, coronal_coords = _coordinates(depth, size, size, patch, device)
    sagittal_coords = sagittal_coords.expand(batch_size, -1, -1).contiguous()
    coronal_coords = coronal_coords.expand(batch_size, -1, -1).contiguous()
    labels = [
        {
            "module1_labels": [
                {"plane": plane, "slice_index": depth // 2, "bbox": box, "meniscus_side": side, "abnormality": abnormality}
                for plane in ("sagittal", "coronal")
                for side, box, abnormality in (
                    ("medial", [80.0, 180.0, 230.0, 310.0], "abnormal"),
                    ("lateral", [285.0, 180.0, 430.0, 310.0], "normal"),
                )
            ]
        }
        for _ in range(batch_size)
    ]
    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        predictions = model(sagittal, coronal, sagittal_coords, coronal_coords)
        loss = criterion(predictions, labels)["loss"]
    if backward:
        loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"module1 shape={batch_size}x1x{depth}x{size}x{size} backward={backward} "
        f"loss={float(loss.detach()):.6f} elapsed={elapsed:.2f}s peak_allocated={peak_gib:.2f}GiB"
    )


def smoke_module2(config_path: str, backward: bool, batch_size: int, roi_index: str | None = None) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device("cuda")
    depth, height, width = map(int, config["data"]["roi_shape"])
    model = build_module2_model(config).to(device).train(backward)
    if roi_index:
        dataset = ROIClassificationDataset(roi_index, task="module2")
        if len(dataset) < batch_size:
            raise ValueError(f"ROI smoke index contains {len(dataset)} side samples, need {batch_size}")
        batch = roi_collate([dataset[index] for index in range(batch_size)])
        sagittal_roi = batch["sagittal_roi"].to(device)
        coronal_roi = batch["coronal_roi"].to(device)
        labels = batch["label"].to(device)
        source = roi_index
    else:
        sagittal_roi = torch.rand(batch_size, 1, depth, height, width, device=device)
        coronal_roi = sagittal_roi
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        source = "synthetic"
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    context = torch.enable_grad() if backward else torch.inference_mode()
    with context, torch.autocast("cuda", dtype=torch.float16):
        logits = model(sagittal_roi, coronal_roi)
        loss = torch.nn.functional.cross_entropy(logits, labels)
    if backward:
        loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"module2 source={source} shape={tuple(sagittal_roi.shape)} logits={tuple(logits.shape)} "
        f"backward={backward} loss={float(loss.detach()):.6f} elapsed={elapsed:.2f}s "
        f"peak_allocated={peak_gib:.2f}GiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module1-config", default="configs/module1_meniscovit_cop.yaml")
    parser.add_argument("--module2-config", default="configs/module2_swin.yaml")
    parser.add_argument("--module1-backward", action="store_true")
    parser.add_argument("--module1-batch-size", type=int, default=1)
    parser.add_argument("--module2-backward", action="store_true")
    parser.add_argument("--module2-batch-size", type=int, default=1)
    parser.add_argument("--module2-roi-index", default=None, help="Optional real generated ROI JSONL for Module 2.")
    parser.add_argument("--skip-module1", action="store_true")
    parser.add_argument("--skip-module2", action="store_true")
    args = parser.parse_args()
    if not args.skip_module1:
        smoke_module1(args.module1_config, backward=args.module1_backward, batch_size=args.module1_batch_size)
    if not args.skip_module2:
        smoke_module2(
            args.module2_config,
            backward=args.module2_backward,
            batch_size=args.module2_batch_size,
            roi_index=args.module2_roi_index,
        )


if __name__ == "__main__":
    main()
