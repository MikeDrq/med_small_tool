"""Run one full Module 1 optimizer step on real converted DICOM data."""

from __future__ import annotations

import argparse
import json
import time

import torch

from engine.train_module1 import (
    build_loss,
    build_optimizer,
    create_train_loader,
    load_config,
    set_seed,
)
from engine.validate_module1 import build_module1_model, move_batch_to_device


def run(config_path: str) -> dict[str, object]:
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires CUDA")
    device = torch.device("cuda")
    loader = create_train_loader(config)
    load_start = time.perf_counter()
    batch = next(iter(loader))
    load_seconds = time.perf_counter() - load_start

    model = build_module1_model(config).to(device).train()
    criterion = build_loss(config)
    criterion.set_epoch(1)
    optimizer = build_optimizer(config, model)
    batch = move_batch_to_device(batch, device)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    step_start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        predictions = model(
            batch["sagittal_volume"],
            batch["coronal_volume"],
            batch["sagittal_coords"],
            batch["coronal_coords"],
        )
        losses = criterion(predictions, batch["labels"])
    losses["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    torch.cuda.synchronize(device)
    step_seconds = time.perf_counter() - step_start

    mapping_totals = {
        key: sum(int(labels["module1_label_mapping"].get(key, 0)) for labels in batch["labels"])
        for key in (
            "raw_label_count",
            "exact_source_label_count",
            "remapped_source_label_count",
            "dropped_source_label_count",
            "processed_label_count",
        )
    }
    report = getattr(model.backbone, "pretrained_load_report", None)
    return {
        "config": config_path,
        "cases": list(batch["case_id"]),
        "batch_size": int(batch["sagittal_volume"].shape[0]),
        "volume_shape": list(batch["sagittal_volume"].shape),
        "train_records": len(loader.dataset.records),
        "train_virtual_samples": len(loader.dataset),
        "load_seconds": load_seconds,
        "optimizer_step_seconds": step_seconds,
        "peak_cuda_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "pretrained": report,
        "label_mapping": mapping_totals,
        "losses": {
            key: float(value.detach().float().cpu())
            for key, value in losses.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/module1_meniscovit_cop.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
