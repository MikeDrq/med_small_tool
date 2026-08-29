"""Training entrypoint for Module 3 bucket-handle Video Swin classifier."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.roi_classification_dataset import ROIClassificationDataset
from datasets.loader_utils import worker_loader_kwargs
from datasets.transforms import RandomROIAugmentation
from engine.validate_module2 import build_module2_model
from engine.validate_module3 import evaluate_module3, move_batch_to_device, roi_binary_collate
from losses.focal_loss import BinaryFocalLossWithLogits
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logging import format_metrics, get_logger


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _record_label(record: dict[str, Any]) -> int:
    if "module3_label" in record:
        return int(record["module3_label"])
    payload = torch.load(record["roi_path"], map_location="cpu", weights_only=True)
    return int(payload["labels"][f"{record['side']}_bucket_handle"])


def build_sampler(
    dataset: ROIClassificationDataset,
    positive_fraction: float = 0.25,
) -> WeightedRandomSampler:
    labels = [_record_label(record) for record in dataset.records]
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    class_mass = np.asarray([1.0 - positive_fraction, positive_fraction], dtype=np.float64)
    weights_per_class = class_mass / np.maximum(counts, 1.0)
    sample_weights = [float(weights_per_class[label]) for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def create_train_loader(config: dict[str, Any]) -> DataLoader:
    augmentation = None
    if bool(config["data"].get("train_augmentation", False)):
        augmentation = RandomROIAugmentation(**dict(config["data"].get("train_augmentation_params", {})))
    dataset = ROIClassificationDataset(
        config["data"]["train_rois"],
        task="module3",
        sides=tuple(config["data"].get("sides", ["medial", "lateral"])),
        module3_tear_only=bool(config["data"].get("tear_only", True)),
        transform=augmentation,
    )
    sampler = None
    if config["data"].get("oversampling", True):
        sampler = build_sampler(
            dataset,
            positive_fraction=float(config["data"].get("positive_sampling_fraction", 0.25)),
        )
    return DataLoader(
        dataset,
        batch_size=config["train"].get("batch_size", 2),
        shuffle=sampler is None,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
        collate_fn=roi_binary_collate,
        drop_last=False,
        **worker_loader_kwargs(config["data"]),
    )


def create_val_loader(config: dict[str, Any]) -> DataLoader:
    dataset = ROIClassificationDataset(
        config["data"]["val_rois"],
        task="module3",
        sides=tuple(config["data"].get("sides", ["medial", "lateral"])),
        module3_tear_only=bool(config["data"].get("tear_only", True)),
    )
    return DataLoader(
        dataset,
        batch_size=config.get("validate", {}).get("batch_size", 2),
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=roi_binary_collate,
        **worker_loader_kwargs(config["data"]),
    )


def build_loss(config: dict[str, Any]) -> BinaryFocalLossWithLogits:
    loss_cfg = config["loss"]
    return BinaryFocalLossWithLogits(alpha=float(loss_cfg.get("focal_alpha", 0.25)), gamma=float(loss_cfg.get("focal_gamma", 2.0)))


def build_optimizer(config: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    train_cfg = config["train"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.05),
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
    )


def build_scheduler(config: dict[str, Any], optimizer: torch.optim.Optimizer, steps_per_epoch: int) -> torch.optim.lr_scheduler.LambdaLR:
    train_cfg = config["train"]
    total_steps = max(1, int(train_cfg["epochs"]) * steps_per_epoch)
    warmup_steps = max(1, int(train_cfg.get("warmup_epochs", 2)) * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: torch.nn.Module,
    criterion: BinaryFocalLossWithLogits,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    config: dict[str, Any],
    logger: Any,
) -> float:
    model.train()
    grad_accum_steps = int(config["train"].get("grad_accum_steps", 1))
    amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    device_type = "cuda" if device.type == "cuda" else "cpu"
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp):
            logits = model(batch["sagittal_roi"], batch["coronal_roi"]).squeeze(-1)
            loss = criterion(logits, batch["label"]) / grad_accum_steps
        scaler.scale(loss).backward()
        if step % grad_accum_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        full_loss = float((loss * grad_accum_steps).detach().cpu())
        running += full_loss
        if step % int(config["train"].get("log_interval", 10)) == 0:
            logger.info(f"epoch={epoch} step={step}/{len(loader)} lr={scheduler.get_last_lr()[0]:.6g} {format_metrics({'loss': full_loss})}")
    return running / max(len(loader), 1)


def train_module3(config_path: str, resume: str | None = None) -> None:
    config = load_config(config_path)
    if resume is not None:
        config["train"]["resume"] = resume
    set_seed(int(config.get("seed", 42)))
    output_dir = Path(config["train"].get("output_dir", "checkpoints/module3"))
    logger = get_logger("train_module3", log_file=output_dir / "train.log")
    device_name = config.get("device", "cuda")
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    train_loader = create_train_loader(config)
    val_loader = create_val_loader(config)
    model = build_module2_model(config).to(device)
    criterion = build_loss(config)
    optimizer = build_optimizer(config, model)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / int(config["train"].get("grad_accum_steps", 1))))
    scheduler = build_scheduler(config, optimizer, steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["train"].get("amp", True)) and device.type == "cuda")

    start_epoch = 1
    best_metric = -float("inf")
    resume = config["train"].get("resume")
    if resume:
        metadata = load_checkpoint(resume, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, map_location=device)
        start_epoch = int(metadata.get("epoch", 0)) + 1
        best_metric = float(metadata.get("best_metric", best_metric))
        logger.info(f"resumed from {resume} at epoch {start_epoch}")

    monitor = config["train"].get("save_best_metric", "overall_roc_auc")
    threshold_cfg = config.get("validate", {}).get("threshold_selection", {})
    for epoch in range(start_epoch, int(config["train"]["epochs"]) + 1):
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, scheduler, scaler, device, epoch, config, logger)
        metrics = evaluate_module3(
            model,
            val_loader,
            device,
            threshold=float(config.get("validate", {}).get("threshold", 0.5)),
            amp=config["train"].get("amp", True),
            bootstrap_samples=0,
            optimize_threshold=bool(threshold_cfg.get("enabled", False)),
            threshold_minimum=float(threshold_cfg.get("minimum", 0.05)),
            threshold_maximum=float(threshold_cfg.get("maximum", 0.95)),
        )
        metrics["train_loss"] = train_loss
        logger.info(f"epoch={epoch} validation {format_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})}")

        current = float(metrics.get(monitor, -float("inf")))
        is_best = current == current and current > best_metric
        if is_best:
            best_metric = current
        decision_threshold = float(metrics["decision_threshold"])
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_metric,
            decision_threshold=decision_threshold,
            metrics=metrics,
        )
        if is_best:
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                decision_threshold=decision_threshold,
                metrics=metrics,
            )
            logger.info(
                f"saved best checkpoint: {output_dir / 'best.pt'} "
                f"{monitor}={best_metric:.4f} threshold={decision_threshold:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/module3_bucket_swin.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train_module3(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
