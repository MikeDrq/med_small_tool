"""Training entrypoint for Module 1 MeniscoViT-CoP."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.meniscus_detection_dataset import MeniscusDetectionDataset
from datasets.loader_utils import worker_loader_kwargs
from datasets.transforms import RandomTrainingAugmentation
from engine.validate_module1 import build_module1_model, evaluate_module1, module1_collate, move_batch_to_device
from losses.detection_loss import Module1DetectionLoss
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


def build_training_transform(config: dict[str, Any]) -> RandomTrainingAugmentation | None:
    """Build a coordinate-safe training transform from backward-compatible config."""

    data_cfg = config["data"]
    if not bool(data_cfg.get("train_augmentation", True)):
        return None
    params = dict(data_cfg.get("train_augmentation_params", {}))
    return RandomTrainingAugmentation(
        seed=int(config.get("seed", 42)),
        mode=str(data_cfg.get("train_augmentation_mode", "photometric")),
        **params,
    )


def create_train_loader(config: dict[str, Any]) -> DataLoader:
    data_cfg = config["data"]
    train_cfg = config["train"]
    transform = build_training_transform(config)
    dataset = MeniscusDetectionDataset(
        index_jsonl=data_cfg["train_index"],
        output_size=data_cfg["output_size"],
        target_slices=data_cfg["num_slices"],
        patch_size=data_cfg["patch_size"],
        slice_selection=data_cfg.get("train_slice_selection", data_cfg.get("slice_selection")),
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 1),
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        collate_fn=module1_collate,
        drop_last=False,
        **worker_loader_kwargs(data_cfg),
    )


def create_val_loader(config: dict[str, Any]) -> DataLoader:
    data_cfg = config["data"]
    val_cfg = config.get("validate", {})
    dataset = MeniscusDetectionDataset(
        index_jsonl=data_cfg["val_index"],
        output_size=data_cfg["output_size"],
        target_slices=data_cfg["num_slices"],
        patch_size=data_cfg["patch_size"],
        slice_selection=data_cfg.get("val_slice_selection", data_cfg.get("slice_selection")),
        transform=None,
    )
    return DataLoader(
        dataset,
        batch_size=val_cfg.get("batch_size", 1),
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=module1_collate,
        drop_last=False,
        **worker_loader_kwargs(data_cfg),
    )


def build_loss(config: dict[str, Any]) -> Module1DetectionLoss:
    loss_cfg = config["loss"]
    use_bbox_quality_head = bool(config.get("model", {}).get("use_bbox_quality_head", False))
    return Module1DetectionLoss(
        objectness_weight=loss_cfg.get("objectness_weight", 1.0),
        bbox_weight=loss_cfg.get("bbox_weight", 1.0),
        side_weight=loss_cfg.get("side_weight", 0.5),
        abnormality_weight=loss_cfg.get("abnormality_weight", 1.0),
        bbox_loss_type=loss_cfg.get("bbox_loss_type", "smooth_l1"),
        objectness_loss_type=loss_cfg.get("objectness_loss", loss_cfg.get("objectness_loss_type", "focal")),
        objectness_positive_weight=loss_cfg.get("objectness_positive_weight", 1.0),
        objectness_negative_weight=loss_cfg.get("objectness_negative_weight", 1.0),
        objectness_normalization=loss_cfg.get("objectness_normalization", "positive_count"),
        objectness_target_mode=loss_cfg.get("objectness_target_mode", "assigned"),
        quality_iou_threshold=loss_cfg.get("quality_iou_threshold", 0.3),
        quality_iou_start_epoch=loss_cfg.get("quality_iou_start_epoch", 30),
        quality_iou_stop_gradient=loss_cfg.get("quality_iou_stop_gradient", True),
        task_aligned_alpha=loss_cfg.get("task_aligned_alpha", 1.0),
        task_aligned_beta=loss_cfg.get("task_aligned_beta", 6.0),
        task_aligned_topk=loss_cfg.get("task_aligned_topk", 10),
        task_aligned_start_epoch=loss_cfg.get("task_aligned_start_epoch", 30),
        bbox_quality_loss_weight=loss_cfg.get("bbox_quality_loss_weight", 1.0 if use_bbox_quality_head else 0.0),
        bbox_quality_loss_type=loss_cfg.get("bbox_quality_loss_type", "bce"),
        bbox_quality_target_mode=loss_cfg.get("bbox_quality_target_mode", "iou"),
        bbox_quality_train_mask=loss_cfg.get("bbox_quality_train_mask", "topk_by_alignment"),
        bbox_quality_iou_threshold=loss_cfg.get("bbox_quality_iou_threshold", 0.3),
        bbox_quality_topk=loss_cfg.get("bbox_quality_topk", 100),
        bbox_quality_start_epoch=loss_cfg.get("bbox_quality_start_epoch", 30),
        quality_pos_iou_threshold=loss_cfg.get("quality_pos_iou_threshold", 0.3),
        quality_neg_iou_threshold=loss_cfg.get("quality_neg_iou_threshold", 0.1),
        quality_num_pos=loss_cfg.get("quality_num_pos", 256),
        quality_num_neg=loss_cfg.get("quality_num_neg", 512),
        quality_num_mid=loss_cfg.get("quality_num_mid", 256),
        quality_hard_negative_source=loss_cfg.get("quality_hard_negative_source", "objectness_or_alignment"),
        quality_mid_weight=loss_cfg.get("quality_mid_weight", 0.25),
        quality_neg_target_mode=loss_cfg.get("quality_neg_target_mode", "zero"),
        quality_sampling_per_group=loss_cfg.get("quality_sampling_per_group", True),
        quality_sample_total=loss_cfg.get("quality_sample_total", 1024),
        quality_high_iou_fraction=loss_cfg.get("quality_high_iou_fraction", 0.25),
        quality_mid_iou_fraction=loss_cfg.get("quality_mid_iou_fraction", 0.25),
        quality_low_iou_fraction=loss_cfg.get("quality_low_iou_fraction", 0.50),
        use_quality_ranking_loss=loss_cfg.get("use_quality_ranking_loss", False),
        quality_ranking_loss_weight=loss_cfg.get("quality_ranking_loss_weight", 0.5),
        ranking_pos_iou_threshold=loss_cfg.get("ranking_pos_iou_threshold", 0.3),
        ranking_neg_iou_threshold=loss_cfg.get("ranking_neg_iou_threshold", 0.1),
        ranking_margin=loss_cfg.get("ranking_margin", 0.2),
        ranking_num_pairs=loss_cfg.get("ranking_num_pairs", 512),
        quality_ranking_score_space=loss_cfg.get("quality_ranking_score_space", "logits"),
        bbox_quality_varifocal_alpha=loss_cfg.get("bbox_quality_varifocal_alpha", loss_cfg.get("focal_alpha", 0.75)),
        bbox_quality_varifocal_gamma=loss_cfg.get("bbox_quality_varifocal_gamma", loss_cfg.get("focal_gamma", 2.0)),
        bbox_quality_qfl_beta=loss_cfg.get("bbox_quality_qfl_beta", 2.0),
        focal_alpha=loss_cfg.get("focal_alpha", 0.25),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
        positive_radius_tokens=loss_cfg.get("positive_radius_tokens", 0),
    )


def build_optimizer(config: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    train_cfg = config["train"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.005),
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
    )


def build_scheduler(config: dict[str, Any], optimizer: torch.optim.Optimizer, steps_per_epoch: int) -> torch.optim.lr_scheduler.LambdaLR:
    train_cfg = config["train"]
    if str(train_cfg.get("scheduler", "cosine")).lower() == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
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
    criterion: Module1DetectionLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    config: dict[str, Any],
    logger: Any,
) -> dict[str, float]:
    model.train()
    grad_accum_steps = int(config["train"].get("grad_accum_steps", 1))
    amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    device_type = "cuda" if device.type == "cuda" else "cpu"
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_metrics: dict[str, float] = {}
    criterion.set_epoch(epoch)

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp):
            predictions = model(batch["sagittal_volume"], batch["coronal_volume"], batch["sagittal_coords"], batch["coronal_coords"])
            loss_dict = criterion(predictions, batch["labels"])
            loss = loss_dict["loss"] / grad_accum_steps
        scaler.scale(loss).backward()
        if step % grad_accum_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        running_loss += float(loss_dict["loss"].detach().cpu())
        for key, value in loss_dict.items():
            running_metrics[key] = running_metrics.get(key, 0.0) + float(value.detach().cpu())
        if step % int(config["train"].get("log_interval", 10)) == 0:
            log_values = {key: float(value.detach().cpu()) for key, value in loss_dict.items()}
            logger.info(f"epoch={epoch} step={step}/{len(loader)} lr={scheduler.get_last_lr()[0]:.6g} {format_metrics(log_values)}")
    averaged = {key: value / max(len(loader), 1) for key, value in running_metrics.items()}
    averaged["loss"] = running_loss / max(len(loader), 1)
    return averaged


def train_module1(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))
    output_dir = Path(config["train"].get("output_dir", "checkpoints/module1"))
    logger = get_logger("train_module1", log_file=output_dir / "train.log")
    device_name = config.get("device", "cuda")
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")
    logger.info(
        "train_augmentation="
        + (
            str(config["data"].get("train_augmentation_mode", "photometric"))
            if config["data"].get("train_augmentation", True)
            else "none"
        )
    )

    train_loader = create_train_loader(config)
    val_loader = create_val_loader(config)
    logger.info(
        f"train_records={len(train_loader.dataset.records)} "
        f"train_virtual_samples={len(train_loader.dataset)} "
        f"val_records={len(val_loader.dataset.records)} val_samples={len(val_loader.dataset)}"
    )
    model = build_module1_model(config).to(device)
    criterion = build_loss(config)
    optimizer = build_optimizer(config, model)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / int(config["train"].get("grad_accum_steps", 1))))
    scheduler = build_scheduler(config, optimizer, steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["train"].get("amp", True)) and device.type == "cuda")

    start_epoch = 1
    best_metric = -float("inf")
    resume = config["train"].get("resume")
    if resume:
        if bool(config["train"].get("resume_weights_only", False)):
            metadata = load_checkpoint(resume, model, map_location=device)
            logger.info(f"loaded model weights from {resume}; optimizer/scheduler/scaler reset")
        else:
            metadata = load_checkpoint(resume, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, map_location=device)
            start_epoch = int(metadata.get("epoch", 0)) + 1
            best_metric = float(metadata.get("best_metric", best_metric))
            logger.info(f"resumed from {resume} at epoch {start_epoch}")

    epochs = int(config["train"]["epochs"])
    monitor = config["train"].get("save_best_metric", "objectness_token_pr_auc")
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, scheduler, scaler, device, epoch, config, logger)
        logger.info(
            f"epoch={epoch} train objectness_target_mode={criterion.effective_objectness_target_mode()} "
            f"{format_metrics(train_metrics)}"
        )
        metrics = evaluate_module1(
            model,
            val_loader,
            device,
            score_threshold=config.get("validate", {}).get("score_threshold", 0.5),
            iou_threshold=config.get("validate", {}).get("iou_threshold", 0.45),
            iou_thresholds=config.get("validate", {}).get("iou_thresholds"),
            topk_per_image=config.get("validate", {}).get("topk_per_image"),
            nms_threshold=config.get("validate", {}).get("nms_threshold"),
            final_score_mode=config.get("validate", {}).get("final_score_mode", "objectness"),
            quality_score_alpha=config.get("validate", {}).get("quality_score_alpha", 1.0),
            quality_score_beta=config.get("validate", {}).get("quality_score_beta", 1.0),
            amp=config["train"].get("amp", True),
            positive_radius_tokens=config.get("loss", {}).get("positive_radius_tokens", 0),
            quality_iou_threshold=config.get("loss", {}).get("quality_iou_threshold", 0.3),
            logger=logger,
            log_detection_diagnostics=bool(config.get("validate", {}).get("log_detection_diagnostics", False)),
            compute_extended_diagnostics=bool(
                config.get("validate", {}).get("compute_extended_diagnostics_during_training", False)
            ),
            progress_interval=int(config.get("validate", {}).get("progress_interval", 50)),
        )
        metrics["train_loss"] = train_metrics.get("loss", float("nan"))
        metrics["train_bbox_loss"] = train_metrics.get("loss_bbox", float("nan"))
        metrics["train_positive_bbox_loss"] = train_metrics.get("positive_bbox_loss", train_metrics.get("loss_bbox", float("nan")))
        logger.info(f"epoch={epoch} validation {format_metrics(metrics)}")

        current = metrics.get(monitor, -float("inf"))
        is_best = current == current and current > best_metric
        if is_best:
            best_metric = current
        save_checkpoint(output_dir / "last.pt", model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, best_metric=best_metric, metrics=metrics)
        if is_best:
            save_checkpoint(output_dir / "best.pt", model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, best_metric=best_metric, metrics=metrics)
            logger.info(f"saved best checkpoint: {output_dir / 'best.pt'} {monitor}={best_metric:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/module1_meniscovit_cop.yaml")
    args = parser.parse_args()
    train_module1(args.config)


if __name__ == "__main__":
    main()
