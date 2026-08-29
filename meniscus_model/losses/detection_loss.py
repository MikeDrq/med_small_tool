"""Loss and target assignment for Module 1 MeniscoViT-CoP.

paper-specified:
    Module 1 performs detection, medial/lateral classification, normal/abnormal
    classification, and abnormality localization.

paper-inferred implementation:
    The paper does not release detection-head assignment details. We assign
    positive labels to patch tokens whose normalized patch centers fall inside a
    labeled bbox on the matching plane/slice. If no token center lies inside a
    bbox, the nearest token on that slice is assigned positive. The optional
    positive radius expands each positive token to neighboring grid tokens on
    the same slice for tiny-data overfit debugging.

    The quality-aware and task-aligned objectness modes are experimental,
    paper-inferred engineering additions for Module 1 detection debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .focal_loss import BinaryFocalLossWithLogits


SIDE_TO_ID = {"medial": 0, "lateral": 1}
ABNORMALITY_TO_ID = {"normal": 0, "abnormal": 1}
PLANE_TO_ID = {"sagittal": 0, "coronal": 1}
BBOX_QUALITY_TRAIN_MASKS = {
    "assigned_positive",
    "task_aligned_positive",
    "iou_above_threshold",
    "topk_by_alignment",
    "hard_pos_neg_iou_sampling",
    "sampled_soft_iou",
}
BBOX_QUALITY_LOSS_TYPES = {
    "bce",
    "bce_soft_iou",
    "mse",
    "smooth_l1",
    "varifocal",
    "quality_focal",
    "pairwise_ranking",
    "varifocal_plus_ranking",
}
QUALITY_HARD_NEGATIVE_SOURCES = {
    "objectness",
    "task_alignment",
    "alignment",
    "quality",
    "objectness_or_alignment",
    "objectness_or_quality",
    "alignment_or_quality",
    "objectness_or_alignment_or_quality",
}


@dataclass(frozen=True)
class DetectionTargetConfig:
    """Target assignment configuration."""

    positive_radius_tokens: int = 0


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0)
    return wh[..., 0] * wh[..., 1]


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for normalized xyxy boxes."""

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = box_area(torch.cat([lt, rb], dim=-1))
    union = box_area(boxes1)[:, None] + box_area(boxes2)[None, :] - inter
    return inter / union.clamp_min(1e-6)


def generalized_box_iou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Mean GIoU loss for aligned normalized xyxy boxes."""

    if pred_boxes.numel() == 0:
        return pred_boxes.sum() * 0.0
    lt = torch.maximum(pred_boxes[:, :2], target_boxes[:, :2])
    rb = torch.minimum(pred_boxes[:, 2:], target_boxes[:, 2:])
    inter = box_area(torch.cat([lt, rb], dim=-1))
    area_pred = box_area(pred_boxes)
    area_tgt = box_area(target_boxes)
    union = area_pred + area_tgt - inter
    iou = inter / union.clamp_min(1e-6)

    c_lt = torch.minimum(pred_boxes[:, :2], target_boxes[:, :2])
    c_rb = torch.maximum(pred_boxes[:, 2:], target_boxes[:, 2:])
    c_area = box_area(torch.cat([c_lt, c_rb], dim=-1)).clamp_min(1e-6)
    giou = iou - (c_area - union) / c_area
    return (1.0 - giou).mean()


def _grid_centers(shape: dict[str, Any], device: torch.device) -> torch.Tensor:
    rows = (torch.arange(shape["grid_height"], device=device, dtype=torch.float32) + 0.5) / float(shape["grid_height"])
    cols = (torch.arange(shape["grid_width"], device=device, dtype=torch.float32) + 0.5) / float(shape["grid_width"])
    y, x = torch.meshgrid(rows, cols, indexing="ij")
    centers_2d = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)
    return centers_2d.repeat(shape["slices"], 1)


def _slice_indices(shape: dict[str, Any], device: torch.device) -> torch.Tensor:
    per_slice = int(shape["patches_per_slice"])
    return torch.arange(shape["slices"], device=device).repeat_interleave(per_slice)


def _token_plane_and_slice_ids(shape_metadata: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sagittal_shape = shape_metadata["sagittal"]
    coronal_shape = shape_metadata["coronal"]
    sag_slices = _slice_indices(sagittal_shape, device=device)
    cor_slices = _slice_indices(coronal_shape, device=device)
    plane_ids = torch.cat(
        [
            torch.zeros_like(sag_slices, dtype=torch.long),
            torch.ones_like(cor_slices, dtype=torch.long),
        ],
        dim=0,
    )
    slice_indices = torch.cat([sag_slices, cor_slices], dim=0)
    return plane_ids, slice_indices


def _expand_indices_by_radius(
    local_indices: torch.Tensor,
    shape: dict[str, Any],
    slice_index: int,
    radius: int,
    device: torch.device,
) -> torch.Tensor:
    if radius <= 0 or local_indices.numel() == 0:
        return local_indices

    grid_height = int(shape["grid_height"])
    grid_width = int(shape["grid_width"])
    per_slice = int(shape["patches_per_slice"])
    slice_start = int(slice_index) * per_slice
    expanded: list[int] = []
    for local_index in local_indices.detach().cpu().tolist():
        index_in_slice = int(local_index) - slice_start
        if index_in_slice < 0 or index_in_slice >= per_slice:
            continue
        row = index_in_slice // grid_width
        col = index_in_slice % grid_width
        row_start = max(0, row - radius)
        row_end = min(grid_height - 1, row + radius)
        col_start = max(0, col - radius)
        col_end = min(grid_width - 1, col + radius)
        for out_row in range(row_start, row_end + 1):
            for out_col in range(col_start, col_end + 1):
                expanded.append(slice_start + out_row * grid_width + out_col)
    if not expanded:
        return local_indices
    return torch.unique(torch.tensor(expanded, device=device, dtype=torch.long))


def _normalize_bbox(label: dict[str, Any], shape: dict[str, Any], device: torch.device) -> torch.Tensor:
    x1, y1, x2, y2 = [float(v) for v in label["bbox"]]
    width = float(shape["width"])
    height = float(shape["height"])
    box = torch.tensor([x1 / width, y1 / height, x2 / width, y2 / height], device=device, dtype=torch.float32)
    x1y1 = torch.minimum(box[:2], box[2:])
    x2y2 = torch.maximum(box[:2], box[2:])
    return torch.cat([x1y1, x2y2]).clamp(0.0, 1.0)


def build_module1_targets(
    labels: list[dict[str, Any]],
    shape_metadata: dict[str, Any],
    device: torch.device,
    target_config: DetectionTargetConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Build token-level targets from one batch of JSONL labels.

    Args:
        labels: Batch list where each item contains ``module1_labels``.
        shape_metadata: Backbone metadata with sagittal/coronal patch grids.
        device: Output device.

    Returns:
        Dict with objectness, boxes, side, abnormality, and positive mask.
    """

    target_config = target_config or DetectionTargetConfig()
    batch_size = len(labels)
    sagittal_shape = shape_metadata["sagittal"]
    coronal_shape = shape_metadata["coronal"]
    sag_tokens = int(sagittal_shape["slices"] * sagittal_shape["patches_per_slice"])
    cor_tokens = int(coronal_shape["slices"] * coronal_shape["patches_per_slice"])
    total_tokens = sag_tokens + cor_tokens

    objectness = torch.zeros(batch_size, total_tokens, device=device)
    bbox_targets = torch.zeros(batch_size, total_tokens, 4, device=device)
    side_targets = torch.zeros(batch_size, total_tokens, dtype=torch.long, device=device)
    abnormality_targets = torch.zeros(batch_size, total_tokens, dtype=torch.long, device=device)
    positive_mask = torch.zeros(batch_size, total_tokens, dtype=torch.bool, device=device)

    centers_by_plane = {
        "sagittal": _grid_centers(sagittal_shape, device),
        "coronal": _grid_centers(coronal_shape, device),
    }
    slices_by_plane = {
        "sagittal": _slice_indices(sagittal_shape, device),
        "coronal": _slice_indices(coronal_shape, device),
    }
    offsets = {"sagittal": 0, "coronal": sag_tokens}
    shapes = {"sagittal": sagittal_shape, "coronal": coronal_shape}

    for batch_idx, case_labels in enumerate(labels):
        for raw_label in case_labels.get("module1_labels", []):
            plane = raw_label["plane"]
            if plane not in PLANE_TO_ID:
                continue
            shape = shapes[plane]
            box = _normalize_bbox(raw_label, shape, device)
            slice_index = int(raw_label["slice_index"])
            centers = centers_by_plane[plane]
            slice_indices = slices_by_plane[plane]
            in_slice = slice_indices == slice_index
            in_box = (
                (centers[:, 0] >= box[0])
                & (centers[:, 0] <= box[2])
                & (centers[:, 1] >= box[1])
                & (centers[:, 1] <= box[3])
                & in_slice
            )
            local_indices = torch.nonzero(in_box, as_tuple=False).flatten()
            if local_indices.numel() == 0:
                same_slice_indices = torch.nonzero(in_slice, as_tuple=False).flatten()
                if same_slice_indices.numel() == 0:
                    continue
                box_center = (box[:2] + box[2:]) / 2.0
                distances = torch.norm(centers[same_slice_indices] - box_center, dim=1)
                local_indices = same_slice_indices[distances.argmin()].view(1)
            local_indices = _expand_indices_by_radius(
                local_indices,
                shape=shape,
                slice_index=slice_index,
                radius=int(target_config.positive_radius_tokens),
                device=device,
            )

            indices = local_indices + offsets[plane]
            objectness[batch_idx, indices] = 1.0
            bbox_targets[batch_idx, indices] = box
            side_targets[batch_idx, indices] = int(raw_label.get("meniscus_side_id", SIDE_TO_ID[raw_label["meniscus_side"]]))
            abnormality_targets[batch_idx, indices] = int(raw_label.get("abnormality_id", ABNORMALITY_TO_ID[raw_label["abnormality"]]))
            positive_mask[batch_idx, indices] = True

    return {
        "objectness": objectness,
        "bbox_targets": bbox_targets,
        "side_targets": side_targets,
        "abnormality_targets": abnormality_targets,
        "positive_mask": positive_mask,
    }


def build_pred_iou_objectness_targets(
    labels: list[dict[str, Any]],
    shape_metadata: dict[str, Any],
    pred_boxes: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Soft objectness targets from predicted-box IoU on matching plane/slice."""

    batch_size, total_tokens = pred_boxes.shape[:2]
    sagittal_shape = shape_metadata["sagittal"]
    coronal_shape = shape_metadata["coronal"]
    sag_tokens = int(sagittal_shape["slices"] * sagittal_shape["patches_per_slice"])
    targets = torch.zeros(batch_size, total_tokens, device=device, dtype=pred_boxes.dtype)
    slices_by_plane = {
        "sagittal": _slice_indices(sagittal_shape, device),
        "coronal": _slice_indices(coronal_shape, device),
    }
    offsets = {"sagittal": 0, "coronal": sag_tokens}
    shapes = {"sagittal": sagittal_shape, "coronal": coronal_shape}
    detached_boxes = pred_boxes.detach()

    for batch_idx, case_labels in enumerate(labels):
        for raw_label in case_labels.get("module1_labels", []):
            plane = raw_label["plane"]
            if plane not in PLANE_TO_ID:
                continue
            slice_index = int(raw_label["slice_index"])
            local_indices = torch.nonzero(slices_by_plane[plane] == slice_index, as_tuple=False).flatten()
            if local_indices.numel() == 0:
                continue
            indices = local_indices + offsets[plane]
            box = _normalize_bbox(raw_label, shapes[plane], device).to(dtype=detached_boxes.dtype)
            ious = box_iou(detached_boxes[batch_idx, indices].float(), box.view(1, 4).float()).flatten()
            targets[batch_idx, indices] = torch.maximum(targets[batch_idx, indices], ious.to(dtype=targets.dtype))

    return targets.clamp(0.0, 1.0)


def build_strict_pred_iou_targets(
    labels: list[dict[str, Any]],
    shape_metadata: dict[str, Any],
    pred_boxes: torch.Tensor,
    side_logits: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Max predicted-box IoU within matching case/plane/slice/predicted-side."""

    batch_size, total_tokens = pred_boxes.shape[:2]
    sagittal_shape = shape_metadata["sagittal"]
    coronal_shape = shape_metadata["coronal"]
    sag_tokens = int(sagittal_shape["slices"] * sagittal_shape["patches_per_slice"])
    targets = torch.zeros(batch_size, total_tokens, device=device, dtype=pred_boxes.dtype)
    slices_by_plane = {
        "sagittal": _slice_indices(sagittal_shape, device),
        "coronal": _slice_indices(coronal_shape, device),
    }
    offsets = {"sagittal": 0, "coronal": sag_tokens}
    shapes = {"sagittal": sagittal_shape, "coronal": coronal_shape}
    detached_boxes = pred_boxes.detach()
    side_ids = side_logits.detach().argmax(dim=-1)

    for batch_idx, case_labels in enumerate(labels):
        for raw_label in case_labels.get("module1_labels", []):
            plane = raw_label["plane"]
            if plane not in PLANE_TO_ID:
                continue
            slice_index = int(raw_label["slice_index"])
            side_id = int(raw_label.get("meniscus_side_id", SIDE_TO_ID[raw_label["meniscus_side"]]))
            local_indices = torch.nonzero(slices_by_plane[plane] == slice_index, as_tuple=False).flatten()
            if local_indices.numel() == 0:
                continue
            indices = local_indices + offsets[plane]
            matching_side = side_ids[batch_idx, indices] == side_id
            indices = indices[matching_side]
            if indices.numel() == 0:
                continue
            box = _normalize_bbox(raw_label, shapes[plane], device).to(dtype=detached_boxes.dtype)
            ious = box_iou(detached_boxes[batch_idx, indices].float(), box.view(1, 4).float()).flatten()
            targets[batch_idx, indices] = torch.maximum(targets[batch_idx, indices], ious.to(dtype=targets.dtype))

    return targets.clamp(0.0, 1.0)


def build_quality_iou_objectness_targets(
    labels: list[dict[str, Any]],
    shape_metadata: dict[str, Any],
    pred_boxes: torch.Tensor,
    side_logits: torch.Tensor,
    assigned_objectness: torch.Tensor,
    assigned_positive_mask: torch.Tensor,
    device: torch.device,
    quality_iou_threshold: float = 0.3,
) -> dict[str, torch.Tensor]:
    """Build IoU-aware soft objectness targets.

    Assigned-positive tokens keep target 1.0. Non-assigned tokens become soft
    positives only when their detached predicted box has sufficient strict-key
    IoU with a GT box from the same case, plane, processed slice, and side.
    """

    max_pred_iou = build_strict_pred_iou_targets(
        labels,
        shape_metadata,
        pred_boxes,
        side_logits,
        device=device,
    )
    objectness_targets = assigned_objectness.to(dtype=max_pred_iou.dtype).clone()
    objectness_targets[assigned_positive_mask] = 1.0
    quality_mask = (~assigned_positive_mask) & (max_pred_iou >= float(quality_iou_threshold))
    quality_targets = torch.zeros_like(objectness_targets)
    quality_targets[quality_mask] = max_pred_iou[quality_mask]
    objectness_targets[quality_mask] = max_pred_iou[quality_mask]
    return {
        "objectness_targets": objectness_targets.clamp(0.0, 1.0),
        "quality_targets": quality_targets.clamp(0.0, 1.0),
        "quality_mask": quality_mask,
        "max_pred_iou": max_pred_iou,
    }


def build_task_aligned_objectness_targets(
    labels: list[dict[str, Any]],
    shape_metadata: dict[str, Any],
    pred_boxes: torch.Tensor,
    objectness_logits: torch.Tensor,
    side_logits: torch.Tensor,
    assigned_objectness: torch.Tensor,
    assigned_positive_mask: torch.Tensor,
    device: torch.device,
    alpha: float = 1.0,
    beta: float = 6.0,
    topk: int = 10,
) -> dict[str, torch.Tensor]:
    """Basic TOOD-style task-aligned objectness targets.

    This is a paper-inferred experimental implementation. For each GT box, it
    selects top-k tokens in the same case, plane, processed slice, and predicted
    side by ``sigmoid(objectness)^alpha * IoU^beta``. Targets are the detached
    predicted IoU, while assigned-positive tokens still take priority as 1.0.
    """

    batch_size, total_tokens = pred_boxes.shape[:2]
    sagittal_shape = shape_metadata["sagittal"]
    coronal_shape = shape_metadata["coronal"]
    sag_tokens = int(sagittal_shape["slices"] * sagittal_shape["patches_per_slice"])
    objectness_targets = assigned_objectness.to(dtype=pred_boxes.dtype).clone()
    task_aligned_targets = torch.zeros(batch_size, total_tokens, device=device, dtype=pred_boxes.dtype)
    task_aligned_mask = torch.zeros(batch_size, total_tokens, device=device, dtype=torch.bool)

    slices_by_plane = {
        "sagittal": _slice_indices(sagittal_shape, device),
        "coronal": _slice_indices(coronal_shape, device),
    }
    offsets = {"sagittal": 0, "coronal": sag_tokens}
    shapes = {"sagittal": sagittal_shape, "coronal": coronal_shape}
    detached_boxes = pred_boxes.detach()
    objectness_prob = torch.sigmoid(objectness_logits).detach()
    side_ids = side_logits.detach().argmax(dim=-1)
    topk = max(0, int(topk))

    for batch_idx, case_labels in enumerate(labels):
        for raw_label in case_labels.get("module1_labels", []):
            plane = raw_label["plane"]
            if plane not in PLANE_TO_ID:
                continue
            slice_index = int(raw_label["slice_index"])
            side_id = int(raw_label.get("meniscus_side_id", SIDE_TO_ID[raw_label["meniscus_side"]]))
            local_indices = torch.nonzero(slices_by_plane[plane] == slice_index, as_tuple=False).flatten()
            if local_indices.numel() == 0 or topk == 0:
                continue
            indices = local_indices + offsets[plane]
            indices = indices[side_ids[batch_idx, indices] == side_id]
            if indices.numel() == 0:
                continue

            box = _normalize_bbox(raw_label, shapes[plane], device).to(dtype=detached_boxes.dtype)
            pred_iou = box_iou(detached_boxes[batch_idx, indices].float(), box.view(1, 4).float()).flatten()
            pred_iou = pred_iou.to(dtype=task_aligned_targets.dtype).clamp(0.0, 1.0)
            alignment_score = objectness_prob[batch_idx, indices].to(dtype=pred_iou.dtype).pow(float(alpha))
            alignment_score = alignment_score * pred_iou.pow(float(beta))
            k = min(topk, int(indices.numel()))
            selected_local = torch.topk(alignment_score, k=k, largest=True).indices
            selected_indices = indices[selected_local]
            selected_iou = pred_iou[selected_local]
            task_aligned_mask[batch_idx, selected_indices] = True
            task_aligned_targets[batch_idx, selected_indices] = torch.maximum(
                task_aligned_targets[batch_idx, selected_indices],
                selected_iou,
            )
            objectness_targets[batch_idx, selected_indices] = torch.maximum(
                objectness_targets[batch_idx, selected_indices],
                selected_iou,
            )

    objectness_targets[assigned_positive_mask] = 1.0
    return {
        "objectness_targets": objectness_targets.clamp(0.0, 1.0),
        "task_aligned_targets": task_aligned_targets.clamp(0.0, 1.0),
        "task_aligned_mask": task_aligned_mask,
    }


def _sample_indices(indices: torch.Tensor, count: int) -> torch.Tensor:
    count = max(0, int(count))
    if count == 0 or indices.numel() == 0:
        return indices.new_empty((0,), dtype=torch.long)
    if indices.numel() <= count:
        return indices
    order = torch.randperm(indices.numel(), device=indices.device)[:count]
    return indices[order]


def _topk_indices(indices: torch.Tensor, scores: torch.Tensor, count: int) -> torch.Tensor:
    count = max(0, int(count))
    if count == 0 or indices.numel() == 0:
        return indices.new_empty((0,), dtype=torch.long)
    k = min(count, int(indices.numel()))
    selected = torch.topk(scores, k=k, largest=True).indices
    return indices[selected]


def _distributed_budget(total: int, groups: int) -> int:
    total = max(0, int(total))
    groups = max(1, int(groups))
    if total == 0:
        return 0
    return max(1, (total + groups - 1) // groups)


def _quality_group_indices(
    shape_metadata: dict[str, Any],
    side_logits: torch.Tensor,
    batch_idx: int,
    device: torch.device,
    per_group: bool,
) -> list[torch.Tensor]:
    total_tokens = int(side_logits.shape[1])
    all_indices = torch.arange(total_tokens, device=device, dtype=torch.long)
    if not per_group:
        return [all_indices]
    plane_ids, slice_indices = _token_plane_and_slice_ids(shape_metadata, device)
    side_ids = side_logits.detach().argmax(dim=-1)
    group_keys = torch.stack([plane_ids, slice_indices, side_ids[batch_idx]], dim=1)
    groups: list[torch.Tensor] = []
    for key in torch.unique(group_keys, dim=0):
        group = torch.nonzero((group_keys == key).all(dim=1), as_tuple=False).flatten()
        if group.numel() > 0:
            groups.append(group)
    return groups or [all_indices]


def _hard_negative_scores(
    objectness_prob: torch.Tensor,
    alignment_targets: torch.Tensor,
    source: str,
    quality_prob: torch.Tensor | None = None,
) -> torch.Tensor:
    alignment_scores = alignment_targets
    if source == "alignment":
        source = "task_alignment"
    if source == "objectness":
        return objectness_prob
    if source == "task_alignment":
        return alignment_scores
    if source == "quality":
        if quality_prob is None:
            raise ValueError("quality_hard_negative_source=quality requires bbox_quality_logits")
        return quality_prob.to(dtype=alignment_scores.dtype)
    if source == "objectness_or_alignment":
        return torch.maximum(objectness_prob.to(dtype=alignment_scores.dtype), alignment_scores)
    if source == "objectness_or_quality":
        if quality_prob is None:
            raise ValueError("quality_hard_negative_source=objectness_or_quality requires bbox_quality_logits")
        return torch.maximum(objectness_prob.to(dtype=alignment_scores.dtype), quality_prob.to(dtype=alignment_scores.dtype))
    if source == "alignment_or_quality":
        if quality_prob is None:
            raise ValueError("quality_hard_negative_source=alignment_or_quality requires bbox_quality_logits")
        return torch.maximum(alignment_scores, quality_prob.to(dtype=alignment_scores.dtype))
    if source == "objectness_or_alignment_or_quality":
        if quality_prob is None:
            raise ValueError("quality_hard_negative_source=objectness_or_alignment_or_quality requires bbox_quality_logits")
        return torch.maximum(
            torch.maximum(objectness_prob.to(dtype=alignment_scores.dtype), alignment_scores),
            quality_prob.to(dtype=alignment_scores.dtype),
        )
    raise ValueError(
        "quality_hard_negative_source must be objectness, task_alignment, alignment, quality, "
        "objectness_or_alignment, objectness_or_quality, alignment_or_quality, "
        "or objectness_or_alignment_or_quality"
    )


def build_bbox_quality_targets(
    labels: list[dict[str, Any]],
    shape_metadata: dict[str, Any],
    pred_boxes: torch.Tensor,
    objectness_logits: torch.Tensor,
    side_logits: torch.Tensor,
    assigned_positive_mask: torch.Tensor,
    device: torch.device,
    target_mode: str = "iou",
    train_mask: str = "topk_by_alignment",
    iou_threshold: float = 0.3,
    topk: int = 100,
    alpha: float = 1.0,
    beta: float = 6.0,
    task_aligned_mask: torch.Tensor | None = None,
    quality_pos_iou_threshold: float = 0.3,
    quality_neg_iou_threshold: float = 0.1,
    quality_num_pos: int = 256,
    quality_num_neg: int = 512,
    quality_num_mid: int = 256,
    quality_hard_negative_source: str = "objectness_or_alignment",
    quality_mid_weight: float = 0.25,
    quality_neg_target_mode: str = "zero",
    quality_sampling_per_group: bool = True,
    quality_sample_total: int = 1024,
    quality_high_iou_fraction: float = 0.25,
    quality_mid_iou_fraction: float = 0.25,
    quality_low_iou_fraction: float = 0.50,
    bbox_quality_logits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build detached bbox-quality targets for the experimental quality head."""

    if target_mode not in {"iou", "task_alignment"}:
        raise ValueError("bbox_quality_target_mode must be iou or task_alignment")
    if train_mask not in BBOX_QUALITY_TRAIN_MASKS:
        raise ValueError(
            "bbox_quality_train_mask must be assigned_positive, task_aligned_positive, "
            "iou_above_threshold, topk_by_alignment, hard_pos_neg_iou_sampling, or sampled_soft_iou"
        )
    if quality_neg_target_mode not in {"zero", "iou"}:
        raise ValueError("quality_neg_target_mode must be zero or iou")

    max_pred_iou = build_strict_pred_iou_targets(
        labels,
        shape_metadata,
        pred_boxes,
        side_logits,
        device=device,
    ).detach()
    objectness_prob = torch.sigmoid(objectness_logits).detach()
    quality_prob = torch.sigmoid(bbox_quality_logits).detach() if bbox_quality_logits is not None else None
    alignment_targets = objectness_prob.to(dtype=max_pred_iou.dtype).pow(float(alpha)) * max_pred_iou.pow(float(beta))
    quality_targets = max_pred_iou if target_mode == "iou" else alignment_targets.clamp(0.0, 1.0)
    quality_weights = torch.zeros_like(max_pred_iou)
    sample_kind = torch.zeros_like(max_pred_iou, dtype=torch.long)
    quality_mask = torch.zeros_like(assigned_positive_mask, dtype=torch.bool)

    if train_mask == "assigned_positive":
        quality_mask = assigned_positive_mask.detach().clone()
    elif train_mask == "task_aligned_positive":
        if task_aligned_mask is None:
            assigned_objectness = assigned_positive_mask.to(dtype=pred_boxes.dtype)
            task_result = build_task_aligned_objectness_targets(
                labels,
                shape_metadata,
                pred_boxes,
                objectness_logits,
                side_logits,
                assigned_objectness,
                assigned_positive_mask,
                device=device,
                alpha=alpha,
                beta=beta,
                topk=topk,
            )
            quality_mask = task_result["task_aligned_mask"]
        else:
            quality_mask = task_aligned_mask.detach().clone()
    elif train_mask == "iou_above_threshold":
        quality_mask = max_pred_iou >= float(iou_threshold)
    elif train_mask == "topk_by_alignment":
        task_result = build_task_aligned_objectness_targets(
            labels,
            shape_metadata,
            pred_boxes,
            objectness_logits,
            side_logits,
            assigned_positive_mask.to(dtype=pred_boxes.dtype),
            assigned_positive_mask,
            device=device,
            alpha=alpha,
            beta=beta,
            topk=topk,
        )
        quality_mask = task_result["task_aligned_mask"]
    elif train_mask == "hard_pos_neg_iou_sampling":
        quality_targets = max_pred_iou.detach().clone()
        hard_scores = _hard_negative_scores(
            objectness_prob.to(dtype=max_pred_iou.dtype),
            alignment_targets.detach(),
            quality_hard_negative_source,
            quality_prob=quality_prob,
        )
        pos_thr = float(quality_pos_iou_threshold)
        neg_thr = float(quality_neg_iou_threshold)
        mid_weight = float(quality_mid_weight)
        for batch_idx in range(max_pred_iou.shape[0]):
            groups = _quality_group_indices(
                shape_metadata,
                side_logits,
                batch_idx=batch_idx,
                device=device,
                per_group=bool(quality_sampling_per_group),
            )
            pos_budget = _distributed_budget(quality_num_pos, len(groups)) if quality_sampling_per_group else max(0, int(quality_num_pos))
            neg_budget = _distributed_budget(quality_num_neg, len(groups)) if quality_sampling_per_group else max(0, int(quality_num_neg))
            mid_budget = _distributed_budget(quality_num_mid, len(groups)) if quality_sampling_per_group else max(0, int(quality_num_mid))
            batch_iou = max_pred_iou[batch_idx]
            for group in groups:
                group_iou = batch_iou[group]
                pos = group[group_iou >= pos_thr]
                neg = group[group_iou <= neg_thr]
                mid = group[(group_iou > neg_thr) & (group_iou < pos_thr)]
                selected_pos = _sample_indices(pos, pos_budget)
                selected_neg = _topk_indices(neg, hard_scores[batch_idx, neg], neg_budget)
                selected_mid = _sample_indices(mid, mid_budget)
                if selected_pos.numel() > 0:
                    quality_mask[batch_idx, selected_pos] = True
                    quality_weights[batch_idx, selected_pos] = 1.0
                    sample_kind[batch_idx, selected_pos] = 1
                if selected_neg.numel() > 0:
                    quality_mask[batch_idx, selected_neg] = True
                    quality_weights[batch_idx, selected_neg] = 1.0
                    sample_kind[batch_idx, selected_neg] = 2
                    if quality_neg_target_mode == "zero":
                        quality_targets[batch_idx, selected_neg] = 0.0
                if selected_mid.numel() > 0:
                    quality_mask[batch_idx, selected_mid] = True
                    quality_weights[batch_idx, selected_mid] = mid_weight
                    sample_kind[batch_idx, selected_mid] = 3
    else:
        quality_targets = max_pred_iou.detach().clone()
        total = max(0, int(quality_sample_total))
        high_budget = max(0, int(round(total * float(quality_high_iou_fraction))))
        mid_budget = max(0, int(round(total * float(quality_mid_iou_fraction))))
        low_budget = max(0, int(round(total * float(quality_low_iou_fraction))))
        if high_budget + mid_budget + low_budget > total:
            low_budget = max(0, total - high_budget - mid_budget)
        elif high_budget + mid_budget + low_budget < total:
            low_budget += total - high_budget - mid_budget - low_budget
        pos_thr = float(quality_pos_iou_threshold)
        neg_thr = float(quality_neg_iou_threshold)
        for batch_idx in range(max_pred_iou.shape[0]):
            batch_iou = max_pred_iou[batch_idx]
            high = torch.nonzero(batch_iou >= pos_thr, as_tuple=False).flatten()
            mid = torch.nonzero((batch_iou > neg_thr) & (batch_iou < pos_thr), as_tuple=False).flatten()
            low = torch.nonzero(batch_iou <= neg_thr, as_tuple=False).flatten()
            selected_high = _sample_indices(high, high_budget)
            selected_mid = _sample_indices(mid, mid_budget)
            selected_low = _sample_indices(low, low_budget)
            if selected_high.numel() > 0:
                quality_mask[batch_idx, selected_high] = True
                quality_weights[batch_idx, selected_high] = 1.0
                sample_kind[batch_idx, selected_high] = 1
            if selected_low.numel() > 0:
                quality_mask[batch_idx, selected_low] = True
                quality_weights[batch_idx, selected_low] = 1.0
                sample_kind[batch_idx, selected_low] = 2
            if selected_mid.numel() > 0:
                quality_mask[batch_idx, selected_mid] = True
                quality_weights[batch_idx, selected_mid] = 1.0
                sample_kind[batch_idx, selected_mid] = 3

    if train_mask in {"assigned_positive", "task_aligned_positive", "iou_above_threshold", "topk_by_alignment"}:
        quality_weights = quality_mask.to(dtype=max_pred_iou.dtype)
        sample_kind = torch.zeros_like(max_pred_iou, dtype=torch.long)
        sample_kind[quality_mask & (max_pred_iou >= float(quality_pos_iou_threshold))] = 1
        sample_kind[quality_mask & (max_pred_iou <= float(quality_neg_iou_threshold))] = 2
        sample_kind[quality_mask & (max_pred_iou > float(quality_neg_iou_threshold)) & (max_pred_iou < float(quality_pos_iou_threshold))] = 3

    return {
        "quality_targets": quality_targets.detach().clamp(0.0, 1.0),
        "quality_mask": quality_mask.detach(),
        "quality_weights": quality_weights.detach(),
        "max_pred_iou": max_pred_iou.detach().clamp(0.0, 1.0),
        "alignment_targets": alignment_targets.detach().clamp(0.0, 1.0),
        "num_quality_pos_samples": (sample_kind == 1).sum().detach(),
        "num_quality_neg_samples": (sample_kind == 2).sum().detach(),
        "num_quality_mid_samples": (sample_kind == 3).sum().detach(),
    }


class Module1DetectionLoss(nn.Module):
    """Multi-task Module 1 loss."""

    def __init__(
        self,
        objectness_weight: float = 1.0,
        bbox_weight: float = 1.0,
        side_weight: float = 0.5,
        abnormality_weight: float = 1.0,
        bbox_loss_type: str = "smooth_l1",
        objectness_loss_type: str = "focal",
        objectness_positive_weight: float = 1.0,
        objectness_negative_weight: float = 1.0,
        objectness_normalization: str = "positive_count",
        objectness_target_mode: str = "assigned",
        quality_iou_threshold: float = 0.3,
        quality_iou_start_epoch: int = 30,
        quality_iou_stop_gradient: bool = True,
        task_aligned_alpha: float = 1.0,
        task_aligned_beta: float = 6.0,
        task_aligned_topk: int = 10,
        task_aligned_start_epoch: int = 30,
        bbox_quality_loss_weight: float = 0.0,
        bbox_quality_loss_type: str = "bce",
        bbox_quality_target_mode: str = "iou",
        bbox_quality_train_mask: str = "topk_by_alignment",
        bbox_quality_iou_threshold: float = 0.3,
        bbox_quality_topk: int = 100,
        bbox_quality_start_epoch: int = 30,
        quality_pos_iou_threshold: float = 0.3,
        quality_neg_iou_threshold: float = 0.1,
        quality_num_pos: int = 256,
        quality_num_neg: int = 512,
        quality_num_mid: int = 256,
        quality_hard_negative_source: str = "objectness_or_alignment",
        quality_mid_weight: float = 0.25,
        quality_neg_target_mode: str = "zero",
        quality_sampling_per_group: bool = True,
        quality_sample_total: int = 1024,
        quality_high_iou_fraction: float = 0.25,
        quality_mid_iou_fraction: float = 0.25,
        quality_low_iou_fraction: float = 0.50,
        use_quality_ranking_loss: bool = False,
        quality_ranking_loss_weight: float = 0.5,
        ranking_pos_iou_threshold: float = 0.3,
        ranking_neg_iou_threshold: float = 0.1,
        ranking_margin: float = 0.2,
        ranking_num_pairs: int = 512,
        quality_ranking_score_space: str = "logits",
        bbox_quality_varifocal_alpha: float = 0.75,
        bbox_quality_varifocal_gamma: float = 2.0,
        bbox_quality_qfl_beta: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        positive_radius_tokens: int = 0,
    ):
        super().__init__()
        if bbox_loss_type not in {"smooth_l1", "giou", "smooth_l1_giou"}:
            raise ValueError("bbox_loss_type must be smooth_l1, giou, or smooth_l1_giou")
        if objectness_loss_type not in {"focal", "weighted_bce", "bce"}:
            raise ValueError("objectness_loss_type must be focal, weighted_bce, or bce")
        if objectness_normalization not in {"positive_count", "all_tokens"}:
            raise ValueError("objectness_normalization must be positive_count or all_tokens")
        if objectness_target_mode not in {"assigned", "assigned_or_pred_iou", "quality_iou", "task_aligned"}:
            raise ValueError("objectness_target_mode must be assigned, assigned_or_pred_iou, quality_iou, or task_aligned")
        if objectness_target_mode != "assigned" and objectness_loss_type not in {"weighted_bce", "bce"}:
            raise ValueError("soft objectness targets require weighted_bce or bce")
        if bbox_quality_loss_type not in BBOX_QUALITY_LOSS_TYPES:
            raise ValueError(
                "bbox_quality_loss_type must be bce, bce_soft_iou, mse, smooth_l1, "
                "varifocal, quality_focal, pairwise_ranking, or varifocal_plus_ranking"
            )
        if bbox_quality_target_mode not in {"iou", "task_alignment"}:
            raise ValueError("bbox_quality_target_mode must be iou or task_alignment")
        if bbox_quality_train_mask not in BBOX_QUALITY_TRAIN_MASKS:
            raise ValueError(
                "bbox_quality_train_mask must be assigned_positive, task_aligned_positive, "
                "iou_above_threshold, topk_by_alignment, hard_pos_neg_iou_sampling, or sampled_soft_iou"
            )
        if quality_hard_negative_source not in QUALITY_HARD_NEGATIVE_SOURCES:
            raise ValueError(
                "quality_hard_negative_source must be objectness, task_alignment, alignment, quality, "
                "objectness_or_alignment, objectness_or_quality, alignment_or_quality, "
                "or objectness_or_alignment_or_quality"
            )
        if quality_neg_target_mode not in {"zero", "iou"}:
            raise ValueError("quality_neg_target_mode must be zero or iou")
        if quality_ranking_score_space not in {"logits", "sigmoid"}:
            raise ValueError("quality_ranking_score_space must be logits or sigmoid")
        if not quality_iou_stop_gradient:
            raise ValueError("quality_iou_stop_gradient must stay true so objectness targets do not backprop through bbox IoU")
        self.objectness_loss_type = objectness_loss_type
        self.objectness_target_mode = objectness_target_mode
        self.objectness_positive_weight = float(objectness_positive_weight)
        self.objectness_negative_weight = float(objectness_negative_weight)
        self.objectness_normalization = objectness_normalization
        self.quality_iou_threshold = float(quality_iou_threshold)
        self.quality_iou_start_epoch = int(quality_iou_start_epoch)
        self.task_aligned_alpha = float(task_aligned_alpha)
        self.task_aligned_beta = float(task_aligned_beta)
        self.task_aligned_topk = int(task_aligned_topk)
        self.task_aligned_start_epoch = int(task_aligned_start_epoch)
        self.bbox_quality_loss_weight = float(bbox_quality_loss_weight)
        self.bbox_quality_loss_type = bbox_quality_loss_type
        self.bbox_quality_target_mode = bbox_quality_target_mode
        self.bbox_quality_train_mask = bbox_quality_train_mask
        self.bbox_quality_iou_threshold = float(bbox_quality_iou_threshold)
        self.bbox_quality_topk = int(bbox_quality_topk)
        self.bbox_quality_start_epoch = int(bbox_quality_start_epoch)
        self.quality_pos_iou_threshold = float(quality_pos_iou_threshold)
        self.quality_neg_iou_threshold = float(quality_neg_iou_threshold)
        self.quality_num_pos = int(quality_num_pos)
        self.quality_num_neg = int(quality_num_neg)
        self.quality_num_mid = int(quality_num_mid)
        self.quality_hard_negative_source = quality_hard_negative_source
        self.quality_mid_weight = float(quality_mid_weight)
        self.quality_neg_target_mode = quality_neg_target_mode
        self.quality_sampling_per_group = bool(quality_sampling_per_group)
        self.quality_sample_total = int(quality_sample_total)
        self.quality_high_iou_fraction = float(quality_high_iou_fraction)
        self.quality_mid_iou_fraction = float(quality_mid_iou_fraction)
        self.quality_low_iou_fraction = float(quality_low_iou_fraction)
        self.use_quality_ranking_loss = bool(use_quality_ranking_loss)
        self.quality_ranking_loss_weight = float(quality_ranking_loss_weight)
        self.ranking_pos_iou_threshold = float(ranking_pos_iou_threshold)
        self.ranking_neg_iou_threshold = float(ranking_neg_iou_threshold)
        self.ranking_margin = float(ranking_margin)
        self.ranking_num_pairs = int(ranking_num_pairs)
        self.quality_ranking_score_space = quality_ranking_score_space
        self.bbox_quality_varifocal_alpha = float(bbox_quality_varifocal_alpha)
        self.bbox_quality_varifocal_gamma = float(bbox_quality_varifocal_gamma)
        self.bbox_quality_qfl_beta = float(bbox_quality_qfl_beta)
        self.current_epoch = 0
        self.objectness_loss = BinaryFocalLossWithLogits(alpha=focal_alpha, gamma=focal_gamma)
        self.objectness_weight = objectness_weight
        self.bbox_weight = bbox_weight
        self.side_weight = side_weight
        self.abnormality_weight = abnormality_weight
        self.bbox_loss_type = bbox_loss_type
        self.target_config = DetectionTargetConfig(positive_radius_tokens=int(positive_radius_tokens))

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def forward(self, predictions: dict[str, torch.Tensor | dict[str, Any]], labels: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        shape_metadata = predictions["shape_metadata"]
        if not isinstance(shape_metadata, dict):
            raise TypeError("predictions['shape_metadata'] must be a dict")
        targets = build_module1_targets(labels, shape_metadata, device=predictions["objectness_logits"].device, target_config=self.target_config)

        effective_target_mode = self._effective_objectness_target_mode()
        objectness_targets = targets["objectness"]
        quality_targets = torch.zeros_like(objectness_targets)
        quality_mask = torch.zeros_like(targets["positive_mask"])
        task_aligned_targets = torch.zeros_like(objectness_targets)
        task_aligned_mask = torch.zeros_like(targets["positive_mask"])
        if effective_target_mode == "assigned_or_pred_iou":
            iou_targets = build_pred_iou_objectness_targets(
                labels,
                shape_metadata,
                predictions["bbox_pred"],
                device=predictions["objectness_logits"].device,
            )
            objectness_targets = torch.maximum(objectness_targets.to(dtype=iou_targets.dtype), iou_targets)
            quality_targets = torch.where(targets["positive_mask"], torch.zeros_like(iou_targets), iou_targets)
        elif effective_target_mode == "quality_iou":
            quality_result = build_quality_iou_objectness_targets(
                labels,
                shape_metadata,
                predictions["bbox_pred"],
                predictions["side_logits"],
                targets["objectness"],
                targets["positive_mask"],
                device=predictions["objectness_logits"].device,
                quality_iou_threshold=self.quality_iou_threshold,
            )
            objectness_targets = quality_result["objectness_targets"]
            quality_targets = quality_result["quality_targets"]
            quality_mask = quality_result["quality_mask"]
        elif effective_target_mode == "task_aligned":
            task_result = build_task_aligned_objectness_targets(
                labels,
                shape_metadata,
                predictions["bbox_pred"],
                predictions["objectness_logits"],
                predictions["side_logits"],
                targets["objectness"],
                targets["positive_mask"],
                device=predictions["objectness_logits"].device,
                alpha=self.task_aligned_alpha,
                beta=self.task_aligned_beta,
                topk=self.task_aligned_topk,
            )
            objectness_targets = task_result["objectness_targets"]
            task_aligned_targets = task_result["task_aligned_targets"]
            task_aligned_mask = task_result["task_aligned_mask"]

        positive_mask = targets["positive_mask"]
        negative_mask = objectness_targets.detach() <= 0
        soft_positive_mask = (~positive_mask) & (~negative_mask)
        objectness_token_loss = self._objectness_elementwise_loss(predictions["objectness_logits"], objectness_targets)
        if self.objectness_normalization == "positive_count":
            # Dense detectors sum focal/BCE losses and normalize by foreground
            # anchors. Averaging over every token would dilute the positive
            # signal by O(H*W*T / foreground), which is ~250x for the paper
            # configuration and prevents the objectness head from learning.
            objectness_normalizer = positive_mask.sum().clamp_min(1).to(dtype=objectness_token_loss.dtype)
        else:
            objectness_normalizer = objectness_token_loss.new_tensor(objectness_token_loss.numel())
        objectness_loss = objectness_token_loss.sum() / objectness_normalizer
        zero = predictions["objectness_logits"].sum() * 0.0
        positive_objectness_loss = objectness_token_loss[positive_mask].mean() if positive_mask.any() else zero
        negative_objectness_loss = objectness_token_loss[negative_mask].mean() if negative_mask.any() else zero
        soft_positive_objectness_loss = objectness_token_loss[soft_positive_mask].mean() if soft_positive_mask.any() else zero
        positive_objectness_loss_contribution = objectness_token_loss[positive_mask].sum() / objectness_normalizer if positive_mask.any() else zero
        negative_objectness_loss_contribution = objectness_token_loss[negative_mask].sum() / objectness_normalizer if negative_mask.any() else zero
        soft_positive_objectness_loss_contribution = (
            objectness_token_loss[soft_positive_mask].sum() / objectness_normalizer if soft_positive_mask.any() else zero
        )
        positive_objectness_tokens = positive_mask.sum()
        negative_objectness_tokens = negative_mask.sum()
        if positive_mask.any():
            pred_boxes = predictions["bbox_pred"][positive_mask]
            target_boxes = targets["bbox_targets"][positive_mask]
            if self.bbox_loss_type == "smooth_l1":
                bbox_loss = F.smooth_l1_loss(pred_boxes, target_boxes)
            elif self.bbox_loss_type == "giou":
                bbox_loss = generalized_box_iou_loss(pred_boxes, target_boxes)
            else:
                bbox_loss = F.smooth_l1_loss(pred_boxes, target_boxes) + generalized_box_iou_loss(pred_boxes, target_boxes)
            side_loss = F.cross_entropy(predictions["side_logits"][positive_mask], targets["side_targets"][positive_mask])
            abnormality_loss = F.cross_entropy(predictions["abnormality_logits"][positive_mask], targets["abnormality_targets"][positive_mask])
        else:
            bbox_loss = zero
            side_loss = zero
            abnormality_loss = zero

        bbox_quality_targets = torch.zeros_like(objectness_targets)
        bbox_quality_mask = torch.zeros_like(targets["positive_mask"])
        bbox_quality_weights = torch.zeros_like(objectness_targets)
        bbox_quality_loss = zero
        quality_ranking_loss = zero
        quality_ranking_pairs = torch.zeros((), device=predictions["objectness_logits"].device)
        quality_sample_counts = {
            "num_quality_pos_samples": torch.zeros((), device=predictions["objectness_logits"].device),
            "num_quality_neg_samples": torch.zeros((), device=predictions["objectness_logits"].device),
            "num_quality_mid_samples": torch.zeros((), device=predictions["objectness_logits"].device),
        }
        ranking_loss_requested = self.use_quality_ranking_loss or self.bbox_quality_loss_type in {
            "pairwise_ranking",
            "varifocal_plus_ranking",
        }
        quality_supervision_active = (
            (
                self.bbox_quality_loss_weight > 0.0
                or (
                    ranking_loss_requested
                    and self.bbox_quality_loss_type != "pairwise_ranking"
                    and self.quality_ranking_loss_weight > 0.0
                )
            )
            and self.current_epoch >= self.bbox_quality_start_epoch
        )
        if quality_supervision_active:
            if "bbox_quality_logits" not in predictions:
                raise KeyError("bbox_quality_logits is required when bbox quality supervision is enabled")
            quality_result = build_bbox_quality_targets(
                labels,
                shape_metadata,
                predictions["bbox_pred"],
                predictions["objectness_logits"],
                predictions["side_logits"],
                targets["positive_mask"],
                device=predictions["objectness_logits"].device,
                target_mode=self.bbox_quality_target_mode,
                train_mask=self.bbox_quality_train_mask,
                iou_threshold=self.bbox_quality_iou_threshold,
                topk=self.bbox_quality_topk,
                alpha=self.task_aligned_alpha,
                beta=self.task_aligned_beta,
                task_aligned_mask=task_aligned_mask,
                quality_pos_iou_threshold=self.quality_pos_iou_threshold,
                quality_neg_iou_threshold=self.quality_neg_iou_threshold,
                quality_num_pos=self.quality_num_pos,
                quality_num_neg=self.quality_num_neg,
                quality_num_mid=self.quality_num_mid,
                quality_hard_negative_source=self.quality_hard_negative_source,
                quality_mid_weight=self.quality_mid_weight,
                quality_neg_target_mode=self.quality_neg_target_mode,
                quality_sampling_per_group=self.quality_sampling_per_group,
                quality_sample_total=self.quality_sample_total,
                quality_high_iou_fraction=self.quality_high_iou_fraction,
                quality_mid_iou_fraction=self.quality_mid_iou_fraction,
                quality_low_iou_fraction=self.quality_low_iou_fraction,
                bbox_quality_logits=predictions["bbox_quality_logits"],
            )
            bbox_quality_targets = quality_result["quality_targets"]
            bbox_quality_mask = quality_result["quality_mask"]
            bbox_quality_weights = quality_result["quality_weights"]
            quality_sample_counts = {
                "num_quality_pos_samples": quality_result["num_quality_pos_samples"],
                "num_quality_neg_samples": quality_result["num_quality_neg_samples"],
                "num_quality_mid_samples": quality_result["num_quality_mid_samples"],
            }
            if ranking_loss_requested:
                ranking_result = self._quality_ranking_loss(
                    predictions["bbox_quality_logits"],
                    predictions["objectness_logits"],
                    quality_result["max_pred_iou"],
                    quality_result["alignment_targets"],
                )
                quality_ranking_loss = ranking_result["loss"]
                quality_ranking_pairs = ranking_result["num_pairs"]
            if self.bbox_quality_loss_weight > 0.0 and self.bbox_quality_loss_type == "pairwise_ranking":
                bbox_quality_loss = quality_ranking_loss
            elif self.bbox_quality_loss_weight > 0.0 and bbox_quality_mask.any():
                bbox_quality_loss = self._bbox_quality_loss(
                    predictions["bbox_quality_logits"][bbox_quality_mask],
                    bbox_quality_targets[bbox_quality_mask],
                    bbox_quality_weights[bbox_quality_mask],
                )

        total = zero
        if self.objectness_weight != 0.0:
            total = total + self.objectness_weight * objectness_loss
        if self.bbox_weight != 0.0:
            total = total + self.bbox_weight * bbox_loss
        if self.side_weight != 0.0:
            total = total + self.side_weight * side_loss
        if self.abnormality_weight != 0.0:
            total = total + self.abnormality_weight * abnormality_loss
        if self.bbox_quality_loss_weight != 0.0:
            total = total + self.bbox_quality_loss_weight * bbox_quality_loss
        if (
            (self.use_quality_ranking_loss or self.bbox_quality_loss_type == "varifocal_plus_ranking")
            and self.bbox_quality_loss_type != "pairwise_ranking"
            and self.quality_ranking_loss_weight != 0.0
        ):
            total = total + self.quality_ranking_loss_weight * quality_ranking_loss
        return {
            "loss": total,
            "loss_objectness": objectness_loss.detach(),
            "loss_objectness_positive": positive_objectness_loss.detach(),
            "loss_objectness_negative": negative_objectness_loss.detach(),
            "loss_objectness_soft_positive": soft_positive_objectness_loss.detach(),
            "loss_objectness_positive_contribution": positive_objectness_loss_contribution.detach(),
            "loss_objectness_negative_contribution": negative_objectness_loss_contribution.detach(),
            "loss_objectness_soft_positive_contribution": soft_positive_objectness_loss_contribution.detach(),
            "objectness_normalizer": objectness_normalizer.detach(),
            "loss_bbox": bbox_loss.detach(),
            "loss_bbox_quality": bbox_quality_loss.detach(),
            "loss_quality_ranking": quality_ranking_loss.detach(),
            "positive_bbox_loss": bbox_loss.detach(),
            "loss_side": side_loss.detach(),
            "loss_abnormality": abnormality_loss.detach(),
            "objectness_positive_tokens": positive_objectness_tokens.detach(),
            "objectness_negative_tokens": negative_objectness_tokens.detach(),
            "objectness_target_mean": objectness_targets.detach().mean(),
            "objectness_target_max": objectness_targets.detach().max(),
            "objectness_soft_positive_tokens": (objectness_targets.detach() >= 0.3).sum(),
            "num_assigned_positive": positive_objectness_tokens.detach(),
            "num_quality_positive": (quality_targets.detach() >= self.quality_iou_threshold).sum(),
            "num_task_aligned_positive": task_aligned_mask.detach().sum(),
            "quality_target_mean": quality_targets.detach().mean(),
            "quality_target_max": quality_targets.detach().max(),
            "task_aligned_target_mean": task_aligned_targets.detach().mean(),
            "task_aligned_target_max": task_aligned_targets.detach().max(),
            "bbox_quality_target_mean": bbox_quality_targets.detach().mean(),
            "bbox_quality_target_max": bbox_quality_targets.detach().max(),
            "num_bbox_quality_tokens": bbox_quality_mask.detach().sum(),
            "bbox_quality_weight_mean": bbox_quality_weights.detach()[bbox_quality_mask].mean() if bbox_quality_mask.any() else zero.detach(),
            "num_quality_pos_samples": quality_sample_counts["num_quality_pos_samples"].detach(),
            "num_quality_neg_samples": quality_sample_counts["num_quality_neg_samples"].detach(),
            "num_quality_mid_samples": quality_sample_counts["num_quality_mid_samples"].detach(),
            "num_quality_ranking_pairs": quality_ranking_pairs.detach(),
            "bbox_quality_positive_prob_mean": self._masked_prob_mean(
                predictions.get("bbox_quality_logits", predictions["objectness_logits"]),
                bbox_quality_mask,
                zero,
            ).detach(),
            "positive_objectness_prob_mean": self._masked_prob_mean(predictions["objectness_logits"], positive_mask, zero).detach(),
            "negative_objectness_prob_mean": self._masked_prob_mean(predictions["objectness_logits"], negative_mask, zero).detach(),
            "quality_positive_objectness_prob_mean": self._masked_prob_mean(predictions["objectness_logits"], quality_mask, zero).detach(),
            "task_aligned_positive_objectness_prob_mean": self._masked_prob_mean(predictions["objectness_logits"], task_aligned_mask, zero).detach(),
            **self._oracle_bbox_metrics(predictions["bbox_pred"], targets["bbox_targets"], positive_mask, zero),
        }

    def _objectness_elementwise_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.objectness_loss_type == "focal":
            return self.objectness_loss.elementwise_loss(logits, targets)
        target_values = targets.to(dtype=logits.dtype)
        if self.objectness_loss_type == "bce":
            return F.binary_cross_entropy_with_logits(logits, target_values, reduction="none")
        weights = self.objectness_negative_weight + target_values * (self.objectness_positive_weight - self.objectness_negative_weight)
        return F.binary_cross_entropy_with_logits(logits, target_values, weight=weights, reduction="none")

    def _bbox_quality_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_values = targets.detach().to(dtype=logits.dtype)
        sample_weights = weights.to(dtype=logits.dtype) if weights is not None else None
        if self.bbox_quality_loss_type in {"bce", "bce_soft_iou"}:
            loss = F.binary_cross_entropy_with_logits(logits, target_values, reduction="none")
            return self._weighted_mean_loss(loss, sample_weights)
        if self.bbox_quality_loss_type in {"varifocal", "varifocal_plus_ranking"}:
            probs = torch.sigmoid(logits)
            positive = target_values > 0.0
            focal_weight = torch.where(
                positive,
                target_values.clamp(0.0, 1.0),
                float(self.bbox_quality_varifocal_alpha) * probs.pow(float(self.bbox_quality_varifocal_gamma)),
            )
            loss = F.binary_cross_entropy_with_logits(logits, target_values, reduction="none") * focal_weight
            return self._weighted_mean_loss(loss, sample_weights)
        if self.bbox_quality_loss_type == "quality_focal":
            probs = torch.sigmoid(logits)
            focal_weight = (target_values - probs).abs().pow(float(self.bbox_quality_qfl_beta))
            loss = F.binary_cross_entropy_with_logits(logits, target_values, reduction="none") * focal_weight
            return self._weighted_mean_loss(loss, sample_weights)
        if self.bbox_quality_loss_type == "pairwise_ranking":
            return logits.sum() * 0.0
        probs = torch.sigmoid(logits)
        if self.bbox_quality_loss_type == "mse":
            loss = F.mse_loss(probs, target_values, reduction="none")
        else:
            loss = F.smooth_l1_loss(probs, target_values, reduction="none")
        return self._weighted_mean_loss(loss, sample_weights)

    @staticmethod
    def _weighted_mean_loss(loss: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
        if weights is None:
            return loss.mean()
        weighted = loss * weights
        return weighted.sum() / weights.sum().clamp_min(1e-6)

    def _quality_ranking_loss(
        self,
        bbox_quality_logits: torch.Tensor,
        objectness_logits: torch.Tensor,
        max_pred_iou: torch.Tensor,
        alignment_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = bbox_quality_logits.sum() * 0.0
        max_iou = max_pred_iou.detach()
        objectness_prob = torch.sigmoid(objectness_logits.detach()).to(dtype=max_iou.dtype)
        quality_prob = torch.sigmoid(bbox_quality_logits.detach()).to(dtype=max_iou.dtype)
        hard_scores = _hard_negative_scores(
            objectness_prob,
            alignment_targets.detach().to(dtype=max_iou.dtype),
            self.quality_hard_negative_source,
            quality_prob=quality_prob,
        )
        pair_losses: list[torch.Tensor] = []
        pair_count = 0
        batch_budget = _distributed_budget(self.ranking_num_pairs, int(max_iou.shape[0]))
        for batch_idx in range(max_iou.shape[0]):
            pos = torch.nonzero(max_iou[batch_idx] >= self.ranking_pos_iou_threshold, as_tuple=False).flatten()
            neg = torch.nonzero(max_iou[batch_idx] <= self.ranking_neg_iou_threshold, as_tuple=False).flatten()
            k = min(batch_budget, int(pos.numel()), int(neg.numel()))
            if k == 0:
                continue
            selected_pos = _sample_indices(pos, k)
            selected_neg = _topk_indices(neg, hard_scores[batch_idx, neg], k)
            pos_score = bbox_quality_logits[batch_idx, selected_pos]
            neg_score = bbox_quality_logits[batch_idx, selected_neg]
            if self.quality_ranking_score_space == "sigmoid":
                pos_score = torch.sigmoid(pos_score)
                neg_score = torch.sigmoid(neg_score)
            pair_losses.append(F.relu(float(self.ranking_margin) - pos_score + neg_score))
            pair_count += int(k)
        if not pair_losses:
            return {
                "loss": zero,
                "num_pairs": torch.zeros((), device=bbox_quality_logits.device),
            }
        return {
            "loss": torch.cat(pair_losses).mean(),
            "num_pairs": torch.tensor(float(pair_count), device=bbox_quality_logits.device),
        }

    def _effective_objectness_target_mode(self) -> str:
        if self.objectness_target_mode == "quality_iou" and self.current_epoch < self.quality_iou_start_epoch:
            return "assigned"
        if self.objectness_target_mode == "task_aligned" and self.current_epoch < self.task_aligned_start_epoch:
            return "assigned"
        return self.objectness_target_mode

    def effective_objectness_target_mode(self) -> str:
        return self._effective_objectness_target_mode()

    @staticmethod
    def _masked_prob_mean(logits: torch.Tensor, mask: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.detach())
        return probs[mask].mean() if mask.any() else zero.detach()

    @staticmethod
    def _oracle_bbox_metrics(
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        positive_mask: torch.Tensor,
        zero: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not positive_mask.any():
            zero_detached = zero.detach()
            return {
                "oracle_pos_bbox_mean_iou": zero_detached,
                "oracle_pos_bbox_max_iou": zero_detached,
                "oracle_pos_bbox_recall@0.1": zero_detached,
                "oracle_pos_bbox_recall@0.3": zero_detached,
                "oracle_pos_bbox_recall@0.45": zero_detached,
            }
        pred = pred_boxes.detach()[positive_mask]
        target = target_boxes.detach()[positive_mask]
        ious = box_iou(pred.float(), target.float()).diag()
        return {
            "oracle_pos_bbox_mean_iou": ious.mean().detach(),
            "oracle_pos_bbox_max_iou": ious.max().detach(),
            "oracle_pos_bbox_recall@0.1": (ious >= 0.1).float().mean().detach(),
            "oracle_pos_bbox_recall@0.3": (ious >= 0.3).float().mean().detach(),
            "oracle_pos_bbox_recall@0.45": (ious >= 0.45).float().mean().detach(),
        }


# Backward-compatible name used by earlier skeleton code.
DetectionLoss = Module1DetectionLoss
