"""Module 4: rule-based discoid meniscus detection.

paper-specified:
    In the coronal plane, measure tibia width ``w1`` and meniscus width ``w2``.
    Compute ``beta = w2 / w1``. ``beta > 0.4`` indicates discoid meniscus.

paper-inferred implementation:
    The manuscript does not publish width extraction code. This module provides
    practical interfaces for three tibia-width sources and two meniscus-width
    sources while keeping the beta rule exactly configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage


@dataclass(frozen=True)
class WidthMeasurement:
    """One width measurement in millimeters or consistent pixel units."""

    width: float
    source: str
    unit: str = "mm"
    slice_index: int | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class DiscoidMeasurement:
    """Rule-based discoid meniscus output."""

    meniscus_width: float
    tibia_width: float
    beta: float
    probability: float
    predicted_label: int
    threshold: float
    meniscus_source: str
    tibia_source: str
    unit: str = "mm"

    @property
    def is_discoid(self) -> bool:
        return bool(self.predicted_label)


@dataclass(frozen=True)
class PairedWidthMeasurement:
    """Meniscus and tibia widths measured on the same coronal slice."""

    meniscus: WidthMeasurement
    tibia: WidthMeasurement
    aggregation_strategy: str = "per_slice"
    candidate_count: int = 1
    eligible_candidate_count: int = 1

    @property
    def beta(self) -> float:
        return compute_beta(self.meniscus.width, self.tibia.width)


PAIRED_WIDTH_AGGREGATION_STRATEGIES = {
    "max_w2",
    "max_w1",
    "median_beta",
    "q25_beta",
    "central3_median",
}


def compute_beta(meniscus_width: float, tibia_width: float) -> float:
    """Compute ``beta = meniscus_width / tibia_width``."""

    if tibia_width <= 0:
        raise ValueError("tibia_width must be positive")
    if meniscus_width < 0:
        raise ValueError("meniscus_width must be non-negative")
    return meniscus_width / tibia_width


def beta_to_probability(beta: float, threshold: float = 0.4, temperature: float = 0.05) -> float:
    """Map beta to a smooth probability-like score for ROC-AUC.

    The paper-specified decision is still ``beta > threshold``. This sigmoid is
    a paper-inferred convenience for ranking and visualization.
    """

    if temperature <= 0:
        return float(beta > threshold)
    return float(1.0 / (1.0 + np.exp(-(beta - threshold) / temperature)))


def detect_discoid_meniscus(
    meniscus_width: float,
    tibia_width: float,
    threshold: float = 0.4,
    probability_temperature: float = 0.05,
    meniscus_source: str = "unknown",
    tibia_source: str = "unknown",
    unit: str = "mm",
) -> DiscoidMeasurement:
    """Classify discoid meniscus using the beta rule."""

    beta = compute_beta(meniscus_width, tibia_width)
    return DiscoidMeasurement(
        meniscus_width=meniscus_width,
        tibia_width=tibia_width,
        beta=beta,
        probability=beta_to_probability(beta, threshold, probability_temperature),
        predicted_label=int(beta > threshold),
        threshold=threshold,
        meniscus_source=meniscus_source,
        tibia_source=tibia_source,
        unit=unit,
    )


def tibia_width_from_annotation(record: dict[str, Any]) -> WidthMeasurement | None:
    """Read manually annotated tibia width from JSONL ``module4_labels``."""

    labels = record.get("module4_labels", {})
    value = labels.get("tibia_width")
    if value is None:
        return None
    return WidthMeasurement(width=float(value), source="annotation", unit=labels.get("tibia_width_unit", "mm"))


def load_mask(mask_path: str | Path) -> np.ndarray:
    """Load a tibia segmentation mask from ``.npy`` or torch ``.pt``."""

    path = Path(mask_path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            value = value.get("mask", value.get("tibia_mask"))
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)
    raise ValueError(f"Unsupported mask format: {path}. Use .npy, .pt, or .pth.")


def tibia_widths_from_array(
    mask: np.ndarray | torch.Tensor,
    pixel_spacing_col: float = 1.0,
    unit: str = "mm",
    source: str = "mask",
    min_slice_pixels: int = 1,
) -> dict[int, WidthMeasurement]:
    """Measure the largest tibia component separately on every slice.

    Width is measured as the maximum foreground x-span across coronal slices.
    Only the largest connected component on each slice is retained so isolated
    segmentation artifacts cannot expand the measured width.
    For a 2D mask, shape is ``[H,W]``. For a 3D mask, shape is ``[T,H,W]``.
    """

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask) > 0
    if mask.ndim == 2:
        masks = mask[None, ...]
    elif mask.ndim == 3:
        masks = mask
    else:
        raise ValueError(f"Expected 2D or 3D tibia mask, got shape {mask.shape}")

    measurements: dict[int, WidthMeasurement] = {}
    for slice_idx, slice_mask in enumerate(masks):
        component_labels, component_count = ndimage.label(
            slice_mask,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        if component_count == 0:
            continue
        component_sizes = np.bincount(component_labels.ravel())
        component_sizes[0] = 0
        largest_component = component_labels == int(component_sizes.argmax())
        ys, xs = np.where(largest_component)
        if xs.size < int(min_slice_pixels):
            continue
        width_px = float(xs.max() - xs.min() + 1)
        measurements[slice_idx] = WidthMeasurement(
            width=width_px * float(pixel_spacing_col),
            source=source,
            unit=unit,
            slice_index=slice_idx,
            bbox_xyxy=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
        )
    return measurements


def tibia_width_from_array(
    mask: np.ndarray | torch.Tensor,
    pixel_spacing_col: float = 1.0,
    unit: str = "mm",
    source: str = "mask",
    min_slice_pixels: int = 1,
) -> WidthMeasurement:
    """Return the maximum per-slice tibia width from a segmentation mask."""

    measurements = tibia_widths_from_array(
        mask,
        pixel_spacing_col=pixel_spacing_col,
        unit=unit,
        source=source,
        min_slice_pixels=min_slice_pixels,
    )
    if not measurements:
        raise ValueError("Tibia mask has no usable foreground pixels")
    return max(measurements.values(), key=lambda item: item.width)


def tibia_width_from_mask(mask_path: str | Path, pixel_spacing_col: float = 1.0, unit: str = "mm") -> WidthMeasurement:
    """Load a tibia mask and compute its maximum coronal width."""

    return tibia_width_from_array(load_mask(mask_path), pixel_spacing_col=pixel_spacing_col, unit=unit, source="mask")


def tibia_width_from_unet(
    volume: torch.Tensor,
    model: Any | None = None,
    pixel_spacing_col: float = 1.0,
    min_slice_pixels: int = 1,
) -> WidthMeasurement:
    """Predict a tibia mask with U2Net and measure its maximum width.

    Args:
        volume: Coronal volume tensor, for example ``[1,T,H,W]``.
        model: Segmentation adapter exposing ``predict_mask(volume)``.
        pixel_spacing_col: Column spacing used to convert mask width to mm.
    """

    if model is None:
        raise ValueError("A tibia segmentation model is required")
    if not hasattr(model, "predict_mask"):
        raise TypeError("Tibia segmentation model must expose predict_mask(volume)")
    mask = model.predict_mask(volume)
    return tibia_width_from_array(
        mask,
        pixel_spacing_col=pixel_spacing_col,
        source=str(getattr(model, "source", "unet")),
        min_slice_pixels=min_slice_pixels,
    )


def meniscus_width_from_bbox(
    bbox_xyxy: list[float] | tuple[float, float, float, float] | torch.Tensor,
    pixel_spacing_col: float = 1.0,
    source: str = "bbox",
    unit: str = "mm",
    slice_index: int | None = None,
) -> WidthMeasurement:
    """Measure meniscus width from a coronal bbox."""

    box = torch.as_tensor(bbox_xyxy, dtype=torch.float32)
    x1, x2 = float(torch.minimum(box[0], box[2])), float(torch.maximum(box[0], box[2]))
    return WidthMeasurement(width=max(0.0, x2 - x1) * float(pixel_spacing_col), source=source, unit=unit, slice_index=slice_index, bbox_xyxy=tuple(float(v) for v in box.tolist()))


def lateral_coronal_gt_meniscus_width(record: dict[str, Any], pixel_spacing_col: float = 1.0) -> WidthMeasurement | None:
    """Measure lateral meniscus width from GT coronal bbox labels."""

    best: WidthMeasurement | None = None
    for label in record.get("module1_labels", []):
        if label.get("plane") != "coronal" or label.get("meniscus_side") != "lateral":
            continue
        measurement = meniscus_width_from_bbox(label["bbox"], pixel_spacing_col=pixel_spacing_col, source="gt_coronal_lateral_bbox", slice_index=int(label.get("slice_index", 0)))
        if best is None or measurement.width > best.width:
            best = measurement
    return best


def lateral_coronal_predicted_meniscus_width_from_roi_cache(roi_payload: dict[str, Any], image_width: int | None = None, pixel_spacing_col: float = 1.0) -> WidthMeasurement | None:
    """Measure lateral meniscus width from generated ROI cache predicted boxes.

    ROI cache boxes are normalized ``xyxy``. ``image_width`` is needed to convert
    normalized width to pixels before applying spacing.
    """

    lateral = roi_payload.get("lateral", {})
    boxes = lateral.get("boxes", {}).get("coronal", [])
    measured_boxes = [
        (index, item)
        for index, item in enumerate(boxes)
        if not str(item.get("source", "")).startswith("filled_")
        and str(item.get("source", "")) != "fallback_center"
    ]
    if not measured_boxes:
        return None
    if image_width is None:
        coronal_roi = lateral.get("coronal_roi")
        image_width = int(coronal_roi.shape[-1]) if isinstance(coronal_roi, torch.Tensor) else 1
    best_index, best_box = max(
        measured_boxes,
        key=lambda pair: abs(float(pair[1]["box"][2]) - float(pair[1]["box"][0])),
    )
    box = torch.as_tensor(best_box["box"], dtype=torch.float32)
    if float(box.max()) <= 1.0:
        box = box * torch.tensor([image_width, 1.0, image_width, 1.0])
    return meniscus_width_from_bbox(
        box,
        pixel_spacing_col=pixel_spacing_col,
        source=f"roi_cache_{best_box.get('source', 'pred')}_max_width",
        slice_index=best_index,
    )


def paired_width_candidates_from_mask_and_roi_cache(
    mask: np.ndarray | torch.Tensor,
    roi_payload: dict[str, Any],
    image_width: int,
    pixel_spacing_col: float = 1.0,
    tibia_source: str = "mask",
    min_slice_pixels: int = 1,
) -> list[PairedWidthMeasurement]:
    """Return all real same-slice ``w1``/``w2`` candidates.

    Figure 11 of the manuscript draws the tibial plateau width and the lateral
    meniscus bounding-box width on the same coronal image. Filled/fallback ROI
    boxes are excluded because they are crop conveniences rather than actual
    Module 1 detections. Aggregation is deliberately handled separately so one
    U2Net pass can be audited with several paper-inferred slice-selection rules.
    """

    tibia_by_slice = tibia_widths_from_array(
        mask,
        pixel_spacing_col=pixel_spacing_col,
        source=tibia_source,
        min_slice_pixels=min_slice_pixels,
    )
    boxes = roi_payload.get("lateral", {}).get("boxes", {}).get("coronal", [])
    candidates: list[PairedWidthMeasurement] = []
    for slice_index, item in enumerate(boxes):
        source = str(item.get("source", ""))
        if source.startswith("filled_") or source == "fallback_center":
            continue
        tibia = tibia_by_slice.get(slice_index)
        if tibia is None:
            continue
        box = torch.as_tensor(item["box"], dtype=torch.float32)
        if float(box.max()) <= 1.0:
            box = box * torch.tensor([image_width, 1.0, image_width, 1.0])
        meniscus = meniscus_width_from_bbox(
            box,
            pixel_spacing_col=pixel_spacing_col,
            source=f"roi_cache_{source}_same_slice",
            slice_index=slice_index,
        )
        if meniscus.width > 0:
            candidates.append(PairedWidthMeasurement(meniscus=meniscus, tibia=tibia))
    return candidates


def _nearest_beta_quantile(
    candidates: list[PairedWidthMeasurement],
    quantile: float,
) -> PairedWidthMeasurement:
    ordered = sorted(
        candidates,
        key=lambda item: (item.beta, int(item.meniscus.slice_index or 0)),
    )
    rank = int(np.floor(float(quantile) * (len(ordered) - 1) + 0.5))
    return ordered[rank]


def select_paired_width_candidate(
    candidates: list[PairedWidthMeasurement],
    aggregation_strategy: str = "max_w2",
    min_plateau_width_fraction: float = 0.95,
    min_candidate_slices: int = 1,
    central_window_radius: int = 1,
) -> PairedWidthMeasurement | None:
    """Select one interpretable measured slice from same-slice candidates.

    ``median_beta`` and ``q25_beta`` select the nearest actual candidate rather
    than returning an interpolated synthetic width pair. ``central3_median``
    uses the maximum-plateau slice and its immediate neighbours by default.
    """

    if not candidates:
        return None
    strategy = str(aggregation_strategy).lower()
    if strategy not in PAIRED_WIDTH_AGGREGATION_STRATEGIES:
        raise ValueError(
            "aggregation_strategy must be one of "
            f"{sorted(PAIRED_WIDTH_AGGREGATION_STRATEGIES)}, got {aggregation_strategy!r}"
        )
    fraction = float(min_plateau_width_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("min_plateau_width_fraction must be in (0, 1]")
    if int(min_candidate_slices) <= 0:
        raise ValueError("min_candidate_slices must be positive")
    if int(central_window_radius) < 0:
        raise ValueError("central_window_radius must be non-negative")

    minimum_width = max(item.tibia.width for item in candidates) * fraction
    eligible = [item for item in candidates if item.tibia.width >= minimum_width]
    if strategy == "central3_median":
        anchor = max(candidates, key=lambda item: item.tibia.width)
        anchor_index = int(anchor.tibia.slice_index or 0)
        eligible = [
            item
            for item in eligible
            if abs(int(item.tibia.slice_index or 0) - anchor_index) <= int(central_window_radius)
        ]
    if len(eligible) < int(min_candidate_slices):
        return None

    if strategy == "max_w2":
        selected = max(eligible, key=lambda item: item.meniscus.width)
    elif strategy == "max_w1":
        selected = max(eligible, key=lambda item: item.tibia.width)
    elif strategy in {"median_beta", "central3_median"}:
        selected = _nearest_beta_quantile(eligible, 0.5)
    else:
        selected = _nearest_beta_quantile(eligible, 0.25)
    return PairedWidthMeasurement(
        meniscus=selected.meniscus,
        tibia=selected.tibia,
        aggregation_strategy=strategy,
        candidate_count=len(candidates),
        eligible_candidate_count=len(eligible),
    )


def paired_widths_from_mask_and_roi_cache(
    mask: np.ndarray | torch.Tensor,
    roi_payload: dict[str, Any],
    image_width: int,
    pixel_spacing_col: float = 1.0,
    tibia_source: str = "mask",
    min_slice_pixels: int = 1,
    min_plateau_width_fraction: float = 0.95,
    aggregation_strategy: str = "max_w2",
    min_candidate_slices: int = 1,
    central_window_radius: int = 1,
) -> PairedWidthMeasurement | None:
    """Measure one same-slice pair using a configurable aggregation rule."""

    candidates = paired_width_candidates_from_mask_and_roi_cache(
        mask,
        roi_payload,
        image_width=image_width,
        pixel_spacing_col=pixel_spacing_col,
        tibia_source=tibia_source,
        min_slice_pixels=min_slice_pixels,
    )
    return select_paired_width_candidate(
        candidates,
        aggregation_strategy=aggregation_strategy,
        min_plateau_width_fraction=min_plateau_width_fraction,
        min_candidate_slices=min_candidate_slices,
        central_window_radius=central_window_radius,
    )


def paired_widths_from_unet_and_roi_cache(
    volume: torch.Tensor,
    model: Any,
    roi_payload: dict[str, Any],
    image_width: int,
    pixel_spacing_col: float = 1.0,
    min_slice_pixels: int = 1,
    min_plateau_width_fraction: float = 0.95,
    aggregation_strategy: str = "max_w2",
    min_candidate_slices: int = 1,
    central_window_radius: int = 1,
) -> PairedWidthMeasurement | None:
    """Segment tibia and measure same-slice ``w1``/``w2`` pairs."""

    if model is None or not hasattr(model, "predict_mask"):
        raise TypeError("Tibia segmentation model must expose predict_mask(volume)")
    mask = model.predict_mask(volume)
    return paired_widths_from_mask_and_roi_cache(
        mask,
        roi_payload,
        image_width=image_width,
        pixel_spacing_col=pixel_spacing_col,
        tibia_source=str(getattr(model, "source", "unet")),
        min_slice_pixels=min_slice_pixels,
        min_plateau_width_fraction=min_plateau_width_fraction,
        aggregation_strategy=aggregation_strategy,
        min_candidate_slices=min_candidate_slices,
        central_window_radius=central_window_radius,
    )
