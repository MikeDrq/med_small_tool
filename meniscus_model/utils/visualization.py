"""Visualization helpers for detection boxes and cached ROI sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def _as_numpy_image(image: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")
    return image


def save_slice_overlay(
    image_or_volume: np.ndarray | torch.Tensor,
    output_path: str | Path,
    boxes_xyxy: Sequence[Sequence[float]] | None = None,
    slice_index: int | None = None,
    title: str | None = None,
) -> None:
    """Save original image plus normalized or pixel-space bbox overlays.

    Args:
        image_or_volume: 2D image ``[H,W]`` or volume ``[T,H,W]``/``[1,T,H,W]``.
        output_path: Destination PNG path.
        boxes_xyxy: Boxes in pixel-space. If all values are in ``[0,1]``, they
            are interpreted as normalized ``xyxy``.
        slice_index: Slice to visualize when a volume is given.
        title: Optional figure title.
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image = image_or_volume.detach().cpu().numpy() if isinstance(image_or_volume, torch.Tensor) else np.asarray(image_or_volume)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3:
        slice_index = image.shape[0] // 2 if slice_index is None else slice_index
        image = image[slice_index]
    image = _as_numpy_image(image)
    height, width = image.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    for box in boxes_xyxy or []:
        x1, y1, x2, y2 = [float(v) for v in box]
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="tab:red", linewidth=2))
    if title:
        ax.set_title(title)
    ax.axis("off")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roi_preview(
    sagittal_roi: torch.Tensor,
    coronal_roi: torch.Tensor,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Save a two-panel preview of center sagittal/coronal ROI slices.

    Args:
        sagittal_roi: Tensor ``[C,T,H,W]``.
        coronal_roi: Tensor ``[C,T,H,W]``.
        output_path: Destination PNG path.
    """

    import matplotlib.pyplot as plt

    def center_slice(roi: torch.Tensor) -> np.ndarray:
        roi_cpu = roi.detach().cpu()
        return roi_cpu[0, roi_cpu.shape[1] // 2].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(center_slice(sagittal_roi), cmap="gray")
    axes[0].set_title("sagittal ROI")
    axes[1].imshow(center_slice(coronal_roi), cmap="gray")
    axes[1].set_title("coronal ROI")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_module4_measurement_plot(
    output_path: str | Path,
    tibia_width: float,
    meniscus_width: float,
    beta: float,
    threshold: float = 0.4,
    title: str | None = None,
) -> None:
    """Visualize Module 4 width measurements as comparable w1/w2 bars."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(["w1 tibia", "w2 meniscus"], [tibia_width, meniscus_width], color=["tab:blue", "tab:orange"])
    ax.set_xlabel("width")
    ax.set_title(title or f"beta={beta:.3f}, threshold={threshold:.3f}")
    ax.text(tibia_width, 0, f" {tibia_width:.2f}", va="center")
    ax.text(meniscus_width, 1, f" {meniscus_width:.2f}", va="center")
    ax.text(0.02, 0.92, f"beta = w2 / w1 = {beta:.3f}", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_token_heatmap(
    image_or_volume: np.ndarray | torch.Tensor,
    heatmap_hw: np.ndarray | torch.Tensor,
    output_path: str | Path,
    slice_index: int | None = None,
    title: str | None = None,
) -> None:
    """Save a token-score heatmap over an image slice.

    This is used by the full pipeline as an attention/activation-style
    visualization when explicit attention tensors are not requested from the
    backbone.
    """

    import matplotlib.pyplot as plt

    image = image_or_volume.detach().cpu().numpy() if isinstance(image_or_volume, torch.Tensor) else np.asarray(image_or_volume)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3:
        slice_index = image.shape[0] // 2 if slice_index is None else slice_index
        image = image[slice_index]
    image = _as_numpy_image(image)
    heatmap = heatmap_hw.detach().cpu().numpy() if isinstance(heatmap_hw, torch.Tensor) else np.asarray(heatmap_hw)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    ax.imshow(heatmap, cmap="magma", alpha=0.45, extent=(0, image.shape[1], image.shape[0], 0))
    if title:
        ax.set_title(title)
    ax.axis("off")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
