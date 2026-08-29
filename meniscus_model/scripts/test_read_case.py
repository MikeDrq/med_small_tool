"""Minimal smoke test for DICOM reading, coordinates, labels, and visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from datasets.meniscus_detection_dataset import MeniscusDetectionDataset


def _draw_bbox(ax: plt.Axes, bbox: list[float], color: str = "tab:red") -> None:
    x1, y1, x2, y2 = bbox
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))


def test_read_case(index_jsonl: str, case_id: str | None, output_png: str, output_size: int, target_slices: int, patch_size: int) -> None:
    dataset = MeniscusDetectionDataset(
        index_jsonl=index_jsonl,
        output_size=output_size,
        target_slices=target_slices,
        patch_size=patch_size,
        transform=None,
    )
    sample = None
    for idx in range(len(dataset)):
        candidate = dataset[idx]
        if case_id is None or candidate["case_id"] == case_id:
            sample = candidate
            break
    if sample is None:
        raise ValueError(f"case_id={case_id} not found in {index_jsonl}")

    print("case_id:", sample["case_id"])
    print("sagittal volume shape:", tuple(sample["sagittal_volume"].shape))
    print("coronal volume shape:", tuple(sample["coronal_volume"].shape))
    print("sagittal coords shape:", tuple(sample["sagittal_coords"].shape))
    print("coronal coords shape:", tuple(sample["coronal_coords"].shape))
    print("bbox labels:", sample["labels"]["module1_labels"])

    labels = sample["labels"]["module1_labels"]
    if labels:
        label = labels[0]
        plane = label["plane"]
        slice_index = int(label["slice_index"])
        volume = sample[f"{plane}_volume"][0]
        image = volume[slice_index].numpy()
    else:
        plane = "sagittal"
        slice_index = sample["sagittal_volume"].shape[1] // 2
        image = sample["sagittal_volume"][0, slice_index].numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    for label in labels:
        if label["plane"] == plane and int(label["slice_index"]) == slice_index:
            _draw_bbox(ax, label["bbox"])
    ax.set_title(f"{sample['case_id']} | {plane} slice {slice_index}")
    ax.axis("off")
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print("saved visualization:", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-jsonl", required=True)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--output-png", default="outputs/test_read_case.png")
    parser.add_argument("--output-size", type=int, default=224, choices=[224, 256])
    parser.add_argument("--target-slices", type=int, default=16, choices=[16, 32, 48])
    parser.add_argument("--patch-size", type=int, default=16)
    args = parser.parse_args()
    test_read_case(args.index_jsonl, args.case_id, args.output_png, args.output_size, args.target_slices, args.patch_size)


if __name__ == "__main__":
    main()
