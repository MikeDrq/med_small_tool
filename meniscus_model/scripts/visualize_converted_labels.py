"""Sample converted JSONL labels and save DICOM bbox overlay PNGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.meniscus_detection_dataset import MeniscusDetectionDataset, read_jsonl
from utils.visualization import save_slice_overlay


DEFAULT_CATEGORIES = ("normal", "tear", "degeneration", "medial", "lateral")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "case"


def categories_for_record(record: dict[str, Any]) -> set[str]:
    categories: set[str] = set()
    for label in record.get("module1_labels", []):
        side = label.get("meniscus_side")
        if side in {"medial", "lateral"}:
            categories.add(side)
        abnormality = label.get("abnormality")
        if abnormality == "normal":
            categories.add("normal")

    for status in record.get("module2_labels", {}).values():
        if status in {"normal", "tear", "degeneration"}:
            categories.add(status)
    return categories


def select_record_indices(
    records: list[dict[str, Any]],
    categories: list[str],
    per_category: int,
    seed: int,
) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    selected: list[tuple[str, int]] = []
    memberships = [categories_for_record(record) for record in records]
    for category in categories:
        candidates = [idx for idx, item_categories in enumerate(memberships) if category in item_categories]
        rng.shuffle(candidates)
        selected.extend((category, idx) for idx in candidates[:per_category])
    return selected


def _status_side_for_category(labels: dict[str, Any], category: str) -> str | None:
    module2 = labels.get("module2_labels", {})
    for side in ("medial", "lateral"):
        if module2.get(f"{side}_status") == category:
            return side
    return None


def _choose_focus_label(labels: dict[str, Any], category: str) -> dict[str, Any] | None:
    module1_labels = labels.get("module1_labels", [])
    if category in {"medial", "lateral"}:
        for label in module1_labels:
            if label.get("meniscus_side") == category:
                return label
    if category in {"tear", "degeneration"}:
        status_side = _status_side_for_category(labels, category)
        if status_side is not None:
            for label in module1_labels:
                if label.get("meniscus_side") == status_side:
                    return label
    if category == "normal":
        for label in module1_labels:
            if label.get("abnormality") == "normal":
                return label
        status_side = _status_side_for_category(labels, "normal")
        if status_side is not None:
            for label in module1_labels:
                if label.get("meniscus_side") == status_side:
                    return label
    return module1_labels[0] if module1_labels else None


def render_selected_overlays(
    index_jsonl: str | Path,
    output_dir: str | Path,
    categories: list[str] | None = None,
    per_category: int = 8,
    seed: int = 42,
    output_size: int = 224,
    target_slices: int = 16,
    patch_size: int = 16,
) -> dict[str, Any]:
    categories = list(categories or DEFAULT_CATEGORIES)
    records = read_jsonl(index_jsonl)
    selections = select_record_indices(records, categories, per_category=per_category, seed=seed)
    dataset = MeniscusDetectionDataset(
        index_jsonl=index_jsonl,
        output_size=output_size,
        target_slices=target_slices,
        patch_size=patch_size,
        transform=None,
    )

    output_root = Path(output_dir)
    saved: list[dict[str, Any]] = []
    counts = {category: 0 for category in categories}
    for category, record_index in selections:
        sample = dataset[record_index]
        labels = sample["labels"]
        focus = _choose_focus_label(labels, category)
        if focus is None:
            plane = "sagittal"
            slice_index = sample["sagittal_volume"].shape[1] // 2
            boxes: list[list[float]] = []
        else:
            plane = str(focus["plane"])
            slice_index = int(focus["slice_index"])
            boxes = [
                label["bbox"]
                for label in labels.get("module1_labels", [])
                if label.get("plane") == plane and int(label.get("slice_index", -1)) == slice_index
            ]

        case_id = str(sample["case_id"])
        filename = f"{counts[category]:03d}_{_safe_name(case_id)}_{plane}_slice{slice_index}.png"
        output_path = output_root / category / filename
        volume = sample[f"{plane}_volume"][0]
        save_slice_overlay(
            volume,
            output_path=output_path,
            boxes_xyxy=boxes,
            slice_index=slice_index,
            title=f"{case_id} | {category} | {plane} slice {slice_index}",
        )
        counts[category] += 1
        saved.append({"category": category, "case_id": case_id, "plane": plane, "slice_index": slice_index, "path": str(output_path)})

    summary = {
        "index_jsonl": str(index_jsonl),
        "output_dir": str(output_root),
        "seed": seed,
        "per_category": per_category,
        "counts": counts,
        "saved": saved,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "label_overlay_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-jsonl", default="data/index_all_center_wh.jsonl")
    parser.add_argument("--output-dir", default="outputs/label_overlays_center_wh")
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES), help="Comma-separated category names.")
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--target-slices", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=16)
    args = parser.parse_args()

    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    summary = render_selected_overlays(
        index_jsonl=args.index_jsonl,
        output_dir=args.output_dir,
        categories=categories,
        per_category=args.per_category,
        seed=args.seed,
        output_size=args.output_size,
        target_slices=args.target_slices,
        patch_size=args.patch_size,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "saved"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
