"""Build case-level JSONL indexes for the meniscus MRI pipeline.

The preferred input is an annotation JSON/JSONL file that already contains
``case_id`` and labels. This script attaches sagittal/coronal DICOM directories
found under ``raw_root`` and writes one normalized JSON object per line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODULE2 = {"medial_status": "normal", "lateral_status": "normal"}
DEFAULT_MODULE3 = {"medial_bucket_handle": 0, "lateral_bucket_handle": 0}
DEFAULT_MODULE4 = {"lateral_discoid": 0, "tibia_width": None, "tibia_mask_path": None}


def _read_annotation_records(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    annotation_path = Path(path)
    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)
    text = annotation_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if annotation_path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "cases" in data:
        return list(data["cases"])
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported annotation format: {annotation_path}")


def _contains_dicom_files(path: Path) -> bool:
    return any(child.is_file() for child in path.iterdir()) if path.exists() and path.is_dir() else False


def _find_plane_dir(case_dir: Path, plane: str) -> str | None:
    candidates = []
    for path in case_dir.rglob("*"):
        if not path.is_dir() or not _contains_dicom_files(path):
            continue
        name = path.name.lower()
        if plane in name:
            candidates.append(path)
    if not candidates:
        direct = case_dir / plane
        if _contains_dicom_files(direct):
            return str(direct)
        return None
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return str(candidates[0])


def _case_dirs(raw_root: Path) -> dict[str, Path]:
    dirs = {}
    for path in raw_root.iterdir():
        if path.is_dir():
            dirs[path.name] = path
    return dirs


def _default_record(case_id: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "module1_labels": [],
        "module2_labels": dict(DEFAULT_MODULE2),
        "module3_labels": dict(DEFAULT_MODULE3),
        "module4_labels": dict(DEFAULT_MODULE4),
    }


def normalize_record(record: dict[str, Any], raw_root: Path, case_dir: Path | None = None) -> dict[str, Any]:
    """Normalize one case record to the requested JSONL schema."""

    case_id = str(record["case_id"])
    case_dir = case_dir or raw_root / case_id
    sagittal = record.get("sagittal_dicom_dir") or _find_plane_dir(case_dir, "sagittal")
    coronal = record.get("coronal_dicom_dir") or _find_plane_dir(case_dir, "coronal")
    if sagittal is None or coronal is None:
        raise FileNotFoundError(f"Could not find sagittal/coronal DICOM dirs for case_id={case_id} under {case_dir}")

    normalized = _default_record(case_id)
    normalized.update(
        {
            "sagittal_dicom_dir": str(Path(sagittal).expanduser()),
            "coronal_dicom_dir": str(Path(coronal).expanduser()),
            "module1_labels": record.get("module1_labels", []),
            "module2_labels": {**DEFAULT_MODULE2, **record.get("module2_labels", {})},
            "module3_labels": {**DEFAULT_MODULE3, **record.get("module3_labels", {})},
            "module4_labels": {**DEFAULT_MODULE4, **record.get("module4_labels", {})},
        }
    )
    return normalized


def prepare_index(raw_root: str, annotations: str | None, output_jsonl: str) -> None:
    """Create a case-level JSONL index.

    If annotations are omitted or empty, each immediate subdirectory of
    ``raw_root`` is treated as one unlabeled case.
    """

    raw_root_path = Path(raw_root).expanduser()
    records = _read_annotation_records(annotations)
    case_dirs = _case_dirs(raw_root_path)
    if not records:
        records = [_default_record(case_id) for case_id in sorted(case_dirs)]

    normalized_records = []
    for record in records:
        case_id = str(record["case_id"])
        normalized_records.append(normalize_record(record, raw_root_path, case_dirs.get(case_id)))

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in normalized_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--annotations", default=None, help="Optional JSON or JSONL annotation file.")
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()
    prepare_index(args.raw_root, args.annotations, args.output_jsonl)


if __name__ == "__main__":
    main()
