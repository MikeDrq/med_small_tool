"""Rewrite JSONL index paths after moving data to another machine.

Example:
    python scripts/remap_index_paths.py \
        --input data/center_wh/train.jsonl \
        --output data/center_wh_server/train.jsonl \
        --new-root /srv/meniscus_data \
        --keep-from-segment data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_FIELDS = {
    "annotation_json",
    "sagittal_dicom_dir",
    "coronal_dicom_dir",
    "tibia_mask_path",
}


def normalize_parts(path: str) -> list[str]:
    return [part for part in path.replace("\\", "/").split("/") if part]


def remap_path(path: str, new_root: str, keep_from_segment: str | None = None) -> str:
    parts = normalize_parts(path)
    if keep_from_segment:
        lowered = [part.lower() for part in parts]
        marker = keep_from_segment.lower()
        if marker in lowered:
            parts = parts[len(lowered) - 1 - lowered[::-1].index(marker) :]
    return str(Path(new_root, *parts).as_posix())


def remap_record(value: Any, fields: set[str], new_root: str, keep_from_segment: str | None) -> Any:
    if isinstance(value, dict):
        updated: dict[str, Any] = {}
        for key, item in value.items():
            if key in fields and isinstance(item, str) and item:
                updated[key] = remap_path(item, new_root, keep_from_segment)
            else:
                updated[key] = remap_record(item, fields, new_root, keep_from_segment)
        return updated
    if isinstance(value, list):
        return [remap_record(item, fields, new_root, keep_from_segment) for item in value]
    return value


def remap_jsonl(input_path: Path, output_path: Path, fields: set[str], new_root: str, keep_from_segment: str | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            dst.write(json.dumps(remap_record(record, fields, new_root, keep_from_segment), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite path fields in a JSONL index for a new data root.")
    parser.add_argument("--input", required=True, help="Input JSONL index.")
    parser.add_argument("--output", required=True, help="Output JSONL index.")
    parser.add_argument("--new-root", required=True, help="New root path on the target machine.")
    parser.add_argument(
        "--keep-from-segment",
        default=None,
        help="Keep only the suffix starting from this path segment, e.g. data.",
    )
    parser.add_argument(
        "--field",
        action="append",
        default=[],
        help="Additional JSON key to rewrite. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fields = set(DEFAULT_FIELDS)
    fields.update(args.field)
    remap_jsonl(
        Path(args.input),
        Path(args.output),
        fields,
        args.new_root,
        args.keep_from_segment,
    )


if __name__ == "__main__":
    main()
