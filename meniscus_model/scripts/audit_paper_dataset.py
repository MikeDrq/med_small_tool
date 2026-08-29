"""Audit converted/split labels before expensive paper-reproduction training."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    module2: Counter[str] = Counter()
    module3: Counter[str] = Counter()
    module4: Counter[str] = Counter()
    module1: Counter[str] = Counter()
    invalid_boxes = 0
    unmatched_lesions = 0
    lesion_boxes = 0
    depths: dict[str, list[int]] = {"sagittal": [], "coronal": []}
    group_ids: list[str] = []
    case_ids: list[str] = []
    study_uids: list[str] = []
    missing_patient_groups = 0
    logical_inconsistencies: Counter[str] = Counter()
    distinct_labeled_slices: dict[str, list[int]] = {"sagittal": [], "coronal": []}

    for record in records:
        case_ids.append(str(record.get("case_id", "")))
        study_uids.append(str(record.get("study_uid", "")))
        patient_group_id = record.get("patient_group_id")
        if not patient_group_id:
            missing_patient_groups += 1
        group_ids.append(str(patient_group_id or record.get("case_id") or record.get("study_uid")))
        abnormal_sides: set[str] = set()
        for side in ("medial", "lateral"):
            status = str(record.get("module2_labels", {}).get(f"{side}_status", "missing"))
            module2[f"{side}:{status}"] += 1
            bucket = int(record.get("module3_labels", {}).get(f"{side}_bucket_handle", 0))
            module3[f"{side}:{bucket}"] += 1
            if bucket and status != "tear":
                logical_inconsistencies["bucket_handle_without_tear_status"] += 1
        discoid = int(record.get("module4_labels", {}).get("lateral_discoid", 0))
        module4[f"lateral_discoid:{discoid}"] += 1

        conversion = record.get("conversion_metadata", {})
        seen = int(conversion.get("lesion_boxes_seen", 0))
        matched = int(conversion.get("lesion_boxes_matched", 0))
        lesion_boxes += seen
        unmatched_lesions += max(seen - matched, 0)
        for label in record.get("module1_labels", []):
            plane = str(label.get("plane", "missing"))
            side = str(label.get("meniscus_side", "missing"))
            abnormality = str(label.get("abnormality", "missing"))
            module1[f"{plane}:{side}:{abnormality}"] += 1
            if abnormality == "abnormal":
                abnormal_sides.add(side)
            box = label.get("bbox", [])
            if len(box) != 4 or float(box[2]) <= float(box[0]) or float(box[3]) <= float(box[1]):
                invalid_boxes += 1
        for side in ("medial", "lateral"):
            status = str(record.get("module2_labels", {}).get(f"{side}_status", "missing"))
            if status == "normal" and side in abnormal_sides:
                logical_inconsistencies["normal_status_with_abnormal_slice"] += 1
            if status in {"degeneration", "tear"} and side not in abnormal_sides:
                logical_inconsistencies["abnormal_status_without_abnormal_slice"] += 1
        for plane in ("sagittal", "coronal"):
            distinct_labeled_slices[plane].append(
                len({
                    int(label["slice_index"])
                    for label in record.get("module1_labels", [])
                    if label.get("plane") == plane
                })
            )
            series_dir = Path(str(record.get(f"{plane}_dicom_dir", "")))
            if series_dir.is_dir():
                depths[plane].append(sum(1 for item in series_dir.iterdir() if item.is_file()))

    def depth_stats(values: list[int]) -> dict[str, float | int | None]:
        if not values:
            return {"min": None, "median": None, "max": None}
        ordered = sorted(values)
        return {
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "max": ordered[-1],
            "over_32": sum(value > 32 for value in values),
        }

    patient_counts = Counter(group_ids)
    return {
        "records": len(records),
        "unique_case_ids": len(set(case_ids)),
        "unique_study_uids": len(set(study_uids)),
        "unique_patient_groups": len(patient_counts),
        "repeated_patient_groups": sum(count > 1 for count in patient_counts.values()),
        "records_in_repeated_patient_groups": sum(count for count in patient_counts.values() if count > 1),
        "missing_patient_group_ids": missing_patient_groups,
        "module1_box_counts": dict(sorted(module1.items())),
        "module2_case_counts": dict(sorted(module2.items())),
        "module3_case_counts": dict(sorted(module3.items())),
        "module4_case_counts": dict(sorted(module4.items())),
        "lesion_boxes": lesion_boxes,
        "unmatched_lesion_boxes": unmatched_lesions,
        "invalid_boxes": invalid_boxes,
        "logical_inconsistencies": dict(sorted(logical_inconsistencies.items())),
        "series_depth": {plane: depth_stats(values) for plane, values in depths.items()},
        "distinct_labeled_slice_count": {
            plane: depth_stats(values) for plane, values in distinct_labeled_slices.items()
        },
    }


def audit(
    inputs: list[str],
    output: str | None = None,
    annotation_root: str | None = None,
) -> dict[str, Any]:
    records_by_split = {Path(path).stem: read_jsonl(path) for path in inputs}
    summaries = {name: summarize_records(records) for name, records in records_by_split.items()}
    group_sets = {
        name: {
            str(record.get("patient_group_id") or record.get("case_id") or record.get("study_uid"))
            for record in records
        }
        for name, records in records_by_split.items()
    }
    leakage: dict[str, int] = {}
    names = list(group_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            leakage[f"{left}__{right}"] = len(group_sets[left] & group_sets[right])
    report: dict[str, Any] = {"splits": summaries, "patient_group_overlap": leakage}
    if annotation_root:
        root = Path(annotation_root)
        annotation_paths = [root] if root.is_file() else sorted(root.rglob("*.json"))
        valid_studies: set[str] = set()
        invalid_annotations: list[str] = []
        for path in annotation_paths:
            try:
                annotation = json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception:
                invalid_annotations.append(str(path))
                continue
            study_uid = str(annotation.get("study_uid") or "")
            if study_uid:
                valid_studies.add(study_uid)
        indexed_studies = {
            str(record.get("study_uid") or "")
            for records in records_by_split.values()
            for record in records
            if record.get("study_uid")
        }
        missing_studies = sorted(valid_studies - indexed_studies)
        report["source_availability"] = {
            "annotation_files": len(annotation_paths),
            "valid_annotation_studies": len(valid_studies),
            "invalid_annotation_files": len(invalid_annotations),
            "indexed_studies": len(indexed_studies),
            "annotations_without_usable_index_record": len(missing_studies),
            "coverage_fraction": len(indexed_studies) / max(len(valid_studies), 1),
            "invalid_examples": invalid_annotations[:20],
            "missing_study_examples": missing_studies[:20],
        }
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="One or more converted JSONL files")
    parser.add_argument("--output", default=None)
    parser.add_argument("--annotation-root", default=None, help="Optional source annotations for conversion-coverage auditing.")
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.inputs, output=args.output, annotation_root=args.annotation_root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
