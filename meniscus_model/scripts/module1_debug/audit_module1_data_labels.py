"""Audit Module 1 data/label geometry without changing training behavior.

The audit intentionally loads samples through :class:`MeniscusDetectionDataset`
with the same train/validation slice-selection policy as the requested config.
It writes only diagnostic artifacts under ``--output-dir``:

* ``resolved_config.yaml``: the exact audit inputs and resolved slice policy;
* ``summary.json``: aggregate schema, slice-mapping, duplication, and leakage checks;
* ``per_gt.csv``: one row per raw-GT-to-processed-slice mapping occurrence;
* ``overlays/*.png``: processed images with exact mappings in green and optional
  legacy nearest-slice mappings in red.

The pure helpers near the top of this file do not read DICOM and are unit tested
separately. This makes the most important mapping and leakage contracts testable
even when the private dataset is unavailable.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import posixpath
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable, Mapping, Sequence
import warnings

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.meniscus_detection_dataset import MeniscusDetectionDataset, read_jsonl


VALID_PLANES = {"sagittal", "coronal"}
VALID_SIDES = {"medial", "lateral"}
VALID_ABNORMALITY = {"normal", "abnormal"}
REQUIRED_RECORD_KEYS = {"case_id", "sagittal_dicom_dir", "coronal_dicom_dir", "module1_labels"}
REQUIRED_LABEL_KEYS = {"plane", "slice_index", "bbox", "meniscus_side", "abnormality"}

PER_GT_COLUMNS = [
    "dataset_index",
    "record_index",
    "chunk_index",
    "chunk_count",
    "case_id",
    "study_uid",
    "raw_gt_index",
    "raw_gt_id",
    "plane",
    "side",
    "abnormality",
    "source_series_uid",
    "source_instance_uid",
    "source_slice_index",
    "processed_slice_index",
    "sampled_source_slice_index",
    "slice_delta",
    "mapping_kind",
    "exact_slice_match",
    "replication_factor",
    "mapped_gt_count",
    "mapped_source_slice_count",
    "mapping_collision",
    "source_slice_collision",
    "exact_duplicate_count",
    "exact_duplicate",
    "bbox_source_xyxy",
    "bbox_processed_xyxy",
    "bbox_numeric",
    "bbox_finite",
    "bbox_strict_xyxy",
    "bbox_source_bounds_ok",
    "bbox_processed_bounds_ok",
    "source_slice_in_range",
    "identifiers_valid",
    "record_schema_valid",
    "label_schema_valid",
    "valid_for_training",
    "record_issues",
    "label_issues",
    "dataset_contract_match",
]


warnings.filterwarnings("ignore", message="Invalid value for VR UI:*")


def resolve_usage(split: str, usage: str = "auto") -> str:
    """Resolve ``auto`` to the same policy used by train/validation entrypoints."""

    normalized_split = str(split).lower()
    normalized_usage = str(usage).lower()
    if normalized_usage == "auto":
        return "train" if normalized_split == "train" else "val"
    if normalized_usage not in {"train", "val"}:
        raise ValueError("usage must be auto, train, or val")
    return normalized_usage


def resolve_slice_selection(config: Mapping[str, Any], split: str, usage: str = "auto") -> tuple[str, dict[str, Any]]:
    """Return ``(resolved_usage, slice_selection)`` without mutating config."""

    resolved_usage = resolve_usage(split, usage)
    data_cfg = dict(config.get("data", {}))
    key = "train_slice_selection" if resolved_usage == "train" else "val_slice_selection"
    selection = data_cfg.get(key, data_cfg.get("slice_selection"))
    return resolved_usage, dict(selection or {})


def select_case_indices(total: int, max_cases: int | None, seed: int) -> list[int]:
    """Select a deterministic random subset, returned in dataset order."""

    if total < 0:
        raise ValueError("total must be non-negative")
    if max_cases is None or int(max_cases) <= 0 or int(max_cases) >= total:
        return list(range(total))
    rng = random.Random(int(seed))
    return sorted(rng.sample(range(total), int(max_cases)))


def canonicalize_path(value: Any) -> str:
    """Normalize path separators for split-leakage comparison without I/O."""

    if not isinstance(value, (str, Path)) or not str(value).strip():
        return ""
    normalized = str(value).strip().replace("\\", "/")
    return posixpath.normpath(normalized)


def inspect_record_schema(record: Any) -> list[str]:
    """Return case-record schema issues; an empty list means structurally valid."""

    if not isinstance(record, dict):
        return ["record_not_mapping"]
    issues = [f"missing_record_key:{key}" for key in sorted(REQUIRED_RECORD_KEYS - set(record))]
    if "case_id" in record and not str(record.get("case_id", "")).strip():
        issues.append("empty_case_id")
    for key in ("sagittal_dicom_dir", "coronal_dicom_dir"):
        if key in record and not isinstance(record.get(key), (str, Path)):
            issues.append(f"{key}_not_path_string")
        elif key in record and not str(record.get(key, "")).strip():
            issues.append(f"empty_{key}")
    if "module1_labels" in record and not isinstance(record.get("module1_labels"), list):
        issues.append("module1_labels_not_list")
    return issues


def _coerce_bbox(bbox: Any) -> list[float] | None:
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        return None
    if any(isinstance(value, bool) for value in bbox):
        return None
    try:
        return [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None


def validate_bbox_xyxy(bbox: Any, width: int | float | None = None, height: int | float | None = None) -> dict[str, Any]:
    """Validate finite, strict pixel-space ``xyxy`` and optional image bounds."""

    values = _coerce_bbox(bbox)
    numeric = values is not None
    finite = bool(numeric and all(math.isfinite(value) for value in values))
    strict = bool(finite and values[0] < values[2] and values[1] < values[3])
    bounds_ok: bool | None = None
    if finite and width is not None and height is not None:
        bounds_ok = bool(
            float(width) > 0
            and float(height) > 0
            and values[0] >= 0.0
            and values[1] >= 0.0
            and values[2] <= float(width)
            and values[3] <= float(height)
        )
    issues: list[str] = []
    if not numeric:
        issues.append("bbox_not_numeric_length4")
    elif not finite:
        issues.append("bbox_not_finite")
    elif not strict:
        issues.append("bbox_not_strict_xyxy")
    if bounds_ok is False:
        issues.append("bbox_out_of_bounds")
    return {
        "values": values,
        "numeric": numeric,
        "finite": finite,
        "strict_xyxy": strict,
        "bounds_ok": bounds_ok,
        "issues": issues,
    }


def analyze_source_slice_mapping(
    source_slice_index: int,
    sampled_source_indices: Sequence[int],
    unmatched_label_policy: str = "drop",
) -> list[dict[str, Any]]:
    """Mirror Dataset source->processed mapping and expose its semantic delta."""

    sampled = [int(value) for value in sampled_source_indices]
    if not sampled:
        return []
    policy = str(unmatched_label_policy).lower()
    if policy not in {"drop", "nearest", "error"}:
        raise ValueError(f"unsupported unmatched_label_policy={unmatched_label_policy!r}")
    source = int(source_slice_index)
    exact_positions = [idx for idx, value in enumerate(sampled) if value == source]
    if exact_positions:
        positions = exact_positions
    elif policy == "drop":
        return [
            {
                "processed_slice_index": None,
                "sampled_source_slice_index": None,
                "slice_delta": None,
                "exact_slice_match": False,
                "mapping_kind": "dropped",
            }
        ]
    elif policy == "error":
        raise ValueError(f"source label slice {source} is not sampled: {sampled}")
    else:
        distances = [abs(value - source) for value in sampled]
        positions = [int(min(range(len(distances)), key=lambda idx: distances[idx]))]
    rows = []
    for processed_index in positions:
        sampled_source = sampled[processed_index]
        delta = int(sampled_source - source)
        rows.append(
            {
                "processed_slice_index": int(processed_index),
                "sampled_source_slice_index": sampled_source,
                "slice_delta": delta,
                "exact_slice_match": delta == 0,
                "mapping_kind": "exact" if delta == 0 else "nearest",
            }
        )
    return rows


def scale_bbox_to_processed(
    bbox: Any,
    source_shape: Sequence[int],
    processed_shape: Sequence[int],
) -> list[float] | None:
    """Scale pixel ``xyxy`` exactly as ``MeniscusDetectionDataset`` does."""

    values = _coerce_bbox(bbox)
    if values is None or len(source_shape) != 3 or len(processed_shape) != 3:
        return None
    _, source_h, source_w = [int(value) for value in source_shape]
    _, processed_h, processed_w = [int(value) for value in processed_shape]
    if source_h <= 0 or source_w <= 0:
        return None
    return [
        values[0] * processed_w / float(source_w),
        values[1] * processed_h / float(source_h),
        values[2] * processed_w / float(source_w),
        values[3] * processed_h / float(source_h),
    ]


def canonical_label_key(label: Mapping[str, Any], precision: int = 6) -> tuple[Any, ...]:
    """Build an exact-label identity robust to int/float JSON spelling."""

    bbox = _coerce_bbox(label.get("bbox"))
    bbox_key: tuple[Any, ...]
    if bbox is None:
        bbox_key = (repr(label.get("bbox")),)
    else:
        bbox_key = tuple(round(value, precision) if math.isfinite(value) else repr(value) for value in bbox)
    try:
        slice_index: Any = int(label.get("slice_index"))
    except (TypeError, ValueError):
        slice_index = repr(label.get("slice_index"))
    return (
        str(label.get("plane")),
        slice_index,
        str(label.get("meniscus_side")),
        str(label.get("abnormality")),
        bbox_key,
        str(label.get("source_series_uid", "")),
        str(label.get("source_instance_uid", "")),
    )


def exact_duplicate_groups(labels: Sequence[Any]) -> dict[int, list[int]]:
    """Map each duplicated raw-label index to all identical label indices."""

    grouped: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        if isinstance(label, Mapping):
            grouped[canonical_label_key(label)].append(index)
    output: dict[int, list[int]] = {}
    for indices in grouped.values():
        if len(indices) > 1:
            for index in indices:
                output[index] = list(indices)
    return output


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    return None


def _json_list(values: Sequence[float] | None) -> str:
    return "" if values is None else json.dumps([float(value) for value in values], ensure_ascii=False)


def build_gt_audit_rows(
    record: Mapping[str, Any],
    metadata_by_plane: Mapping[str, Mapping[str, Any]],
    record_index: int,
    unmatched_label_policy: str = "drop",
    dataset_index: int | None = None,
    chunk_index: int = 0,
    chunk_count: int = 1,
    allowed_source_label_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Create per-GT mapping rows from raw record plus processed DICOM metadata."""

    record_issues = inspect_record_schema(record)
    raw_labels = record.get("module1_labels", []) if isinstance(record, Mapping) else []
    if not isinstance(raw_labels, list):
        return []
    duplicate_map = exact_duplicate_groups(raw_labels)
    rows: list[dict[str, Any]] = []

    for gt_index, raw_label in enumerate(raw_labels):
        if allowed_source_label_indices is not None and gt_index not in allowed_source_label_indices:
            continue
        label = raw_label if isinstance(raw_label, Mapping) else {}
        label_issues = [f"missing_label_key:{key}" for key in sorted(REQUIRED_LABEL_KEYS - set(label))]
        plane = str(label.get("plane", ""))
        side = str(label.get("meniscus_side", ""))
        abnormality = str(label.get("abnormality", ""))
        identifiers_valid = plane in VALID_PLANES and side in VALID_SIDES and abnormality in VALID_ABNORMALITY
        if plane not in VALID_PLANES:
            label_issues.append("invalid_plane")
        if side not in VALID_SIDES:
            label_issues.append("invalid_side")
        if abnormality not in VALID_ABNORMALITY:
            label_issues.append("invalid_abnormality")

        source_slice = _safe_int(label.get("slice_index"))
        if source_slice is None:
            label_issues.append("slice_index_not_integer")

        plane_metadata = metadata_by_plane.get(plane, {})
        original_shape = plane_metadata.get("original_shape")
        processed_shape = plane_metadata.get("processed_shape")
        sampled_indices = plane_metadata.get("sampled_source_indices", [])
        source_h = source_w = source_depth = None
        valid_original_shape = isinstance(original_shape, Sequence) and not isinstance(original_shape, (str, bytes)) and len(original_shape) == 3
        valid_processed_shape = isinstance(processed_shape, Sequence) and not isinstance(processed_shape, (str, bytes)) and len(processed_shape) == 3
        if valid_original_shape:
            try:
                source_depth, source_h, source_w = [int(value) for value in original_shape]
            except (TypeError, ValueError, OverflowError):
                valid_original_shape = False
                source_depth = source_h = source_w = None
        source_slice_in_range: bool | None = None
        if source_slice is not None and source_depth is not None:
            source_slice_in_range = 0 <= source_slice < source_depth
            if not source_slice_in_range:
                label_issues.append("source_slice_out_of_range")
        elif not plane_metadata:
            label_issues.append("plane_metadata_unavailable")

        bbox_source = validate_bbox_xyxy(label.get("bbox"), width=source_w, height=source_h)
        label_issues.extend(bbox_source["issues"])
        processed_bbox = None
        bbox_processed = validate_bbox_xyxy(None)
        if valid_original_shape and valid_processed_shape:
            processed_bbox = scale_bbox_to_processed(label.get("bbox"), original_shape, processed_shape)
            try:
                processed_h, processed_w = int(processed_shape[1]), int(processed_shape[2])
            except (TypeError, ValueError, OverflowError):
                processed_h = processed_w = None
            if processed_h is not None and processed_w is not None:
                bbox_processed = validate_bbox_xyxy(processed_bbox, width=processed_w, height=processed_h)
                if bbox_processed["bounds_ok"] is False:
                    label_issues.append("processed_bbox_out_of_bounds")

        mappings = []
        if source_slice is not None and isinstance(sampled_indices, Sequence):
            mappings = analyze_source_slice_mapping(
                source_slice,
                sampled_indices,
                unmatched_label_policy=unmatched_label_policy,
            )
        if not mappings:
            mappings = [
                {
                    "processed_slice_index": None,
                    "sampled_source_slice_index": None,
                    "slice_delta": None,
                    "exact_slice_match": False,
                    "mapping_kind": "unavailable",
                }
            ]
            if "plane_metadata_unavailable" not in label_issues:
                label_issues.append("slice_mapping_unavailable")

        duplicate_indices = duplicate_map.get(gt_index, [gt_index])
        raw_gt_id = f"{int(record_index)}:{gt_index}"
        unique_label_issues = sorted(set(label_issues))
        for mapping in mappings:
            row = {
                "dataset_index": int(record_index if dataset_index is None else dataset_index),
                "record_index": int(record_index),
                "chunk_index": int(chunk_index),
                "chunk_count": int(chunk_count),
                "case_id": str(record.get("case_id", "")),
                "study_uid": str(record.get("study_uid", "")),
                "raw_gt_index": gt_index,
                "raw_gt_id": raw_gt_id,
                "plane": plane,
                "side": side,
                "abnormality": abnormality,
                "source_series_uid": str(label.get("source_series_uid", "")),
                "source_instance_uid": str(label.get("source_instance_uid", "")),
                "source_slice_index": source_slice,
                "processed_slice_index": mapping["processed_slice_index"],
                "sampled_source_slice_index": mapping["sampled_source_slice_index"],
                "slice_delta": mapping["slice_delta"],
                "mapping_kind": mapping["mapping_kind"],
                "exact_slice_match": bool(mapping["exact_slice_match"]),
                "replication_factor": (
                    len(mappings) if mapping["mapping_kind"] in {"exact", "nearest"} else 0
                ),
                "mapped_gt_count": 0,
                "mapped_source_slice_count": 0,
                "mapping_collision": False,
                "source_slice_collision": False,
                "exact_duplicate_count": len(duplicate_indices),
                "exact_duplicate": len(duplicate_indices) > 1,
                "bbox_source_xyxy": _json_list(bbox_source["values"]),
                "bbox_processed_xyxy": _json_list(processed_bbox),
                "bbox_numeric": bool(bbox_source["numeric"]),
                "bbox_finite": bool(bbox_source["finite"]),
                "bbox_strict_xyxy": bool(bbox_source["strict_xyxy"]),
                "bbox_source_bounds_ok": bbox_source["bounds_ok"],
                "bbox_processed_bounds_ok": bbox_processed["bounds_ok"],
                "source_slice_in_range": source_slice_in_range,
                "identifiers_valid": identifiers_valid,
                "record_schema_valid": not record_issues,
                "label_schema_valid": not unique_label_issues,
                # Schema validity and semantic mapping validity are kept
                # separate above, but this convenience flag is intentionally
                # strict: a GT attached to a neighboring source image should
                # not be treated as training-ready merely because its bbox is
                # numerically valid.
                "valid_for_training": (
                    not record_issues
                    and not unique_label_issues
                    and mapping["mapping_kind"] == "exact"
                ),
                "record_issues": "|".join(sorted(record_issues)),
                "label_issues": "|".join(unique_label_issues),
                "dataset_contract_match": None,
                "_processed_bbox_values": processed_bbox,
            }
            rows.append(row)

    grouped_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["processed_slice_index"] is None:
            continue
        key = (row["dataset_index"], row["plane"], row["processed_slice_index"], row["side"])
        grouped_rows[key].append(row)
    for members in grouped_rows.values():
        raw_gt_ids = {row["raw_gt_id"] for row in members}
        source_slices = {row["source_slice_index"] for row in members}
        for row in members:
            row["mapped_gt_count"] = len(raw_gt_ids)
            row["mapped_source_slice_count"] = len(source_slices)
            row["mapping_collision"] = len(raw_gt_ids) > 1
            row["source_slice_collision"] = len(source_slices) > 1
    return rows


def _rounded_box_key(box: Any, precision: int = 5) -> tuple[Any, ...]:
    values = _coerce_bbox(box)
    if values is None:
        return (repr(box),)
    return tuple(round(value, precision) for value in values)


def annotate_dataset_contract(rows: list[dict[str, Any]], processed_labels: Sequence[Any]) -> dict[str, Any]:
    """Compare expected audit mappings with Dataset-emitted processed labels."""

    expected: Counter[tuple[Any, ...]] = Counter()
    for row in rows:
        if row["processed_slice_index"] is None or row["_processed_bbox_values"] is None:
            continue
        key = (
            row["plane"],
            row["source_slice_index"],
            row["processed_slice_index"],
            row["side"],
            _rounded_box_key(row["_processed_bbox_values"]),
        )
        expected[key] += 1

    actual: Counter[tuple[Any, ...]] = Counter()
    for label in processed_labels:
        if not isinstance(label, Mapping):
            continue
        key = (
            str(label.get("plane", "")),
            _safe_int(label.get("source_slice_index")),
            _safe_int(label.get("slice_index")),
            str(label.get("meniscus_side", "")),
            _rounded_box_key(label.get("bbox")),
        )
        actual[key] += 1

    mismatches = []
    all_keys = set(expected) | set(actual)
    for key in sorted(all_keys, key=repr):
        if expected[key] != actual[key]:
            mismatches.append({"key": repr(key), "expected": expected[key], "actual": actual[key]})
    matching_keys = {key for key in all_keys if expected[key] == actual[key]}
    for row in rows:
        if row["processed_slice_index"] is None or row["_processed_bbox_values"] is None:
            row["dataset_contract_match"] = row["mapping_kind"] == "dropped"
            continue
        key = (
            row["plane"],
            row["source_slice_index"],
            row["processed_slice_index"],
            row["side"],
            _rounded_box_key(row["_processed_bbox_values"]),
        )
        row["dataset_contract_match"] = key in matching_keys
    return {
        "expected_processed_labels": sum(expected.values()),
        "actual_processed_labels": sum(actual.values()),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
    }


def collect_split_identifiers(records: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    """Collect case/study/path/UID identifiers used for split-overlap checks."""

    fields: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if not isinstance(record, Mapping):
            continue
        case_id = str(record.get("case_id", "")).strip()
        study_uid = str(record.get("study_uid", "")).strip()
        if case_id:
            fields["case_id"].add(case_id)
        if study_uid:
            fields["study_uid"].add(study_uid)
        for key in ("sagittal_dicom_dir", "coronal_dicom_dir", "annotation_json"):
            value = canonicalize_path(record.get(key))
            if value:
                fields[key].add(value)
                if key.endswith("_dicom_dir"):
                    fields["dicom_dir_any"].add(value)
        labels = record.get("module1_labels", [])
        if not isinstance(labels, list):
            continue
        for label in labels:
            if not isinstance(label, Mapping):
                continue
            for key in ("source_series_uid", "source_instance_uid"):
                value = str(label.get(key, "")).strip()
                if value:
                    fields[key].add(value)
    return dict(fields)


def audit_split_leakage(split_records: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Find pairwise overlap by case, study, paths, and source DICOM UIDs."""

    identifiers = {split: collect_split_identifiers(records) for split, records in split_records.items()}
    split_names = sorted(identifiers)
    comparisons: dict[str, Any] = {}
    total_overlap = 0
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            pair_name = f"{left}__{right}"
            pair: dict[str, Any] = {}
            fields = sorted(set(identifiers[left]) | set(identifiers[right]))
            for field in fields:
                overlap = sorted(identifiers[left].get(field, set()) & identifiers[right].get(field, set()))
                pair[field] = {"count": len(overlap), "examples": overlap[:20]}
                total_overlap += len(overlap)
            comparisons[pair_name] = pair
    return {
        "splits_checked": split_names,
        "record_counts": {split: len(split_records[split]) for split in split_names},
        "comparisons": comparisons,
        "total_identifier_overlaps": total_overlap,
        "has_overlap": total_overlap > 0,
        "patient_level_note": "PatientID is not stored in the JSONL schema; this audit cannot exclude patient-level leakage across different case_id values.",
    }


def _load_available_leakage_indexes(index_paths: Mapping[str, str | Path | None]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    loaded: dict[str, list[dict[str, Any]]] = {}
    status: dict[str, Any] = {}
    for split, raw_path in sorted(index_paths.items()):
        if raw_path is None or not str(raw_path).strip():
            continue
        path = Path(raw_path)
        if not path.exists():
            status[split] = {"path": str(path), "status": "missing"}
            continue
        try:
            records = read_jsonl(path)
        except Exception as exc:  # keep the primary dataset audit usable
            status[split] = {"path": str(path), "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            continue
        loaded[split] = records
        status[split] = {"path": str(path), "status": "loaded", "records": len(records)}
    return loaded, status


def summarize_gt_rows(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    selected_indices: Sequence[int],
    case_errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-occurrence rows without double-counting replicated GTs."""

    by_gt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_gt[str(row["raw_gt_id"])].append(row)

    nonexact_gt = 0
    exact_gt = 0
    dropped_gt = 0
    unavailable_gt = 0
    replicated_gt = 0
    replication_factors: list[int] = []
    duplicate_gt_ids: set[str] = set()
    invalid_gt_ids: set[str] = set()
    signed_deltas: list[int] = []
    for gt_rows in by_gt.values():
        kinds = {str(row["mapping_kind"]) for row in gt_rows}
        if "nearest" in kinds:
            nonexact_gt += 1
        elif kinds == {"exact"}:
            exact_gt += 1
        elif kinds == {"dropped"}:
            dropped_gt += 1
        else:
            unavailable_gt += 1
        factor = max(int(row["replication_factor"]) for row in gt_rows)
        replication_factors.append(factor)
        if factor > 1:
            replicated_gt += 1
        if any(bool(row["exact_duplicate"]) for row in gt_rows):
            duplicate_gt_ids.add(str(gt_rows[0]["raw_gt_id"]))
        if any(not bool(row["valid_for_training"]) for row in gt_rows):
            invalid_gt_ids.add(str(gt_rows[0]["raw_gt_id"]))
        for row in gt_rows:
            if row["slice_delta"] is not None:
                signed_deltas.append(int(row["slice_delta"]))

    collision_groups = {
        (row["dataset_index"], row["plane"], row["processed_slice_index"], row["side"])
        for row in rows
        if bool(row["mapping_collision"])
    }
    source_collision_groups = {
        (row["dataset_index"], row["plane"], row["processed_slice_index"], row["side"])
        for row in rows
        if bool(row["source_slice_collision"])
    }
    empty_cases = []
    invalid_record_cases = []
    exact_duplicate_group_count = 0
    record_issue_counts: Counter[str] = Counter()
    for index in selected_indices:
        record = records[index]
        labels = record.get("module1_labels", []) if isinstance(record, Mapping) else []
        case_name = str(record.get("case_id", index)) if isinstance(record, Mapping) else str(index)
        if isinstance(labels, list) and not labels:
            empty_cases.append(case_name)
        record_issues = inspect_record_schema(record)
        record_issue_counts.update(record_issues)
        if record_issues:
            invalid_record_cases.append(case_name)
        if isinstance(labels, list):
            duplicate_map = exact_duplicate_groups(labels)
            exact_duplicate_group_count += len({tuple(indices) for indices in duplicate_map.values()})

    label_issue_counts: Counter[str] = Counter()
    for gt_rows in by_gt.values():
        label_issue_counts.update(item for item in str(gt_rows[0]["label_issues"]).split("|") if item)

    absolute_deltas = [abs(value) for value in signed_deltas]
    deltas_sorted = sorted(absolute_deltas)
    p95_delta = 0
    if deltas_sorted:
        p95_delta = deltas_sorted[min(len(deltas_sorted) - 1, math.ceil(0.95 * len(deltas_sorted)) - 1)]
    expected_raw_gt_count = sum(
        len(records[index].get("module1_labels", []))
        for index in selected_indices
        if isinstance(records[index].get("module1_labels", []), list)
    )
    error_record_indices = {int(item["record_index"]) for item in case_errors}
    return {
        "cases_selected": len(selected_indices),
        "cases_loaded": len(selected_indices) - len(error_record_indices),
        "cases_with_load_errors": len(error_record_indices),
        "case_load_errors": len(case_errors),
        "case_error_examples": list(case_errors)[:20],
        "empty_label_cases": len(empty_cases),
        "empty_label_case_examples": empty_cases[:20],
        "invalid_record_schema_cases": len(invalid_record_cases),
        "invalid_record_case_examples": invalid_record_cases[:20],
        "raw_gt_count": expected_raw_gt_count,
        "audited_raw_gt_count": len(by_gt),
        "unplanned_raw_gt_count": max(expected_raw_gt_count - len(by_gt), 0),
        "mapping_occurrence_count": len(rows),
        "exact_mapped_gt": exact_gt,
        "nonexact_nearest_mapped_gt": nonexact_gt,
        "dropped_unsampled_gt": dropped_gt,
        "unavailable_mapping_gt": unavailable_gt,
        "exact_mapping_rate": exact_gt / max(exact_gt + nonexact_gt, 1),
        "source_slice_sampling_recall": exact_gt / max(expected_raw_gt_count, 1),
        "replicated_gt": replicated_gt,
        "replication_factor_mean": sum(replication_factors) / max(len(replication_factors), 1),
        "replication_factor_max": max(replication_factors, default=0),
        "mapping_collision_groups": len(collision_groups),
        "source_slice_collision_groups": len(source_collision_groups),
        "raw_gt_in_exact_duplicate_group": len(duplicate_gt_ids),
        "exact_duplicate_groups": exact_duplicate_group_count,
        "invalid_gt": len(invalid_gt_ids),
        "record_issue_counts": dict(sorted(record_issue_counts.items())),
        "label_issue_counts": dict(sorted(label_issue_counts.items())),
        "replication_factor_histogram": dict(sorted(Counter(str(value) for value in replication_factors).items())),
        "signed_slice_delta_histogram": dict(sorted(Counter(str(value) for value in signed_deltas).items(), key=lambda item: int(item[0]))),
        "absolute_slice_delta_mean": sum(absolute_deltas) / max(len(absolute_deltas), 1),
        "absolute_slice_delta_p95": p95_delta,
        "absolute_slice_delta_max": max(absolute_deltas, default=0),
        "dataset_contract_mismatch_occurrences": sum(row.get("dataset_contract_match") is False for row in rows),
    }


def choose_overlay_groups(rows: Sequence[Mapping[str, Any]], count: int, seed: int) -> list[tuple[int, str, int]]:
    """Prefer non-exact groups, then fill deterministically with exact groups."""

    if count <= 0:
        return []
    groups: dict[tuple[int, str, int], bool] = {}
    for row in rows:
        processed = row.get("processed_slice_index")
        plane = str(row.get("plane", ""))
        if processed is None or plane not in VALID_PLANES:
            continue
        key = (int(row["dataset_index"]), plane, int(processed))
        groups[key] = groups.get(key, False) or not bool(row.get("exact_slice_match"))
    nonexact = [key for key, value in groups.items() if value]
    exact = [key for key, value in groups.items() if not value]
    rng = random.Random(int(seed) + 1701)
    rng.shuffle(nonexact)
    rng.shuffle(exact)
    return (nonexact + exact)[: int(count)]


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "case"


def render_overlays(
    dataset: MeniscusDetectionDataset,
    rows: Sequence[Mapping[str, Any]],
    groups: Sequence[tuple[int, str, int]],
    output_dir: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Render processed GT overlays; red explicitly means non-exact mapping."""

    if not groups:
        return [], []
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows_by_group: dict[tuple[int, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("processed_slice_index") is None:
            continue
        key = (int(row["dataset_index"]), str(row["plane"]), int(row["processed_slice_index"]))
        rows_by_group[key].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    errors: list[dict[str, Any]] = []
    for dataset_index, plane, processed_index in groups:
        key = (dataset_index, plane, processed_index)
        try:
            sample = dataset[dataset_index]
            image = sample[f"{plane}_volume"][0, processed_index].detach().cpu().numpy()
            group_rows = rows_by_group[key]
            nonexact = any(not bool(row["exact_slice_match"]) for row in group_rows)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(image, cmap="gray")
            legend_labels: set[str] = set()
            for row in group_rows:
                box = row.get("_processed_bbox_values")
                if not isinstance(box, Sequence) or len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
                    continue
                color = "red" if not bool(row["exact_slice_match"]) else "lime"
                x1, y1, x2, y2 = [float(value) for value in box]
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2.4))
                label_text = (
                    f"{row['side']} src={row['source_slice_index']} -> "
                    f"sampled={row['sampled_source_slice_index']} delta={row['slice_delta']}"
                )
                ax.text(
                    x1,
                    max(0.0, y1 - 3.0),
                    label_text,
                    color=color,
                    fontsize=7,
                    weight="bold",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5},
                )
                legend_labels.add("NON-EXACT nearest mapping" if color == "red" else "exact mapping")
            case_id = str(sample.get("case_id", dataset_index))
            prefix = "NON-EXACT MAPPING" if nonexact else "EXACT MAPPING"
            ax.set_title(
                f"{prefix}\n{case_id} | {plane} processed={processed_index}\n" + ", ".join(sorted(legend_labels)),
                color="red" if nonexact else "darkgreen",
                weight="bold",
            )
            ax.axis("off")
            path = output_dir / (
                f"{dataset_index:05d}_{_safe_filename(case_id)}_"
                f"chunk{int(sample.get('chunk_index', 0)):02d}_{plane}_processed{processed_index:02d}.png"
            )
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            saved.append(str(path))
        except Exception as exc:
            errors.append(
                {
                    "dataset_index": dataset_index,
                    "record_index": int(dataset.sample_plans[dataset_index]["record_index"]),
                    "plane": plane,
                    "processed_slice_index": processed_index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return saved, errors


def write_per_gt_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_GT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in PER_GT_COLUMNS})


def _parse_leak_index_specs(specs: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--leak-index must be SPLIT=PATH, got {spec!r}")
        split, path = spec.split("=", 1)
        if not split.strip() or not path.strip():
            raise ValueError(f"--leak-index must be SPLIT=PATH, got {spec!r}")
        parsed[split.strip()] = path.strip()
    return parsed


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    data_cfg = dict(config.get("data", {}))
    resolved_usage, slice_selection = resolve_slice_selection(config, args.split, args.usage)
    index_jsonl = args.index_jsonl or data_cfg.get(f"{args.split}_index")
    if not index_jsonl:
        raise ValueError(f"No index configured for split={args.split}; pass --index-jsonl")
    index_path = Path(index_jsonl)
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    seed = int(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(index_path)
    selected_indices = select_case_indices(len(records), args.max_cases, seed)

    resolved_config = {
        "audit": {
            "config_path": str(Path(args.config)),
            "split": args.split,
            "usage": resolved_usage,
            "index_jsonl": str(index_path),
            "slice_selection": slice_selection,
            "seed": seed,
            "max_cases": int(args.max_cases),
            "selected_record_indices": selected_indices,
            "num_overlays": int(args.num_overlays),
            "output_dir": str(output_dir),
        },
        "resolved_data": {
            "output_size": data_cfg.get("output_size"),
            "num_slices": data_cfg.get("num_slices"),
            "patch_size": data_cfg.get("patch_size", config.get("model", {}).get("patch_size", 16)),
            "train_slice_selection": data_cfg.get("train_slice_selection"),
            "val_slice_selection": data_cfg.get("val_slice_selection"),
            "fallback_slice_selection": data_cfg.get("slice_selection"),
        },
        "base_config": config,
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    dataset = MeniscusDetectionDataset(
        index_jsonl=index_path,
        output_size=data_cfg["output_size"],
        target_slices=data_cfg["num_slices"],
        patch_size=data_cfg.get("patch_size", config.get("model", {}).get("patch_size", 16)),
        slice_selection=slice_selection,
        transform=None,
    )
    unmatched_label_policy = dataset.unmatched_label_policy

    all_rows: list[dict[str, Any]] = []
    case_errors: list[dict[str, Any]] = []
    case_contracts: list[dict[str, Any]] = []
    for record_index in selected_indices:
        record = records[record_index]
        dataset_indices = dataset.dataset_indices_for_record(record_index)
        for dataset_index in dataset_indices:
            metadata_by_plane: dict[str, Mapping[str, Any]] = {}
            processed_labels: Sequence[Any] = []
            mapping_summary: Mapping[str, Any] = {}
            try:
                sample = dataset[dataset_index]
                metadata_by_plane = {
                    "sagittal": sample["sagittal_metadata"],
                    "coronal": sample["coronal_metadata"],
                }
                labels_dict = sample.get("labels", {})
                processed_labels = labels_dict.get("module1_labels", [])
                mapping_summary = labels_dict.get("module1_label_mapping", {})
            except Exception as exc:
                plan = dataset.sample_plans[dataset_index]
                case_errors.append(
                    {
                        "dataset_index": dataset_index,
                        "record_index": record_index,
                        "chunk_index": int(plan["chunk_index"]),
                        "case_id": str(record.get("case_id", "")) if isinstance(record, Mapping) else "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            plan = dataset.sample_plans[dataset_index]
            active_source_label_indices = set(
                mapping_summary.get(
                    "active_source_label_indices",
                    plan["active_source_label_indices"],
                )
            )
            rows = build_gt_audit_rows(
                record,
                metadata_by_plane,
                record_index=record_index,
                unmatched_label_policy=unmatched_label_policy,
                dataset_index=dataset_index,
                chunk_index=int(plan["chunk_index"]),
                chunk_count=int(plan["chunk_count"]),
                allowed_source_label_indices=active_source_label_indices,
            )
            contract = annotate_dataset_contract(rows, processed_labels) if metadata_by_plane else {
                "expected_processed_labels": 0,
                "actual_processed_labels": 0,
                "mismatch_count": 0,
                "mismatches": [],
            }
            case_contracts.append(
                {
                    "dataset_index": dataset_index,
                    "record_index": record_index,
                    "chunk_index": int(plan["chunk_index"]),
                    "case_id": str(record.get("case_id", "")) if isinstance(record, Mapping) else "",
                    **contract,
                }
            )
            all_rows.extend(rows)

    write_per_gt_csv(output_dir / "per_gt.csv", all_rows)

    leak_paths: dict[str, str | Path | None] = {
        "train": args.train_index or data_cfg.get("train_index"),
        "val": args.val_index or data_cfg.get("val_index"),
        "test": args.test_index or data_cfg.get("test_index"),
    }
    if not leak_paths["test"] and leak_paths["val"]:
        sibling_test = Path(leak_paths["val"]).with_name("test.jsonl")
        if sibling_test.exists():
            leak_paths["test"] = sibling_test
    leak_paths.update(_parse_leak_index_specs(args.leak_index))
    leak_paths[args.split] = str(index_path)
    resolved_config["audit"]["leakage_indexes"] = {
        split: None if path is None else str(path) for split, path in sorted(leak_paths.items())
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    leakage_records, leakage_index_status = _load_available_leakage_indexes(leak_paths)
    leakage = audit_split_leakage(leakage_records)
    leakage["index_status"] = leakage_index_status

    overlay_groups = choose_overlay_groups(all_rows, int(args.num_overlays), seed)
    try:
        overlay_paths, overlay_errors = render_overlays(dataset, all_rows, overlay_groups, output_dir / "overlays")
    except Exception as exc:
        overlay_paths = []
        overlay_errors = [{"error": f"{type(exc).__name__}: {exc}"}]

    summary = {
        "audit_type": "module1_data_labels",
        "config": str(args.config),
        "split": args.split,
        "usage": resolved_usage,
        "index_jsonl": str(index_path),
        "slice_selection": slice_selection,
        "seed": seed,
        "max_cases": int(args.max_cases),
        "dataset_records": len(records),
        "dataset_virtual_samples": len(dataset),
        "selected_virtual_samples": sum(
            len(dataset.dataset_indices_for_record(record_index))
            for record_index in selected_indices
        ),
        "selected_record_indices": selected_indices,
        "metrics": summarize_gt_rows(all_rows, records, selected_indices, case_errors),
        "dataset_contract": {
            "cases_with_mismatch": sum(item["mismatch_count"] > 0 for item in case_contracts),
            "total_mismatch_keys": sum(item["mismatch_count"] for item in case_contracts),
            "case_details": [item for item in case_contracts if item["mismatch_count"] > 0][:50],
        },
        "split_leakage": leakage,
        "overlays": {
            "requested": int(args.num_overlays),
            "saved": len(overlay_paths),
            "paths": overlay_paths,
            "errors": overlay_errors,
        },
        "limitations": [
            "PatientID/laterality are not stored in the project JSONL schema, so patient-level leakage "
            "and left/right-knee consistency require direct DICOM-tag access.",
            "Dropped GTs are excluded from processed-label detection metrics; use source_slice_sampling_recall "
            "to report how much of the raw GT set was actually evaluated.",
            "Legacy nearest mapping is reported as non-exact even when the neighboring slice looks visually "
            "similar; only SOP/source-index equality counts as exact.",
            "Spatial augmentation is disabled for this audit so overlays represent deterministic preprocessing and configured slice selection.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    metrics = summary["metrics"]
    print(f"saved audit: {output_dir}")
    print(
        "cases="
        f"{metrics['cases_loaded']}/{metrics['cases_selected']} raw_gt={metrics['raw_gt_count']} "
        f"audited_gt={metrics['audited_raw_gt_count']} unplanned={metrics['unplanned_raw_gt_count']} "
        f"exact={metrics['exact_mapped_gt']} nonexact={metrics['nonexact_nearest_mapped_gt']} "
        f"dropped={metrics['dropped_unsampled_gt']} sampling_recall={metrics['source_slice_sampling_recall']:.3f} "
        f"replicated={metrics['replicated_gt']} collisions={metrics['mapping_collision_groups']} "
        f"invalid_gt={metrics['invalid_gt']} empty_cases={metrics['empty_label_cases']}"
    )
    print(
        f"split_overlap={summary['split_leakage']['total_identifier_overlaps']} "
        f"overlays={summary['overlays']['saved']} per_gt={output_dir / 'per_gt.csv'}"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/module1_meniscovit_cop.yaml", help="Base Module 1 YAML config.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Split whose samples are audited.")
    parser.add_argument(
        "--usage",
        choices=["auto", "train", "val"],
        default="auto",
        help="Slice-selection usage. auto resolves train->train and val/test->val.",
    )
    parser.add_argument("--index-jsonl", default=None, help="Override the selected split JSONL.")
    parser.add_argument("--train-index", default=None, help="Optional train JSONL override for leakage checks.")
    parser.add_argument("--val-index", default=None, help="Optional validation JSONL override for leakage checks.")
    parser.add_argument("--test-index", default=None, help="Optional test JSONL override for leakage checks.")
    parser.add_argument(
        "--leak-index",
        action="append",
        default=[],
        metavar="SPLIT=PATH",
        help="Additional/override JSONL for leakage checks; may be repeated.",
    )
    parser.add_argument("--max-cases", type=int, default=32, help="Deterministic random case limit; <=0 audits all records.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed sampling/overlay seed.")
    parser.add_argument("--num-overlays", type=int, default=12, help="Number of processed GT overlay PNGs; <=0 disables rendering.")
    parser.add_argument("--output-dir", default="outputs/module1_data_label_audit", help="Independent diagnostic output directory.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
