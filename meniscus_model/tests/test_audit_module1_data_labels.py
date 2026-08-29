"""Pure-function tests for the Module 1 data/label audit."""

from __future__ import annotations

import math

from scripts.module1_debug.audit_module1_data_labels import (
    analyze_source_slice_mapping,
    audit_split_leakage,
    build_gt_audit_rows,
    exact_duplicate_groups,
    resolve_slice_selection,
    select_case_indices,
    summarize_gt_rows,
    validate_bbox_xyxy,
)
from scripts.module1_debug.check_module1_detection_contract import _slice_mapping_report


def _record(labels: list[dict] | None = None, case_id: str = "case_a", study_uid: str = "study_a") -> dict:
    return {
        "case_id": case_id,
        "study_uid": study_uid,
        "sagittal_dicom_dir": f"/data/{case_id}/sagittal",
        "coronal_dicom_dir": f"/data/{case_id}/coronal",
        "module1_labels": labels or [],
    }


def _label(source_slice: int, bbox: list[float] | None = None) -> dict:
    return {
        "plane": "sagittal",
        "slice_index": source_slice,
        "bbox": bbox or [10.0, 20.0, 50.0, 60.0],
        "meniscus_side": "medial",
        "abnormality": "normal",
        "source_series_uid": "series_sag",
        "source_instance_uid": f"instance_{source_slice}",
    }


def test_resolve_slice_selection_uses_train_and_val_policies_without_mutation() -> None:
    config = {
        "data": {
            "slice_selection": {"mode": "uniform"},
            "train_slice_selection": {"mode": "label"},
            "val_slice_selection": {"mode": "trimmed_uniform", "exclude_head": 2},
        }
    }

    train_usage, train_selection = resolve_slice_selection(config, split="train", usage="auto")
    val_usage, val_selection = resolve_slice_selection(config, split="test", usage="auto")
    train_selection["mode"] = "changed"

    assert train_usage == "train"
    assert val_usage == "val"
    assert val_selection == {"mode": "trimmed_uniform", "exclude_head": 2}
    assert config["data"]["train_slice_selection"]["mode"] == "label"


def test_source_slice_mapping_reports_exact_drop_and_opt_in_nearest() -> None:
    exact = analyze_source_slice_mapping(4, [4, 4, 9, 20])
    dropped = analyze_source_slice_mapping(5, [0, 2, 4, 8])
    nearest = analyze_source_slice_mapping(5, [0, 2, 4, 8], unmatched_label_policy="nearest")

    assert [row["processed_slice_index"] for row in exact] == [0, 1]
    assert all(row["exact_slice_match"] for row in exact)
    assert dropped[0]["mapping_kind"] == "dropped"
    assert dropped[0]["processed_slice_index"] is None
    assert nearest == [
        {
            "processed_slice_index": 2,
            "sampled_source_slice_index": 4,
            "slice_delta": -1,
            "exact_slice_match": False,
            "mapping_kind": "nearest",
        }
    ]


def test_bbox_validation_requires_finite_strict_xyxy_and_bounds() -> None:
    valid = validate_bbox_xyxy([0, 1, 20, 30], width=64, height=64)
    zero_width = validate_bbox_xyxy([10, 1, 10, 30], width=64, height=64)
    nonfinite = validate_bbox_xyxy([0, 1, math.nan, 30], width=64, height=64)
    outside = validate_bbox_xyxy([-1, 1, 20, 30], width=64, height=64)

    assert valid["strict_xyxy"] and valid["bounds_ok"]
    assert not zero_width["strict_xyxy"]
    assert not nonfinite["finite"]
    assert outside["strict_xyxy"] and outside["bounds_ok"] is False


def test_fractional_slice_index_is_rejected_instead_of_silently_truncated() -> None:
    label = _label(1)
    label["slice_index"] = 1.5
    metadata = {
        "sagittal": {
            "original_shape": (4, 100, 200),
            "processed_shape": (2, 50, 100),
            "sampled_source_indices": [1, 3],
        }
    }

    rows = build_gt_audit_rows(_record([label]), metadata, record_index=0)

    assert len(rows) == 1
    assert rows[0]["mapping_kind"] == "unavailable"
    assert "slice_index_not_integer" in rows[0]["label_issues"]
    assert not rows[0]["valid_for_training"]


def test_exact_duplicate_groups_identifies_only_identical_raw_labels() -> None:
    labels = [_label(1), _label(1), _label(2)]

    duplicates = exact_duplicate_groups(labels)

    assert duplicates == {0: [0, 1], 1: [0, 1]}


def test_build_gt_rows_exposes_replication_nearest_collision_and_duplicates() -> None:
    labels = [_label(1), _label(1), _label(2)]
    record = _record(labels)
    metadata = {
        "sagittal": {
            "original_shape": (4, 100, 200),
            "processed_shape": (3, 50, 100),
            "sampled_source_indices": [1, 1, 3],
        }
    }

    rows = build_gt_audit_rows(record, metadata, record_index=7, unmatched_label_policy="nearest")

    # Each duplicated source=1 GT is copied to both exact processed positions;
    # source=2 uses the first nearest position, producing five occurrences.
    assert len(rows) == 5
    source_one = [row for row in rows if row["source_slice_index"] == 1]
    source_two = [row for row in rows if row["source_slice_index"] == 2]
    assert all(row["replication_factor"] == 2 for row in source_one)
    assert all(row["exact_duplicate_count"] == 2 for row in source_one)
    assert source_two[0]["mapping_kind"] == "nearest"
    assert source_two[0]["slice_delta"] == -1
    assert source_two[0]["label_schema_valid"]
    assert not source_two[0]["valid_for_training"]

    processed_zero = [row for row in rows if row["processed_slice_index"] == 0]
    assert len(processed_zero) == 3
    assert all(row["mapped_gt_count"] == 3 for row in processed_zero)
    assert all(row["mapped_source_slice_count"] == 2 for row in processed_zero)
    assert all(row["mapping_collision"] for row in processed_zero)
    assert all(row["source_slice_collision"] for row in processed_zero)
    assert all(row["bbox_processed_xyxy"] == "[5.0, 10.0, 25.0, 30.0]" for row in rows)


def test_build_gt_rows_drops_unsampled_source_by_default() -> None:
    metadata = {
        "sagittal": {
            "original_shape": (8, 100, 200),
            "processed_shape": (4, 50, 100),
            "sampled_source_indices": [0, 2, 4, 6],
        }
    }

    rows = build_gt_audit_rows(_record([_label(5)]), metadata, record_index=0)

    assert len(rows) == 1
    assert rows[0]["mapping_kind"] == "dropped"
    assert rows[0]["processed_slice_index"] is None
    assert rows[0]["replication_factor"] == 0
    assert rows[0]["dataset_contract_match"] is None
    assert not rows[0]["valid_for_training"]


def test_detection_contract_accepts_intentionally_dropped_unsampled_gt() -> None:
    raw_record = _record([_label(5)])
    sample = {
        "sagittal_metadata": {"sampled_source_indices": [0, 2, 4, 6]},
        "labels": {
            "module1_label_mapping": {
                "unmatched_label_policy": "drop",
                "raw_label_count": 1,
                "matched_source_label_count": 0,
                "exact_source_label_count": 0,
                "remapped_source_label_count": 0,
                "dropped_source_label_count": 1,
                "processed_label_count": 0,
            }
        },
    }

    report = _slice_mapping_report(raw_record, [], sample)

    assert report["mismatch_count"] == 0
    assert report["label_mapping_summary"]["dropped_source_label_count"] == 1


def test_chunk_rows_aggregate_against_original_raw_gt_denominator() -> None:
    record = _record([_label(1), _label(2)])
    metadata = {
        "sagittal": {
            "original_shape": (4, 100, 200),
            "processed_shape": (2, 50, 100),
            "sampled_source_indices": [1, 2],
        }
    }
    first = build_gt_audit_rows(
        record,
        metadata,
        record_index=0,
        dataset_index=0,
        chunk_index=0,
        chunk_count=2,
        allowed_source_label_indices={0},
    )
    second = build_gt_audit_rows(
        record,
        metadata,
        record_index=0,
        dataset_index=1,
        chunk_index=1,
        chunk_count=2,
        allowed_source_label_indices={1},
    )

    summary = summarize_gt_rows(first + second, [record], [0], [])

    assert summary["raw_gt_count"] == 2
    assert summary["audited_raw_gt_count"] == 2
    assert summary["unplanned_raw_gt_count"] == 0
    assert summary["exact_mapped_gt"] == 2
    assert summary["source_slice_sampling_recall"] == 1.0


def test_split_leakage_detects_case_study_path_and_source_uid_overlap() -> None:
    train = _record([_label(1)], case_id="shared_case", study_uid="shared_study")
    val = _record([_label(1)], case_id="shared_case", study_uid="shared_study")
    # Exercise slash normalization as well as exact UID checks.
    val["sagittal_dicom_dir"] = "\\data\\shared_case\\sagittal"
    train["sagittal_dicom_dir"] = "/data/shared_case/sagittal"
    val["coronal_dicom_dir"] = "/different/coronal"

    report = audit_split_leakage({"train": [train], "val": [val]})
    comparison = report["comparisons"]["train__val"]

    assert report["has_overlap"]
    assert comparison["case_id"]["count"] == 1
    assert comparison["study_uid"]["count"] == 1
    assert comparison["sagittal_dicom_dir"]["count"] == 1
    assert comparison["source_series_uid"]["count"] == 1
    assert comparison["source_instance_uid"]["count"] == 1


def test_case_selection_is_seeded_and_all_mode_preserves_order() -> None:
    first = select_case_indices(total=20, max_cases=5, seed=42)
    second = select_case_indices(total=20, max_cases=5, seed=42)

    assert first == second
    assert first == sorted(first)
    assert select_case_indices(total=4, max_cases=0, seed=999) == [0, 1, 2, 3]
