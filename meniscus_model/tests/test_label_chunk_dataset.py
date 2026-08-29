"""Integration tests for exact-preserving Module 1 label chunks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from datasets.dicom_reader import DicomSeries, select_slice_indices
from datasets.meniscus_detection_dataset import MeniscusDetectionDataset


def _label(plane: str, source_slice: int, side: str) -> dict:
    return {
        "plane": plane,
        "slice_index": source_slice,
        "bbox": [4.0, 5.0, 20.0, 24.0],
        "meniscus_side": side,
        "abnormality": "normal",
    }


def _fake_read_dicom_series(
    series_dir: str | Path,
    output_size: int,
    target_slices: int,
    slice_selection: dict,
) -> DicomSeries:
    del series_dir
    depth = 32
    indices = select_slice_indices(
        depth=depth,
        target_slices=target_slices,
        **slice_selection,
    )
    size = int(output_size)
    slices = [
        {
            "ImageOrientationPatient": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "ImagePositionPatient": [0.0, 0.0, float(source_index)],
            "PixelSpacing": [1.0, 1.0],
        }
        for source_index in indices
    ]
    metadata = {
        "original_shape": (depth, size, size),
        "processed_shape": (len(indices), size, size),
        "sampled_source_indices": indices.astype(int).tolist(),
        "slices": slices,
    }
    return DicomSeries(
        volume=np.zeros((len(indices), size, size), dtype=np.float32),
        affine=np.eye(4, dtype=np.float32),
        instance_paths=[Path(f"slice_{index}.dcm") for index in indices],
        metadata=metadata,
    )


def test_label_chunks_cover_dense_case_once_without_repeating_required_slices(
    tmp_path: Path,
    monkeypatch,
) -> None:
    labels = [_label("sagittal", source_slice, "medial") for source_slice in range(18)]
    labels.extend(_label("coronal", source_slice, "lateral") for source_slice in (3, 7))
    record = {
        "case_id": "dense_case",
        "sagittal_dicom_dir": "/fake/sagittal",
        "coronal_dicom_dir": "/fake/coronal",
        "module1_labels": labels,
    }
    index_path = tmp_path / "index.jsonl"
    index_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "datasets.meniscus_detection_dataset.read_dicom_series",
        _fake_read_dicom_series,
    )

    dataset = MeniscusDetectionDataset(
        index_jsonl=index_path,
        output_size=32,
        target_slices=16,
        patch_size=16,
        slice_selection={"mode": "label_chunks", "unmatched_label_policy": "drop"},
    )

    assert len(dataset.records) == 1
    assert len(dataset) == 2
    samples = [dataset[index] for index in range(len(dataset))]
    assigned_indices = [
        label["source_label_index"]
        for sample in samples
        for label in sample["labels"]["module1_labels"]
    ]
    assert sorted(assigned_indices) == list(range(len(labels)))
    assert len(assigned_indices) == len(set(assigned_indices))
    assert sum(
        sample["labels"]["module1_label_mapping"]["dropped_source_label_count"]
        for sample in samples
    ) == 0
    assert sum(
        sample["labels"]["module1_label_mapping"]["raw_label_count"]
        for sample in samples
    ) == len(labels)

    first_sagittal = samples[0]["sagittal_metadata"]["sampled_source_indices"]
    second_sagittal = samples[1]["sagittal_metadata"]["sampled_source_indices"]
    assert set(range(16)).issubset(first_sagittal)
    assert {16, 17}.issubset(second_sagittal)
    assert len(first_sagittal) == len(set(first_sagittal)) == 16
    assert len(second_sagittal) == len(set(second_sagittal)) == 16
    assert samples[1]["labels"]["module1_label_mapping"]["active_source_label_indices"] == [16, 17]


def test_coordinate_output_can_be_disabled_for_roi_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    record = {
        "case_id": "roi_case",
        "sagittal_dicom_dir": "/fake/sagittal",
        "coronal_dicom_dir": "/fake/coronal",
        "module1_labels": [_label("sagittal", 2, "medial")],
    }
    index_path = tmp_path / "index.jsonl"
    index_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "datasets.meniscus_detection_dataset.read_dicom_series",
        _fake_read_dicom_series,
    )
    dataset = MeniscusDetectionDataset(
        index_jsonl=index_path,
        output_size=32,
        target_slices=4,
        patch_size=4,
        return_coordinates=False,
    )

    sample = dataset[0]

    assert "sagittal_coords" not in sample
    assert "coronal_coords" not in sample
    assert sample["sagittal_volume"].shape == (1, 4, 32, 32)
