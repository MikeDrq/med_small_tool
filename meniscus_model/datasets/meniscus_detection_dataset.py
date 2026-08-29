"""JSONL dataset for Module 1 MeniscoViT-CoP data preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from datasets.coordinate_utils import volume_patch_grid_to_world_coordinates
from datasets.dicom_reader import DicomSeries, read_dicom_series


SIDE_TO_ID = {"medial": 0, "lateral": 1}
ABNORMALITY_TO_ID = {"normal": 0, "abnormal": 1}
UNMATCHED_LABEL_POLICIES = {"drop", "nearest", "error"}
LABEL_CHUNK_MODES = {"label_chunks", "label_chunk", "chunked_labels"}
PLANES = ("sagittal", "coronal")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a case-level JSONL index."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def _nearest_processed_slice(source_slice_index: int, sampled_source_indices: list[int]) -> int:
    distances = [abs(int(src) - int(source_slice_index)) for src in sampled_source_indices]
    return int(min(range(len(distances)), key=lambda idx: distances[idx]))


def _processed_slices_for_source(
    source_slice_index: int,
    sampled_source_indices: list[int],
    unmatched_label_policy: str = "drop",
) -> list[int]:
    """Map one source label only to processed slices containing that image.

    ``drop`` is the safe default: a box is never attached to a different MRI
    slice. ``nearest`` is retained only for reproducing legacy experiments.
    ``error`` turns incomplete slice coverage into an explicit data-contract
    failure.
    """

    policy = str(unmatched_label_policy).lower()
    if policy not in UNMATCHED_LABEL_POLICIES:
        raise ValueError(
            f"unmatched_label_policy must be one of {sorted(UNMATCHED_LABEL_POLICIES)}, "
            f"got {unmatched_label_policy}"
        )
    exact_matches = [
        idx
        for idx, source_idx in enumerate(sampled_source_indices)
        if int(source_idx) == int(source_slice_index)
    ]
    if exact_matches:
        return exact_matches
    if policy == "drop":
        return []
    if policy == "error":
        raise ValueError(
            f"source label slice {int(source_slice_index)} is not present in "
            f"sampled_source_indices={list(map(int, sampled_source_indices))}"
        )
    if not sampled_source_indices:
        raise ValueError("cannot use unmatched_label_policy='nearest' with no sampled source slices")
    return [_nearest_processed_slice(source_slice_index, sampled_source_indices)]


def _scale_bbox_to_processed(
    bbox: list[float],
    source_shape: tuple[int, int, int],
    processed_shape: tuple[int, int, int],
) -> list[float]:
    _, source_h, source_w = source_shape
    _, processed_h, processed_w = processed_shape
    x_scale = processed_w / float(source_w)
    y_scale = processed_h / float(source_h)
    x1, y1, x2, y2 = bbox
    return [float(x1 * x_scale), float(y1 * y_scale), float(x2 * x_scale), float(y2 * y_scale)]


def _prepare_module1_labels(
    record: dict[str, Any],
    series_by_plane: dict[str, DicomSeries],
    unmatched_label_policy: str = "drop",
    allowed_source_label_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    labels = []
    for source_label_index, raw_label in enumerate(record.get("module1_labels", [])):
        if allowed_source_label_indices is not None and source_label_index not in allowed_source_label_indices:
            continue
        plane = raw_label["plane"]
        if plane not in series_by_plane:
            raise KeyError(f"Label plane '{plane}' is not available in case {record.get('case_id')}")
        series = series_by_plane[plane]
        source_slice_index = int(raw_label["slice_index"])
        bbox = _scale_bbox_to_processed(
            raw_label["bbox"],
            series.metadata["original_shape"],
            series.metadata["processed_shape"],
        )
        processed_slices = _processed_slices_for_source(
            source_slice_index,
            series.metadata["sampled_source_indices"],
            unmatched_label_policy=unmatched_label_policy,
        )
        for processed_slice_index in processed_slices:
            sampled_source_slice_index = int(series.metadata["sampled_source_indices"][processed_slice_index])
            exact_slice_match = sampled_source_slice_index == source_slice_index
            labels.append(
                {
                    "plane": plane,
                    "slice_index": processed_slice_index,
                    "source_slice_index": source_slice_index,
                    "source_label_index": int(source_label_index),
                    "sampled_source_slice_index": sampled_source_slice_index,
                    "exact_slice_match": exact_slice_match,
                    "slice_mapping_kind": "exact" if exact_slice_match else "nearest",
                    "bbox": bbox,
                    "meniscus_side": raw_label["meniscus_side"],
                    "meniscus_side_id": SIDE_TO_ID[raw_label["meniscus_side"]],
                    "abnormality": raw_label["abnormality"],
                    "abnormality_id": ABNORMALITY_TO_ID[raw_label["abnormality"]],
                }
            )
    return labels


def build_label_chunk_sample_plans(
    records: list[dict[str, Any]],
    target_slices: int,
) -> list[dict[str, Any]]:
    """Partition every raw GT into exactly one fixed-capacity virtual sample."""

    if int(target_slices) <= 0:
        raise ValueError("target_slices must be positive for label_chunks mode")
    plans: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        raw_labels = record.get("module1_labels", [])
        if not isinstance(raw_labels, list):
            raise TypeError(f"module1_labels must be a list in case {record.get('case_id')}")
        labels_by_plane_slice: dict[str, dict[int, list[int]]] = {
            plane: {} for plane in PLANES
        }
        for label_index, label in enumerate(raw_labels):
            plane = str(label.get("plane", ""))
            if plane not in labels_by_plane_slice:
                raise ValueError(
                    f"Unsupported label plane {plane!r} in case {record.get('case_id')}"
                )
            source_slice = int(label["slice_index"])
            labels_by_plane_slice[plane].setdefault(source_slice, []).append(label_index)

        chunks_by_plane: dict[str, list[tuple[int, ...]]] = {}
        for plane in PLANES:
            source_slices = sorted(labels_by_plane_slice[plane])
            chunks_by_plane[plane] = [
                tuple(source_slices[start : start + int(target_slices)])
                for start in range(0, len(source_slices), int(target_slices))
            ]

        chunk_count = max([1] + [len(chunks_by_plane[plane]) for plane in PLANES])
        for chunk_index in range(chunk_count):
            plane_source_indices: dict[str, tuple[int, ...] | None] = {}
            active_label_indices: list[int] = []
            for plane in PLANES:
                plane_chunks = chunks_by_plane[plane]
                source_chunk = plane_chunks[chunk_index] if chunk_index < len(plane_chunks) else None
                plane_source_indices[plane] = source_chunk
                if source_chunk is not None:
                    for source_slice in source_chunk:
                        active_label_indices.extend(labels_by_plane_slice[plane][source_slice])
            plans.append(
                {
                    "record_index": int(record_index),
                    "chunk_index": int(chunk_index),
                    "chunk_count": int(chunk_count),
                    "plane_source_indices": plane_source_indices,
                    "active_source_label_indices": tuple(sorted(active_label_indices)),
                }
            )
    return plans


class MeniscusDetectionDataset(Dataset):
    """Loads paired sagittal/coronal MRI and Module 1 labels.

    Args:
        index_jsonl: Case-level JSONL file.
        output_size: In-plane output size, usually ``224`` or ``256``.
        target_slices: Fixed number of slices, usually ``16``.
        patch_size: Module 1 in-plane patch size used to emit token coordinates.
        slice_selection: Optional slice-selection policy passed to the DICOM
            reader. ``mode: label`` uses Module 1 label source slices per plane.
            ``mode: label_chunks`` expands dense records into virtual samples,
            retains at most ``target_slices`` required label slices per plane,
            and fills remaining positions with background source slices.
            It may include ``unmatched_label_policy`` (``drop``, ``nearest``,
            or ``error``); this key is consumed by the Dataset, not the reader.
        transform: Optional sample transform, typically ``RandomTrainingAugmentation``.
        unmatched_label_policy: Optional explicit override for the policy in
            ``slice_selection``. The default is the safe ``drop`` behavior.
        return_coordinates: Emit the DICOM world-coordinate patch grids used
            by MeniscoViT-CoP. ROI-cache generation from GT boxes can disable
            this expensive output because it only needs image volumes.

    Returns sample dict:
        ``sagittal_volume``: FloatTensor ``[1,T,H,W]``.
        ``coronal_volume``: FloatTensor ``[1,T,H,W]``.
        ``sagittal_coords``: FloatTensor ``[T*num_patches,3]``.
        ``coronal_coords``: FloatTensor ``[T*num_patches,3]``.
        ``labels``: dict containing transformed labels and downstream labels.
    """

    def __init__(
        self,
        index_jsonl: str | Path,
        output_size: int | tuple[int, int] = 224,
        target_slices: int = 16,
        patch_size: int | tuple[int, int] = 16,
        slice_selection: dict[str, Any] | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        unmatched_label_policy: str | None = None,
        return_coordinates: bool = True,
    ):
        self.index_jsonl = Path(index_jsonl)
        self.output_size = output_size
        self.target_slices = target_slices
        self.patch_size = patch_size
        self.slice_selection = dict(slice_selection or {})
        configured_policy = self.slice_selection.pop("unmatched_label_policy", None)
        selected_policy = configured_policy if unmatched_label_policy is None else unmatched_label_policy
        self.unmatched_label_policy = str(selected_policy or "drop").lower()
        if self.unmatched_label_policy not in UNMATCHED_LABEL_POLICIES:
            raise ValueError(
                f"unmatched_label_policy must be one of {sorted(UNMATCHED_LABEL_POLICIES)}, "
                f"got {self.unmatched_label_policy}"
            )
        self.transform = transform
        self.return_coordinates = bool(return_coordinates)
        self.records = read_jsonl(self.index_jsonl)
        self.slice_selection_mode = str(self.slice_selection.get("mode", "uniform")).lower()
        if self.slice_selection_mode in LABEL_CHUNK_MODES:
            self.sample_plans = build_label_chunk_sample_plans(self.records, int(self.target_slices))
        else:
            self.sample_plans = [
                {
                    "record_index": int(record_index),
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "plane_source_indices": {plane: None for plane in PLANES},
                    "active_source_label_indices": tuple(range(len(record.get("module1_labels", [])))),
                }
                for record_index, record in enumerate(self.records)
            ]

    def __len__(self) -> int:
        return len(self.sample_plans)

    def dataset_indices_for_record(self, record_index: int) -> list[int]:
        """Return virtual dataset indices belonging to one source JSONL row."""

        return [
            dataset_index
            for dataset_index, plan in enumerate(self.sample_plans)
            if int(plan["record_index"]) == int(record_index)
        ]

    @staticmethod
    def _label_source_indices(record: dict[str, Any], plane: str) -> list[int]:
        return sorted(
            {
                int(label["slice_index"])
                for label in record.get("module1_labels", [])
                if label.get("plane") == plane
            }
        )

    def _slice_selection_for_plane(
        self,
        record: dict[str, Any],
        plane: str,
        sample_plan: dict[str, Any],
    ) -> dict[str, Any]:
        selection = dict(self.slice_selection)
        normalized_mode = str(selection.get("mode", "uniform")).lower()
        if normalized_mode in LABEL_CHUNK_MODES:
            required = sample_plan["plane_source_indices"].get(plane)
            if required is None:
                return {"mode": "uniform"}
            return {"mode": "include", "candidate_indices": list(required)}
        if normalized_mode in {"label", "labels", "labeled", "label_only"}:
            selection["candidate_indices"] = self._label_source_indices(record, plane)
        return selection

    def _load_series(
        self,
        record: dict[str, Any],
        sample_plan: dict[str, Any],
    ) -> dict[str, DicomSeries]:
        return {
            "sagittal": read_dicom_series(
                record["sagittal_dicom_dir"],
                output_size=self.output_size,
                target_slices=self.target_slices,
                slice_selection=self._slice_selection_for_plane(record, "sagittal", sample_plan),
            ),
            "coronal": read_dicom_series(
                record["coronal_dicom_dir"],
                output_size=self.output_size,
                target_slices=self.target_slices,
                slice_selection=self._slice_selection_for_plane(record, "coronal", sample_plan),
            ),
        }

    def _coords_for_series(self, series: DicomSeries) -> torch.Tensor:
        _, height, width = series.volume.shape
        coords = volume_patch_grid_to_world_coordinates(
            image_size=(height, width),
            patch_size=self.patch_size,
            num_slices=series.volume.shape[0],
            dicom_metadata=series.metadata,
        )
        return torch.from_numpy(coords).float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_plan = self.sample_plans[index]
        record_index = int(sample_plan["record_index"])
        record = self.records[record_index]
        series = self._load_series(record, sample_plan)
        active_source_label_indices = set(sample_plan["active_source_label_indices"])
        module1_labels = _prepare_module1_labels(
            record,
            series,
            unmatched_label_policy=self.unmatched_label_policy,
            allowed_source_label_indices=active_source_label_indices,
        )
        matched_source_label_indices = {int(label["source_label_index"]) for label in module1_labels}
        exact_source_label_indices = {
            int(label["source_label_index"])
            for label in module1_labels
            if bool(label["exact_slice_match"])
        }
        raw_module1_label_count = len(active_source_label_indices)
        case_raw_module1_label_count = len(record.get("module1_labels", []))
        labels = {
            "case_id": record["case_id"],
            "module1_labels": module1_labels,
            "module1_label_mapping": {
                "unmatched_label_policy": self.unmatched_label_policy,
                "slice_selection_mode": self.slice_selection_mode,
                "record_index": record_index,
                "chunk_index": int(sample_plan["chunk_index"]),
                "chunk_count": int(sample_plan["chunk_count"]),
                "active_source_label_indices": sorted(active_source_label_indices),
                "case_raw_label_count": case_raw_module1_label_count,
                "raw_label_count": raw_module1_label_count,
                "matched_source_label_count": len(matched_source_label_indices),
                "exact_source_label_count": len(exact_source_label_indices),
                "remapped_source_label_count": len(matched_source_label_indices - exact_source_label_indices),
                "dropped_source_label_count": max(raw_module1_label_count - len(matched_source_label_indices), 0),
                "processed_label_count": len(module1_labels),
            },
            "module2_labels": record.get("module2_labels", {}),
            "module3_labels": record.get("module3_labels", {}),
            "module4_labels": record.get("module4_labels", {}),
        }
        sample: dict[str, Any] = {
            "case_id": record["case_id"],
            "group_id": record.get("patient_group_id") or record["case_id"],
            "sample_id": f"{record['case_id']}::chunk{int(sample_plan['chunk_index'])}",
            "record_index": record_index,
            "chunk_index": int(sample_plan["chunk_index"]),
            "chunk_count": int(sample_plan["chunk_count"]),
            "sagittal_volume": torch.from_numpy(series["sagittal"].volume).unsqueeze(0).float(),
            "coronal_volume": torch.from_numpy(series["coronal"].volume).unsqueeze(0).float(),
            "sagittal_metadata": series["sagittal"].metadata,
            "coronal_metadata": series["coronal"].metadata,
            "labels": labels,
        }
        if self.return_coordinates:
            sample["sagittal_coords"] = self._coords_for_series(series["sagittal"])
            sample["coronal_coords"] = self._coords_for_series(series["coronal"])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
