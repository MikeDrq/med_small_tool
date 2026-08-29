"""DICOM series loading for sagittal/coronal knee MRI.

This module prepares the data contract needed by MeniscoViT-CoP:

* read one DICOM folder;
* sort slices by ``ImagePositionPatient`` when available, otherwise by
  ``InstanceNumber``;
* stack ``pixel_array`` into a volume shaped ``[T, H, W]``;
* preserve per-slice geometry tags needed for real-world coordinates;
* percentile clip and normalize intensities to ``[0, 1]``;
* optionally resize in-plane resolution and sample/pad to a fixed slice count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DicomSeries:
    """Loaded and preprocessed MRI series.

    Attributes:
        volume: Float32 array with shape ``[T, H, W]`` normalized to ``[0, 1]``.
        affine: Float32 array with shape ``[4, 4]`` mapping ``[slice,row,col]``
            to DICOM patient-world millimeters for the first slice spacing.
        instance_paths: DICOM files in processed slice order. Padded slices repeat
            the source file path.
        metadata: Geometry and preprocessing metadata. Important keys:
            ``slices`` per processed slice, ``original_shape``,
            ``processed_shape``, ``sampled_source_indices``, ``resize_scale``.
    """

    volume: np.ndarray
    affine: np.ndarray
    instance_paths: list[Path]
    metadata: dict[str, Any]


def _require_pydicom() -> Any:
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("pydicom is required to read DICOM folders. Install requirements.txt first.") from exc
    # The supplied anonymized data uses stable UUID strings in UI fields.
    # Ignore that grammar-only validation during reads; file/pixel decoding
    # errors continue to be raised and handled normally.
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
    return pydicom


def _dicom_files(series_dir: Path) -> list[Path]:
    files = [path for path in series_dir.rglob("*") if path.is_file() and not path.name.startswith(".")]
    if not files:
        raise FileNotFoundError(f"No files found in DICOM folder: {series_dir}")
    return files


def _as_float_list(value: Any, default: list[float] | None = None) -> list[float]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [float(part) for part in value.replace("\\", ",").split(",") if part]
    try:
        return [float(item) for item in value]
    except TypeError:
        return [float(value)]


def _slice_sort_key(item: tuple[Path, Any]) -> tuple[int, float]:
    _, ds = item
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    if ipp is not None and iop is not None:
        orientation = np.asarray(_as_float_list(iop), dtype=np.float64)
        position = np.asarray(_as_float_list(ipp), dtype=np.float64)
        if orientation.size == 6 and position.size == 3:
            normal = np.cross(orientation[:3], orientation[3:])
            return (0, float(np.dot(position, normal)))
    if ipp is not None:
        position = np.asarray(_as_float_list(ipp), dtype=np.float64)
        if position.size == 3:
            return (0, float(position[2]))
    instance = getattr(ds, "InstanceNumber", 0)
    return (1, float(instance))


def _metadata_from_dataset(ds: Any) -> dict[str, Any]:
    return {
        "PixelSpacing": _as_float_list(getattr(ds, "PixelSpacing", None), [1.0, 1.0])[:2],
        "SliceThickness": float(getattr(ds, "SliceThickness", 1.0)),
        "ImageOrientationPatient": _as_float_list(getattr(ds, "ImageOrientationPatient", None), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])[:6],
        "ImagePositionPatient": _as_float_list(getattr(ds, "ImagePositionPatient", None), [0.0, 0.0, 0.0])[:3],
        "InstanceNumber": int(getattr(ds, "InstanceNumber", 0)),
        "Rows": int(getattr(ds, "Rows", 0)),
        "Columns": int(getattr(ds, "Columns", 0)),
        "SOPInstanceUID": str(getattr(ds, "SOPInstanceUID", "")),
        "WindowCenter": _as_float_list(getattr(ds, "WindowCenter", None), []),
        "WindowWidth": _as_float_list(getattr(ds, "WindowWidth", None), []),
        "VOILUTFunction": str(getattr(ds, "VOILUTFunction", "LINEAR")),
        "PhotometricInterpretation": str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")),
    }


def _pixel_array(ds: Any) -> np.ndarray:
    array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return array * slope + intercept


def _first_float(value: Any) -> float | None:
    values = _as_float_list(value, [])
    if not values:
        return None
    result = float(values[0])
    return result if np.isfinite(result) else None


def apply_dicom_window(array: np.ndarray, ds: Any) -> np.ndarray | None:
    """Apply the DICOM WindowCenter/WindowWidth transform to ``[0, 1]``.

    This implements the standard LINEAR/LINEAR_EXACT and SIGMOID VOI window
    functions after modality rescale.  ``None`` is returned when a usable
    window is absent so callers can explicitly fall back to percentile
    normalization.  MONOCHROME1 images are inverted after windowing.
    """

    center = _first_float(getattr(ds, "WindowCenter", None))
    width = _first_float(getattr(ds, "WindowWidth", None))
    if center is None or width is None or width <= 0:
        return None

    values = np.asarray(array, dtype=np.float32)
    voi_function = str(getattr(ds, "VOILUTFunction", "LINEAR")).upper()
    if voi_function == "SIGMOID":
        clipped_argument = np.clip(-4.0 * (values - center) / width, -80.0, 80.0)
        normalized = 1.0 / (1.0 + np.exp(clipped_argument))
    elif voi_function == "LINEAR_EXACT":
        lower = center - width / 2.0
        upper = center + width / 2.0
        normalized = np.clip((values - lower) / max(upper - lower, 1e-6), 0.0, 1.0)
    else:
        # DICOM PS3.3 C.11.2.1.2 default LINEAR function.
        if width <= 1.0:
            normalized = (values > center - 0.5).astype(np.float32)
        else:
            lower = center - 0.5 - (width - 1.0) / 2.0
            upper = center - 0.5 + (width - 1.0) / 2.0
            normalized = np.empty_like(values, dtype=np.float32)
            normalized[values <= lower] = 0.0
            normalized[values > upper] = 1.0
            middle = (values > lower) & (values <= upper)
            normalized[middle] = ((values[middle] - (center - 0.5)) / (width - 1.0)) + 0.5

    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        normalized = 1.0 - normalized
    return np.asarray(normalized, dtype=np.float32)


def apply_paper_window(array: np.ndarray, ds: Any) -> np.ndarray | None:
    """Apply the exact clipping equations published in the supplement.

    ``WindowMin = Center - Width/2`` and ``WindowMax = Center + Width/2``.
    The clipped values are linearly mapped to ``[0,1]`` for the tensor input;
    this affine rescaling preserves the paper's clipped image information.
    """

    center = _first_float(getattr(ds, "WindowCenter", None))
    width = _first_float(getattr(ds, "WindowWidth", None))
    if center is None or width is None or width <= 0:
        return None
    lower = center - width / 2.0
    upper = center + width / 2.0
    normalized = (np.clip(np.asarray(array, dtype=np.float32), lower, upper) - lower) / width
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        normalized = 1.0 - normalized
    return np.asarray(normalized, dtype=np.float32)


def normalize_intensity(volume: np.ndarray, clip_percentiles: tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    """Percentile clip and normalize a volume to ``[0, 1]``."""

    lo, hi = np.percentile(volume, clip_percentiles)
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    volume = np.clip(volume, lo, hi)
    return ((volume - lo) / (hi - lo)).astype(np.float32)


def resize_volume_inplane(volume: np.ndarray, output_size: int | tuple[int, int]) -> np.ndarray:
    """Resize ``[T, H, W]`` volume to a fixed in-plane size."""

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("torch is required for in-plane DICOM resizing.") from exc

    if isinstance(output_size, int):
        output_hw = (output_size, output_size)
    else:
        output_hw = tuple(output_size)
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(volume.shape[0], output_hw[0], output_hw[1]), mode="trilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)


def _valid_slice_indices(indices: list[int] | tuple[int, ...] | np.ndarray | None, depth: int) -> np.ndarray:
    if indices is None:
        return np.asarray([], dtype=np.int64)
    values = sorted({int(idx) for idx in indices if 0 <= int(idx) < depth})
    return np.asarray(values, dtype=np.int64)


def _fit_indices_to_target(indices: np.ndarray, target_slices: int | None) -> np.ndarray:
    if indices.size == 0:
        raise ValueError("indices must not be empty")
    if target_slices is None or target_slices == indices.size:
        return indices.astype(np.int64)
    if target_slices <= 0:
        raise ValueError("target_slices must be positive")
    if indices.size > target_slices:
        positions = np.linspace(0, indices.size - 1, target_slices).round().astype(np.int64)
        return indices[positions].astype(np.int64)

    pad = target_slices - indices.size
    left = pad // 2
    right = pad - left
    return np.pad(indices.astype(np.int64), (left, right), mode="edge")


def _resample_indices_evenly(indices: np.ndarray, target_slices: int | None) -> np.ndarray:
    if indices.size == 0:
        raise ValueError("indices must not be empty")
    if target_slices is None or target_slices == indices.size:
        return indices.astype(np.int64)
    if target_slices <= 0:
        raise ValueError("target_slices must be positive")
    positions = np.linspace(0, indices.size - 1, target_slices).round().astype(np.int64)
    return indices[positions].astype(np.int64)


def _include_required_indices(
    depth: int,
    required_indices: list[int] | tuple[int, ...] | np.ndarray | None,
    target_slices: int | None,
) -> np.ndarray:
    """Select slices while retaining every valid required source index once."""

    required = _valid_slice_indices(required_indices, depth)
    all_indices = np.arange(depth, dtype=np.int64)
    if target_slices is None:
        return all_indices
    if target_slices <= 0:
        raise ValueError("target_slices must be positive")
    if required.size > target_slices:
        raise ValueError(
            f"required slice count {required.size} exceeds target_slices={target_slices}; "
            "split required indices into label chunks first"
        )
    if target_slices >= depth:
        return _fit_indices_to_target(all_indices, target_slices)
    remaining = np.setdiff1d(all_indices, required, assume_unique=True)
    fill_count = int(target_slices - required.size)
    if fill_count == 0:
        return required
    fill = _fit_indices_to_target(remaining, fill_count)
    return np.sort(np.concatenate([required, fill])).astype(np.int64)


def _trimmed_uniform_candidates(depth: int, exclude_head: int, exclude_tail: int, exclude_center: int) -> np.ndarray:
    mask = np.ones(depth, dtype=bool)
    head = max(0, min(int(exclude_head), depth))
    tail = max(0, min(int(exclude_tail), depth - head))
    if head:
        mask[:head] = False
    if tail:
        mask[depth - tail :] = False
    center_count = max(0, min(int(exclude_center), depth))
    if center_count:
        center_start = max(0, depth // 2 - center_count // 2)
        center_end = min(depth, center_start + center_count)
        mask[center_start:center_end] = False
    candidates = np.nonzero(mask)[0].astype(np.int64)
    if candidates.size == 0:
        return np.arange(depth, dtype=np.int64)
    return candidates


def select_slice_indices(
    depth: int,
    target_slices: int | None,
    mode: str = "uniform",
    candidate_indices: list[int] | tuple[int, ...] | np.ndarray | None = None,
    exclude_head: int = 0,
    exclude_tail: int = 0,
    exclude_center: int = 0,
) -> np.ndarray:
    """Select fixed-length source slice indices for one DICOM series.

    Modes:
        ``uniform`` samples across the full depth, matching the original code.
        ``label`` samples only from supplied labeled source-slice candidates.
        ``include`` retains every supplied required slice and fills remaining
        slots uniformly from the rest of the series.
        ``trimmed_uniform`` drops configurable head/tail/center windows first.
    """

    if depth <= 0:
        raise ValueError("depth must be positive")
    normalized_mode = str(mode).lower()
    if normalized_mode in {"label", "labels", "labeled", "label_only"}:
        candidates = _valid_slice_indices(candidate_indices, depth)
        if candidates.size == 0:
            candidates = np.arange(depth, dtype=np.int64)
        return _resample_indices_evenly(candidates, target_slices)
    elif normalized_mode in {"include", "required", "include_required"}:
        return _include_required_indices(depth, candidate_indices, target_slices)
    elif normalized_mode in {"trimmed_uniform", "trimmed", "validation"}:
        candidates = _trimmed_uniform_candidates(depth, exclude_head, exclude_tail, exclude_center)
    elif normalized_mode == "uniform":
        candidates = _valid_slice_indices(candidate_indices, depth)
        if candidates.size == 0:
            candidates = np.arange(depth, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported slice selection mode: {mode}")
    return _fit_indices_to_target(candidates, target_slices)


def sample_slices(
    volume: np.ndarray,
    target_slices: int | None,
    mode: str = "uniform",
    candidate_indices: list[int] | tuple[int, ...] | np.ndarray | None = None,
    exclude_head: int = 0,
    exclude_tail: int = 0,
    exclude_center: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample or edge-pad a volume along T.

    Returns:
        ``(sampled_volume, source_indices)`` where ``source_indices`` maps each
        processed slice back to the original sorted slice index.
    """

    depth = volume.shape[0]
    indices = select_slice_indices(
        depth=depth,
        target_slices=target_slices,
        mode=mode,
        candidate_indices=candidate_indices,
        exclude_head=exclude_head,
        exclude_tail=exclude_tail,
        exclude_center=exclude_center,
    )
    return volume[indices], indices


def build_affine(metadata: dict[str, Any]) -> np.ndarray:
    """Build an affine mapping ``[slice,row,col,1]`` to patient-world mm."""

    first = metadata["slices"][0]
    iop = np.asarray(first["ImageOrientationPatient"], dtype=np.float64)
    ipp = np.asarray(first["ImagePositionPatient"], dtype=np.float64)
    spacing = np.asarray(first["PixelSpacing"], dtype=np.float64)
    row_direction = iop[:3]
    col_direction = iop[3:]
    normal = np.cross(row_direction, col_direction)

    if len(metadata["slices"]) > 1:
        second_ipp = np.asarray(metadata["slices"][1]["ImagePositionPatient"], dtype=np.float64)
        slice_vector = second_ipp - ipp
        if np.linalg.norm(slice_vector) < 1e-6:
            slice_vector = normal * float(first["SliceThickness"])
    else:
        slice_vector = normal * float(first["SliceThickness"])

    affine = np.eye(4, dtype=np.float32)
    affine[:3, 0] = slice_vector.astype(np.float32)
    affine[:3, 1] = (col_direction * spacing[0]).astype(np.float32)
    affine[:3, 2] = (row_direction * spacing[1]).astype(np.float32)
    affine[:3, 3] = ipp.astype(np.float32)
    return affine


def read_dicom_series(
    series_dir: str | Path,
    output_size: int | tuple[int, int] | None = None,
    target_slices: int | None = None,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
    slice_selection: dict[str, Any] | None = None,
    intensity_mode: str = "paper_window",
) -> DicomSeries:
    """Read, normalize, resize, and sample one DICOM series directory.

    ``paper_window`` follows the supplement's exact WindowCenter/WindowWidth
    clipping equations. ``dicom_window`` retains the DICOM PS3.3 VOI function
    for comparison. If usable window tags are absent, both modes fall back to
    volume-percentile normalization.
    """

    pydicom = _require_pydicom()
    series_dir = Path(series_dir)
    datasets = []
    for path in _dicom_files(series_dir):
        try:
            ds = pydicom.dcmread(path, force=True)
            if hasattr(ds, "PixelData"):
                datasets.append((path, ds))
        except Exception:
            continue
    if not datasets:
        raise FileNotFoundError(f"No readable DICOM image instances found in: {series_dir}")

    datasets.sort(key=_slice_sort_key)
    arrays = [_pixel_array(ds) for _, ds in datasets]
    normalized_mode = str(intensity_mode).lower()
    if normalized_mode not in {"paper_window", "paper", "dicom_window", "window", "percentile"}:
        raise ValueError(f"Unsupported intensity_mode: {intensity_mode}")
    used_window_mode: str | None = None
    if normalized_mode in {"paper_window", "paper"}:
        reference_dataset = next(
            (
                ds
                for _, ds in datasets
                if _first_float(getattr(ds, "WindowCenter", None)) is not None
                and (_first_float(getattr(ds, "WindowWidth", None)) or 0.0) > 0.0
            ),
            None,
        )
        windowed_arrays = (
            [apply_paper_window(array, reference_dataset) for array in arrays]
            if reference_dataset is not None
            else []
        )
        if windowed_arrays and all(array is not None for array in windowed_arrays):
            volume = np.stack(windowed_arrays, axis=0).astype(np.float32)
            used_window_mode = "paper_window"
        else:
            volume = normalize_intensity(np.stack(arrays, axis=0).astype(np.float32), clip_percentiles)
    elif normalized_mode in {"dicom_window", "window"}:
        windowed_arrays = [apply_dicom_window(array, ds) for array, (_, ds) in zip(arrays, datasets)]
        if all(array is not None for array in windowed_arrays):
            volume = np.stack(windowed_arrays, axis=0).astype(np.float32)
            used_window_mode = "dicom_window"
        else:
            volume = np.stack(arrays, axis=0).astype(np.float32)
            volume = normalize_intensity(volume, clip_percentiles)
    else:
        volume = np.stack(arrays, axis=0).astype(np.float32)
        volume = normalize_intensity(volume, clip_percentiles)
    original_shape = tuple(int(v) for v in volume.shape)
    slice_metadata = [_metadata_from_dataset(ds) for _, ds in datasets]
    original_paths = [path for path, _ in datasets]

    resize_scale = {"row": 1.0, "col": 1.0}
    if output_size is not None:
        output_hw = (output_size, output_size) if isinstance(output_size, int) else tuple(output_size)
        old_h, old_w = volume.shape[1:]
        volume = resize_volume_inplane(volume, output_hw)
        resize_scale = {"row": old_h / float(output_hw[0]), "col": old_w / float(output_hw[1])}
        for meta in slice_metadata:
            meta["PixelSpacing"] = [meta["PixelSpacing"][0] * resize_scale["row"], meta["PixelSpacing"][1] * resize_scale["col"]]
            meta["Rows"] = int(output_hw[0])
            meta["Columns"] = int(output_hw[1])

    selection = dict(slice_selection or {})
    selection_target = selection.pop("target_slices", target_slices)
    selection_mode = selection.pop("mode", "uniform")
    candidate_indices = selection.pop("candidate_indices", None)
    exclude_head = int(selection.pop("exclude_head", 0))
    exclude_tail = int(selection.pop("exclude_tail", 0))
    exclude_center = int(selection.pop("exclude_center", 0))
    if selection:
        raise ValueError(f"Unsupported slice_selection keys: {sorted(selection)}")

    volume, sampled_indices = sample_slices(
        volume,
        selection_target,
        mode=selection_mode,
        candidate_indices=candidate_indices,
        exclude_head=exclude_head,
        exclude_tail=exclude_tail,
        exclude_center=exclude_center,
    )
    processed_metadata = [dict(slice_metadata[int(idx)], SourceSliceIndex=int(idx)) for idx in sampled_indices]
    processed_paths = [original_paths[int(idx)] for idx in sampled_indices]
    metadata = {
        "series_dir": str(series_dir),
        "original_shape": original_shape,
        "processed_shape": tuple(int(v) for v in volume.shape),
        "sampled_source_indices": sampled_indices.astype(int).tolist(),
        "slice_selection": {
            "mode": str(selection_mode),
            "target_slices": None if selection_target is None else int(selection_target),
            "candidate_indices": None if candidate_indices is None else [int(idx) for idx in candidate_indices],
            "exclude_head": exclude_head,
            "exclude_tail": exclude_tail,
            "exclude_center": exclude_center,
        },
        "resize_scale": resize_scale,
        "clip_percentiles": list(clip_percentiles),
        "intensity_preprocessing": used_window_mode or "percentile_fallback",
        "slices": processed_metadata,
    }
    return DicomSeries(volume=volume.astype(np.float32), affine=build_affine(metadata), instance_paths=processed_paths, metadata=metadata)


def load_knee_mri_case(
    case_spec: dict[str, str | Path],
    output_size: int | tuple[int, int] | None = None,
    target_slices: int | None = None,
    slice_selection: dict[str, Any] | None = None,
) -> dict[str, DicomSeries]:
    """Load sagittal/coronal series for one knee MRI case."""

    return {
        plane: read_dicom_series(path, output_size=output_size, target_slices=target_slices, slice_selection=slice_selection)
        for plane, path in case_spec.items()
    }
