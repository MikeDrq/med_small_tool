r"""Convert exported meniscus annotation JSON files to the project JSONL format.

The source annotation files are matched to DICOM images by UID:

* annotation ``study_uid`` -> DICOM ``StudyInstanceUID``
* annotation ``boxs[].seriesUid`` -> DICOM ``SeriesInstanceUID``
* annotation ``boxs[].instanceUid`` -> DICOM ``SOPInstanceUID``

Example:

    python -m scripts.convert_labeled_dataset ^
      --dicom-root D:\path\to\labeled_data\data\dicom\site_id ^
      --dicom-root D:\path\to\labeled_data\data\json\site_id ^
      --annotation-root D:\path\to\labeled_data\data\json\site_id ^
      --output-jsonl data\index_all_center_wh.jsonl
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dicom_reader import _as_float_list, _slice_sort_key
from utils.annotation_dedup import select_annotation_files


DEFAULT_MODULE2 = {"medial_status": "normal", "lateral_status": "normal"}
DEFAULT_MODULE3 = {"medial_bucket_handle": 0, "lateral_bucket_handle": 0}
DEFAULT_MODULE4 = {"lateral_discoid": 0, "tibia_width": None, "tibia_mask_path": None}
STATUS_RANK = {"normal": 0, "degeneration": 1, "tear": 2}


@dataclass(frozen=True)
class InstanceInfo:
    path: Path
    sop_uid: str
    rows: int
    columns: int


@dataclass(frozen=True)
class SeriesInfo:
    case_id: str
    series_dir: Path
    study_uid: str
    series_uid: str
    description: str
    plane: str | None
    instances: list[InstanceInfo]

    @property
    def sop_to_index(self) -> dict[str, int]:
        return {instance.sop_uid: idx for idx, instance in enumerate(self.instances)}

    @property
    def sop_to_instance(self) -> dict[str, InstanceInfo]:
        return {instance.sop_uid: instance for instance in self.instances}


def _require_pydicom() -> Any:
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("pydicom is required. Install requirements.txt first.") from exc
    # This anonymized export deliberately stores UUID strings in DICOM UI
    # fields.  They are stable identifiers and are matched verbatim, but do
    # not satisfy the standard numeric-dot UI grammar.  Suppress only value
    # validation while retaining I/O and malformed-file exceptions.
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
    return pydicom


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _contains_files(path: Path) -> bool:
    try:
        return any(child.is_file() for child in path.iterdir())
    except OSError:
        return False


def _iter_candidate_series_dirs(root: Path, case_ids: set[str] | None = None) -> list[tuple[str, Path]]:
    """Return ``(case_id, dir)`` pairs for likely series directories under root."""

    candidates: list[tuple[str, Path]] = []
    if not root.exists():
        raise FileNotFoundError(root)
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if case_ids and case_dir.name not in case_ids:
            continue
        if _contains_files(case_dir):
            candidates.append((case_dir.name, case_dir))
        for series_dir in sorted(path for path in case_dir.rglob("*") if path.is_dir()):
            if _contains_files(series_dir):
                candidates.append((case_dir.name, series_dir))
    return candidates


def _infer_plane(description: str, orientation: list[float]) -> str | None:
    lowered = description.lower()
    if "sag" in lowered:
        return "sagittal"
    if "cor" in lowered:
        return "coronal"
    if "ax" in lowered or "tra" in lowered:
        return "axial"

    if len(orientation) == 6:
        import numpy as np

        row = np.asarray(orientation[:3], dtype=float)
        col = np.asarray(orientation[3:], dtype=float)
        normal = np.abs(np.cross(row, col))
        axis = int(normal.argmax())
        if axis == 0:
            return "sagittal"
        if axis == 1:
            return "coronal"
        if axis == 2:
            return "axial"
    return None


def _scan_series(
    series_dir: Path,
    case_id: str,
    required_series_uids: set[str] | None = None,
) -> SeriesInfo | None:
    """Read one physical series directory.

    A cheap first-header check avoids reading every instance of unannotated
    sequences.  Source series directories are leaf directories in the supplied
    export; only direct files are considered so a case directory can never be
    mistaken for one mixed multi-series volume.
    """

    pydicom = _require_pydicom()
    datasets: list[tuple[Path, Any]] = []
    selected_series_uid: str | None = None
    for path in sorted(child for child in series_dir.iterdir() if child.is_file()):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if not getattr(ds, "SOPInstanceUID", None) or not getattr(ds, "SeriesInstanceUID", None):
            continue
        if not getattr(ds, "StudyInstanceUID", None):
            continue
        if not getattr(ds, "Rows", None) or not getattr(ds, "Columns", None):
            continue
        series_uid = str(ds.SeriesInstanceUID)
        if selected_series_uid is None:
            selected_series_uid = series_uid
            if required_series_uids is not None and series_uid not in required_series_uids:
                return None
        if series_uid != selected_series_uid:
            continue
        datasets.append((path, ds))

    if not datasets:
        return None

    datasets.sort(key=_slice_sort_key)
    first = datasets[0][1]
    description = str(getattr(first, "SeriesDescription", ""))
    orientation = _as_float_list(getattr(first, "ImageOrientationPatient", None), [])
    instances = [
        InstanceInfo(
            path=path,
            sop_uid=str(getattr(ds, "SOPInstanceUID", "")),
            rows=int(getattr(ds, "Rows", 0)),
            columns=int(getattr(ds, "Columns", 0)),
        )
        for path, ds in datasets
    ]
    return SeriesInfo(
        case_id=case_id,
        series_dir=series_dir,
        study_uid=str(getattr(first, "StudyInstanceUID", "")),
        series_uid=str(getattr(first, "SeriesInstanceUID", "")),
        description=description,
        plane=_infer_plane(description, orientation),
        instances=instances,
    )


def scan_dicom_roots(
    dicom_roots: list[str],
    scan_case_ids: set[str] | None = None,
    scan_workers: int = 8,
    required_series_uids: set[str] | None = None,
) -> dict[str, list[SeriesInfo]]:
    """Build a study UID -> series list index from one or more DICOM roots."""

    by_study: dict[str, list[SeriesInfo]] = {}
    seen_dirs: set[Path] = set()
    candidates: list[tuple[str, Path]] = []
    for root_text in dicom_roots:
        root = Path(root_text).expanduser()
        for case_id, series_dir in _iter_candidate_series_dirs(root, case_ids=scan_case_ids):
            resolved = series_dir.resolve()
            if resolved in seen_dirs:
                continue
            seen_dirs.add(resolved)
            candidates.append((case_id, series_dir))

    def scan(candidate: tuple[str, Path]) -> SeriesInfo | None:
        case_id, series_dir = candidate
        return _scan_series(series_dir, case_id, required_series_uids=required_series_uids)

    workers = max(1, int(scan_workers))
    if workers == 1:
        scanned = map(scan, candidates)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dicom-scan")
        scanned = executor.map(scan, candidates)
    try:
        for index, series in enumerate(scanned, start=1):
            if series is not None:
                by_study.setdefault(series.study_uid, []).append(series)
            if index % 500 == 0 or index == len(candidates):
                print(
                    f"DICOM series scan: {index}/{len(candidates)} "
                    f"({len(by_study)} studies)",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return by_study


def _iter_boxes(annotation: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for anno in annotation.get("annos", []):
        info = anno.get("info", {})
        for box in anno.get("boxs", []):
            items.append((info, box))
    return items


def _side_from_info(info: dict[str, Any]) -> str | None:
    """Return anatomical meniscus side from an annotation ``info`` block.

    The source schema uses ``position`` for medial/lateral meniscus (``t9``)
    annotations.  In lesion (``t10``) annotations, ``menicustype`` instead
    means tear/degeneration and must *not* be interpreted as anatomical side.
    Lesion side is recovered geometrically from the same-slice meniscus boxes
    in :func:`convert_annotation`.
    """

    text = " ".join(str(value) for value in info.values())
    position = info.get("position")
    if position == "t1" or "内侧" in text or "medial" in text.lower():
        return "medial"
    if position == "t2" or "外侧" in text or "lateral" in text.lower():
        return "lateral"
    return None


def _is_meniscus_annotation(info: dict[str, Any]) -> bool:
    return info.get("type") == "t9" or info.get("typeName") == "半月板"


def _is_lesion_annotation(info: dict[str, Any]) -> bool:
    return info.get("type") == "t10" or info.get("typeName") == "病灶半月板"


def _lesion_status(info: dict[str, Any]) -> str | None:
    text = " ".join(str(value) for value in info.values()).lower()
    if "桶柄" in text or "bucket" in text:
        return "tear"
    if "损伤" in text or "撕裂" in text or "tear" in text:
        return "tear"
    if "变性" in text or "degeneration" in text:
        return "degeneration"

    # ``menicustype`` is the diagnosis field in this export. ``lesiontype``
    # stores tear subtypes (including the candidate bucket-handle code) and
    # must not be used to choose tear versus degeneration.
    meniscus_type = info.get("menicustype")
    if _coded_flag(meniscus_type, "t1"):
        return "tear"
    if _coded_flag(meniscus_type, "t2"):
        return "degeneration"
    return None


def _coded_flag(value: Any, code: str) -> bool:
    if isinstance(value, dict):
        return bool(value.get(code, False))
    return str(value) == str(code)


def _is_bucket_handle(info: dict[str, Any], bucket_handle_code: str = "t3") -> bool:
    """Identify bucket-handle labels in text or the exported coded schema.

    The supplied export stores generic tear as ``lesiontype.t1`` and the
    bucket-handle subtype as ``lesiontype.t3``.  The code is an explicit
    converter option because the manuscript does not publish this private
    annotation schema.
    """

    text = " ".join(str(value) for value in info.values()).lower()
    return "桶柄" in text or "bucket" in text or _coded_flag(info.get("lesiontype"), bucket_handle_code)


def _is_discoid(info: dict[str, Any]) -> bool:
    text = " ".join(str(value) for value in info.values()).lower()
    # For t9 meniscus annotations in this export, structure=t1 is discoid and
    # structure=t2 is non-discoid.  Preserve support for textual exports too.
    return "盘状" in text or "discoid" in text or (
        _is_meniscus_annotation(info) and str(info.get("structure")) == "t1"
    )


def _choose_best_series(series: list[SeriesInfo], plane: str, referenced_uids: set[str]) -> SeriesInfo | None:
    candidates = [item for item in series if item.plane == plane]
    if not candidates:
        return None

    def score(item: SeriesInfo) -> tuple[int, int, int, int, str]:
        desc = item.description.lower()
        referenced = int(item.series_uid in referenced_uids)
        usable = int(not any(word in desc for word in ("screen", "save", "localizer", "loc", "cal")))
        pd_or_t2 = int("pd" in desc or "t2" in desc)
        return (referenced, usable, pd_or_t2, len(item.instances), str(item.series_dir))

    return max(candidates, key=score)


def _bbox_from_box(box: dict[str, Any], box_format: str) -> list[float]:
    x = float(box["x"])
    y = float(box["y"])
    rx = float(box["rx"])
    ry = float(box["ry"])
    if box_format == "xywh":
        x1, y1, x2, y2 = x, y, x + rx, y + ry
    elif box_format == "center-wh":
        x1, y1, x2, y2 = x - rx / 2.0, y - ry / 2.0, x + rx / 2.0, y + ry / 2.0
    elif box_format == "xyxy":
        x1, y1, x2, y2 = x, y, rx, ry
    else:
        raise ValueError(f"Unsupported box format: {box_format}")
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _clip_bbox(bbox: list[float], rows: int, columns: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [
        max(0.0, min(float(columns), x1)),
        max(0.0, min(float(rows), y1)),
        max(0.0, min(float(columns), x2)),
        max(0.0, min(float(rows), y2)),
    ]


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _intersection_area(first: list[float], second: list[float]) -> float:
    width = max(0.0, min(float(first[2]), float(second[2])) - max(float(first[0]), float(second[0])))
    height = max(0.0, min(float(first[3]), float(second[3])) - max(float(first[1]), float(second[1])))
    return width * height


def _bbox_center_distance(first: list[float], second: list[float]) -> float:
    first_x = 0.5 * (float(first[0]) + float(first[2]))
    first_y = 0.5 * (float(first[1]) + float(first[3]))
    second_x = 0.5 * (float(second[0]) + float(second[2]))
    second_y = 0.5 * (float(second[1]) + float(second[3]))
    return math.hypot(first_x - second_x, first_y - second_y)


def _match_lesion_to_meniscus(
    lesion_bbox: list[float],
    candidates: list[dict[str, Any]],
) -> int | None:
    """Return index of the same-slice meniscus box best matching a lesion.

    Lesion boxes in the supplied data almost always lie inside one t9
    meniscus box.  Intersection-over-lesion-area is therefore the primary
    criterion, with centre distance providing a deterministic fallback/tie
    break for the few boundary annotations that touch both menisci.
    """

    if not candidates:
        return None
    lesion_area = max(_bbox_area(lesion_bbox), 1e-6)
    scored = [
        (
            _intersection_area(lesion_bbox, candidate["bbox"]) / lesion_area,
            -_bbox_center_distance(lesion_bbox, candidate["bbox"]),
            -index,
            index,
        )
        for index, candidate in enumerate(candidates)
    ]
    return max(scored)[-1]


def _patient_group_id(annotation: dict[str, Any]) -> str | None:
    """Create a stable non-reversible group id for patient-level splitting."""

    raw = annotation.get("patientId") or annotation.get("patient_id")
    if raw is None or not str(raw).strip():
        return None
    return hashlib.sha256(str(raw).encode("utf-8")).hexdigest()


def _upgrade_status(current: str, candidate: str | None) -> str:
    if candidate is None:
        return current
    return candidate if STATUS_RANK[candidate] > STATUS_RANK[current] else current


def convert_annotation(
    annotation_path: Path,
    series_for_study: list[SeriesInfo],
    box_format: str,
    clamp_boxes: bool,
    bucket_handle_code: str = "t3",
) -> tuple[dict[str, Any] | None, str | None]:
    annotation = _read_json(annotation_path)
    study_uid = str(annotation.get("study_uid") or annotation.get("StudyInstanceUID") or "")
    if not study_uid:
        return None, "missing study_uid"
    if not series_for_study:
        return None, "no matching DICOM study"

    referenced_uids = {str(box.get("seriesUid")) for _, box in _iter_boxes(annotation) if box.get("seriesUid")}
    sagittal = _choose_best_series(series_for_study, "sagittal", referenced_uids)
    coronal = _choose_best_series(series_for_study, "coronal", referenced_uids)
    if sagittal is None or coronal is None:
        return None, "missing sagittal or coronal series"

    series_to_plane = {sagittal.series_uid: ("sagittal", sagittal), coronal.series_uid: ("coronal", coronal)}
    module2 = dict(DEFAULT_MODULE2)
    module3 = dict(DEFAULT_MODULE3)
    module4 = dict(DEFAULT_MODULE4)

    # Convert t9 boxes first.  They carry the only reliable medial/lateral
    # field in this export and act as same-slice anchors for t10 lesions.
    labels: list[dict[str, Any]] = []
    labels_by_instance: dict[tuple[str, str], list[dict[str, Any]]] = {}
    skipped_boxes = 0
    for info, box in _iter_boxes(annotation):
        if not _is_meniscus_annotation(info):
            continue
        side = _side_from_info(info)
        if side is None:
            skipped_boxes += 1
            continue
        series_uid = str(box.get("seriesUid", ""))
        instance_uid = str(box.get("instanceUid", ""))
        if series_uid not in series_to_plane:
            skipped_boxes += 1
            continue
        plane, series = series_to_plane[series_uid]
        slice_index = series.sop_to_index.get(instance_uid)
        instance = series.sop_to_instance.get(instance_uid)
        if slice_index is None or instance is None:
            skipped_boxes += 1
            continue
        bbox = _bbox_from_box(box, box_format=box_format)
        if clamp_boxes:
            bbox = _clip_bbox(bbox, rows=instance.rows, columns=instance.columns)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            skipped_boxes += 1
            continue
        label = {
            "plane": plane,
            "slice_index": int(slice_index),
            "bbox": [float(value) for value in bbox],
            "meniscus_side": side,
            # Per-slice abnormality is filled from geometrically matched t10
            # lesion boxes below.  Do not mark every slice of an abnormal case.
            "abnormality": "normal",
            "source_series_uid": series_uid,
            "source_instance_uid": instance_uid,
        }
        labels.append(label)
        labels_by_instance.setdefault((series_uid, instance_uid), []).append(label)
        if side == "lateral" and _is_discoid(info):
            module4["lateral_discoid"] = 1

    lesion_boxes_seen = 0
    lesion_boxes_matched = 0
    for info, box in _iter_boxes(annotation):
        if not _is_lesion_annotation(info):
            continue
        series_uid = str(box.get("seriesUid", ""))
        instance_uid = str(box.get("instanceUid", ""))
        if series_uid not in series_to_plane:
            skipped_boxes += 1
            continue
        _, series = series_to_plane[series_uid]
        instance = series.sop_to_instance.get(instance_uid)
        if instance is None:
            skipped_boxes += 1
            continue
        lesion_bbox = _bbox_from_box(box, box_format=box_format)
        if clamp_boxes:
            lesion_bbox = _clip_bbox(lesion_bbox, rows=instance.rows, columns=instance.columns)
        if _bbox_area(lesion_bbox) <= 0:
            skipped_boxes += 1
            continue
        lesion_boxes_seen += 1
        candidates = labels_by_instance.get((series_uid, instance_uid), [])
        matched_index = _match_lesion_to_meniscus(lesion_bbox, candidates)
        if matched_index is None:
            skipped_boxes += 1
            continue
        matched = candidates[matched_index]
        matched["abnormality"] = "abnormal"
        side = str(matched["meniscus_side"])
        module2[f"{side}_status"] = _upgrade_status(module2[f"{side}_status"], _lesion_status(info))
        if _is_bucket_handle(info, bucket_handle_code=bucket_handle_code):
            module3[f"{side}_bucket_handle"] = 1
        lesion_boxes_matched += 1

    if not labels:
        return None, f"no usable meniscus boxes; skipped_boxes={skipped_boxes}"

    case_id = sagittal.case_id if sagittal.case_id == coronal.case_id else study_uid
    record = {
        "case_id": case_id,
        "study_uid": study_uid,
        "patient_group_id": _patient_group_id(annotation),
        "annotation_json": str(annotation_path),
        "sagittal_dicom_dir": str(sagittal.series_dir),
        "coronal_dicom_dir": str(coronal.series_dir),
        "module1_labels": labels,
        "module2_labels": module2,
        "module3_labels": module3,
        "module4_labels": module4,
        "conversion_metadata": {
            "box_format": box_format,
            "bucket_handle_code": bucket_handle_code,
            "lesion_side_assignment": "same_instance_max_intersection_over_lesion_area_then_center_distance",
            "lesion_boxes_seen": lesion_boxes_seen,
            "lesion_boxes_matched": lesion_boxes_matched,
            "skipped_boxes": skipped_boxes,
        },
    }
    return record, None


def convert_dataset(
    dicom_roots: list[str],
    annotation_root: str,
    output_jsonl: str,
    report_json: str | None = None,
    box_format: str = "center-wh",
    clamp_boxes: bool = True,
    case_ids: set[str] | None = None,
    scan_case_ids: set[str] | None = None,
    limit: int | None = None,
    bucket_handle_code: str = "t3",
    scan_workers: int = 8,
) -> dict[str, Any]:
    annotation_paths, deduplication = select_annotation_files(annotation_root)
    referenced_series_uids: set[str] = set()
    for annotation_path in annotation_paths:
        try:
            annotation = _read_json(annotation_path)
        except Exception:
            continue
        referenced_series_uids.update(
            str(box["seriesUid"])
            for _, box in _iter_boxes(annotation)
            if box.get("seriesUid")
        )

    by_study = scan_dicom_roots(
        dicom_roots,
        scan_case_ids=scan_case_ids,
        scan_workers=scan_workers,
        required_series_uids=referenced_series_uids,
    )
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = [
        {"annotation": item["path"], "reason": item["reason"]}
        for item in deduplication["excluded_files"]
    ]

    for annotation_path in annotation_paths:
        try:
            annotation = _read_json(annotation_path)
        except Exception as exc:
            skipped.append({"annotation": str(annotation_path), "reason": f"invalid json: {exc}"})
            continue
        study_uid = str(annotation.get("study_uid") or "")
        if study_uid not in by_study:
            skipped.append({
                "annotation": str(annotation_path),
                "reason": "no referenced DICOM series found for study_uid",
            })
            continue
        record, reason = convert_annotation(
            annotation_path=annotation_path,
            series_for_study=by_study.get(study_uid, []),
            box_format=box_format,
            clamp_boxes=clamp_boxes,
            bucket_handle_code=bucket_handle_code,
        )
        if record is None:
            skipped.append({"annotation": str(annotation_path), "reason": reason or "unknown"})
            continue
        if case_ids and annotation_path.stem not in case_ids and study_uid not in case_ids and record["case_id"] not in case_ids:
            continue
        records.append(record)
        if limit is not None and len(records) >= limit:
            break

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "dicom_roots": dicom_roots,
        "annotation_root": annotation_root,
        "output_jsonl": str(output_path),
        "annotation_files": deduplication["annotation_files_found"],
        "annotation_files_selected": len(annotation_paths),
        "unique_study_uids": deduplication["unique_study_uids"],
        "duplicate_studies": deduplication["duplicate_studies"],
        "duplicate_versions_discarded": deduplication["duplicate_versions_discarded"],
        "studies_scanned": len(by_study),
        "records_written": len(records),
        "annotations_skipped": len(skipped),
        "bucket_handle_code": bucket_handle_code,
        "scan_workers": int(scan_workers),
        "referenced_series_uids": len(referenced_series_uids),
        "deduplication": {
            key: value
            for key, value in deduplication.items()
            if key != "selected_annotations"
        },
        "skipped": skipped[:200],
    }
    if report_json:
        report_path = Path(report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-root", action="append", required=True, help="DICOM root to scan. Repeat this option to scan multiple roots.")
    parser.add_argument("--annotation-root", required=True, help="Directory containing exported annotation JSON files, or one JSON file.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--box-format", choices=["xywh", "center-wh", "xyxy"], default="center-wh")
    parser.add_argument("--no-clamp-boxes", action="store_true")
    parser.add_argument(
        "--bucket-handle-code",
        default="t3",
        help="Coded lesiontype flag used for bucket-handle tear in this annotation export (default: t3).",
    )
    parser.add_argument("--case-id", action="append", default=None, help="Optional case/study id filter. Repeat for multiple cases.")
    parser.add_argument("--scan-case-id", action="append", default=None, help="Optional DICOM case directory filter. Repeat for multiple cases.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after writing this many records.")
    parser.add_argument("--scan-workers", type=int, default=8, help="Parallel DICOM header readers (default: 8).")
    args = parser.parse_args()

    summary = convert_dataset(\
        dicom_roots=args.dicom_root,
        annotation_root=args.annotation_root,
        output_jsonl=args.output_jsonl,
        report_json=args.report_json,
        box_format=args.box_format,
        clamp_boxes=not args.no_clamp_boxes,
        case_ids=set(args.case_id) if args.case_id else None,
        scan_case_ids=set(args.scan_case_id) if args.scan_case_id else None,
        limit=args.limit,
        bucket_handle_code=args.bucket_handle_code,
        scan_workers=args.scan_workers,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "skipped"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
