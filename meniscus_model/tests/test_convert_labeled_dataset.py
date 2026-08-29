"""Regression tests for the private meniscus annotation export converter."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.convert_labeled_dataset import InstanceInfo, SeriesInfo, convert_annotation


def _series(tmp_path: Path, plane: str, uid: str, instance_uid: str) -> SeriesInfo:
    return SeriesInfo(
        case_id="CASE001",
        series_dir=tmp_path / uid,
        study_uid="STUDY001",
        series_uid=uid,
        description=f"{plane} PD FS",
        plane=plane,
        instances=[InstanceInfo(path=tmp_path / uid / "1.dcm", sop_uid=instance_uid, rows=512, columns=512)],
    )


def _box(series_uid: str, instance_uid: str, x: float, y: float, width: float, height: float) -> dict:
    return {
        "seriesUid": series_uid,
        "instanceUid": instance_uid,
        "x": x,
        "y": y,
        "rx": width,
        "ry": height,
    }


def test_lesion_side_is_recovered_from_geometry_not_menicustype(tmp_path: Path) -> None:
    sagittal = _series(tmp_path, "sagittal", "SAG", "SAG1")
    coronal = _series(tmp_path, "coronal", "COR", "COR1")
    annotation = {
        "study_uid": "STUDY001",
        "patientId": "encrypted-patient-value",
        "annos": [
            {
                "info": {"type": "t9", "typeName": "半月板", "position": "t1", "positionName": "内侧", "structure": "t2"},
                "boxs": [_box("SAG", "SAG1", 140, 250, 80, 40)],
            },
            {
                "info": {"type": "t9", "typeName": "半月板", "position": "t2", "positionName": "外侧", "structure": "t1"},
                "boxs": [_box("SAG", "SAG1", 360, 250, 80, 40)],
            },
            {
                # menicustype=t1 means tear, not medial.  Geometry places this
                # lesion inside the lateral meniscus.
                "info": {
                    "type": "t10",
                    "typeName": "病灶半月板",
                    "positionName": "损伤",
                    "menicustype": "t1",
                    "lesiontype": {"t1": True, "t3": True},
                },
                "boxs": [_box("SAG", "SAG1", 360, 250, 30, 20)],
            },
        ],
    }
    path = tmp_path / "annotation.json"
    path.write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8")

    record, reason = convert_annotation(path, [sagittal, coronal], box_format="center-wh", clamp_boxes=True)

    assert reason is None
    assert record is not None
    assert record["module2_labels"] == {"medial_status": "normal", "lateral_status": "tear"}
    assert record["module3_labels"] == {"medial_bucket_handle": 0, "lateral_bucket_handle": 1}
    assert record["module4_labels"]["lateral_discoid"] == 1
    labels_by_side = {item["meniscus_side"]: item for item in record["module1_labels"]}
    assert labels_by_side["medial"]["abnormality"] == "normal"
    assert labels_by_side["lateral"]["abnormality"] == "abnormal"
    assert len(record["patient_group_id"]) == 64
    assert record["conversion_metadata"]["lesion_boxes_matched"] == 1


def test_degeneration_is_assigned_to_overlapping_medial_box(tmp_path: Path) -> None:
    sagittal = _series(tmp_path, "sagittal", "SAG", "SAG1")
    coronal = _series(tmp_path, "coronal", "COR", "COR1")
    annotation = {
        "study_uid": "STUDY001",
        "annos": [
            {
                "info": {"type": "t9", "position": "t1", "positionName": "内侧", "structure": "t2"},
                "boxs": [_box("COR", "COR1", 150, 280, 90, 40)],
            },
            {
                "info": {"type": "t9", "position": "t2", "positionName": "外侧", "structure": "t2"},
                "boxs": [_box("COR", "COR1", 360, 280, 90, 40)],
            },
            {
                # menicustype=t2 means degeneration, not lateral.
                "info": {"type": "t10", "positionName": "变性", "menicustype": "t2", "lesiontype": "t2"},
                "boxs": [_box("COR", "COR1", 150, 280, 25, 15)],
            },
        ],
    }
    path = tmp_path / "annotation.json"
    path.write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8")

    record, reason = convert_annotation(path, [sagittal, coronal], box_format="center-wh", clamp_boxes=True)

    assert reason is None
    assert record is not None
    assert record["module2_labels"] == {"medial_status": "degeneration", "lateral_status": "normal"}
    assert record["module3_labels"] == {"medial_bucket_handle": 0, "lateral_bucket_handle": 0}


def test_lesion_subtype_code_does_not_override_menicustype_diagnosis(tmp_path: Path) -> None:
    sagittal = _series(tmp_path, "sagittal", "SAG", "SAG1")
    coronal = _series(tmp_path, "coronal", "COR", "COR1")
    annotation = {
        "study_uid": "STUDY001",
        "annos": [
            {
                "info": {"type": "t9", "position": "t1", "structure": "t2"},
                "boxs": [_box("SAG", "SAG1", 150, 280, 90, 40)],
            },
            {
                # No Chinese diagnosis text: menicustype=t1 is still a tear,
                # while lesiontype.t2 is only an independent subtype flag.
                "info": {
                    "type": "t10",
                    "menicustype": "t1",
                    "lesiontype": {"t2": True},
                },
                "boxs": [_box("SAG", "SAG1", 150, 280, 25, 15)],
            },
        ],
    }
    path = tmp_path / "annotation.json"
    path.write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8")

    record, reason = convert_annotation(path, [sagittal, coronal], box_format="center-wh", clamp_boxes=True)

    assert reason is None
    assert record is not None
    assert record["module2_labels"]["medial_status"] == "tear"
    assert record["module3_labels"]["medial_bucket_handle"] == 0
