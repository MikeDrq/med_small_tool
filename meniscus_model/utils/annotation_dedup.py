"""Select one authoritative annotation export per MRI study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SELECTION_POLICY = "reviewed_then_latest_update_time_then_box_count"


def _annotation_files(annotation_root: str | Path) -> list[Path]:
    root = Path(annotation_root).expanduser()
    if root.is_file():
        return [root.resolve()]
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(path.resolve() for path in root.rglob("*.json") if path.is_file())


def _version_metadata(path: Path, annotation: dict[str, Any]) -> dict[str, Any]:
    annotations = annotation.get("annos") or []
    try:
        update_time = int(annotation.get("update_time") or 0)
    except (TypeError, ValueError):
        update_time = 0
    return {
        "path": str(path),
        "study_uid": str(annotation.get("study_uid") or annotation.get("StudyInstanceUID") or ""),
        "label_status": str(annotation.get("labelStatus") or ""),
        "update_time": update_time,
        "annotation_count": len(annotations),
        "box_count": sum(len(item.get("boxs") or []) for item in annotations),
    }


def _selection_rank(version: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        int(version["label_status"].upper() == "REVIEWED"),
        int(version["update_time"]),
        int(version["box_count"]),
        int(version["annotation_count"]),
        str(version["path"]),
    )


def select_annotation_files(annotation_root: str | Path) -> tuple[list[Path], dict[str, Any]]:
    """Choose the latest reviewed JSON for each unique StudyInstanceUID."""

    source_paths = _annotation_files(annotation_root)
    versions_by_study: dict[str, list[dict[str, Any]]] = {}
    excluded_files: list[dict[str, str]] = []

    for path in source_paths:
        try:
            annotation = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            excluded_files.append({"path": str(path), "reason": f"invalid json: {exc}"})
            continue
        version = _version_metadata(path, annotation)
        study_uid = version["study_uid"]
        if not study_uid:
            excluded_files.append({"path": str(path), "reason": "missing study_uid"})
            continue
        versions_by_study.setdefault(study_uid, []).append(version)

    selected_versions: list[dict[str, Any]] = []
    duplicate_groups: list[dict[str, Any]] = []
    for study_uid in sorted(versions_by_study):
        versions = sorted(versions_by_study[study_uid], key=_selection_rank, reverse=True)
        selected_versions.append(versions[0])
        if len(versions) > 1:
            duplicate_groups.append(
                {
                    "study_uid": study_uid,
                    "selected": versions[0],
                    "discarded": versions[1:],
                }
            )

    valid_versions = sum(len(versions) for versions in versions_by_study.values())
    report = {
        "selection_policy": SELECTION_POLICY,
        "annotation_root": str(Path(annotation_root).expanduser().resolve()),
        "annotation_files_found": len(source_paths),
        "valid_annotation_files": valid_versions,
        "invalid_or_missing_uid_files": len(excluded_files),
        "unique_study_uids": len(selected_versions),
        "duplicate_studies": len(duplicate_groups),
        "duplicate_versions_discarded": valid_versions - len(selected_versions),
        "selected_annotations": selected_versions,
        "duplicate_groups": duplicate_groups,
        "excluded_files": excluded_files,
    }
    return [Path(version["path"]) for version in selected_versions], report
