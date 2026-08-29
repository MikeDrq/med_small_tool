"""Tests for cross-batch annotation version selection."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from utils.annotation_dedup import select_annotation_files


def _write_version(
    root: Path,
    batch: str,
    study_uid: str,
    *,
    status: str,
    update_time: int,
    boxes: int,
) -> Path:
    path = root / batch / f"{study_uid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "study_uid": study_uid,
        "labelStatus": status,
        "update_time": update_time,
        "annos": [{"boxs": [{"x": index} for index in range(boxes)]}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path.resolve()


class AnnotationDeduplicationTest(unittest.TestCase):
    def test_selects_reviewed_then_latest_then_most_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_version(root, "a", "latest", status="REVIEWED", update_time=10, boxes=5)
            latest = _write_version(root, "b", "latest", status="REVIEWED", update_time=20, boxes=1)

            reviewed = _write_version(root, "a", "reviewed", status="REVIEWED", update_time=10, boxes=1)
            _write_version(root, "b", "reviewed", status="DRAFT", update_time=20, boxes=8)

            _write_version(root, "a", "complete", status="REVIEWED", update_time=10, boxes=1)
            complete = _write_version(root, "b", "complete", status="REVIEWED", update_time=10, boxes=2)

            unique = _write_version(root, "a", "unique", status="REVIEWED", update_time=1, boxes=1)
            (root / "a" / "broken.json").write_text("{", encoding="utf-8")

            selected, report = select_annotation_files(root)

            self.assertEqual(set(selected), {latest, reviewed, complete, unique})
            self.assertEqual(report["annotation_files_found"], 8)
            self.assertEqual(report["valid_annotation_files"], 7)
            self.assertEqual(report["unique_study_uids"], 4)
            self.assertEqual(report["duplicate_studies"], 3)
            self.assertEqual(report["duplicate_versions_discarded"], 3)
            self.assertEqual(report["invalid_or_missing_uid_files"], 1)


if __name__ == "__main__":
    unittest.main()
