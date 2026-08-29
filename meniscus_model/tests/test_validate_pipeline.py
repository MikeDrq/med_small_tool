"""Tests for full pipeline validation metric aggregation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine.validate_pipeline import TASKS, task_score_and_pred, validate_pipeline


class ValidatePipelineTest(unittest.TestCase):
    def test_missing_tibia_measurement_is_not_scored_as_negative_discoid(self) -> None:
        task = next(item for item in TASKS if item["key"] == "lateral_discoid")
        result = {
            "menisci": {
                "lateral": {
                    "discoid": False,
                    "discoid_available": False,
                    "discoid_beta": None,
                    "confidence": {"discoid": None},
                }
            }
        }
        self.assertEqual(task_score_and_pred(result, task), (None, None))

    def test_validate_pipeline_from_existing_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            index = root / "index.jsonl"
            records = [
                {
                    "case_id": "case_1",
                    "sagittal_dicom_dir": "unused",
                    "coronal_dicom_dir": "unused",
                    "module2_labels": {"medial_status": "tear", "lateral_status": "normal"},
                    "module3_labels": {"medial_bucket_handle": 1, "lateral_bucket_handle": 0},
                    "module4_labels": {"lateral_discoid": 1},
                },
                {
                    "case_id": "case_2",
                    "sagittal_dicom_dir": "unused",
                    "coronal_dicom_dir": "unused",
                    "module2_labels": {"medial_status": "normal", "lateral_status": "degeneration"},
                    "module3_labels": {"medial_bucket_handle": 0, "lateral_bucket_handle": 0},
                    "module4_labels": {"lateral_discoid": 0},
                },
            ]
            with index.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            predictions_dir = root / "preds"
            predictions_dir.mkdir()
            for record in records:
                case_dir = predictions_dir / record["case_id"]
                case_dir.mkdir()
                medial_status_probabilities = {
                    "normal": 0.9 if record["module2_labels"]["medial_status"] == "normal" else 0.05,
                    "degeneration": 0.9 if record["module2_labels"]["medial_status"] == "degeneration" else 0.05,
                    "tear": 0.9 if record["module2_labels"]["medial_status"] == "tear" else 0.05,
                }
                if record["module2_labels"]["medial_status"] == "tear":
                    # Argmax predicts tear even though its one-vs-rest score is
                    # below 0.5. Metrics must agree with the saved prediction.
                    medial_status_probabilities = {
                        "normal": 0.35,
                        "degeneration": 0.20,
                        "tear": 0.45,
                    }
                result = {
                    "case_id": record["case_id"],
                    "menisci": {
                        "medial": {
                            "status": record["module2_labels"]["medial_status"],
                            "bucket_handle": bool(record["module3_labels"]["medial_bucket_handle"]),
                            "confidence": {
                                "status": 0.9,
                                "status_probabilities": medial_status_probabilities,
                                "bucket_handle": 0.9 if record["module3_labels"]["medial_bucket_handle"] else 0.1,
                            },
                        },
                        "lateral": {
                            "status": record["module2_labels"]["lateral_status"],
                            "bucket_handle": bool(record["module3_labels"]["lateral_bucket_handle"]),
                            "discoid": bool(record["module4_labels"]["lateral_discoid"]),
                            "discoid_beta": 0.45 if record["module4_labels"]["lateral_discoid"] else 0.25,
                            "confidence": {
                                "status": 0.9,
                                "status_probabilities": {
                                    "normal": 0.9 if record["module2_labels"]["lateral_status"] == "normal" else 0.05,
                                    "degeneration": 0.9 if record["module2_labels"]["lateral_status"] == "degeneration" else 0.05,
                                    "tear": 0.9 if record["module2_labels"]["lateral_status"] == "tear" else 0.05,
                                },
                                "bucket_handle": 0.9 if record["module3_labels"]["lateral_bucket_handle"] else 0.1,
                                "discoid": 0.9 if record["module4_labels"]["lateral_discoid"] else 0.1,
                            },
                        },
                    },
                }
                (case_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

            config = root / "pipeline.yaml"
            config.write_text("pipeline: {}\n", encoding="utf-8")
            out = root / "validation"
            metrics = validate_pipeline(str(config), str(index), str(out), bootstrap=10, seed=1, predictions_dir=str(predictions_dir))
            self.assertEqual(metrics["medial_tear"]["n"], 2)
            self.assertEqual(metrics["medial_tear"]["recall"], 1.0)
            self.assertEqual(metrics["lateral_discoid"]["n"], 2)
            self.assertEqual(metrics["medial_tear"]["bootstrap_group_count"], 2)
            self.assertTrue((out / "metrics.json").exists())
            self.assertTrue((out / "metrics.csv").exists())
            self.assertTrue((out / "confusion_matrices" / "medial_tear_confusion_matrix.csv").exists())


if __name__ == "__main__":
    unittest.main()
