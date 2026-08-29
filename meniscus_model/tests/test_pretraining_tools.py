"""Tests for pre-training data preparation utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from datasets.dicom_reader import apply_dicom_window, apply_paper_window, select_slice_indices
from datasets.meniscus_detection_dataset import (
    _prepare_module1_labels,
    _processed_slices_for_source,
    build_label_chunk_sample_plans,
)
from scripts.convert_labeled_dataset import _bbox_from_box
from scripts.create_case_subset import select_case_subset
from scripts.split_index import split_index, split_records
from scripts.visualize_converted_labels import categories_for_record, select_record_indices


class PretrainingToolsTest(unittest.TestCase):
    def test_paper_window_uses_exact_center_plus_minus_half_width(self) -> None:
        dataset = SimpleNamespace(
            WindowCenter=50.0,
            WindowWidth=100.0,
            PhotometricInterpretation="MONOCHROME2",
        )
        values = np.asarray([-1.0, 0.0, 50.0, 100.0, 101.0], dtype=np.float32)
        windowed = apply_paper_window(values, dataset)
        assert windowed is not None
        np.testing.assert_allclose(windowed, [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-6)

    def test_dicom_linear_window_and_monochrome1_inversion(self) -> None:
        dataset = SimpleNamespace(
            WindowCenter=50.0,
            WindowWidth=101.0,
            VOILUTFunction="LINEAR",
            PhotometricInterpretation="MONOCHROME2",
        )
        values = np.asarray([-1.0, 0.0, 50.0, 100.0, 101.0], dtype=np.float32)
        windowed = apply_dicom_window(values, dataset)
        assert windowed is not None
        np.testing.assert_allclose(windowed, [0.0, 0.005, 0.505, 1.0, 1.0], atol=1e-6)

        dataset.PhotometricInterpretation = "MONOCHROME1"
        inverted = apply_dicom_window(values, dataset)
        assert inverted is not None
        np.testing.assert_allclose(inverted, 1.0 - windowed, atol=1e-6)

    def _records(self) -> list[dict]:
        return [
            {"case_id": "case_a", "study_uid": "study_a", "module1_labels": [{"meniscus_side": "medial"}]},
            {"case_id": "case_a", "study_uid": "study_a_2", "module1_labels": [{"meniscus_side": "lateral"}]},
            {"case_id": "case_b", "study_uid": "study_b", "module2_labels": {"medial_status": "tear"}},
            {"case_id": "case_c", "study_uid": "study_c", "module2_labels": {"lateral_status": "degeneration"}},
            {"case_id": "case_d", "study_uid": "study_d", "module2_labels": {"medial_status": "normal"}},
        ]

    def test_split_records_keeps_case_groups_together(self) -> None:
        splits = split_records(self._records(), train_ratio=0.4, val_ratio=0.2, test_ratio=0.4, seed=7)
        seen: dict[str, str] = {}
        for split_name, records in splits.items():
            for record in records:
                case_id = record["case_id"]
                if case_id in seen:
                    self.assertEqual(seen[case_id], split_name)
                else:
                    seen[case_id] = split_name

    def test_stratified_split_balances_rare_labels_and_repeated_groups(self) -> None:
        records = []
        for index in range(60):
            record = {
                "case_id": f"case_{index}",
                "patient_group_id": f"patient_{index}",
                "module2_labels": {
                    "medial_status": ("normal", "degeneration", "tear")[index % 3],
                    "lateral_status": ("tear", "normal", "degeneration")[index % 3],
                },
                "module3_labels": {
                    "medial_bucket_handle": int(index % 7 == 0),
                    "lateral_bucket_handle": int(index % 11 == 0),
                },
                "module4_labels": {"lateral_discoid": int(index % 9 == 0)},
            }
            records.append(record)
            if index % 10 == 0:
                records.append({**record, "case_id": f"case_{index}_repeat"})

        splits = split_records(records, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=17)
        group_sets = {
            split_name: {record["patient_group_id"] for record in split_records_}
            for split_name, split_records_ in splits.items()
        }
        self.assertFalse(group_sets["train"] & group_sets["val"])
        self.assertFalse(group_sets["train"] & group_sets["test"])
        self.assertFalse(group_sets["val"] & group_sets["test"])
        for split_records_ in splits.values():
            self.assertTrue(any(record["module3_labels"]["medial_bucket_handle"] for record in split_records_))
            counts = {}
            for record in split_records_:
                group_id = record["patient_group_id"]
                counts[group_id] = counts.get(group_id, 0) + 1
            self.assertTrue(any(count > 1 for count in counts.values()))

    def test_case_subset_is_seeded_and_keeps_all_records_for_selected_cases(self) -> None:
        subset, selected_ids = select_case_subset(self._records(), num_cases=2, seed=7)

        self.assertEqual(selected_ids, sorted(selected_ids))
        self.assertEqual({record["case_id"] for record in subset}, set(selected_ids))
        for case_id in selected_ids:
            expected = sum(record["case_id"] == case_id for record in self._records())
            actual = sum(record["case_id"] == case_id for record in subset)
            self.assertEqual(actual, expected)

    def test_split_index_writes_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "index_all.jsonl"
            input_path.write_text("\n".join('{"case_id": "%s"}' % idx for idx in range(6)) + "\n", encoding="utf-8")
            summary = split_index(input_path, root / "out", seed=3)
            self.assertEqual(summary["total"], 6)
            self.assertTrue((root / "out" / "train.jsonl").exists())
            self.assertTrue((root / "out" / "val.jsonl").exists())
            self.assertTrue((root / "out" / "test.jsonl").exists())
            self.assertTrue((root / "out" / "split_report.json").exists())

    def test_visualization_category_selection(self) -> None:
        record = {
            "module1_labels": [{"meniscus_side": "medial", "abnormality": "normal"}],
            "module2_labels": {"medial_status": "normal", "lateral_status": "tear"},
        }
        self.assertEqual(categories_for_record(record), {"normal", "tear", "medial"})
        selected = select_record_indices([record], ["normal", "tear"], per_category=1, seed=42)
        self.assertEqual(selected, [("normal", 0), ("tear", 0)])

    def test_center_wh_conversion_is_one_time_xyxy(self) -> None:
        bbox = _bbox_from_box({"x": 50, "y": 40, "rx": 20, "ry": 10}, box_format="center-wh")
        self.assertEqual(bbox, [40.0, 35.0, 60.0, 45.0])

    def test_prepared_labels_cover_padded_duplicate_slices(self) -> None:
        self.assertEqual(_processed_slices_for_source(0, [0, 0, 1, 2]), [0, 1])
        self.assertEqual(_processed_slices_for_source(5, [0, 2, 4, 8]), [])
        self.assertEqual(
            _processed_slices_for_source(5, [0, 2, 4, 8], unmatched_label_policy="nearest"),
            [2],
        )
        with self.assertRaises(ValueError):
            _processed_slices_for_source(5, [0, 2, 4, 8], unmatched_label_policy="error")
        series = SimpleNamespace(
            metadata={
                "sampled_source_indices": [0, 0, 1, 2],
                "original_shape": (3, 100, 200),
                "processed_shape": (4, 50, 100),
            }
        )
        record = {
            "case_id": "case_a",
            "module1_labels": [
                {
                    "plane": "sagittal",
                    "slice_index": 0,
                    "bbox": [20, 10, 60, 30],
                    "meniscus_side": "medial",
                    "abnormality": "abnormal",
                }
            ],
        }
        labels = _prepare_module1_labels(record, {"sagittal": series})
        self.assertEqual([label["slice_index"] for label in labels], [0, 1])
        self.assertEqual(labels[0]["bbox"], [10.0, 5.0, 30.0, 15.0])
        self.assertTrue(all(label["exact_slice_match"] for label in labels))

        nearest_series = SimpleNamespace(
            metadata={
                "sampled_source_indices": [0, 2, 4, 8],
                "original_shape": (9, 100, 200),
                "processed_shape": (4, 50, 100),
            }
        )
        nearest_record = {
            **record,
            "module1_labels": [{**record["module1_labels"][0], "slice_index": 5}],
        }
        remapped = _prepare_module1_labels(
            nearest_record,
            {"sagittal": nearest_series},
            unmatched_label_policy="nearest",
        )
        self.assertEqual(remapped[0]["sampled_source_slice_index"], 4)
        self.assertFalse(remapped[0]["exact_slice_match"])
        self.assertEqual(remapped[0]["slice_mapping_kind"], "nearest")

    def test_label_slice_selection_uses_only_label_slices(self) -> None:
        indices = select_slice_indices(depth=32, target_slices=16, mode="label", candidate_indices=[4, 9, 20])
        self.assertEqual(len(indices), 16)
        self.assertTrue(set(indices.tolist()).issubset({4, 9, 20}))
        self.assertEqual(set(indices.tolist()), {4, 9, 20})

    def test_trimmed_slice_selection_excludes_head_tail_and_center(self) -> None:
        indices = select_slice_indices(depth=32, target_slices=16, mode="trimmed_uniform", exclude_head=2, exclude_tail=2, exclude_center=2)
        self.assertEqual(len(indices), 16)
        self.assertGreaterEqual(int(indices.min()), 2)
        self.assertLessEqual(int(indices.max()), 29)
        self.assertNotIn(15, indices.tolist())
        self.assertNotIn(16, indices.tolist())

    def test_include_slice_selection_keeps_required_slices_without_duplicates(self) -> None:
        indices = select_slice_indices(
            depth=32,
            target_slices=16,
            mode="include",
            candidate_indices=[1, 5, 10, 20],
        )

        self.assertEqual(len(indices), 16)
        self.assertEqual(len(set(indices.tolist())), 16)
        self.assertTrue({1, 5, 10, 20}.issubset(set(indices.tolist())))

        with self.assertRaises(ValueError):
            select_slice_indices(
                depth=32,
                target_slices=16,
                mode="include",
                candidate_indices=list(range(17)),
            )

    def test_label_chunk_plans_partition_every_raw_gt_once(self) -> None:
        labels = [
            {
                "plane": "sagittal",
                "slice_index": source_slice,
                "bbox": [1, 1, 3, 3],
                "meniscus_side": "medial",
                "abnormality": "normal",
            }
            for source_slice in range(18)
        ]
        labels.extend(
            {
                "plane": "coronal",
                "slice_index": source_slice,
                "bbox": [1, 1, 3, 3],
                "meniscus_side": "lateral",
                "abnormality": "normal",
            }
            for source_slice in (3, 7)
        )

        plans = build_label_chunk_sample_plans(
            [{"case_id": "dense_case", "module1_labels": labels}],
            target_slices=16,
        )

        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0]["plane_source_indices"]["sagittal"], tuple(range(16)))
        self.assertEqual(plans[1]["plane_source_indices"]["sagittal"], (16, 17))
        self.assertEqual(plans[0]["plane_source_indices"]["coronal"], (3, 7))
        self.assertIsNone(plans[1]["plane_source_indices"]["coronal"])
        assigned = [index for plan in plans for index in plan["active_source_label_indices"]]
        self.assertEqual(sorted(assigned), list(range(len(labels))))
        self.assertEqual(len(assigned), len(set(assigned)))


if __name__ == "__main__":
    unittest.main()
