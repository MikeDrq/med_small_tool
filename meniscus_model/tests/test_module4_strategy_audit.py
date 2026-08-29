"""Tests for the one-pass Module 4 A/B strategy audit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_module4_strategies import evaluate_strategies, save_strategy_outputs


def _candidate(slice_index: int, beta: float) -> dict:
    return {
        "slice_index": slice_index,
        "meniscus_width": beta * 100.0,
        "tibia_width": 100.0,
        "beta": beta,
        "meniscus_source": "roi_cache_pred_same_slice",
        "tibia_source": "dummy_u2net",
        "meniscus_bbox_xyxy": None,
        "tibia_bbox_xyxy": None,
        "unit": "mm",
    }


class Module4StrategyAuditTest(unittest.TestCase):
    def test_evaluate_and_save_strategy_sweep(self) -> None:
        cases = [
            {
                "case_id": "negative",
                "group_id": "negative",
                "label": 0,
                "sources": {
                    "gt": [_candidate(0, 0.20), _candidate(1, 0.30)],
                    "pred": [_candidate(0, 0.25), _candidate(1, 0.35)],
                },
            },
            {
                "case_id": "positive",
                "group_id": "positive",
                "label": 1,
                "sources": {
                    "gt": [_candidate(0, 0.45), _candidate(1, 0.55)],
                    "pred": [_candidate(0, 0.42), _candidate(1, 0.50)],
                },
            },
        ]
        metrics, predictions = evaluate_strategies(
            cases,
            strategies=["max_w2", "q25_beta"],
            plateau_fractions=[0.95],
            threshold=0.4,
            min_candidate_slices=1,
            central_window_radius=1,
            bootstrap_samples=0,
            seed=42,
        )
        self.assertEqual(len(metrics), 4)
        self.assertEqual(len(predictions), 8)
        self.assertTrue(all(item["roc_auc"] == 1.0 for item in metrics))
        self.assertTrue(all(item["threshold"] == 0.4 for item in metrics))
        with tempfile.TemporaryDirectory() as tmpdir:
            save_strategy_outputs(tmpdir, metrics, predictions)
            selected = json.loads(
                (Path(tmpdir) / "selected_strategy.json").read_text(encoding="utf-8")
            )
            self.assertEqual(selected["selection_box_source"], "pred")
            self.assertTrue((Path(tmpdir) / "strategy_metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
