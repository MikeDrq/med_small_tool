"""Tests for reproducible Module 1 diagnostic runtime configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import Subset

from scripts.module1_debug.audit_module1_detection_candidates import (
    make_loader as make_candidate_audit_loader,
    resolve_audit_options,
)
from scripts.module1_debug.evaluate_module1_oracle_ranking import make_loader as make_oracle_loader
from scripts.module1_debug.runtime_utils import set_diagnostic_seed
from scripts.module1_debug.sweep_module1_postprocess import make_loader as make_sweep_loader


class Module1DebugRuntimeTest(unittest.TestCase):
    def _index_and_config(self, directory: str) -> tuple[Path, dict]:
        index_path = Path(directory) / "val.jsonl"
        index_path.write_text(
            "".join(json.dumps({"case_id": f"case_{idx}"}) + "\n" for idx in range(4)),
            encoding="utf-8",
        )
        config = {
            "seed": 17,
            "data": {
                "val_index": str(index_path),
                "output_size": 224,
                "num_slices": 16,
                "patch_size": 16,
                "num_workers": 0,
                "slice_selection": {"mode": "uniform"},
                "val_slice_selection": {
                    "mode": "trimmed_uniform",
                    "exclude_head": 2,
                    "exclude_tail": 2,
                },
            },
            "validate": {"batch_size": 1},
        }
        return index_path, config

    def test_debug_loaders_use_validation_slice_selection_and_case_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path, config = self._index_and_config(tmpdir)
            loaders = [
                make_candidate_audit_loader(config, index_path, max_cases=2),
                make_oracle_loader(config, str(index_path), max_cases=2),
                make_sweep_loader(config, str(index_path), max_cases=2),
            ]

            for loader in loaders:
                self.assertIsInstance(loader.dataset, Subset)
                self.assertEqual(len(loader.dataset), 2)
                base_dataset = loader.dataset.dataset
                self.assertEqual(base_dataset.slice_selection, config["data"]["val_slice_selection"])

    def test_candidate_audit_defaults_resolve_from_validation_config(self) -> None:
        config = {
            "seed": 17,
            "data": {"val_slice_selection": {"mode": "trimmed_uniform"}},
            "validate": {
                "score_threshold": 0.02,
                "topk_per_image": 750,
                "nms_threshold": 0.35,
                "final_score_mode": "objectness_quality",
                "quality_score_alpha": 0.75,
                "quality_score_beta": 1.25,
            },
        }
        args = argparse.Namespace(
            score_threshold=None,
            topk_per_image=None,
            nms_threshold=None,
            final_score_mode=None,
            quality_score_alpha=None,
            quality_score_beta=None,
            seed=None,
            max_cases=None,
        )

        resolved = resolve_audit_options(args, config)

        self.assertEqual(resolved.score_threshold, 0.02)
        self.assertEqual(resolved.topk_per_image, 750)
        self.assertEqual(resolved.nms_threshold, 0.35)
        self.assertEqual(resolved.final_score_mode, "objectness_quality")
        self.assertEqual(resolved.quality_score_alpha, 0.75)
        self.assertEqual(resolved.quality_score_beta, 1.25)
        self.assertEqual(resolved.seed, 17)
        self.assertEqual(resolved.resolved_slice_selection, {"mode": "trimmed_uniform"})

    def test_candidate_audit_explicit_overrides_are_preserved(self) -> None:
        config = {
            "seed": 17,
            "data": {},
            "validate": {"score_threshold": 0.5, "topk_per_image": 100, "nms_threshold": 0.5},
        }
        args = argparse.Namespace(
            score_threshold=0.0,
            topk_per_image=1000,
            nms_threshold=0.3,
            final_score_mode="quality",
            quality_score_alpha=2.0,
            quality_score_beta=3.0,
            seed=9,
            max_cases=5,
        )

        resolved = resolve_audit_options(args, config)

        self.assertEqual(resolved.score_threshold, 0.0)
        self.assertEqual(resolved.topk_per_image, 1000)
        self.assertEqual(resolved.nms_threshold, 0.3)
        self.assertEqual(resolved.final_score_mode, "quality")
        self.assertEqual(resolved.quality_score_alpha, 2.0)
        self.assertEqual(resolved.quality_score_beta, 3.0)
        self.assertEqual(resolved.seed, 9)
        self.assertEqual(resolved.max_cases, 5)

    def test_diagnostic_seed_repeats_python_numpy_and_torch(self) -> None:
        set_diagnostic_seed(23)
        first = (random.random(), float(np.random.rand()), float(torch.rand(()).item()))
        set_diagnostic_seed(23)
        second = (random.random(), float(np.random.rand()), float(torch.rand(()).item()))

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
