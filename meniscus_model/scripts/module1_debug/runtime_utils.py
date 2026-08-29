"""Shared runtime helpers for reproducible Module 1 diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def set_diagnostic_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a diagnostic run."""

    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def limit_dataset(dataset: Dataset, max_cases: int | None) -> Dataset:
    """Return the first ``max_cases`` samples without loading any case data."""

    if max_cases is None:
        return dataset
    count = int(max_cases)
    if count < 0:
        raise ValueError("max_cases must be non-negative")
    return Subset(dataset, range(min(count, len(dataset))))


def write_resolved_run_config(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist the effective diagnostic configuration as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
