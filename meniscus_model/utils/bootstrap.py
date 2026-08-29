"""Bootstrap confidence interval utilities."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
import math
import random
from typing import TypeVar


T = TypeVar("T")


def bootstrap_ci(
    values: Sequence[T],
    metric_fn: Callable[[list[T]], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap estimate and percentile confidence interval.

    Args:
        values: Sample-level values. Each bootstrap draw resamples this list.
        metric_fn: Function mapping sampled values to one scalar metric.
        n_boot: Number of bootstrap repetitions.
        alpha: Two-sided tail probability. ``0.05`` gives 95% CI.
        seed: Random seed.

    Returns:
        ``(point_estimate, ci_low, ci_high)``. Returns NaNs when no valid
        estimate can be computed.
    """

    if not values:
        return float("nan"), float("nan"), float("nan")
    values_list = list(values)
    point = metric_fn(values_list)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(max(0, n_boot)):
        sample = [rng.choice(values_list) for _ in values_list]
        value = metric_fn(sample)
        if value == value and not math.isinf(value):
            estimates.append(float(value))
    if not estimates:
        return point, float("nan"), float("nan")
    estimates.sort()
    lo_idx = max(0, int((alpha / 2.0) * len(estimates)))
    hi_idx = min(len(estimates) - 1, int((1.0 - alpha / 2.0) * len(estimates)) - 1)
    return point, estimates[lo_idx], estimates[hi_idx]


def grouped_bootstrap_ci(
    values: Sequence[T],
    group_ids: Sequence[Hashable],
    metric_fn: Callable[[list[T]], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap observations by case/patient rather than independently.

    All observations sharing one ``group_id`` are kept together in every
    redraw.  This is required when, for example, medial and lateral menisci
    or multiple detections originate from the same MRI case.
    """

    if len(values) != len(group_ids):
        raise ValueError("values and group_ids must have identical lengths")
    if not values:
        return float("nan"), float("nan"), float("nan")

    grouped: dict[Hashable, list[T]] = {}
    for value, group_id in zip(values, group_ids):
        grouped.setdefault(group_id, []).append(value)
    groups = list(grouped.values())
    values_list = list(values)
    point = metric_fn(values_list)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(max(0, n_boot)):
        sample: list[T] = []
        for _ in groups:
            sample.extend(rng.choice(groups))
        value = metric_fn(sample)
        if value == value and not math.isinf(value):
            estimates.append(float(value))
    if not estimates:
        return point, float("nan"), float("nan")
    estimates.sort()
    lo_idx = max(0, int((alpha / 2.0) * len(estimates)))
    hi_idx = min(len(estimates) - 1, int((1.0 - alpha / 2.0) * len(estimates)) - 1)
    return point, estimates[lo_idx], estimates[hi_idx]
