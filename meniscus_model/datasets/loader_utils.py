"""Shared DataLoader worker options."""

from __future__ import annotations

from typing import Any


def worker_loader_kwargs(data_config: dict[str, Any]) -> dict[str, Any]:
    """Return valid worker-only DataLoader arguments.

    PyTorch rejects ``persistent_workers`` and ``prefetch_factor`` when
    ``num_workers=0``, which is common in unit tests and CPU debugging.
    """

    workers = int(data_config.get("num_workers", 0))
    kwargs: dict[str, Any] = {"num_workers": workers}
    if workers > 0:
        kwargs["persistent_workers"] = bool(data_config.get("persistent_workers", False))
        kwargs["prefetch_factor"] = int(data_config.get("prefetch_factor", 2))
    return kwargs
