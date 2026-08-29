"""Logging setup for training and inference scripts."""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str = "meniscus_model", level: int = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """Create a logger with console and optional file handlers."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    if not any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path) for handler in logger.handlers):
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


def format_metrics(metrics: dict[str, float]) -> str:
    """Format metrics as compact key=value text."""

    parts = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)
