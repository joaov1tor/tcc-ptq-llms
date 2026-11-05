"""Single logger: stdout + one file per run. No WikiText body in the log."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(log_path: Path, verbose: bool = False, name: str = "ppl") -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    stream.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def slug_model_id(model_id: str) -> str:
    return model_id.replace("/", "_").replace(" ", "_")
