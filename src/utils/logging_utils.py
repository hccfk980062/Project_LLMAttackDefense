"""Centralized logging configuration."""
import logging
import os
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
