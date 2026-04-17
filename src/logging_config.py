"""Centralized loguru configuration."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def configure_logging(output_dir: str, run_id: str, level: str = "INFO") -> str:
    """Configure console and file sinks for a run."""
    logs_dir = Path(output_dir) / "runs" / run_id / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_dir / "run.log"

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    logger.add(
        str(log_file_path),
        level=level,
        rotation="5 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    )
    return str(log_file_path)

