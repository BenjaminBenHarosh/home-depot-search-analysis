"""Centralized loguru configuration."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from tqdm import tqdm


def _tqdm_console_sink(message) -> None:
    # Route console logs through tqdm so progress bars and log lines do not clobber each other.
    tqdm.write(str(message), end="")


def configure_logging(output_dir: str, run_id: str, level: str = "INFO") -> str:
    """Configure console and file sinks for a run."""
    logs_dir = Path(output_dir) / "runs" / run_id / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_dir / "run.log"

    logger.remove()
    logger.add(
        _tqdm_console_sink,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}\n",
    )
    logger.add(
        str(log_file_path),
        level=level,
        rotation="5 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    )
    return str(log_file_path)

