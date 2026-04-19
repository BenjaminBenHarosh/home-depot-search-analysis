"""Validate Home Depot competition data directory layout."""

from __future__ import annotations

from pathlib import Path

REQUIRED_DATA_FILES = (
    "train.csv",
    "test.csv",
    "attributes.csv",
    "product_descriptions.csv",
)


def check_data_dir(data_dir: str) -> Path:
    """
    Resolve data_dir and ensure required CSVs exist.

    Raises ValueError with an actionable message if validation fails.
    """
    path = Path(data_dir).expanduser().resolve()
    if not path.exists():
        raise ValueError(
            f"Data directory does not exist: {path}\n"
            "Fix: pass --data-dir to the folder containing the Kaggle CSV extract "
            "(train.csv, test.csv, attributes.csv, product_descriptions.csv)."
        )
    if not path.is_dir():
        raise ValueError(
            f"Data path is not a directory: {path}\n"
            "Fix: pass --data-dir to a directory, not a file."
        )
    missing = [name for name in REQUIRED_DATA_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(
            f"Missing required file(s) in data directory {path}: {', '.join(missing)}\n"
            "Fix: download the Home Depot competition data from Kaggle and extract all CSVs into this folder."
        )
    return path
