"""Validate the raw CSV and write the cleaned parquet + data-quality report.

Usage (from the project root):
    python -m scripts.prepare_data
"""
from __future__ import annotations

import sys

from src.data.load_data import DatasetNotFoundError, prepare
from src.data.validate_data import DataValidationError
from src.utils.logging import get_logger

log = get_logger("prepare_data")


def main() -> int:
    try:
        df = prepare()
    except DatasetNotFoundError as exc:
        log.error(str(exc))
        return 1
    except DataValidationError as exc:
        log.error("Dataset failed validation: %s", exc)
        return 2
    log.info("Done: %s rows, %s columns", *df.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
