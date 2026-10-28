"""Small logging helper so every module logs with the same format."""
from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(os.getenv("PROPWISE_LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger
