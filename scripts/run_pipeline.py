"""Run the full reproducible workflow end to end.

Usage (from the project root):
    python -m scripts.run_pipeline
Steps: validate + clean → EDA report → train + track (MLflow) → evaluation report.
"""
from __future__ import annotations

import sys

from scripts import evaluate_models, prepare_data, run_eda, train_models
from src.utils.logging import get_logger

log = get_logger("pipeline")


def main() -> int:
    for name, step in [("prepare_data", prepare_data.main), ("run_eda", run_eda.main),
                       ("train_models", lambda: train_models.main([])), ("evaluate_models", evaluate_models.main)]:
        log.info("── %s", name)
        code = step()
        if code:
            log.error("Step %s failed with exit code %s", name, code)
            return code
    log.info("Pipeline complete. Start the app with: streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
