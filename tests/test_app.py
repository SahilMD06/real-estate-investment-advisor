"""Headless smoke tests of every Streamlit page (requires the processed data + model bundle)."""
from __future__ import annotations

import pytest

from src.utils import config as C

pytestmark = pytest.mark.skipif(
    not (C.PROCESSED_DATA_PATH.exists() and C.CLASSIFIER_PATH.exists()),
    reason="Run scripts.prepare_data and scripts.train_models first",
)

PAGES = ["dashboard", "investment_analyzer", "property_comparison", "market_intelligence",
         "model_insights", "methodology"]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_exception(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(C.PROJECT_ROOT / "app.py"), default_timeout=120)
    at.run()
    at.switch_page(f"views/{page}.py").run()
    assert not at.exception, at.exception


def test_analyzer_submit_produces_assessment():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(C.PROJECT_ROOT / "app.py"), default_timeout=120)
    at.run()
    at.switch_page("views/investment_analyzer.py").run()
    at.button[0].click().run()  # the form's "Analyze property" submit button
    assert not at.exception, at.exception
    assert "az_result" in at.session_state
    record, result, _ = at.session_state["az_result"]
    assert result.verdict in ("Potentially good value", "Priced above comparables")


def test_comparison_table_is_arrow_compatible():
    """Mixed text/number columns make Streamlit log an Arrow fallback traceback;
    the comparison table must serialise cleanly."""
    import logging

    import pandas as pd
    from streamlit.testing.v1 import AppTest

    from src.features.engineering import RAW_INPUT_COLUMNS

    records = []
    for _, row in pd.read_parquet(C.PROCESSED_DATA_PATH).head(3).iterrows():
        rec = {k: (v.item() if hasattr(v, "item") else v) for k, v in row[RAW_INPUT_COLUMNS].items()}
        records.append({"label": f"#{row['ID']}", "record": rec})

    messages: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = Capture(level=logging.DEBUG)
    logger = logging.getLogger("streamlit.dataframe_util")
    logger.addHandler(handler)
    try:
        at = AppTest.from_file(str(C.PROJECT_ROOT / "app.py"), default_timeout=120)
        at.session_state["compare"] = records
        at.run()
        at.switch_page("views/property_comparison.py").run()
        assert not at.exception, at.exception
        assert len(at.dataframe) >= 1
    finally:
        logger.removeHandler(handler)
    assert not [m for m in messages if "Arrow" in m], messages
