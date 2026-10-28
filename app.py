"""PropWise AI — Streamlit entry point.

Run from the project root:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.utils.config import PROJECT_ROOT
from src.ui.state import init_session
from src.ui.theme import inject_css
from src.utils.config import APP_NAME

st.set_page_config(
    page_title=APP_NAME,
    page_icon=":material/apartment:",
    layout="wide",
    initial_sidebar_state="auto",
)
inject_css()
init_session()

pages = {
    "Overview": [
        st.Page("views/dashboard.py", title="Executive Dashboard", icon=":material/space_dashboard:", default=True),
    ],
    "Analyze": [
        st.Page("views/investment_analyzer.py", title="Investment Analyzer", icon=":material/query_stats:"),
        st.Page("views/property_comparison.py", title="Property Comparison", icon=":material/compare_arrows:"),
    ],
    "Explore": [
        st.Page("views/market_intelligence.py", title="Market Intelligence", icon=":material/map:"),
        st.Page("views/model_insights.py", title="Model Insights", icon=":material/insights:"),
    ],
    "Reference": [
        st.Page("views/methodology.py", title="Methodology & About", icon=":material/menu_book:"),
    ],
}

st.logo(str(PROJECT_ROOT / "assets" / "logo.svg"), size="large",
        icon_image=str(PROJECT_ROOT / "assets" / "icon.svg"))

nav = st.navigation(pages)
nav.run()

# Rendered after the page so the count reflects additions made on this run.
with st.sidebar:
    n_cmp = len(st.session_state.get("compare", []))
    st.html(
        '<div class="pw-side-note">Static dataset snapshot · 250,000 listings across 42 cities.<br>'
        f'Comparison list: {n_cmp} of 4 properties.<br>Not financial advice.</div>'
    )
