"""Cached loaders and session helpers shared by all pages."""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from src.data.load_data import DatasetNotFoundError, load_processed
from src.models.inference import ModelArtifactsMissing, ModelBundle, load_bundle
from src.utils.config import QUALITY_REPORT_PATH

MAX_COMPARE = 4


@st.cache_data(show_spinner="Loading market data…")
def _data() -> pd.DataFrame:
    return load_processed()


@st.cache_resource(show_spinner="Loading models…")
def _bundle() -> ModelBundle:
    return load_bundle()


def get_data() -> pd.DataFrame:
    try:
        return _data()
    except DatasetNotFoundError as exc:
        st.error(f"**Dataset unavailable.** {exc}", icon=":material/database:")
        st.stop()


def get_bundle() -> ModelBundle:
    try:
        return _bundle()
    except ModelArtifactsMissing as exc:
        st.error(f"**Models unavailable.** {exc}", icon=":material/model_training:")
        st.stop()


@st.cache_data
def get_quality_report() -> dict:
    if QUALITY_REPORT_PATH.exists():
        return json.loads(QUALITY_REPORT_PATH.read_text())
    return {}


@st.cache_data
def localities_for(city: str) -> list[str]:
    df = _data()
    locs = df.loc[df["City"] == city, "Locality"].unique().tolist()
    return sorted(locs, key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 0)


def init_session() -> None:
    st.session_state.setdefault("analyses", [])
    st.session_state.setdefault("compare", [])


def remember_analysis(record: dict, result) -> None:
    init_session()
    st.session_state["analyses"].insert(0, {
        "time": datetime.now().strftime("%H:%M"),
        "label": f"{record['BHK']} BHK {record['Property_Type']}, {record['City']}",
        "record": record,
        "price": record["Price_in_Lakhs"],
        "pps": result.pps,
        "gap": result.gap_pct,
        "verdict": result.verdict,
    })
    st.session_state["analyses"] = st.session_state["analyses"][:10]


def add_to_compare(label: str, record: dict) -> bool:
    init_session()
    items = st.session_state["compare"]
    if any(i["record"] == record for i in items):
        return False
    if len(items) >= MAX_COMPARE:
        return False
    items.append({"label": label, "record": record})
    return True
