"""Smoke tests: every chart builder returns a figure on valid and edge-case data."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.data.load_data import clean
from src.models.scenario import trajectory
from src.visualization import charts as ch


def test_eda_charts_build(raw_df):
    df = clean(raw_df)
    figs = [
        ch.price_distribution(df), ch.size_distribution(df), ch.pps_outliers(df), ch.size_vs_price(df),
        ch.box_from_quantiles(df, "Property_Type", "Price_per_SqFt_INR", "t", "y"),
        ch.bar_mean_ci(df, "Facing", "Price_per_SqFt_INR", "t", "v"),
        ch.bar_mean_ci(df, "City", "Price_in_Lakhs", "t", "v", horizontal=True),
        ch.count_bar(df, "Owner_Type", "t"),
        ch.line_mean_ci(df, "Nearby_Schools", "Price_per_SqFt_INR", "t", "x", "y"),
        ch.correlation_heatmap(df, ["Price_in_Lakhs", "Size_in_SqFt", "BHK"]),
        ch.bhk_by_city(df), ch.amenities_effect(df),
    ]
    assert all(isinstance(f, go.Figure) for f in figs)


def test_app_charts_build():
    assert isinstance(ch.cohort_position(np.linspace(1000, 20000, 200), 8000, 9000), go.Figure)
    assert isinstance(ch.cohort_position(np.linspace(1000, 20000, 200), 15000, 9000), go.Figure)
    assert isinstance(ch.scenario_trajectory(trajectory(100, {"A": 0.02, "B": 0.04, "C": 0.06})), go.Figure)
    assert isinstance(ch.confusion_chart([[10, 2], [3, 9]], "cm"), go.Figure)


def test_legend_and_title_do_not_collide():
    fig = ch.scenario_trajectory(trajectory(100, {"A": 0.02, "B": 0.04}))
    fig = ch.finish(fig, "Title", 300)
    assert fig.layout.margin.t >= 70
