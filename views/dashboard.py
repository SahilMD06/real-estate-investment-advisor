"""Executive Dashboard."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui import components as ui
from src.ui.state import get_bundle, get_data, get_quality_report
from src.visualization import charts as ch

df = get_data()
bundle = get_bundle()
quality = get_quality_report()

ui.page_header(
    "Indian residential market at a glance",
    "Check whether a listing is priced below comparable properties, and see what it could be worth in five "
    "years under transparent growth scenarios. Every number traces back to the dataset or a cited source.",
    eyebrow="Executive dashboard",
)

ui.kpi_row([
    {"label": "Listings analysed", "value": f"{len(df):,}", "sub": "Static snapshot, no listing dates"},
    {"label": "Coverage", "value": f"{df['City'].nunique()} cities", "sub": f"{df['State'].nunique()} states · {df.groupby(['City', 'Locality']).ngroups:,} localities"},
    {"label": "Median price", "value": ui.lakhs(df["Price_in_Lakhs"].median()), "sub": f"Mean {ui.lakhs(df['Price_in_Lakhs'].mean())}"},
    {"label": "Median ₹ / sqft", "value": ui.inr(df["Price_per_SqFt_INR"].median()), "sub": f"IQR {ui.inr(df['Price_per_SqFt_INR'].quantile(.25))}–{ui.inr(df['Price_per_SqFt_INR'].quantile(.75))}"},
    {"label": "Comparable cohorts", "value": f"{bundle.metadata['cohorts']['n']}", "sub": "City × type × BHK"},
])

with ui.row("card", "dashboard-30"):
    c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    with st.container(border=True, height="stretch"):
        st.markdown("**Analyze a listing**")
        st.caption("Enter a property's details to see how its ₹/sqft compares with similar homes, "
                   "plus three five-year value scenarios.")
        st.page_link("views/investment_analyzer.py", label="Open Investment Analyzer", icon=":material/arrow_forward:")
with c2:
    with st.container(border=True, height="stretch"):
        st.markdown("**Compare side by side**")
        st.caption("Put up to four listings next to each other under the same growth assumption. "
                   "Differences are highlighted; no winner is declared.")
        st.page_link("views/property_comparison.py", label="Open Property Comparison", icon=":material/arrow_forward:")
with c3:
    with st.container(border=True, height="stretch"):
        st.markdown("**Explore the market**")
        st.caption("Filter by state, city, type, BHK, price and size, and browse every listing.")
        st.page_link("views/market_intelligence.py", label="Open Market Intelligence", icon=":material/arrow_forward:")

ui.section("Market shape", "All 250,000 listings")
with ui.row("chart", "dashboard-50"):
    left, right = st.columns([1, 1], gap="medium")
with left:
    with st.container(border=True):
        st.plotly_chart(ch.histogram(df["Price_in_Lakhs"], "Distribution of asking prices", "Price (₹ lakhs)", bins=25, height=430),
                        width="stretch", config={"displaylogo": False})
with right:
    with st.container(border=True):
        city = df.groupby("City")["Price_per_SqFt_INR"].median().sort_values(ascending=False)
        top = pd.concat([city.head(8), city.tail(8)])
        fig = ch.ranked_bar(pd.Series(top.index), top.values, "Median ₹/sqft: 8 highest and 8 lowest cities",
                            "Median ₹ per sqft", height=430,
                            hover="%{y}<br><b>₹%{x:,.0f}/sqft</b><extra></extra>")
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

spread = (city.max() / city.min() - 1) * 100
ui.callout(
    "info", "How to read these numbers",
    f"City medians differ by only {spread:.1f}%, and price is unrelated to size, amenities or transport in this "
    "dataset (see Market Intelligence). That is why PropWise compares each listing with its own cohort "
    "instead of claiming one city or feature is 'better'.",
)

ui.section("Dataset coverage & quality")
q_stats = quality.get("stats", {})
inc = q_stats.get("inconsistency_shares", {})
with ui.row("chart", "dashboard-75"):
    d1, d2 = st.columns([1, 1], gap="medium")
with d1:
    with st.container(border=True, height="stretch"):
        st.markdown("**Coverage & freshness**")
        st.markdown(
            f"- **Source:** `india_housing_prices.csv` ({quality.get('n_rows', len(df)):,} rows, 23 columns)\n"
            f"- **Fingerprint:** `{bundle.metadata['dataset']['sha256'][:12]}…`\n"
            f"- **Age reference year:** {bundle.metadata['dataset']['reference_year']} (Age = 2025 − Year Built in every row)\n"
            f"- **Models trained:** {bundle.metadata['created_at'][:10]}\n"
            "- **Freshness:** one static snapshot with no listing dates, so trends over time are not available"
        )
with d2:
    with st.container(border=True, height="stretch"):
        st.markdown("**Data quality checks**")
        st.markdown(
            f"- Missing values: **{sum(q_stats.get('missing_values', {}).values())}** · duplicate rows: **{q_stats.get('duplicate_rows', 0)}**\n"
            f"- Floor above building height: **{inc.get('floor_above_total_floors', 0):.1%}** of rows (flagged, kept)\n"
            f"- 'Under construction' but built ≤ 2020: **{inc.get('under_construction_built_2020_or_earlier', 0):.1%}**\n"
            f"- Villas/houses with > 4 floors: **{inc.get('house_or_villa_over_4_floors', 0):.1%}**\n"
            f"- Stored price/sqft rounded to 0.00: **{q_stats.get('price_per_sqft_rounded_to_zero', 0):,}** rows (recomputed in ₹)"
        )

analyses = st.session_state.get("analyses", [])
ui.section("Recent analyses", "This browser session only, not saved")
if analyses:
    st.dataframe(
        pd.DataFrame([{k: a[k] for k in ("time", "label", "price", "pps", "gap", "verdict")} for a in analyses]),
        hide_index=True, width="stretch",
        column_config={
            "time": "Time", "label": "Property",
            "price": st.column_config.NumberColumn("Price", format="₹%.1f L"),
            "pps": st.column_config.NumberColumn("₹/sqft", format="₹%,.0f"),
            "gap": st.column_config.NumberColumn("vs cohort median", format="%+.1f%%"),
            "verdict": "Assessment",
        },
    )
else:
    ui.empty_state("No analyses yet", "Properties you analyse in this session will appear here.")

ui.footer()
