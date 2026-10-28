"""Market Intelligence."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui import components as ui
from src.ui.state import get_data
from src.utils import config as C
from src.visualization import charts as ch

df = get_data()
CFG = {"displaylogo": False}

ui.page_header(
    "Market Intelligence",
    "Slice the dataset by location, property type, size and price. Charts show means with 95% confidence "
    "intervals, so you can tell real differences from noise.",
    eyebrow="Explore",
)

with st.container(border=True):
    f1, f2, f3, f4 = st.columns([1.3, 1.3, 1.1, 1])
    states = f1.multiselect("States", sorted(df["State"].unique()), placeholder="All states")
    city_pool = sorted(df.loc[df["State"].isin(states), "City"].unique()) if states else sorted(df["City"].unique())
    cities = f2.multiselect("Cities", city_pool, placeholder="All cities in selection")
    types = f3.multiselect("Property type", list(C.CATEGORY_VALUES["Property_Type"]), placeholder="All types")
    bhks = f4.multiselect("BHK", [1, 2, 3, 4, 5], placeholder="All")
    s1, s2 = st.columns(2)
    price_rng = s1.slider("Price (₹ lakhs)", 10.0, 500.0, (10.0, 500.0), 5.0)
    size_rng = s2.slider("Area (sqft)", 500, 5000, (500, 5000), 100)


@st.cache_data(show_spinner=False)
def apply_filters(states, cities, types, bhks, price_rng, size_rng) -> pd.DataFrame:
    m = df["Price_in_Lakhs"].between(*price_rng) & df["Size_in_SqFt"].between(*size_rng)
    if states:
        m &= df["State"].isin(states)
    if cities:
        m &= df["City"].isin(cities)
    if types:
        m &= df["Property_Type"].isin(types)
    if bhks:
        m &= df["BHK"].isin(bhks)
    return df.loc[m]


view = apply_filters(tuple(states), tuple(cities), tuple(types), tuple(bhks), price_rng, size_rng)

if view.empty:
    ui.empty_state("No listings match these filters", "Widen the price or size range, or clear a location filter.")
    ui.footer()
    st.stop()

ui.kpi_row([
    {"label": "Matching listings", "value": f"{len(view):,}", "sub": f"{len(view) / len(df):.1%} of dataset"},
    {"label": "Median price", "value": ui.lakhs(view["Price_in_Lakhs"].median())},
    {"label": "Median ₹ / sqft", "value": ui.inr(view["Price_per_SqFt_INR"].median())},
    {"label": "Median area", "value": f"{view['Size_in_SqFt'].median():,.0f} sqft"},
    {"label": "Cities in view", "value": f"{view['City'].nunique()}", "sub": f"{view['State'].nunique()} states"},
])
if len(view) < 300:
    ui.callout("warn", "Small selection",
               f"Only {len(view):,} listings match. Group averages below will be noisy, so check the confidence intervals.")

t_loc, t_mix, t_amen, t_table = st.tabs(["Location", "Property mix", "Amenities & access", "Listings"])

with t_loc:
    with ui.row("chart", "market_intelligence-69"):
        c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True):
        order = view.groupby("State")["Price_per_SqFt_INR"].mean().sort_values(ascending=False).index.tolist()
        st.plotly_chart(ch.bar_mean_ci(view, "State", "Price_per_SqFt_INR", "Average ₹/sqft by state", "₹ per sqft",
                                       order=order, horizontal=True, height=max(300, 26 * len(order) + 90)),
                        width="stretch", config=CFG)
    with c2, st.container(border=True):
        g = view.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False)
        shown = g.head(20)
        st.plotly_chart(ch.bar_mean_ci(view[view["City"].isin(shown.index)], "City", "Price_in_Lakhs",
                                       f"Average price by city{' (top 20)' if len(g) > 20 else ''}", "₹ lakhs",
                                       order=shown.index.tolist(), horizontal=True, fmt=",.1f",
                                       height=max(300, 26 * len(shown) + 90)),
                        width="stretch", config=CFG)
    with st.container(border=True):
        loc = view.groupby(["City", "Locality"])["Price_per_SqFt_INR"].agg(["median", "size"])
        loc = loc[loc["size"] >= 10].sort_values("median", ascending=False).head(12)
        if loc.empty:
            st.caption("Not enough listings per locality (need ≥ 10) to rank localities in this selection.")
        else:
            st.plotly_chart(ch.ranked_bar(pd.Series([f"{c} / {l}" for c, l in loc.index]), loc["median"],
                                          "Highest median ₹/sqft localities (≥ 10 listings)", "Median ₹ per sqft",
                                          customdata=loc["size"].values,
                                          hover="%{y}<br><b>₹%{x:,.0f}/sqft</b><br>n = %{customdata}<extra></extra>"),
                            width="stretch", config=CFG)
            st.caption("With about 12 listings per locality, these rankings mostly reflect sampling noise; "
                       "a random shuffle of prices produces the same spread (see EDA report §10).")

with t_mix:
    with ui.row("chart", "market_intelligence-98"):
        c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True):
        st.plotly_chart(ch.box_from_quantiles(view, "Property_Type", "Price_per_SqFt_INR", "₹/sqft by property type",
                                              "₹ per sqft", order=[t for t in C.CATEGORY_VALUES["Property_Type"]
                                                                    if t in set(view["Property_Type"])]),
                        width="stretch", config=CFG)
    with c2, st.container(border=True):
        from src.features.engineering import SIZE_BAND_LABELS
        st.plotly_chart(ch.bar_mean_ci(view, "Size_Band", "Price_per_SqFt_INR", "₹/sqft by size band (sqft)",
                                       "₹ per sqft", order=[b for b in SIZE_BAND_LABELS if b in set(view["Size_Band"])]),
                        width="stretch", config=CFG)
        st.caption("Smaller homes show far higher ₹/sqft only because price does not rise with size in this data.")
    with st.container(border=True):
        tab = pd.crosstab(view["City"], view["BHK"], normalize="index") * 100
        fig = ch.heatmap(tab.sort_index(), "BHK mix by city (% of each city's listings)", "% of city", fmt=".0f",
                         height=max(260, 22 * len(tab) + 120))
        fig.update_xaxes(title="BHK")
        st.plotly_chart(fig, width="stretch", config=CFG)

with t_amen:
    with ui.row("chart", "market_intelligence-118"):
        c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True):
        st.plotly_chart(ch.amenities_effect(view), width="stretch", config=CFG)
    with c2, st.container(border=True):
        st.plotly_chart(ch.bar_mean_ci(view, "Public_Transport_Accessibility", "Price_per_SqFt_INR",
                                       "₹/sqft by public transport access", "₹ per sqft",
                                       order=[t for t in ("Low", "Medium", "High")
                                              if t in set(view["Public_Transport_Accessibility"])]),
                        width="stretch", config=CFG)
    with ui.row("chart", "market_intelligence-127"):
        c3, c4 = st.columns(2, gap="medium")
    with c3, st.container(border=True):
        st.plotly_chart(ch.line_mean_ci(view, "Nearby_Schools", "Price_per_SqFt_INR", "₹/sqft by nearby schools",
                                        "Nearby schools", "₹ per sqft"), width="stretch", config=CFG)
    with c4, st.container(border=True):
        st.plotly_chart(ch.bar_mean_ci(view, "Parking_Space", "Price_in_Lakhs", "Average price by parking",
                                       "₹ lakhs", order=[p for p in ("No", "Yes") if p in set(view["Parking_Space"])],
                                       fmt=",.1f"), width="stretch", config=CFG)
    ui.callout("neutral", "Association, not causation",
               "None of these features shows a measurable price effect in this dataset. In real markets, transport "
               "access and amenities often matter, so treat these charts as a description of this data only.")

with t_table:
    q1, q2 = st.columns([2, 1], vertical_alignment="bottom")
    query = q1.text_input("Search by listing ID or locality", placeholder="e.g. 1042 or Locality_84")
    sort_by = q2.selectbox("Sort by", ["₹/sqft (low → high)", "₹/sqft (high → low)", "Price (low → high)",
                                       "Price (high → low)", "Area (large → small)"])
    t = view
    if query.strip():
        qs = query.strip()
        t = t[t["ID"].astype(str).eq(qs) | t["Locality"].str.contains(qs, case=False, regex=False)]
    key, asc = {"₹/sqft (low → high)": ("Price_per_SqFt_INR", True), "₹/sqft (high → low)": ("Price_per_SqFt_INR", False),
                "Price (low → high)": ("Price_in_Lakhs", True), "Price (high → low)": ("Price_in_Lakhs", False),
                "Area (large → small)": ("Size_in_SqFt", False)}[sort_by]
    t = t.sort_values(key, ascending=asc)
    cols = ["ID", "City", "State", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
            "Price_per_SqFt_INR", "Year_Built", "Furnished_Status", "Availability_Status", "amenity_count",
            "Public_Transport_Accessibility", "n_quality_flags"]
    if t.empty:
        ui.empty_state("No matching listings", "Try another ID or locality name.")
    else:
        limit = 2000
        st.caption(f"Showing {min(limit, len(t)):,} of {len(t):,} matching listings. Click a column header to re-sort. "
                   "Use a listing ID in Property Comparison.")
        st.dataframe(
            t[cols].head(limit), hide_index=True, width="stretch", height=480,
            column_config={
                "ID": st.column_config.NumberColumn("ID", format="%d"),
                "Property_Type": "Type",
                "Size_in_SqFt": st.column_config.NumberColumn("Area (sqft)", format="%d"),
                "Price_in_Lakhs": st.column_config.NumberColumn("Price (₹ L)", format="%.1f"),
                "Price_per_SqFt_INR": st.column_config.NumberColumn("₹ / sqft", format="%d"),
                "Year_Built": st.column_config.NumberColumn("Built", format="%d"),
                "Furnished_Status": "Furnishing",
                "Availability_Status": "Availability",
                "amenity_count": st.column_config.ProgressColumn("Amenities", min_value=0, max_value=5, format="%d"),
                "Public_Transport_Accessibility": "Transport",
                "n_quality_flags": st.column_config.NumberColumn("Data flags", help="Number of internal inconsistencies detected (0–3)"),
            },
        )
        st.download_button("Download filtered listings (CSV)", t[cols].to_csv(index=False).encode(),
                           "propwise_filtered_listings.csv", "text/csv", icon=":material/download:")

ui.footer()
