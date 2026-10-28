"""Property Comparison."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features.engineering import RAW_INPUT_COLUMNS
from src.models.inference import InvalidPropertyInput, analyze_property
from src.ui import components as ui
from src.ui.state import MAX_COMPARE, add_to_compare, get_bundle, get_data, init_session
from src.utils import config as C
from src.visualization import charts as ch

df = get_data()
bundle = get_bundle()
init_session()

ui.page_header(
    "Property Comparison",
    "Line up to four listings under the same growth assumption. Differences are highlighted, "
    "and no single winner is declared, because the right choice depends on your goals.",
    eyebrow="Analyze",
)

with st.container(border=True):
    st.markdown("**Add properties**")
    a1, a2, a3 = st.columns([1.2, 1, 1.3], vertical_alignment="bottom")
    listing_id = a1.number_input("Dataset listing ID", 1, int(df["ID"].max()), 1, step=1,
                                 help=f"Any ID from 1 to {int(df['ID'].max()):,}. Browse IDs in Market Intelligence → Listings.")
    if a2.button("Add listing", icon=":material/add:", width="stretch"):
        row = df.loc[df["ID"] == listing_id]
        if row.empty:
            st.toast("No listing with that ID", icon=":material/error:")
        else:
            rec = row.iloc[0][RAW_INPUT_COLUMNS].to_dict()
            rec = {k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}
            label = f"#{listing_id} · {rec['BHK']} BHK {rec['Property_Type']}, {rec['City']}"
            if not add_to_compare(label, rec):
                st.toast("Already added, or the list is full (max 4)", icon=":material/info:")
    analyses = st.session_state.get("analyses", [])
    if analyses:
        options = {f"{a['time']} · {a['label']} · ₹{a['price']:,.0f} L": a for a in analyses}
        pick = a3.selectbox("Or from this session's analyses", list(options), index=None, placeholder="Choose an analysis")
        if pick and st.button("Add selected analysis", icon=":material/history:"):
            a = options[pick]
            if not add_to_compare(f"{a['label']} · ₹{a['price']:,.0f} L", a["record"]):
                st.toast("Already added, or the list is full (max 4)", icon=":material/info:")
    else:
        a3.caption("Analyses you run in the Investment Analyzer can be added here too.")

items = st.session_state["compare"]
top = st.columns([3, 1], vertical_alignment="bottom")
rate = top[0].slider("Shared annual growth assumption", C.MIN_RATE * 100, C.MAX_RATE * 100,
                     C.SCENARIO_RATES["Baseline"] * 100, 0.5, format="%.1f%%",
                     help="Applied to every property so projections are directly comparable.") / 100
if top[1].button("Clear list", icon=":material/delete:", width="stretch", disabled=not items):
    st.session_state["compare"] = []
    st.rerun()

if not items:
    ui.empty_state("Nothing to compare yet",
                   f"Add up to {MAX_COMPARE} properties by dataset ID, or from the Investment Analyzer.")
    ui.footer()
    st.stop()

results, labels = [], []
for it in items:
    try:
        results.append(analyze_property(it["record"], bundle, {"Shared": rate}))
        labels.append(it["label"])
    except InvalidPropertyInput as exc:
        ui.callout("danger", f"Could not analyse {it['label']}", bullets=exc.errors)

if not results:
    st.stop()

short = [f"P{i + 1}" for i in range(len(results))]
st.html("<div style='display:flex;flex-wrap:wrap;gap:8px;margin:.5rem 0'>" +
        "".join(ui.pill(f"{s} · {lab}") for s, lab in zip(short, labels)) + "</div>")

rows = {
    "Location": [f"{r.record['City']}, {r.record['State']}" for r in results],
    "Type": [f"{r.record['BHK']} BHK {r.record['Property_Type']}" for r in results],
    "Asking price (₹ L)": [r.record["Price_in_Lakhs"] for r in results],
    "Area (sqft)": [r.record["Size_in_SqFt"] for r in results],
    "₹ / sqft": [round(r.pps) for r in results],
    "Comparable median ₹/sqft": [round(r.benchmark_pps) for r in results],
    "Gap to median (%)": [round(r.gap_pct, 1) for r in results],
    "Assessment": [r.verdict for r in results],
    "Age (years)": [C.REFERENCE_YEAR - r.record["Year_Built"] for r in results],
    "Furnishing": [r.record["Furnished_Status"] for r in results],
    "Availability": [r.record["Availability_Status"].replace("_", " ") for r in results],
    "Amenities": [r.record["Amenities"] or "None" for r in results],
    "Transport access": [r.record["Public_Transport_Accessibility"] for r in results],
    f"Value in 5 yrs @ {rate:.1%} (₹ L)": [round(r.scenarios[0].future_value, 1) for r in results],
    "Projected gain (₹ L)": [round(r.scenarios[0].gain, 1) for r in results],
}
table = pd.DataFrame(rows, index=short).T
LOW_IS_NOTABLE = {"₹ / sqft", "Gap to median (%)", "Asking price (₹ L)"}


def _style(row: pd.Series):
    numeric = pd.to_numeric(row, errors="coerce")
    if numeric.isna().any() or numeric.nunique() <= 1:
        return [""] * len(row)
    lo, hi = numeric.min(), numeric.max()
    out = []
    for v in numeric:
        if v == lo:
            out.append("background-color:#e7f5ef;font-weight:600" if row.name in LOW_IS_NOTABLE else "font-weight:600")
        elif v == hi:
            out.append("background-color:#fff7e6" if row.name in LOW_IS_NOTABLE else "font-weight:600")
        else:
            out.append("")
    return out


ui.section("Side by side", "Green = lowest price / ₹ per sqft / gap; amber = highest. Bold marks the range ends of other numeric rows.")
FORMATS = {"Asking price (₹ L)": "₹{:,.1f} L", "Area (sqft)": "{:,.0f}", "₹ / sqft": "₹{:,.0f}",
           "Comparable median ₹/sqft": "₹{:,.0f}", "Gap to median (%)": "{:+.1f}%", "Age (years)": "{:.0f}",
           f"Value in 5 yrs @ {rate:.1%} (₹ L)": "₹{:,.1f} L", "Projected gain (₹ L)": "+₹{:,.1f} L"}
# Display an all-text table (Arrow-compatible: one dtype per column) and compute
# the highlights from the numeric values in `table`.
display = table.copy().astype(object)
for row, fmt in FORMATS.items():
    display.loc[row] = [fmt.format(v) for v in table.loc[row]]
display = display.astype(str)
styler = display.style.apply(lambda r: _style(table.loc[r.name]), axis=1)
st.dataframe(styler, width="stretch", height="content")

with ui.row("chart", "property_comparison-132"):
    c1, c2 = st.columns(2, gap="medium")
with c1:
    with st.container(border=True):
        fig = go.Figure([
            go.Bar(name="This listing", x=short, y=[r.pps for r in results],
                   marker=dict(color=ch.EMERALD, cornerradius=3),
                   hovertemplate="%{x}<br>Listing ₹%{y:,.0f}/sqft<extra></extra>"),
            go.Bar(name="Comparable median", x=short, y=[r.benchmark_pps for r in results],
                   marker=dict(color=ch.SLATE, cornerradius=3),
                   hovertemplate="%{x}<br>Comparable median ₹%{y:,.0f}/sqft<extra></extra>"),
        ])
        fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
        fig.update_yaxes(title="₹ per sqft", rangemode="tozero")
        st.plotly_chart(ch.finish(fig, "₹/sqft vs each listing's own comparables", 340), width="stretch",
                        config={"displaylogo": False})
with c2:
    with st.container(border=True):
        fig = go.Figure([
            go.Bar(name="Today", x=short, y=[r.record["Price_in_Lakhs"] for r in results],
                   marker=dict(color=ch.SLATE, cornerradius=3),
                   hovertemplate="%{x}<br>Today ₹%{y:,.1f} L<extra></extra>"),
            go.Bar(name=f"In 5 years @ {rate:.1%}", x=short, y=[r.scenarios[0].future_value for r in results],
                   marker=dict(color=ch.EMERALD, cornerradius=3),
                   hovertemplate="%{x}<br>Year 5 ₹%{y:,.1f} L<extra></extra>"),
        ])
        fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
        fig.update_yaxes(title="₹ lakhs", rangemode="tozero")
        st.plotly_chart(ch.finish(fig, "Value today and in five years (same assumption)", 340), width="stretch",
                        config={"displaylogo": False})

ui.callout("info", "Reading the comparison",
           "Identical growth rates mean projected gains scale with today's price, so they cannot separate the "
           "properties. The meaningful difference is each listing's price relative to its own comparables, "
           "plus the qualitative factors (age, availability, access) that matter to you.")

with st.expander("Remove a property"):
    for i, it in enumerate(list(items)):
        cols = st.columns([5, 1], vertical_alignment="center")
        cols[0].write(f"{short[i] if i < len(short) else ''} · {it['label']}")
        if cols[1].button("Remove", key=f"rm_{i}", icon=":material/close:"):
            st.session_state["compare"].pop(i)
            st.rerun()

ui.footer()
