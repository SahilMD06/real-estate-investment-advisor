"""Property Investment Analyzer."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.models.inference import InvalidPropertyInput, analyze_property
from src.models.scenario import trajectory
from src.ui import components as ui
from src.ui.state import add_to_compare, get_bundle, get_data, localities_for, remember_analysis
from src.utils import config as C
from src.visualization import charts as ch

df = get_data()
bundle = get_bundle()
city_state = bundle.city_state
states = sorted(set(city_state.values()))

ui.page_header(
    "Investment Analyzer",
    "Enter a listing's details. PropWise compares its ₹/sqft with similar properties (same city, type and BHK) "
    "and projects its value under three transparent five-year growth scenarios.",
    eyebrow="Analyze",
)

# --- Location (outside the form so city options follow the chosen state) ---
with st.container(border=True):
    st.markdown("**1 · Location**")
    l1, l2, l3 = st.columns(3)
    state = l1.selectbox("State", states, index=states.index("Karnataka"), key="az_state",
                         help="20 states are covered by the training data.")
    cities = sorted(c for c, s in city_state.items() if s == state)
    city = l2.selectbox("City", cities, key=f"az_city_{state}",
                        help="Only cities in the selected state are listed.")
    locs = ["Not specified"] + localities_for(city)
    locality = l3.selectbox("Locality (optional)", locs, key=f"az_loc_{city}",
                            help="Locality IDs are anonymised in the source data. They are recorded for reference "
                                 "only and are not used in the assessment (about 12 listings per locality is "
                                 "too few for a reliable benchmark).")

with st.form("analyzer", border=False):
    with st.container(border=True):
        st.markdown("**2 · Property & price**")
        a1, a2, a3, a4, a5 = st.columns(5)
        ptype = a1.selectbox("Property type", C.CATEGORY_VALUES["Property_Type"])
        bhk = a2.number_input("BHK", 1, 10, 3, step=1, help="Bedrooms. Training data covers 1–5.")
        size = a3.number_input("Built-up area (sqft)", 100, 20000, 1800, step=50,
                               help="Training data covers 500–5,000 sqft.")
        price = a4.number_input("Asking price (₹ lakhs)", 1.0, 5000.0, 140.0, step=5.0, format="%.1f",
                                help="1 lakh = ₹1,00,000. Training data covers ₹10–500 lakhs.")
        year = a5.number_input("Year built", 1950, C.REFERENCE_YEAR + 5, 2015, step=1)

    with st.container(border=True):
        st.markdown("**3 · Building & listing**")
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        floor = b1.number_input("Floor", 0, 100, 6, step=1, help="0 = ground floor.")
        total = b2.number_input("Total floors", 1, 100, 14, step=1)
        furnished = b3.selectbox("Furnishing", C.CATEGORY_VALUES["Furnished_Status"], index=1)
        avail = b4.selectbox("Availability", C.CATEGORY_VALUES["Availability_Status"],
                             format_func=lambda s: s.replace("_", " "))
        facing = b5.selectbox("Facing", C.CATEGORY_VALUES["Facing"])
        owner = b6.selectbox("Listed by", C.CATEGORY_VALUES["Owner_Type"], index=2)

    with st.container(border=True):
        st.markdown("**4 · Amenities & access**")
        c1, c2, c3, c4, c5, c6 = st.columns([2.2, 1, 1, 1, 1, 1])
        amenities = c1.multiselect("Amenities", list(C.AMENITY_TOKENS), default=["Gym", "Pool"])
        parking = c2.selectbox("Parking", C.CATEGORY_VALUES["Parking_Space"], index=1)
        security = c3.selectbox("Security", C.CATEGORY_VALUES["Security"], index=1)
        transport = c4.selectbox("Public transport", C.CATEGORY_VALUES["Public_Transport_Accessibility"], index=2)
        schools = c5.number_input("Nearby schools", 1, 10, 7, help="Count/score as recorded in the dataset (1–10).")
        hospitals = c6.number_input("Nearby hospitals", 1, 10, 4, help="Count/score as recorded in the dataset (1–10).")

    with st.expander("Five-year growth assumptions (annual, nominal)", icon=":material/tune:"):
        st.caption(C.RATE_RATIONALE)
        r1, r2, r3 = st.columns(3)
        rates = {
            name: col.number_input(f"{name} (% per year)", C.MIN_RATE * 100, C.MAX_RATE * 100,
                                   C.SCENARIO_RATES[name] * 100, step=0.5, format="%.1f") / 100
            for col, name in zip((r1, r2, r3), C.SCENARIO_RATES)
        }

    submitted = st.form_submit_button("Analyze property", type="primary", icon=":material/query_stats:",
                                      width="stretch")

if submitted:
    record = {
        "State": state, "City": city, "Locality": locality if locality != "Not specified" else "Unspecified",
        "Property_Type": ptype, "BHK": bhk, "Size_in_SqFt": size, "Price_in_Lakhs": price, "Year_Built": year,
        "Furnished_Status": furnished, "Floor_No": floor, "Total_Floors": total, "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals, "Public_Transport_Accessibility": transport, "Parking_Space": parking,
        "Security": security, "Amenities": ", ".join(amenities), "Facing": facing, "Owner_Type": owner,
        "Availability_Status": avail,
    }
    try:
        with st.spinner("Analysing against comparable properties…"):
            result = analyze_property(record, bundle, rates)
        st.session_state["az_result"] = (record, result, rates)
        remember_analysis(record, result)
    except InvalidPropertyInput as exc:
        st.session_state.pop("az_result", None)
        ui.callout("danger", "Please fix the following before analysing", bullets=exc.errors)
    except ValueError as exc:
        st.session_state.pop("az_result", None)
        ui.callout("danger", "Invalid assumption", str(exc))

if "az_result" not in st.session_state:
    if not submitted:
        ui.empty_state("No analysis yet", "Complete the form and press Analyze property to see the assessment.")
    ui.footer()
    st.stop()

record, res, rates = st.session_state["az_result"]
meta = bundle.metadata["classifier"]

# --- Results ---------------------------------------------------------------
ui.section("Assessment", f"{record['BHK']} BHK {record['Property_Type']} · {record['City']}, {record['State']}")
good = res.rule_label == 1
detail = (f"₹{res.pps:,.0f}/sqft vs a comparable median of ₹{res.benchmark_pps:,.0f}/sqft "
          f"({res.gap_pct:+.1f}%). Comparables: {record['BHK']} BHK {record['Property_Type'].lower()}s in "
          f"{record['City']} (n = {res.cohort_n} training listings).")
ui.verdict(good, res.verdict, detail)

if res.warnings:
    ui.callout("warn", "Check these inputs", bullets=res.warnings)
if res.borderline:
    ui.callout("neutral", "Borderline result",
               "The price is within 5% of the comparable median, so small changes in price or area could flip the assessment.")

prob_value = f"{res.model_probability:.1%}" if res.show_probability else "Not shown"
prob_sub = (f"Calibrated on test data (ECE {meta['test_ece']:.3f})" if res.show_probability
            else "Calibration not validated")
pct_sub = f"Percentile {res.cohort_percentile:.0f} of comparables" if res.cohort_percentile is not None else "Cohort too small for a percentile"
ui.kpi_row([
    {"label": "Asking ₹ / sqft", "value": ui.inr(res.pps), "sub": f"{ui.lakhs(record['Price_in_Lakhs'])} ÷ {record['Size_in_SqFt']:,} sqft"},
    {"label": "Comparable median", "value": ui.inr(res.benchmark_pps), "sub": f"{res.benchmark_level.title()}-level benchmark"},
    {"label": "Gap to median", "value": f"{res.gap_pct:+.1f}%", "sub": pct_sub, "tone": "good" if good else "bad"},
    {"label": "Model probability", "value": prob_value, "sub": prob_sub},
    {"label": "Rule vs model", "value": "Agree" if res.models_agree else "Disagree",
     "sub": "Model reproduces the rule" if res.models_agree else "Near the threshold, so treat as borderline"},
])

tab_val, tab_scen, tab_fac, tab_lim = st.tabs(
    ["Value vs comparables", "Five-year scenarios", "Factors", "Limitations"]
)

with tab_val:
    key = (record["City"], record["Property_Type"], int(record["BHK"]))
    values = bundle.cohort_stats.cohort_values.get(key)
    with ui.row("chart", "investment_analyzer-150"):
        left, right = st.columns([1.6, 1], gap="medium")
    with left:
        if values is not None:
            st.plotly_chart(
                ch.cohort_position(values, res.pps, res.benchmark_pps,
                                   f"Position among {len(values)} comparables"),
                width="stretch", config={"displaylogo": False})
        else:
            ui.empty_state("No comparable distribution", "This combination is outside the training cohorts; a broader benchmark was used.")
    with right:
        reg = bundle.evaluation["regression"]["results"]["cohort_median_benchmark"]["test"]
        st.markdown("**Price implied by the comparable median**")
        st.metric("Median ₹/sqft × your area", ui.lakhs(res.benchmark_value_lakhs),
                  delta=f"{record['Price_in_Lakhs'] - res.benchmark_value_lakhs:+,.1f} L asking vs implied",
                  delta_color="inverse")
        st.caption(
            f"A reference point, **not a valuation**. As a price predictor it scores R² = {reg['r2']:.2f} on held-out "
            "listings, worse than guessing the average, because price is unrelated to size in this data. "
            "No learned valuation model beat the average either (see Model Insights)."
        )
        if res.city_median_pps:
            st.markdown(f"**{record['City']} city median:** ₹{res.city_median_pps:,.0f}/sqft")

with tab_scen:
    ui.scenario_cards(res.scenarios, ch.CATEGORICAL)
    st.plotly_chart(ch.scenario_trajectory(trajectory(record["Price_in_Lakhs"], rates)), width="stretch",
                    config={"displaylogo": False})
    st.caption(
        f"Future value = asking price × (1 + annual rate)^{C.HORIZON_YEARS}. Rates are assumptions anchored to "
        "recent national house-price indices (see Methodology), not a forecast for this property or city."
    )
    ui.callout("neutral", "Not included in these figures", bullets=[
        "Stamp duty, registration and brokerage on purchase or sale",
        "Maintenance, property tax and insurance",
        "Rental income and vacancy",
        "Loan interest and financing costs",
        "Inflation adjustment (figures are nominal) and capital-gains tax",
    ])

with tab_fac:
    with ui.row("card", "investment_analyzer-190"):
        f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        st.markdown("**Supports value**")
        ui.factor_list(res.positives, "up")
    with f2:
        st.markdown("**Weighs against value**")
        ui.factor_list(res.negatives, "down")
    with f3:
        st.markdown("**Context (not used in the verdict)**")
        ui.factor_list(res.context, "dot")
    imp = bundle.evaluation["classification"]["importance"][meta["champion"]]
    st.plotly_chart(ch.importance_chart(imp, "What the classifier relies on overall (permutation importance)", top=8),
                    width="stretch", config={"displaylogo": False})
    st.caption("Importance shows which inputs the model uses to reproduce the rule. It is not evidence that "
               "these features cause price differences.")

with tab_lim:
    ui.callout("warn", "What this assessment is, and what it isn't", bullets=[
        "‘Good value’ means the asking ₹/sqft is at or below the median of comparable training listings. "
        "It is a documented rule, not a measured investment return.",
        "The classifier is trained to reproduce that rule (test ROC-AUC "
        f"{bundle.evaluation['classification']['results'][meta['champion']]['test']['roc_auc']:.3f}). "
        "Its accuracy shows how well it copies the rule, not that it predicts profit.",
        "Because price is independent of size in this data, larger homes tend to look cheaper per sqft.",
        "Five-year values are scenario arithmetic with assumed growth rates. No historical price series exists in the data.",
        "The dataset appears synthetic. Do not use these results for real purchase decisions.",
    ])

b1, b2 = st.columns([1, 3])
if b1.button("Add to comparison", icon=":material/add:", width="stretch"):
    label = f"{record['BHK']} BHK {record['Property_Type']}, {record['City']} · ₹{record['Price_in_Lakhs']:,.0f} L"
    if add_to_compare(label, record):
        st.toast("Added to comparison", icon=":material/check:")
    else:
        st.toast("Already in the list, or the list is full (max 4)", icon=":material/info:")
b2.page_link("views/property_comparison.py", label="Go to Property Comparison", icon=":material/compare_arrows:")

ui.footer()
