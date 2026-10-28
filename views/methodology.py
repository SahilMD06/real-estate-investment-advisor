"""Methodology & About."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui import components as ui
from src.ui.state import get_bundle
from src.utils import config as C

bundle = get_bundle()
meta = bundle.metadata

ui.page_header(
    "Methodology & About",
    "How PropWise reaches its assessments, what it assumes, and where it should not be trusted.",
    eyebrow="Reference",
)

ui.callout("danger", "Financial disclaimer",
           "PropWise AI is an educational analytics project. It is not investment, legal or tax advice and makes no "
           "guarantee of returns. The underlying dataset appears to be synthetic. Verify any real property "
           "independently (title, RERA registration, approvals, comparable sales) and consult a registered "
           "adviser before buying.")

t1, t2, t3, t4, t5 = st.tabs(["Good-value assessment", "Five-year valuation", "Data & limitations",
                              "Assumptions register", "About"])

with t1:
    st.markdown(f"""
#### What question is answered?
**Is this listing priced below comparable properties?** The dataset contains no investment outcomes (no resale
prices, rental yields or holding-period returns), so profitability cannot be learned. PropWise answers this narrower,
verifiable question instead.

#### The rule
A property is labelled **good value** when

> ₹ per sqft ≤ {C.GOOD_VALUE_RATIO:g} × median ₹ per sqft of its comparable cohort

- **Cohort:** same **City × Property type × BHK** ({meta['cohorts']['n']} cohorts, each with at least
  {meta['cohorts']['min_train_size']} training listings).
- **Fallback:** if a cohort has fewer than {C.MIN_COHORT_SIZE} training listings, the city median is used, then the
  national median.
- **Leakage control:** medians come from the **training split only** and are then applied to validation, test and
  user inputs. The cohort median itself is never a model input.
- **Threshold:** the median splits each cohort roughly 50/50 (train {meta['label']['positive_rates']['train']:.1%}
  positive), so there is no class imbalance.

#### The classifier
Logistic Regression, Random Forest and XGBoost were trained on two input sets and logged to MLflow:

| Track | Inputs | What it shows |
|---|---|---|
| Listing | all property fields **including price** | how faithfully a model reproduces the rule (≈ 0.999 AUC) |
| Attributes only | all fields **except price** | how much is predictable without the price (≈ 0.78 AUC, mostly area) |

The champion (**{ {'xgboost': 'XGBoost', 'random_forest': 'Random Forest', 'logistic_regression': 'Logistic Regression'}.get(meta['classifier']['champion'].split('/')[1], meta['classifier']['champion']) }**) was chosen on validation
ROC-AUC, with ties broken by Brier score. Its probability is displayed because its test calibration error (ECE
{meta['classifier']['test_ece']:.4f}) is below the {C.MAX_DISPLAY_ECE} threshold. **The verdict shown to users is the
exact rule.** The model's probability is a cross-check, and any disagreement is flagged as borderline.

#### Explanations
Positive and negative factors come from the rule itself (gap to cohort median, city median, cohort percentile).
Global feature importance uses permutation importance on held-out data. These are statements about the models,
not causal claims about prices.
""")

with t2:
    rows = pd.DataFrame([{"Scenario": k, "Annual rate": f"{v:.1%}",
                          "5-year multiple": f"×{(1 + v) ** C.HORIZON_YEARS:.3f}",
                          "Total appreciation": f"{((1 + v) ** C.HORIZON_YEARS - 1):.1%}"}
                         for k, v in C.SCENARIO_RATES.items()])
    st.markdown(f"""
#### Why scenarios instead of a forecast model
The dataset is a **single snapshot**: no listing dates, no repeat sales, no price history. A supervised model of
"price in five years" would need observed future prices to learn from and validate against. Inventing those labels
(for example by applying a growth rate and then "predicting" it) would only re-learn the made-up formula, while
reporting impressive but meaningless accuracy.

PropWise therefore uses transparent compounding:

> **Future value = current price × (1 + r)<sup>{C.HORIZON_YEARS}</sup>**
""", unsafe_allow_html=True)
    st.dataframe(rows, hide_index=True, width="stretch")
    st.markdown("**Where the default rates come from**")
    for src in C.RATE_SOURCES:
        st.markdown(f"- [{src['name']}]({src['url']}): **{src['value']}**")
    st.caption(C.RATE_RATIONALE + " Users can change every rate in the analyzer.")
    st.markdown("""
#### Excluded from projections
Stamp duty and registration charges (rates vary by state), brokerage, maintenance,
property tax, rental income, vacancy, financing costs, inflation (all figures are nominal) and capital-gains tax.
Because these are excluded, a projected gain is **not** a net return.

#### Valuation signal test (regression)
To check whether a learned valuation model could support the analysis, Mean, Ridge, Random Forest and XGBoost
regressors were trained to predict **current** price from property attributes. All reach R² ≈ 0 on held-out data,
and the comparable-median rule scores worse than the mean. Every result is logged in MLflow and shown in Model Insights.
""")

with t3:
    with ui.row("chart", "methodology-103"):
        q = st.columns(2)
    with q[0]:
        st.markdown("""
#### Dataset
`india_housing_prices.csv`: 250,000 listings, 23 columns, 20 states, 42 cities, 500 anonymised locality IDs
(which repeat in every city, giving 20,999 City–Locality pairs).

#### Cleaning
- `Price_per_SqFt` is stored in lakhs/sqft rounded to 2 dp (2,534 rows show 0.00), so it is **recomputed in ₹**.
- Amenities (325 strings, 31 actual sets) are parsed into five flags plus a count.
- No missing values or duplicates were found. The pipeline still imputes (median / most frequent) so that
  future data with gaps works.
- Outliers in ₹/sqft (7.9% above the IQR fence) are genuine small-home listings and are kept.
""")
    with q[1]:
        st.markdown("""
#### Known limitations
- **Synthetic-looking data:** every column behaves like an independent random draw, so prices have no relationship
  with location, size or amenities. Real-market conclusions must not be drawn.
- **Internal inconsistencies** are flagged, not deleted, because the faulty field cannot be identified: floor above
  building height (46%), "under construction" but built ≤ 2020 (46%), villas/houses with > 4 floors (58%).
- **Size effect:** large homes look cheap per sqft by construction.
- **No time dimension:** no trends, seasonality or price momentum can be analysed.
- **Not captured:** legal title, RERA status, crime, infrastructure projects, builder reputation. The project brief
  mentions these, but they are not in the data and were not invented.
""")

with t4:
    st.dataframe(pd.DataFrame([
        {"Assumption": "Age reference year", "Value": str(C.REFERENCE_YEAR), "Why": "Age = 2025 − Year Built in every row of the source data"},
        {"Assumption": "Comparable cohort", "Value": " × ".join(C.COHORT_KEYS), "Why": "Like-for-like comparison with ≥ 165 training rows per cohort"},
        {"Assumption": "Good-value threshold", "Value": f"≤ {C.GOOD_VALUE_RATIO:g} × cohort median", "Why": "Median is robust to the ₹/sqft right tail"},
        {"Assumption": "Minimum cohort size", "Value": str(C.MIN_COHORT_SIZE), "Why": "Below this, fall back to city/global median"},
        {"Assumption": "Probability display gate", "Value": f"ECE ≤ {C.MAX_DISPLAY_ECE}", "Why": "Only show probabilities that are calibrated on test data"},
        {"Assumption": "Scenario rates", "Value": ", ".join(f"{k} {v:.0%}" for k, v in C.SCENARIO_RATES.items()), "Why": "Bracket the latest RBI (3.6%) and NHB (5.0%) all-India readings"},
        {"Assumption": "Horizon", "Value": f"{C.HORIZON_YEARS} years", "Why": "Project brief"},
        {"Assumption": "Split", "Value": "70/15/15, stratified by city, seed 42", "Why": "Independent rows; reproducible"},
        {"Assumption": "Infrastructure index", "Value": "Equal-weight schools, hospitals, transport (0–1)", "Why": "Descriptive only; weights are not validated"},
    ]), hide_index=True, width="stretch")

with t5:
    st.markdown(f"""
#### {C.APP_NAME}
{C.APP_TAGLINE}. Built with Python, pandas, scikit-learn, XGBoost, MLflow, Plotly and Streamlit.

| Component | Version |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in meta["versions"].items()) + f"""

Model bundle created **{meta['created_at']}** · registered model
`{(meta['classifier'].get('registry') or {}).get('name', 'n/a')}` v{(meta['classifier'].get('registry') or {}).get('version', '–')}.

Source code, tests and documentation: see the project README.
""")

ui.footer()
