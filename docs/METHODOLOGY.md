# Methodology, Assumptions & Limitations

## 1. Questions answered
1. **Is this listing priced below comparable properties?** A relative-value classification.
2. **What could it be worth in five years under stated assumptions?** A scenario valuation.

PropWise does **not** claim to predict investment profitability. The dataset contains no resale prices, rents,
returns or dates, so profitability cannot be learned or validated.

## 2. Good-value classification

**Rule (the label):** `good_value = 1` when ₹/sqft ≤ 1.0 × the training-split median ₹/sqft of the property's
**City × Property_Type × BHK** cohort.

- 630 cohorts, each with at least 165 training listings. If a cohort is ever smaller than 30 rows, the city
  median is used, then the global median.
- The median threshold gives a balanced label (≈50% positive), so accuracy is meaningful and no
  resampling is needed.
- Cohort medians are computed on the 70% training split only and then applied to validation (15%), test
  (15%) and user inputs. The split is random and stratified by city: rows are independent listings with no
  repeats and no time index, so group- or time-aware validation is not required.

**Models.** Logistic Regression (scaled), Random Forest and XGBoost, each in a pipeline:
`build_model_frame → [impute (+scale) | impute + one-hot] → model`. Two input tracks:

| Track | Inputs | Purpose |
|---|---|---|
| Listing | Every field incl. asking price | The app's model; measures rule reproduction |
| Attributes only | Every field except price | Measures what is learnable without the price |

The cohort median is never an input. Selection: highest validation ROC-AUC on the listing track; ties within
0.001 are broken by lower Brier score. **Result:** XGBoost, test ROC-AUC 0.9991, F1 0.984, ECE 0.0034.

**How to read this.** The label is arithmetic on price and area, so ≈0.999 AUC means "the model learned the
rule". It is not skill at picking investments. The attributes-only track reaches ≈0.78 AUC, driven almost entirely
by area: because price is independent of area here, bigger homes have lower ₹/sqft.

**In the app** the verdict is the exact rule. The model probability is shown only because test calibration
error is below 0.03, and any disagreement between rule and model is flagged as borderline.

**Explanations.** Factors come from the rule decomposition (gap to cohort median, gap to city median,
cohort percentile). Global permutation importance shows what the model uses, not what causes prices.
Context items (construction status, age, transport, amenities) are shown but not used in the verdict.

## 3. Five-year valuation

`Future value = current asking price × (1 + r)^5`

| Scenario | r | 5-year multiple |
|---|---|---|
| Conservative | 2% | ×1.104 |
| Baseline | 4% | ×1.217 |
| Optimistic | 6% | ×1.338 |

**Sources for the defaults:**
- [RBI All-India House Price Index, Q1 2026-27](https://rbi.org.in/SCRIPTS/BS_PressReleaseDisplay.aspx?prid=63436) (released 24 Aug 2026): 3.6% YoY
- [NHB RESIDEX, Q3 FY2025-26](https://www.nhb.org.in/wp-content/uploads/2026/03/National-Press-Release-RESIDEX-Dec25-DV002.pdf): 50-city HPI at assessment prices, 5.0% YoY

The baseline sits between these two readings and the other scenarios bracket it by ±2 points. These are recent
single readings of national indices, not forecasts. City-level readings vary widely (NHB, Dec 2025: −8.9% to
+22.8%), so city-specific rates were deliberately **not** extrapolated over five years. Users can set any rate
from −5% to 15%.

**Excluded:** stamp duty and registration, brokerage, maintenance, property tax, insurance, rental income,
vacancy, loan costs, inflation (figures are nominal) and capital-gains tax. A projected gain is not a net return.

**Why not a regression forecast?** A supervised five-year model needs observed future prices. Generating
labels with a growth formula and then "predicting" them would only re-learn the formula while reporting
meaningless accuracy. Instead, regressors were trained on **current** price from attributes (the valuation
signal test). All score R² ≈ 0, and the comparable-median rule scores R² ≈ −0.74. These results are logged
in MLflow and shown in the app.

## 4. Data handling
- Validation separates errors (missing columns, empty data, wrong types, non-positive price or size) from
  warnings (unknown categories, out-of-range values, internal inconsistencies).
- Inconsistent records are kept and flagged rather than deleted, because the wrong field cannot be identified.
- Price-per-sqft outliers (7.9% above the IQR fence) are genuine small homes and are kept. Medians and tree
  models are robust to them.
- Imputers stay in the pipeline even though there are no missing values, so future data with gaps still works.

## 5. Limitations
- The data appears synthetic, so no conclusion transfers to real Indian housing markets.
- No time dimension: no trends, seasonality or momentum.
- Locality IDs are anonymised and have about 12 rows each, too few for locality benchmarks.
- Legal title, RERA registration, builder track record, crime, infrastructure pipeline and micro-location are not
  captured.
- Out-of-range inputs (e.g. > 5,000 sqft) are allowed but flagged as extrapolation.

## 6. Financial disclaimer
PropWise AI is an educational analytics project. It is not investment, legal or tax advice, and it makes no
guarantee of returns. Verify any real property independently and consult a registered adviser before buying.
