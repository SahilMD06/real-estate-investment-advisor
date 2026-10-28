# PropWise AI — Real Estate Investment Advisor

**Evidence-first property value analytics for Indian residential real estate.**

**Live demo:** https://propwise-ai.streamlit.app · **Source:** https://github.com/SahilMD06/real-estate-investment-advisor

PropWise AI tells you whether a listing is priced below comparable properties and what it could be worth in five
years under transparent, sourced growth scenarios. Every number traces back to the dataset or a cited source.

Built with Python · pandas · scikit-learn · XGBoost · MLflow · Plotly · Streamlit.

> **Not financial advice.** This is an educational project built on a dataset that appears to be synthetic.
> See [Methodology](docs/METHODOLOGY.md) for assumptions and limitations.

---

## What it does

| Page | Purpose |
|---|---|
| **Executive Dashboard** | Dataset KPIs, price distribution, city comparison, data-quality checks, recent session analyses |
| **Investment Analyzer** | Multi-section form → value verdict vs comparables, calibrated model probability, five-year scenarios, factors, limitations, out-of-range warnings |
| **Property Comparison** | Up to four listings side by side under one shared growth assumption, with differences highlighted and no "winner" declared |
| **Market Intelligence** | Filterable state/city/locality, property-mix and amenity analysis with 95% CIs; searchable, downloadable listings table |
| **Model Insights** | Metrics for 6 classifiers and 5 regressors, ROC, calibration, confusion matrix, permutation importance, MLflow run IDs, training coverage |
| **Methodology & About** | Label definition, scenario method and sources, limitations, assumptions register, disclaimer |

## Key findings that shaped the design

Profiling `india_housing_prices.csv` (250,000 rows × 23 columns) showed:

1. **The data is clean but behaves like independent random columns.** Price is unrelated to size, location,
   BHK, amenities or transport (all |r| < 0.01). Learned valuation models score **R² ≈ 0** on held-out data.
2. **There are no outcomes and no time dimension.** Without resale prices, returns or listing dates, neither
   "profitability" nor a five-year price can be learned honestly.

So PropWise:

- **Classifies relative value with a documented rule.** "Good value" means ₹/sqft at or below the
  training-split median of the same City × Property type × BHK cohort. XGBoost, Random Forest and Logistic
  Regression are trained and tracked in MLflow. The champion reproduces the rule (test ROC-AUC 0.999) and is
  reported as exactly that, not as a profit predictor. An *attributes-only* track (price removed) shows what is
  learnable without the price (AUC ≈ 0.78, almost all from area).
- **Projects five-year values with explicit scenarios.** Future = Price × (1 + r)⁵, with Conservative 2%,
  Baseline 4% and Optimistic 6%. These bracket the latest RBI HPI (3.6%) and NHB RESIDEX (5.0%) all-India
  readings, and users can change them.
- **Logs the regression experiments as a "valuation signal test"**, reporting the null result openly.

Full numbers: [Model Evaluation Report (PDF)](reports/Model_Evaluation_Report.pdf) · EDA: [EDA Report (PDF)](reports/EDA_Report.pdf) · All docs: [Project Documentation (PDF)](reports/Project_Documentation.pdf)
(interactive: `reports/eda_report.html`).

---

## Quick start

Requires **Python 3.11–3.13** (tested on 3.13). Commands run from the project folder.

### Windows (PowerShell)
```powershell
cd C:\Labmentix\real-estate-investment-advisor
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
streamlit run app.py                    # app works immediately: data + model bundle are included
```

### macOS / Linux
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Open http://localhost:8501.

### Rebuild everything from the raw CSV
Put `india_housing_prices.csv` in `data/raw/`, then:

```bash
python -m playwright install chromium  # once: headless browser used to print the PDF reports
python -m scripts.run_pipeline      # = the five steps below
python -m scripts.prepare_data      # validate + clean → data/processed/housing_clean.parquet
python -m scripts.run_eda           # → reports/EDA_Report.pdf (+ interactive eda_report.html)
python -m scripts.train_models      # train 11 models, log to MLflow, register champion, write models/
python -m scripts.evaluate_models   # → reports/Model_Evaluation_Report.pdf (+ MLflow cross-check)
python -m scripts.build_docs_pdf    # → reports/Project_Documentation.pdf (all docs/ in one PDF)
pytest                              # 65 tests
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # browse experiments
```

Expected run time on a laptop: prepare ≈ 15 s, EDA ≈ 5 s, training ≈ 3 min.

---

## Project structure

```
real-estate-investment-advisor/
├── app.py                     # entry point: theme, navigation (st.navigation)
├── views/                     # the six pages (named views/, not pages/, so Streamlit's
│                              #   legacy auto-discovery cannot override st.navigation)
├── src/
│   ├── data/                  # load_data.py (load/clean/prepare), validate_data.py
│   ├── features/engineering.py# row features, build_model_frame, CohortStats (train-only)
│   ├── models/                # target.py, classification.py, regression.py, scenario.py,
│   │                          #   preprocessing.py, inference.py (the app's only model API)
│   ├── evaluation/metrics.py
│   ├── visualization/charts.py# Plotly chart library (validated palette)
│   ├── ui/                    # theme.py (CSS tokens), components.py, state.py (caching)
│   └── utils/                 # config.py (all paths/assumptions), logging.py
├── scripts/                   # prepare_data, run_eda, train_models, evaluate_models, run_pipeline
├── data/raw/                  # india_housing_prices.csv (git-ignored, 41 MB)
├── data/processed/            # housing_clean.parquet (9 MB) + data_quality_report.json
├── models/                    # classifier.joblib + classifier_xgb.json, cohort_stats.joblib, metadata.json, evaluation.json
├── reports/                   # EDA report (HTML + Markdown)
├── docs/                      # architecture, data dictionary, methodology, model report,
│                              #   deployment, demo script
├── tests/                     # pytest suite (unit + headless app tests)
├── assets/                    # logo
└── .streamlit/config.toml     # theme
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the data flow.

## Documentation

| Doc | Contents |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components, data flow, design decisions |
| [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) | Every raw and engineered field |
| [METHODOLOGY.md](docs/METHODOLOGY.md) | Label, models, scenarios, assumptions, limitations, disclaimer |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | GitHub setup, Streamlit Cloud, MLflow, troubleshooting |
| [DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) | 6–8 minute walkthrough script |

**Reports (PDF, generated by the pipeline)**

| Report | Contents |
|---|---|
| [EDA_Report.pdf](reports/EDA_Report.pdf) | 20 analyses with charts, findings and limitations |
| [Model_Evaluation_Report.pdf](reports/Model_Evaluation_Report.pdf) | Metrics, ROC/calibration/importance charts, regression signal test, scenarios, MLflow cross-check and screenshots |
| [Project_Documentation.pdf](reports/Project_Documentation.pdf) | The documents above combined for submission |

## Installed versions (tested)

streamlit 1.66.1 · pandas 3.1.0 · numpy 2.5.4 · scikit-learn 1.9.1 · xgboost 3.4.1 · plotly 7.1.0 ·
pyarrow 25.0.1 · joblib 1.6.0 · mlflow 3.17.2 · matplotlib 3.11.2 · pytest 9.1.1 · Python 3.13.

## License
MIT. See [LICENSE](LICENSE).
