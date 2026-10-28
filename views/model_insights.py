"""Model Insights — reads the evaluation bundle written by scripts/train_models.py.

No MLflow server is needed: the app relies on models/evaluation.json and
models/metadata.json, which record the run IDs and registry version.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import components as ui
from src.ui.state import get_bundle
from src.visualization import charts as ch

bundle = get_bundle()
meta, ev = bundle.metadata, bundle.evaluation
clf, reg = ev["classification"], ev["regression"]
champ = meta["classifier"]["champion"]
champ_res = clf["results"][champ]
CFG = {"displaylogo": False}

PRETTY = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost",
          "mean_baseline": "Mean baseline", "ridge": "Ridge", "cohort_median_benchmark": "Comparable-median rule"}


def pretty(key: str) -> str:
    track, _, name = key.partition("/")
    return PRETTY.get(name or track, key)


ui.page_header(
    "Model Insights",
    "How the models were evaluated, why their scores mean what they mean, and where every number comes from.",
    eyebrow="Explore",
)

reg_info = meta["classifier"].get("registry") or {}
ui.kpi_row([
    {"label": "Champion classifier", "value": pretty(champ), "sub": "Listing track"},
    {"label": "Test ROC-AUC", "value": f"{champ_res['test']['roc_auc']:.4f}", "sub": f"F1 {champ_res['test']['f1']:.3f}"},
    {"label": "Calibration (ECE)", "value": f"{meta['classifier']['test_ece']:.4f}",
     "sub": "Probabilities shown" if meta["classifier"]["show_probability"] else "Probabilities hidden"},
    {"label": "Best valuation R²", "value": f"{reg['results'][reg['diagnostics']['best_learned_model']]['test']['r2']:+.4f}",
     "sub": "No price signal found"},
    {"label": "Registry", "value": f"v{reg_info.get('version', '–')}", "sub": f"alias '{reg_info.get('alias', '–')}'"},
])

ui.callout(
    "info", "Why the classifier scores near 100%, and why that is not a profitability claim",
    "The ‘good value’ label is defined by a rule: ₹/sqft at or below the comparable median. With price and area "
    "as inputs, the model only has to learn that division, so near-perfect scores are expected. The "
    "attributes-only track removes price and shows what is learnable without it (≈0.78 AUC, almost all from area).",
)

t_clf, t_reg, t_mlf, t_cov = st.tabs(["Classification", "Valuation signal test", "Experiment tracking", "Training coverage"])

with t_clf:
    rows = []
    for key, r in clf["results"].items():
        rows.append({"Track": r["track"].replace("_", " "), "Model": pretty(key),
                     "Val ROC-AUC": r["val"]["roc_auc"], "Test ROC-AUC": r["test"]["roc_auc"],
                     "Precision": r["test"]["precision"], "Recall": r["test"]["recall"], "F1": r["test"]["f1"],
                     "Accuracy": r["test"]["accuracy"], "Brier": r["test"]["brier"], "ECE": r["test"]["ece"],
                     "Fit (s)": r["fit_seconds"], "Champion": "★" if key == champ else ""})
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f") for c in
                                ["Val ROC-AUC", "Test ROC-AUC", "Precision", "Recall", "F1", "Accuracy", "Brier", "ECE"]})
    st.caption(f"Selection rule: {clf['selection']['criterion']}. Test-split metrics are reported but were not "
               f"used for selection. Label share 'good value': train {clf['selection']['label_rates']['train']:.1%}, "
               f"test {clf['selection']['label_rates']['test']:.1%} (balanced, so accuracy is meaningful).")

    with ui.row("chart", "model_insights-74"):
        c1, c2 = st.columns(2, gap="medium")
    listing = {pretty(k): v["curves"]["roc"] for k, v in clf["results"].items() if v["track"] == "listing"}
    attr = {pretty(k): v["curves"]["roc"] for k, v in clf["results"].items() if v["track"] == "attributes_only"}
    with c1, st.container(border=True):
        st.plotly_chart(ch.roc_chart(listing, "ROC: listing track (price included)"), width="stretch", config=CFG)
    with c2, st.container(border=True):
        st.plotly_chart(ch.roc_chart(attr, "ROC: attributes only (price removed)"), width="stretch", config=CFG)

    with ui.row("chart", "model_insights-82"):
        c3, c4 = st.columns(2, gap="medium")
    with c3, st.container(border=True):
        cal = {pretty(k): v["curves"]["calibration"] for k, v in clf["results"].items() if v["track"] == "listing"}
        st.plotly_chart(ch.calibration_chart(cal, "Calibration: listing track"), width="stretch", config=CFG)
    with c4, st.container(border=True):
        st.plotly_chart(ch.confusion_chart(champ_res["curves"]["confusion_matrix"],
                                           f"Confusion matrix: {pretty(champ)} (test)"), width="stretch", config=CFG)

    best_attr = clf["selection"]["best_attributes_only"]
    with ui.row("chart", "model_insights-91"):
        c5, c6 = st.columns(2, gap="medium")
    with c5, st.container(border=True):
        st.plotly_chart(ch.importance_chart(clf["importance"][champ], f"Permutation importance: {pretty(champ)}, listing"),
                        width="stretch", config=CFG)
    with c6, st.container(border=True):
        st.plotly_chart(ch.importance_chart(clf["importance"][best_attr],
                                            f"Permutation importance: {pretty(best_attr)}, attributes only"),
                        width="stretch", config=CFG)
    st.caption("Importance = drop in test ROC-AUC when one input is randomly shuffled (5,000-row test sample, 3 "
               "repeats). It shows what the model uses, not what causes prices. Area dominates the attributes-only "
               "model because ₹/sqft = price ÷ area.")

with t_reg:
    ui.callout("warn", "This is not a five-year forecast",
               "No future or historical prices exist in the data, so a five-year model cannot be trained or "
               "validated. These experiments ask a narrower question: can today's price be predicted from the "
               "property's attributes? The answer is no, so the analyzer uses scenario arithmetic for five-year values.")
    rows = [{"Model": pretty(k), "Val R²": r["val"]["r2"], "Test R²": r["test"]["r2"], "Test MAE (₹ L)": r["test"]["mae"],
             "Test RMSE (₹ L)": r["test"]["rmse"]} for k, r in reg["results"].items()]
    rt = pd.DataFrame(rows)
    st.dataframe(rt, hide_index=True, width="stretch",
                 column_config={"Val R²": st.column_config.NumberColumn(format="%+.4f"),
                                "Test R²": st.column_config.NumberColumn(format="%+.4f"),
                                "Test MAE (₹ L)": st.column_config.NumberColumn(format="%.2f"),
                                "Test RMSE (₹ L)": st.column_config.NumberColumn(format="%.2f")})
    d = reg["diagnostics"]
    with ui.row("chart", "model_insights-117"):
        c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True):
        st.plotly_chart(ch.actual_vs_predicted(d["actual_vs_pred"]["actual"], d["actual_vs_pred"]["predicted"],
                                               f"Actual vs predicted: {pretty(d['best_learned_model'])} (2,000 test listings)"),
                        width="stretch", config=CFG)
    with c2, st.container(border=True):
        st.plotly_chart(ch.residual_hist(d["residual_hist"]["counts"], d["residual_hist"]["edges"],
                                         f"Residuals: {pretty(d['best_learned_model'])} (all test listings)"),
                        width="stretch", config=CFG)
    with st.container(border=True):
        bc = pd.DataFrame(d["mae_by_city"])
        st.plotly_chart(ch.ranked_bar(bc["City"], bc["mae"], "Mean absolute error by city (test)", "MAE (₹ lakhs)",
                                      customdata=bc["n"].values, fmt=",.1f",
                                      hover="%{y}<br><b>MAE ₹%{x:,.1f} L</b><br>n = %{customdata:,}<extra></extra>"),
                        width="stretch", config=CFG)
    st.caption("Predictions cluster around the average price (₹≈254 L) whatever the actual price, and error is the "
               "same in every city. That is the signature of a target with no learnable relationship to the inputs.")

with t_mlf:
    st.markdown("**Experiments** (local MLflow store: `mlflow.db`, artefacts in `mlartifacts/`)")
    runs = [{"Experiment": "classification", "Run": k, "Run ID": v["run_id"]} for k, v in clf["results"].items()]
    runs += [{"Experiment": "valuation signal test", "Run": k, "Run ID": v["run_id"]} for k, v in reg["results"].items()]
    st.dataframe(pd.DataFrame(runs), hide_index=True, width="stretch")
    st.markdown(
        f"- **Registered model:** `{reg_info.get('name', 'n/a')}` version **{reg_info.get('version', '–')}**, "
        f"alias `@{reg_info.get('alias', '–')}`\n"
        f"- **Dataset fingerprint (SHA-256):** `{meta['dataset']['sha256']}`\n"
        f"- **Trained:** {meta['created_at']} · splits {meta['dataset']['splits']}\n"
        "- **Library versions:** " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items())
    )
    st.markdown("Open the MLflow UI locally (from the project folder):")
    st.code("mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000", language="bash")
    st.caption("The deployed app does not need MLflow. It reads the approved bundle in models/, whose run IDs are listed above.")

with t_cov:
    ds = meta["dataset"]
    ui.kpi_row([
        {"label": "Training rows", "value": f"{ds['splits']['train']:,}"},
        {"label": "Validation rows", "value": f"{ds['splits']['val']:,}"},
        {"label": "Test rows", "value": f"{ds['splits']['test']:,}"},
        {"label": "Cohorts", "value": f"{meta['cohorts']['n']}",
         "sub": f"min {meta['cohorts']['min_train_size']} · median {meta['cohorts']['median_train_size']} rows"},
    ])
    ranges = pd.DataFrame([{"Field": k.replace("_", " "), "Min": v[0], "Max": v[1]}
                           for k, v in meta["domain"]["numeric_ranges"].items()])
    with ui.row("chart", "model_insights-162"):
        c1, c2 = st.columns([1, 1.4], gap="medium")
    with c1:
        st.markdown("**Numeric ranges seen in training**")
        st.dataframe(ranges, hide_index=True, width="stretch")
        st.caption("Inputs outside these ranges trigger an extrapolation warning in the analyzer.")
    with c2:
        st.markdown("**Label definition**")
        st.write(meta["label"]["definition"])
        st.markdown("**Split strategy**")
        st.write("Random 70 / 15 / 15 split stratified by city. Rows are independent listings with no repeated "
                 "properties or time index, so neither group-aware nor time-aware validation is needed. Cohort "
                 "medians are computed on the training split only and then applied to validation, test and user input.")

ui.footer()
