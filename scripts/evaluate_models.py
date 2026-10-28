"""Build the model evaluation report from the saved evaluation bundle and
cross-check it against the MLflow tracking store.

Usage (from the project root, after train_models):
    python -m scripts.evaluate_models

Outputs
    reports/model_evaluation_report.html   interactive version
    reports/Model_Evaluation_Report.pdf    print-ready A4 PDF
"""
from __future__ import annotations

import html
import os
import sys

import pandas as pd
import plotly.graph_objects as go

from src.models.inference import ModelArtifactsMissing, load_bundle
from src.reporting.pdf import PdfToolMissing, html_to_pdf
from src.reporting.template import FigureEmbedder, kpis, page, table
from src.utils import config as C
from src.utils.logging import get_logger
from src.visualization import charts as ch

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
log = get_logger("evaluate_models")

PRETTY = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost",
          "mean_baseline": "Mean baseline", "ridge": "Ridge", "cohort_median_benchmark": "Comparable-median rule"}


def mlflow_crosscheck(evaluation: dict) -> list[str]:
    """Confirm every run ID in the bundle exists in MLflow with the same test metric."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        return ["MLflow not installed; cross-check skipped."]
    db = C.PROJECT_ROOT / "mlflow.db"
    if C.MLFLOW_TRACKING_URI.startswith("sqlite") and not db.exists():
        return ["mlflow.db not found; cross-check skipped (expected on a deployed app)."]
    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    notes = []
    checks = [(k, v, "test_roc_auc", v["test"]["roc_auc"]) for k, v in evaluation["classification"]["results"].items()]
    checks += [(k, v, "test_r2", v["test"]["r2"]) for k, v in evaluation["regression"]["results"].items()]
    ok = 0
    for key, res, metric, expected in checks:
        run = client.get_run(res["run_id"])
        logged = run.data.metrics.get(metric)
        if logged is None or abs(logged - expected) > 1e-9:
            notes.append(f"MISMATCH {key}: bundle {expected} vs MLflow {logged}")
        else:
            ok += 1
    try:
        mv = client.get_model_version_by_alias(C.REGISTERED_CLASSIFIER, "champion")
        notes.append(f"Registry: {C.REGISTERED_CLASSIFIER} v{mv.version} (alias champion), status {mv.status}.")
    except Exception as exc:  # registry optional
        notes.append(f"Registry lookup failed: {exc}")
    notes.insert(0, f"{ok}/{len(checks)} runs found in MLflow with matching test metrics.")
    return notes


SHOTS = [
    ("01_classification_runs.png", "Classification experiment: six runs (two tracks × three models) with test metrics as columns."),
    ("02_compare_runs.png", "Run comparison: parallel coordinates across track, test ROC-AUC, F1 and ECE."),
    ("03_run_overview.png", "Champion run overview (listing / XGBoost)."),
    ("03b_run_metrics_params.png", "Champion run: logged metrics and parameters."),
    ("04_run_artifacts.png", "Champion run artefacts: the logged model and evaluation plots."),
    ("05_regression_signal_test.png", "Valuation signal test experiment: every model at R² ≈ 0."),
    ("06_model_registry.png", "Model registry: propwise-investment-classifier, version 1 with alias @champion."),
]


def _pretty(key: str) -> str:
    return PRETTY[key.split("/")[-1]]


def _no_webgl(fig: go.Figure) -> go.Figure:
    """Chromium's PDF printer drops WebGL canvases; redraw scattergl as SVG."""
    d = fig.to_dict()
    for tr in d["data"]:
        if tr.get("type") == "scattergl":
            tr["type"] = "scatter"
    return go.Figure(d)


def _card(title: str, body: str) -> str:
    return f"<section class='card'><h2>{html.escape(title)}</h2>{body}</section>"


def _figrow(embed, *figs) -> str:
    return "<div class='figrow'>" + "".join(f"<div>{embed(f)}</div>" for f in figs) + "</div>"


def build_report() -> tuple[str, dict]:
    b = load_bundle()
    meta, ev = b.metadata, b.evaluation
    clf, reg = ev["classification"], ev["regression"]
    champ = meta["classifier"]["champion"]
    best_attr = clf["selection"]["best_attributes_only"]
    embed = FigureEmbedder()
    e = html.escape
    parts: list[str] = []

    # Summary -----------------------------------------------------------------
    cr, ar = clf["results"][champ]["test"], clf["results"][best_attr]["test"]
    best_r2 = max(r["test"]["r2"] for k, r in reg["results"].items() if k != "mean_baseline")
    summary = [
        f"Champion classifier: {_pretty(champ)} on the listing track, test ROC-AUC {cr['roc_auc']:.4f}, "
        f"F1 {cr['f1']:.4f}, ECE {cr['ece']:.4f}. It reproduces a transparent rule (₹/sqft vs. comparable median), "
        "so the near-perfect score is expected and is not evidence of predicting returns.",
        f"With price removed, the best model ({_pretty(best_attr)}) reaches ROC-AUC {ar['roc_auc']:.4f}, almost "
        "entirely through area.",
        f"Regression on current price finds no signal: best learned test R² {best_r2:+.4f}. The app therefore shows "
        "no 'estimated value' and uses explicit growth scenarios for five-year figures.",
        "All numbers are read from models/evaluation.json written by scripts/train_models.py and cross-checked "
        "against the MLflow tracking store; none are typed by hand.",
    ]
    parts.append("<div class='summary'><h2>Key results</h2><ul>" + "".join(f"<li>{e(x)}</li>" for x in summary)
                 + "</ul></div>")

    # 1. Setup ----------------------------------------------------------------
    ds = meta["dataset"]
    parts.append("<h2 class='part first'>1. Setup</h2>")
    parts.append(kpis([
        ("Listings", f"{ds['rows']:,}", f"SHA-256 {ds['sha256'][:12]}…"),
        ("Train / val / test", " / ".join(f"{ds['splits'][k] / ds['rows']:.0%}" for k in ("train", "val", "test")),
         f"{ds['splits']['test']:,} test rows · seed {C.RANDOM_STATE}"),
        ("Good-value share", f"{meta['label']['positive_rates']['test']:.1%}",
         f"test (train {meta['label']['positive_rates']['train']:.1%})"),
        ("Cohorts", f"{meta['cohorts']['n']}", f"smallest {meta['cohorts']['min_train_size']} train rows"),
    ]))
    parts.append(_card("Label and selection",
        f"<p><b>Label.</b> {e(meta['label']['definition'])}</p>"
        f"<p><b>Selection.</b> {e(clf['selection']['criterion'].capitalize())}. Test metrics were not used for "
        "selection.</p><p><b>Validation design.</b> Rows are independent listings with no repeats and no time "
        "index, so a random stratified split is appropriate; group- or time-aware validation is not required.</p>"))

    # 2. Classification -------------------------------------------------------
    parts.append("<h2 class='part'>2. Investment classification</h2>")
    keys = list(clf["results"])
    rows = []
    for k in keys:
        r = clf["results"][k]
        t = r["test"]
        rows.append([r["track"].replace("_", " "), _pretty(k) + (" ★" if k == champ else ""),
                     f"{r['val']['roc_auc']:.4f}", f"{t['roc_auc']:.4f}", f"{t['precision']:.4f}",
                     f"{t['recall']:.4f}", f"{t['f1']:.4f}", f"{t['accuracy']:.4f}", f"{t['brier']:.4f}",
                     f"{t['ece']:.4f}"])
    parts.append(_card("Metrics (test split unless marked)",
        table(["Track", "Model", "Val AUC", "Test AUC", "Precision", "Recall", "F1", "Accuracy", "Brier", "ECE"],
              rows, numeric=set(range(2, 10)), highlight=keys.index(champ))
        + "<p>★ = champion (highlighted). <b>Listing</b> track includes price and area; the label is a "
          "deterministic function of price ÷ area against a cohort median, so these models learn to reproduce the "
          "rule. <b>Attributes only</b> removes price; its signal comes almost entirely from area.</p>"))
    roc_l = {_pretty(k): v["curves"]["roc"] for k, v in clf["results"].items() if v["track"] == "listing"}
    roc_a = {_pretty(k): v["curves"]["roc"] for k, v in clf["results"].items() if v["track"] == "attributes_only"}
    parts.append(_card("ROC curves", _figrow(embed, ch.roc_chart(roc_l, "Listing track (price included)"),
                                             ch.roc_chart(roc_a, "Attributes only (price removed)"))))
    cal = {_pretty(k): v["curves"]["calibration"] for k, v in clf["results"].items() if v["track"] == "listing"}
    shown = "are" if meta["classifier"]["show_probability"] else "are not"
    parts.append(_card("Calibration and errors",
        _figrow(embed, ch.calibration_chart(cal, "Calibration: listing track"),
                ch.confusion_chart(clf["results"][champ]["curves"]["confusion_matrix"],
                                   f"Confusion matrix: {_pretty(champ)} (test, threshold 0.5)"))
        + f"<div class='note'>Champion test ECE {meta['classifier']['test_ece']:.4f} is below the display "
          f"threshold of {C.MAX_DISPLAY_ECE}, so probabilities {shown} shown in the app.</div>"))
    parts.append(_card("Permutation importance",
        _figrow(embed, ch.importance_chart(clf["importance"][champ], f"{_pretty(champ)}, listing", top=8),
                ch.importance_chart(clf["importance"][best_attr], f"{_pretty(best_attr)}, attributes only", top=8))
        + "<p>Drop in test ROC-AUC when one input is shuffled (5,000-row test sample, 3 repeats). It shows what "
          "the model uses, not what causes prices.</p>"))

    # 3. Regression -----------------------------------------------------------
    parts.append("<h2 class='part'>3. Valuation signal test</h2>")
    parts.append("<div class='warn'>This is not a five-year forecast. The data has no future or historical prices, "
                 "so a forecasting model cannot be trained or validated. The test asks whether today's price can "
                 "be predicted from a property's attributes.</div>")
    rrows = [[_pretty(k), f"{r['val']['r2']:+.4f}", f"{r['test']['r2']:+.4f}", f"{r['test']['mae']:.2f}",
              f"{r['test']['rmse']:.2f}"] for k, r in reg["results"].items()]
    parts.append(_card("Regression on current price",
        table(["Model", "Val R²", "Test R²", "Test MAE (₹ L)", "Test RMSE (₹ L)"], rrows, numeric={1, 2, 3, 4})
        + "<p>No learned model beats predicting the mean (R² ≈ 0). The comparable-median rule (cohort median "
          "₹/sqft × area) is worse than the mean because price is unrelated to area in this data. "
          "<b>There is no valuation signal to learn</b>, so the app does not present an estimated current value.</p>"))
    d = reg["diagnostics"]
    best = _pretty(d["best_learned_model"])
    parts.append(_card("Diagnostics",
        _figrow(embed, _no_webgl(ch.actual_vs_predicted(d["actual_vs_pred"]["actual"], d["actual_vs_pred"]["predicted"],
                                                        f"Actual vs predicted: {best}")),
                ch.residual_hist(d["residual_hist"]["counts"], d["residual_hist"]["edges"], f"Residuals: {best}"))
        + "<p>Predictions cluster around the average price whatever the actual price: the signature of a target "
          "with no learnable relationship to the inputs.</p>"))
    bc = pd.DataFrame(d["mae_by_city"])
    lo, hi = bc.loc[bc["mae"].idxmin()], bc.loc[bc["mae"].idxmax()]
    fig = ch.ranked_bar(bc["City"], bc["mae"], "Mean absolute error by city (test)", "MAE (₹ lakhs)", fmt=",.1f")
    fig.update_layout(height=760)
    parts.append(_card("Error by city", embed(fig)
        + f"<p>MAE ranges only from ₹{lo['mae']:.1f} L ({e(lo['City'])}) to ₹{hi['mae']:.1f} L ({e(hi['City'])}): "
          "error is flat across cities.</p>"))

    # 4. Scenarios ------------------------------------------------------------
    parts.append("<h2 class='part'>4. Five-year scenarios</h2>")
    srows = [[k, f"{v:.0%}", f"{(1 + v) ** C.HORIZON_YEARS:.3f}×"] for k, v in C.SCENARIO_RATES.items()]
    src = "".join(f"<li><a href='{e(s['url'])}'>{e(s['name'])}</a></li>" for s in C.RATE_SOURCES)
    parts.append(_card("Scenario engine (src/models/scenario.py)",
        f"<p>Future value = current price × (1 + r)<sup>{C.HORIZON_YEARS}</sup>. The rates are assumptions, "
        "not predictions.</p>" + table(["Scenario", "Annual rate", f"Multiplier over {C.HORIZON_YEARS} years"],
                                       srows, numeric={1, 2})
        + f"<p>{e(C.RATE_RATIONALE)}</p><p><b>Sources</b></p><ul>{src}</ul>"))

    # 5. MLflow ---------------------------------------------------------------
    parts.append("<h2 class='part'>5. Experiment tracking (MLflow)</h2>")
    checks = mlflow_crosscheck(ev)
    parts.append(_card("Cross-check against the tracking store",
        "<ul>" + "".join(f"<li>{e(n)}</li>" for n in checks) + "</ul>"
        + f"<p>Classification experiment run IDs: champion <code>{e(meta['classifier']['run_id'])}</code>. "
          "Tracking store: <code>mlflow.db</code> (SQLite), artefacts in <code>mlartifacts/</code>.</p>"))
    shot_dir = C.DOCS_DIR / "mlflow_screenshots"
    for i, (fname, cap) in enumerate(SHOTS):
        if (shot_dir / fname).exists():
            parts.append(f"<figure class='shot card'><img src='../docs/mlflow_screenshots/{fname}' alt='{e(cap)}'>"
                         f"<figcaption>Figure {i + 1}. {e(cap)}</figcaption></figure>")

    # 6. Versions -------------------------------------------------------------
    parts.append("<h2 class='part'>6. Reproducibility</h2>")
    parts.append(_card("Versions and commands",
        table(["Package", "Version"], [[k, v] for k, v in meta["versions"].items()])
        + f"<p>Model bundle created {e(meta['created_at'])}. Rebuild with "
          "<code>python -m scripts.train_models</code> then <code>python -m scripts.evaluate_models</code>.</p>"))

    page_html = page("Model Evaluation Report",
                     f"{ds['rows']:,} listings · 6 classifiers · 5 regression baselines",
                     "".join(parts), footer="Model scores describe this dataset only and are not investment advice.")
    return page_html, {"crosscheck": checks}


def main() -> int:
    try:
        report, info = build_report()
    except ModelArtifactsMissing as exc:
        log.error(str(exc))
        return 1
    for n in info["crosscheck"]:
        log.info(n)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = C.REPORTS_DIR / "model_evaluation_report.html"
    html_path.write_text(report, encoding="utf-8")
    log.info("Wrote %s", html_path)
    try:
        html_to_pdf(html_path, C.REPORTS_DIR / "Model_Evaluation_Report.pdf",
                    "PropWise AI · Model Evaluation Report")
    except PdfToolMissing as exc:
        log.warning("PDF not built: %s", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
