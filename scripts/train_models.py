"""Train, evaluate, track and package all models.

Usage (from the project root):
    python -m scripts.train_models            # full run
    python -m scripts.train_models --sample 40000   # quick smoke run

Outputs
    mlflow.db / mlartifacts/   MLflow tracking store + artefacts (local)
    models/classifier.joblib   champion pipeline (sklearn steps; raw rows in)
    models/classifier_xgb.json XGBoost champion in XGBoost's portable JSON format
    models/cohort_stats.joblib training-only cohort benchmarks
    models/metadata.json       versions, data fingerprint, selection record
    models/evaluation.json     metrics + curves consumed by the app
"""
from __future__ import annotations

import argparse
import os
import json
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.data.load_data import file_fingerprint, load_processed
from src.evaluation.metrics import classification_curves, classification_metrics, regression_metrics
from src.features.engineering import CohortStats, RAW_INPUT_COLUMNS
from src.models.classification import (
    FEATURE_TRACKS, build_pipeline, classifier_candidates, grouped_permutation_importance,
    make_splits, raw_inputs,
)
from src.models.regression import (
    TARGET, build_regression_pipeline, cohort_benchmark_prediction, regressor_candidates,
)
from src.models.persistence import save_classifier
from src.models.target import make_label
from src.utils import config as C
from src.utils.logging import get_logger

log = get_logger("train_models")


SKOPS_TRUSTED = [
    "numpy.dtype",
    "src.features.engineering.build_model_frame",
    "src.models.regression._drop_price",
    "sklearn.tree._tree.Tree",  # trees we trained ourselves in this script
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
    "xgboost.sklearn.XGBRegressor",
]


def _log_sklearn(pipe, sample: pd.DataFrame, output) -> str:
    """Log with skops (MLflow's safe default) and ship src/ so the model is self-contained.
    Returns the logged model URI (MLflow 3 'models:/m-...' form)."""
    info = mlflow.sklearn.log_model(
        pipe, name="model", input_example=sample, signature=infer_signature(sample, output),
        serialization_format="skops", skops_trusted_types=SKOPS_TRUSTED,
        code_paths=[str(C.PROJECT_ROOT / "src")],
    )
    return info.model_uri


def _experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name, artifact_location=C.MLFLOW_ARTIFACT_ROOT)
    return exp.experiment_id


def _log_json(obj, filename: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / filename
        p.write_text(json.dumps(obj, indent=2, default=float))
        mlflow.log_artifact(str(p))


def _common_tags(fingerprint: str, sizes: dict, sample: int | None) -> dict:
    return {
        "dataset_sha256": fingerprint,
        "dataset_rows_train": str(sizes["train"]),
        "dataset_rows_val": str(sizes["val"]),
        "dataset_rows_test": str(sizes["test"]),
        "sample_run": str(bool(sample)),
        "label_definition": (
            f"pps <= {C.GOOD_VALUE_RATIO} x train median pps of {'x'.join(C.COHORT_KEYS)} cohort"
        ),
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def train_classifiers(splits, stats, tags, log_models: bool) -> tuple[dict, dict]:
    y = {name: make_label(getattr(splits, name), stats) for name in ("train", "val", "test")}
    X = {name: raw_inputs(getattr(splits, name)) for name in ("train", "val", "test")}
    label_rates = {k: float(v.mean()) for k, v in y.items()}
    log.info("Label positive rate %s", {k: round(v, 3) for k, v in label_rates.items()})

    exp_id = _experiment(C.CLASSIFICATION_EXPERIMENT)
    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for track, numeric in FEATURE_TRACKS.items():
        for name, (est, scale, params) in classifier_candidates().items():
            key = f"{track}/{name}"
            pipe = build_pipeline(est, numeric, scale)
            t0 = time.time()
            with mlflow.start_run(experiment_id=exp_id, run_name=key) as run:
                mlflow.set_tags({**tags, "track": track, "model_type": name})
                mlflow.log_params({**params, "track": track, "n_numeric": len(numeric), "scaled": scale})
                pipe.fit(X["train"], y["train"])
                fit_s = time.time() - t0
                p_val = pipe.predict_proba(X["val"])[:, 1]
                p_test = pipe.predict_proba(X["test"])[:, 1]
                m_val = classification_metrics(y["val"], p_val)
                m_test = classification_metrics(y["test"], p_test)
                mlflow.log_metrics({f"val_{k}": v for k, v in m_val.items()})
                mlflow.log_metrics({f"test_{k}": v for k, v in m_test.items()})
                mlflow.log_metric("fit_seconds", fit_s)
                curves = classification_curves(y["test"], p_test)
                _log_json(curves, "test_curves.json")
                if log_models:
                    sample = X["val"].head(5)
                    model_uri = _log_sklearn(pipe, sample, pipe.predict_proba(sample)[:, 1])
                else:
                    model_uri = None
                results[key] = {
                    "track": track, "model": name, "run_id": run.info.run_id, "model_uri": model_uri,
                    "params": params, "val": m_val, "test": m_test, "curves": curves, "fit_seconds": round(fit_s, 1),
                }
                fitted[key] = pipe
            log.info("%-36s val AUC %.4f F1 %.4f | test AUC %.4f (%.0fs)",
                     key, m_val["roc_auc"], m_val["f1"], m_test["roc_auc"], fit_s)

    # Champion: listing track, best validation ROC-AUC; ties (<0.001) broken by
    # lower validation Brier score (better calibrated probabilities).
    listing = [k for k in results if results[k]["track"] == "listing"]
    best_auc = max(results[k]["val"]["roc_auc"] for k in listing)
    contenders = [k for k in listing if best_auc - results[k]["val"]["roc_auc"] < 0.001]
    champion = min(contenders, key=lambda k: results[k]["val"]["brier"])
    log.info("Champion classifier: %s", champion)

    attr = [k for k in results if results[k]["track"] == "attributes_only"]
    best_attr = max(attr, key=lambda k: results[k]["val"]["roc_auc"])

    importance = {}
    for key in (champion, best_attr):
        imp = grouped_permutation_importance(fitted[key], X["test"], y["test"])
        importance[key] = imp.round(5).to_dict(orient="records")
        with mlflow.start_run(run_id=results[key]["run_id"]):
            _log_json(importance[key], "permutation_importance.json")

    selection = {
        "champion": champion,
        "criterion": "highest validation ROC-AUC on the listing track; ties within 0.001 broken by lower validation Brier score",
        "contenders": contenders,
        "best_attributes_only": best_attr,
        "label_rates": label_rates,
    }
    return {"results": results, "importance": importance, "selection": selection}, fitted


# ---------------------------------------------------------------------------
# Regression signal test
# ---------------------------------------------------------------------------
def train_regressors(splits, stats, tags, log_models: bool) -> dict:
    exp_id = _experiment(C.REGRESSION_EXPERIMENT)
    Xtr, Xval, Xte = (raw_inputs(getattr(splits, s)) for s in ("train", "val", "test"))
    ytr, yval, yte = (getattr(splits, s)[TARGET] for s in ("train", "val", "test"))
    results: dict[str, dict] = {}
    preds_test: dict[str, np.ndarray] = {}
    for name, (est, scale, params) in regressor_candidates().items():
        pipe = build_regression_pipeline(est, scale)
        t0 = time.time()
        with mlflow.start_run(experiment_id=exp_id, run_name=name) as run:
            mlflow.set_tags({**tags, "model_type": name, "target": TARGET,
                             "purpose": "valuation signal test - not a 5-year forecast"})
            mlflow.log_params(params)
            pipe.fit(Xtr, ytr)
            pv, pt = pipe.predict(Xval), pipe.predict(Xte)
            m_val, m_test = regression_metrics(yval, pv), regression_metrics(yte, pt)
            mlflow.log_metrics({f"val_{k}": v for k, v in m_val.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in m_test.items()})
            if log_models:
                sample = Xval.head(5)
                _log_sklearn(pipe, sample, pipe.predict(sample))
            results[name] = {"run_id": run.info.run_id, "params": params, "val": m_val, "test": m_test,
                             "fit_seconds": round(time.time() - t0, 1)}
            preds_test[name] = pt
        log.info("%-22s val R2 %+.4f RMSE %.2f | test R2 %+.4f", name, m_val["r2"], m_val["rmse"], m_test["r2"])

    # Comparable-median benchmark (what the app shows as a reference value).
    with mlflow.start_run(experiment_id=exp_id, run_name="cohort_median_benchmark") as run:
        mlflow.set_tags({**tags, "model_type": "rule", "target": TARGET})
        pv = cohort_benchmark_prediction(stats, splits.val)
        pt = cohort_benchmark_prediction(stats, splits.test)
        m_val, m_test = regression_metrics(yval, pv), regression_metrics(yte, pt)
        mlflow.log_metrics({f"val_{k}": v for k, v in m_val.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in m_test.items()})
        results["cohort_median_benchmark"] = {"run_id": run.info.run_id, "params": {}, "val": m_val,
                                              "test": m_test, "fit_seconds": 0.0}
        preds_test["cohort_median_benchmark"] = pt
    log.info("cohort benchmark       test R2 %+.4f", m_test["r2"])

    learned = [k for k in results if k not in ("mean_baseline", "cohort_median_benchmark")]
    best = max(learned, key=lambda k: results[k]["val"]["r2"])
    rng = np.random.default_rng(C.RANDOM_STATE)
    idx = rng.choice(len(yte), size=min(2000, len(yte)), replace=False)
    resid = yte.to_numpy() - preds_test[best]
    counts, edges = np.histogram(resid, bins=40)
    by_city = (
        pd.DataFrame({"City": splits.test["City"].to_numpy(), "abs_err": np.abs(resid)})
        .groupby("City")["abs_err"].agg(["mean", "size"]).reset_index()
        .rename(columns={"mean": "mae", "size": "n"}).sort_values("mae")
    )
    diagnostics = {
        "best_learned_model": best,
        "actual_vs_pred": {"actual": yte.to_numpy()[idx].round(2).tolist(),
                           "predicted": preds_test[best][idx].round(2).tolist()},
        "residual_hist": {"counts": counts.tolist(), "edges": edges.round(2).tolist()},
        "mae_by_city": by_city.round(2).to_dict(orient="records"),
    }
    return {"results": results, "diagnostics": diagnostics}


# ---------------------------------------------------------------------------
def register_champion(model_uri: str) -> dict:
    client = MlflowClient()
    mv = mlflow.register_model(model_uri, C.REGISTERED_CLASSIFIER)
    client.set_registered_model_alias(C.REGISTERED_CLASSIFIER, "champion", mv.version)
    client.set_model_version_tag(C.REGISTERED_CLASSIFIER, mv.version, "label", "rule-derived good_value")
    return {"name": C.REGISTERED_CLASSIFIER, "version": str(mv.version), "alias": "champion"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=None, help="train on a random subset (smoke test)")
    ap.add_argument("--no-log-models", action="store_true", help="skip logging model artefacts to MLflow")
    args = ap.parse_args(argv)

    df = load_processed()
    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=C.RANDOM_STATE)
    fingerprint = file_fingerprint(C.RAW_DATA_PATH) if C.RAW_DATA_PATH.exists() else "unavailable"

    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    splits = make_splits(df)
    stats = CohortStats().fit(splits.train)
    log.info("Splits %s | %s cohorts (min train size %s)", splits.sizes(), len(stats.cohort_median),
             min(stats.cohort_count.values()))
    tags = _common_tags(fingerprint, splits.sizes(), args.sample)
    log_models = not args.no_log_models

    clf, fitted = train_classifiers(splits, stats, tags, log_models)
    reg = train_regressors(splits, stats, tags, log_models)

    champion = clf["selection"]["champion"]
    registry = register_champion(clf["results"][champion]["model_uri"]) if log_models else None

    # Probability display gate: only show probabilities if calibration is good.
    ece = clf["results"][champion]["test"]["ece"]
    show_prob = ece <= C.MAX_DISPLAY_ECE

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_classifier(fitted[champion], C.CLASSIFIER_PATH)  # portable: XGBoost part saved as JSON
    joblib.dump(stats, C.COHORT_STATS_PATH, compress=3)

    full = load_processed() if args.sample else df
    city_state = full.groupby("City")["State"].first().to_dict()
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "pandas": pd.__version__, "mlflow": mlflow.__version__,
            "numpy": np.__version__,
        },
        "dataset": {"sha256": fingerprint, "rows": int(len(df)), "splits": splits.sizes(),
                    "sample_run": bool(args.sample), "reference_year": C.REFERENCE_YEAR},
        "label": {
            "name": "good_value",
            "definition": (f"Rupees per sqft at or below {C.GOOD_VALUE_RATIO:g} x the training-split "
                           f"median of its {' x '.join(C.COHORT_KEYS)} cohort "
                           f"(fallback City, then global, when a cohort has < {C.MIN_COHORT_SIZE} rows)."),
            "positive_rates": clf["selection"]["label_rates"],
        },
        "classifier": {
            "champion": champion, "selection": clf["selection"],
            "run_id": clf["results"][champion]["run_id"], "registry": registry,
            "show_probability": show_prob, "test_ece": ece, "max_display_ece": C.MAX_DISPLAY_ECE,
            "input_columns": RAW_INPUT_COLUMNS,
        },
        "regression": {"best_learned_model": reg["diagnostics"]["best_learned_model"],
                       "experiment": C.REGRESSION_EXPERIMENT},
        "cohorts": {"keys": list(C.COHORT_KEYS), "n": len(stats.cohort_median),
                    "min_train_size": int(min(stats.cohort_count.values())),
                    "median_train_size": int(np.median(list(stats.cohort_count.values())))},
        "domain": {
            "city_state": city_state,
            "numeric_ranges": {k: list(v) for k, v in C.NUMERIC_RANGES.items()},
        },
        "mlflow": {"tracking_uri": "sqlite:///mlflow.db (local)",
                   "experiments": [C.CLASSIFICATION_EXPERIMENT, C.REGRESSION_EXPERIMENT]},
    }
    evaluation = {"classification": {k: v for k, v in clf.items()}, "regression": reg}
    C.METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=float))
    C.EVALUATION_PATH.write_text(json.dumps(evaluation, indent=2, default=float))
    log.info("Saved bundle to %s (champion=%s, show_probability=%s, ECE=%.4f)",
             C.MODELS_DIR, champion, show_prob, ece)
    return 0


if __name__ == "__main__":
    sys.exit(main())
