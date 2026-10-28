"""Scenario maths, metrics, inference validation and artefact handling."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import classification_metrics, expected_calibration_error, regression_metrics
from src.features.engineering import RAW_INPUT_COLUMNS
from src.models.inference import (
    InvalidPropertyInput, ModelArtifactsMissing, analyze_property, load_bundle, validate_property,
)
from src.models.scenario import project_value, run_scenarios, trajectory, validate_rate
from src.utils import config as C

CITY_STATE = {"Pune": "Maharashtra", "Mumbai": "Maharashtra", "Chennai": "Tamil Nadu"}


# --- Scenario engine ---------------------------------------------------------
def test_project_value_compounds():
    assert project_value(100, 0.04, 5) == pytest.approx(100 * 1.04 ** 5)
    assert project_value(100, 0.0, 5) == 100
    assert project_value(100, -0.02, 5) < 100


def test_run_scenarios_defaults_and_gain():
    res = run_scenarios(200)
    assert [r.name for r in res] == list(C.SCENARIO_RATES)
    base = next(r for r in res if r.name == "Baseline")
    assert base.gain == pytest.approx(base.future_value - 200)
    assert base.gain_pct == pytest.approx(((1 + C.SCENARIO_RATES["Baseline"]) ** 5 - 1) * 100)


@pytest.mark.parametrize("bad", [0.5, -0.2, 4])
def test_rate_out_of_range_rejected(bad):
    with pytest.raises(ValueError, match="outside the supported range"):
        validate_rate(bad)


def test_rate_type_and_value_errors():
    with pytest.raises(TypeError):
        validate_rate("4%")
    with pytest.raises(ValueError):
        project_value(0, 0.04)
    with pytest.raises(ValueError):
        project_value(100, 0.04, -1)


def test_trajectory_shape():
    t = trajectory(100, {"A": 0.02, "B": 0.05}, years=5)
    assert len(t) == 12 and t["Year"].max() == 5


# --- Metrics -----------------------------------------------------------------
def test_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    m = classification_metrics(y, p)
    assert m["accuracy"] == 1 and m["roc_auc"] == 1
    assert expected_calibration_error(np.array([0, 1] * 50), np.full(100, 0.5)) == pytest.approx(0)
    r = regression_metrics([1, 2, 3], [1, 2, 3])
    assert r["mae"] == 0 and r["r2"] == 1


# --- Input validation --------------------------------------------------------
def test_valid_record_passes(sample_record):
    rec, warnings = validate_property(sample_record, CITY_STATE)
    assert warnings == []
    assert isinstance(rec["BHK"], int)


def test_missing_field(sample_record):
    rec = dict(sample_record)
    rec.pop("Price_in_Lakhs")
    with pytest.raises(InvalidPropertyInput, match="Missing required field"):
        validate_property(rec, CITY_STATE)


@pytest.mark.parametrize("field,value,msg", [
    ("City", "Atlantis", "not covered"),
    ("State", "Tamil Nadu", "is in Maharashtra"),
    ("Facing", "Up", "not supported"),
    ("Price_in_Lakhs", "abc", "must be a number"),
    ("Price_in_Lakhs", -5, "greater than zero"),
    ("Size_in_SqFt", 0, "greater than zero"),
    ("Total_Floors", 0, "at least 1"),
    ("Floor_No", -1, "cannot be negative"),
    ("BHK", float("nan"), "finite"),
])
def test_invalid_inputs(sample_record, field, value, msg):
    with pytest.raises(InvalidPropertyInput, match=msg):
        validate_property(dict(sample_record, **{field: value}), CITY_STATE)


def test_out_of_range_inputs_warn(sample_record):
    _, warnings = validate_property(dict(sample_record, Size_in_SqFt=9000, BHK=7, Floor_No=20), CITY_STATE)
    joined = " ".join(warnings)
    assert "Size in SqFt" in joined and "BHK" in joined and "above the building" in joined


# --- Inference ---------------------------------------------------------------
def test_analyze_property_end_to_end(tiny_bundle, sample_record):
    res = analyze_property(sample_record, tiny_bundle)
    assert res.pps == pytest.approx(90 * 1e5 / 1200)
    assert res.rule_label == int(res.pps <= res.benchmark_pps)
    assert 0 <= res.model_probability <= 1
    assert len(res.scenarios) == 3
    assert res.positives or res.negatives


def test_analyze_rejects_bad_rates(tiny_bundle, sample_record):
    with pytest.raises(ValueError):
        analyze_property(sample_record, tiny_bundle, {"Wild": 0.9})


def test_training_and_inference_preprocessing_match(tiny_bundle, raw_df):
    """The same row scored via the batch training path and the single-record
    inference path must give the same probability."""
    row = raw_df.iloc[[500]]
    batch_prob = tiny_bundle.classifier.predict_proba(row[RAW_INPUT_COLUMNS])[:, 1][0]
    record = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.iloc[0][RAW_INPUT_COLUMNS].items()}
    res = analyze_property(record, tiny_bundle)
    assert res.model_probability == pytest.approx(batch_prob, abs=1e-9)


def test_unsupported_category_rejected_at_inference(tiny_bundle, sample_record):
    with pytest.raises(InvalidPropertyInput):
        analyze_property(dict(sample_record, Property_Type="Castle"), tiny_bundle)


def test_missing_artifacts(tmp_path):
    with pytest.raises(ModelArtifactsMissing, match="train_models"):
        load_bundle(tmp_path)


# --- Shipped bundle (skipped if not trained yet) -------------------------------
@pytest.fixture(scope="module")
def shipped_bundle():
    try:
        return load_bundle()
    except ModelArtifactsMissing:
        pytest.skip("Model bundle not built; run scripts.train_models")


def test_shipped_bundle_metadata(shipped_bundle):
    meta = shipped_bundle.metadata
    assert meta["classifier"]["champion"].startswith("listing/")
    assert meta["cohorts"]["min_train_size"] >= C.MIN_COHORT_SIZE
    assert set(meta["domain"]["city_state"]) and json.dumps(meta)


def test_shipped_bundle_reproduces_rule_on_processed_rows(shipped_bundle):
    path = C.PROCESSED_DATA_PATH
    if not path.exists():
        pytest.skip("Processed data not available")
    df = pd.read_parquet(path).sample(2000, random_state=1)
    prob = shipped_bundle.classifier.predict_proba(df[RAW_INPUT_COLUMNS])[:, 1]
    from src.models.target import make_label
    y = make_label(df, shipped_bundle.cohort_stats)
    assert ((prob >= 0.5).astype(int) == y.to_numpy()).mean() > 0.95


# --- Portable model persistence --------------------------------------------------
@pytest.mark.parametrize("kind", ["xgboost", "logistic_regression"])
def test_classifier_save_load_roundtrip(kind, raw_df, tmp_path):
    """The XGBoost champion must be stored in XGBoost's portable JSON format
    (pickled boosters failed to load on Windows), and reload identically."""
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    from src.features.engineering import LISTING_NUMERIC, CohortStats
    from src.models.classification import build_pipeline, raw_inputs
    from src.models.persistence import XGB_FILENAME, load_classifier, save_classifier
    from src.models.target import make_label

    stats = CohortStats(min_size=5).fit(raw_df)
    y = make_label(raw_df, stats)
    est = XGBClassifier(n_estimators=20, max_depth=3) if kind == "xgboost" else LogisticRegression(max_iter=2000)
    pipe = build_pipeline(est, LISTING_NUMERIC, scale=kind != "xgboost").fit(raw_inputs(raw_df), y)

    path = tmp_path / "classifier.joblib"
    written = save_classifier(pipe, path)
    assert (tmp_path / XGB_FILENAME).exists() == (kind == "xgboost")
    assert len(written) == (2 if kind == "xgboost" else 1)
    if kind == "xgboost":  # the joblib part must not embed an XGBoost booster
        import joblib
        assert not any(isinstance(s, XGBClassifier) for _, s in joblib.load(path).steps)

    reloaded = load_classifier(path)
    np.testing.assert_array_equal(pipe.predict_proba(raw_inputs(raw_df)), reloaded.predict_proba(raw_inputs(raw_df)))


def test_shipped_classifier_uses_portable_format(shipped_bundle):
    from xgboost import XGBClassifier
    model = shipped_bundle.classifier.steps[-1][1]
    if isinstance(model, XGBClassifier):
        assert (C.MODELS_DIR / "classifier_xgb.json").exists()
