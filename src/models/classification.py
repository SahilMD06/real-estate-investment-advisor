"""Investment-classification models.

Two experiment tracks are trained and logged:

* ``listing`` — what the app uses. Inputs include the asking price and size
  (a user evaluating a listing always knows them). Because the label is a
  deterministic function of price/size vs. a cohort median, these models
  learn to *reproduce the rule*; near-perfect scores are expected and are
  not evidence of predictive skill about future returns.
* ``attributes_only`` — the asking price is removed. This measures how much
  of the label can be inferred from property characteristics alone and
  exposes the mechanical size effect (bigger homes → lower ₹/sqft when price
  is unrelated to size).

The cohort median itself is never a model input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

from src.features.engineering import (
    ATTRIBUTE_NUMERIC, LISTING_NUMERIC, MODEL_CATEGORICAL, RAW_INPUT_COLUMNS, build_model_frame,
)
from src.models.preprocessing import build_preprocessor
from src.utils.config import RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE

FEATURE_TRACKS: dict[str, list[str]] = {
    "listing": LISTING_NUMERIC,
    "attributes_only": ATTRIBUTE_NUMERIC,
}


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def make_splits(df: pd.DataFrame, seed: int = RANDOM_STATE) -> Splits:
    """70/15/15 split stratified by City so every city appears in every split.

    Rows are independent property listings (no repeated properties, no time
    index — verified in Phase 1), so a random split is appropriate.
    """
    if len(df) < 20:
        raise ValueError("Need at least 20 rows to split into train/validation/test.")
    strat = df["City"] if df["City"].value_counts().min() >= 4 else None
    train, rest = train_test_split(
        df, test_size=TEST_SIZE + VALIDATION_SIZE, random_state=seed, stratify=strat
    )
    strat_rest = rest["City"] if strat is not None else None
    val, test = train_test_split(
        rest, test_size=TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE), random_state=seed, stratify=strat_rest
    )
    return Splits(train, val, test)


def classifier_candidates(seed: int = RANDOM_STATE) -> dict[str, tuple[object, bool, dict]]:
    """name -> (estimator, needs_scaling, loggable params)."""
    lr = dict(C=1.0, max_iter=2000, solver="lbfgs")
    rf = dict(n_estimators=100, max_depth=12, min_samples_leaf=25, max_features="sqrt")
    xgb = dict(
        n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
        colsample_bytree=0.9, tree_method="hist",
    )
    return {
        "logistic_regression": (LogisticRegression(**lr, random_state=seed), True, lr),
        "random_forest": (RandomForestClassifier(**rf, n_jobs=-1, random_state=seed), False, rf),
        "xgboost": (
            XGBClassifier(**xgb, eval_metric="logloss", n_jobs=-1, random_state=seed),
            False, xgb,
        ),
    }


def build_pipeline(estimator, numeric: list[str], scale: bool) -> Pipeline:
    """Raw-schema rows in → probability out. Feature engineering is inside the
    pipeline, so the saved artefact applies exactly the training transforms."""
    return Pipeline(
        [
            ("features", FunctionTransformer(build_model_frame, validate=False)),
            ("preprocess", build_preprocessor(numeric, MODEL_CATEGORICAL, scale=scale)),
            ("model", estimator),
        ]
    )


def raw_inputs(df: pd.DataFrame) -> pd.DataFrame:
    return df[RAW_INPUT_COLUMNS]


def grouped_permutation_importance(
    pipe: Pipeline, X: pd.DataFrame, y: pd.Series, n_rows: int = 5000, seed: int = RANDOM_STATE,
    scoring: str = "roc_auc",
) -> pd.DataFrame:
    """Permutation importance on *raw* input fields (so e.g. permuting price
    also permutes the derived log-price). Measured on held-out data."""
    if len(X) > n_rows:
        X = X.sample(n_rows, random_state=seed)
        y = y.loc[X.index]
    cols = [c for c in X.columns if c not in ("State", "Locality")]  # unused by models
    result = permutation_importance(
        pipe, X, y, scoring=scoring, n_repeats=3, random_state=seed, n_jobs=1,
    )
    imp = pd.DataFrame(
        {"feature": X.columns, "importance": result.importances_mean, "std": result.importances_std}
    )
    imp = imp[imp["feature"].isin(cols)]
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def predict_proba(pipe: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(raw_inputs(df))[:, 1]
