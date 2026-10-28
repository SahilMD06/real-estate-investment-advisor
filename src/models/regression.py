"""Valuation-signal test (regression).

Question answered: *can current price be predicted from property attributes
in this dataset?* If yes, a learned valuation model could support the
analysis. Phase 1 profiling suggested it cannot (price looks independent of
every attribute); these experiments quantify that formally.

This is NOT a five-year forecast. No future prices exist in the data. The
five-year estimate lives in :mod:`src.models.scenario`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBRegressor

from src.features.engineering import ATTRIBUTE_NUMERIC, MODEL_CATEGORICAL, CohortStats, build_model_frame
from src.models.preprocessing import build_preprocessor
from src.utils.config import LAKH, RANDOM_STATE

TARGET = "Price_in_Lakhs"


def regressor_candidates(seed: int = RANDOM_STATE) -> dict[str, tuple[object, bool, dict]]:
    ridge = dict(alpha=1.0)
    rf = dict(n_estimators=100, max_depth=10, min_samples_leaf=50, max_features="sqrt")
    xgb = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
               tree_method="hist")
    return {
        "mean_baseline": (DummyRegressor(strategy="mean"), False, {"strategy": "mean"}),
        "ridge": (Ridge(**ridge), True, ridge),
        "random_forest": (RandomForestRegressor(**rf, n_jobs=-1, random_state=seed), False, rf),
        "xgboost": (XGBRegressor(**xgb, n_jobs=-1, random_state=seed), False, xgb),
    }


def _drop_price(df: pd.DataFrame) -> pd.DataFrame:
    # build_model_frame needs a price column to exist; feed a constant so no
    # target information can reach the model. ATTRIBUTE_NUMERIC excludes it.
    return build_model_frame(df.assign(Price_in_Lakhs=1.0))


def build_regression_pipeline(estimator, scale: bool) -> Pipeline:
    return Pipeline(
        [
            ("features", FunctionTransformer(_drop_price, validate=False)),
            ("preprocess", build_preprocessor(ATTRIBUTE_NUMERIC, MODEL_CATEGORICAL, scale=scale)),
            ("model", estimator),
        ]
    )


def cohort_benchmark_prediction(stats: CohortStats, df: pd.DataFrame) -> np.ndarray:
    """'Comparable median' valuation: training cohort median ₹/sqft × area."""
    bench = stats.transform(df)["benchmark_pps"].to_numpy()
    return bench * df["Size_in_SqFt"].to_numpy() / LAKH
