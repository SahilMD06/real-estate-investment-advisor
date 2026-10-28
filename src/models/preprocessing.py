"""Reusable scikit-learn preprocessing.

The preprocessor is part of every saved model pipeline, so it is fitted on
training data only and replayed identically at inference time.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    """Median-impute + optional scaling for numbers; mode-impute + one-hot for categories.

    ``scale`` is True only for linear models — tree ensembles are invariant to
    monotonic scaling so we skip it for them.
    """
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), numeric),
            ("cat", Pipeline(cat_steps), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
