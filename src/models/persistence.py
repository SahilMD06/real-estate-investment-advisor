"""Portable save/load for the champion classifier pipeline.

Pickling an XGBoost model embeds XGBoost's internal binary buffer, which the
XGBoost docs do not guarantee to be portable across environments (a pickle
written on Linux raised "input stream corrupted" on Windows with identical
library versions). XGBoost's supported portable format is ``save_model`` to
JSON, so the pipeline is stored in two parts:

* ``classifier.joblib``  – the scikit-learn steps (feature engineering +
  preprocessing) with the final estimator removed when it is XGBoost;
* ``classifier_xgb.json`` – the XGBoost model in XGBoost's own JSON format.

Non-XGBoost champions are stored as a single joblib pipeline, as before.
"""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

XGB_FILENAME = "classifier_xgb.json"


def save_classifier(pipe: Pipeline, path: Path | str) -> list[Path]:
    """Save ``pipe`` to ``path`` (+ an XGBoost JSON file next to it if needed).

    Returns the list of files written.
    """
    from xgboost import XGBClassifier

    path = Path(path)
    xgb_path = path.with_name(XGB_FILENAME)
    name, estimator = pipe.steps[-1]
    if isinstance(estimator, XGBClassifier):
        estimator.save_model(xgb_path)
        joblib.dump(Pipeline(pipe.steps[:-1]), path, compress=3)
        return [path, xgb_path]
    if xgb_path.exists():  # stale file from an earlier XGBoost champion
        xgb_path.unlink()
    joblib.dump(pipe, path, compress=3)
    return [path]


def load_classifier(path: Path | str) -> Pipeline:
    """Inverse of :func:`save_classifier`."""
    path = Path(path)
    pipe = joblib.load(path)
    xgb_path = path.with_name(XGB_FILENAME)
    if xgb_path.exists():
        from xgboost import XGBClassifier

        model = XGBClassifier()
        model.load_model(xgb_path)
        pipe = Pipeline(pipe.steps + [("model", model)])
    return pipe
