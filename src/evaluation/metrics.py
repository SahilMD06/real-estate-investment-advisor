"""Metric helpers shared by training, evaluation scripts and the app."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
)


def expected_calibration_error(y_true, prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def classification_metrics(y_true, prob, threshold: float = 0.5) -> dict[str, float]:
    pred = (np.asarray(prob) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "ece": expected_calibration_error(y_true, prob),
    }


def classification_curves(y_true, prob, max_points: int = 200) -> dict:
    fpr, tpr, _ = roc_curve(y_true, prob)
    step = max(1, len(fpr) // max_points)
    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
    cm = confusion_matrix(y_true, (np.asarray(prob) >= 0.5).astype(int), labels=[0, 1])
    return {
        "roc": {"fpr": fpr[::step].round(4).tolist() + [1.0], "tpr": tpr[::step].round(4).tolist() + [1.0]},
        "calibration": {"mean_pred": mean_pred.round(4).tolist(), "frac_pos": frac_pos.round(4).tolist()},
        "confusion_matrix": cm.tolist(),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
