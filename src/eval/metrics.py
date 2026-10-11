"""Classification metrics + bootstrap confidence intervals."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    positive_class: int = 1,
) -> Dict[str, Any]:
    """Return the metrics commonly reported for binary hoax detection."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
    }


    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[positive_class], average="binary",
        pos_label=positive_class, zero_division=0,
    )
    out["precision_hoax"] = float(p)
    out["recall_hoax"] = float(r)
    out["f1_hoax"] = float(f)


    cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(np.concatenate([y_true, y_pred]))))
    out["confusion_matrix"] = cm.tolist()


    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] >= 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, positive_class]))
            out["pr_auc"] = float(average_precision_score(y_true, y_proba[:, positive_class]))
        except ValueError:
            out["roc_auc"] = float("nan")
            out["pr_auc"] = float("nan")
    return out


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    *,
    metric: str = "f1_macro",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    positive_class: int = 1,
) -> Dict[str, float]:
    """Stratified bootstrap CI for one of the metrics from ``compute_all_metrics``."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    with warnings.catch_warnings():

        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            proba_b = y_proba[idx] if y_proba is not None else None
            m = compute_all_metrics(y_true[idx], y_pred[idx], proba_b, positive_class)
            samples[b] = float(m.get(metric, np.nan))
    samples = samples[~np.isnan(samples)]
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(samples, [100 * alpha, 100 * (1 - alpha)])
    return {
        "metric": metric,
        "point": float(compute_all_metrics(y_true, y_pred, y_proba, positive_class).get(metric, np.nan)),
        "mean": float(samples.mean()),
        "std": float(samples.std(ddof=1)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_bootstrap": int(len(samples)),
        "ci_level": float(ci),
    }
