"""Statistical significance tests for paired model comparison.

- McNemar's test on paired binary predictions
- Paired bootstrap test (two-sided) on any metric
- Multi-seed aggregation (mean / std / 95% normal CI)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy import stats

from src.eval.metrics import compute_all_metrics


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    *,
    continuity: bool = True,
) -> Dict[str, float]:
    """McNemar's chi-square test on the disagreement table.

    Returns the test statistic and a two-sided p-value. With ``continuity=True``
    applies Edwards' correction (recommended for small samples).
    """
    y_true = np.asarray(y_true).astype(int)
    a = np.asarray(y_pred_a).astype(int)
    b = np.asarray(y_pred_b).astype(int)
    a_correct = (a == y_true)
    b_correct = (b == y_true)
    n01 = int(np.sum(a_correct & ~b_correct))
    n10 = int(np.sum(~a_correct & b_correct))
    if n01 + n10 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n01": n01, "n10": n10}
    if continuity:
        statistic = (abs(n01 - n10) - 1.0) ** 2 / (n01 + n10)
    else:
        statistic = (n01 - n10) ** 2 / (n01 + n10)
    p_value = 1.0 - stats.chi2.cdf(statistic, df=1)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n01": n01,
        "n10": n10,
    }


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    *,
    y_proba_a: Optional[np.ndarray] = None,
    y_proba_b: Optional[np.ndarray] = None,
    metric: str = "f1_macro",
    n_bootstrap: int = 1000,
    seed: int = 42,
    positive_class: int = 1,
) -> Dict[str, float]:
    """Two-sided paired bootstrap test of ``metric(B) - metric(A)``.

    Returns the observed delta, its bootstrap mean / std / 95% CI, and a
    two-sided p-value computed as ``2 * min(Pr[delta_b <= 0], Pr[delta_b >= 0])``.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n = len(y_true)


    def _eval(idx: np.ndarray, pred: np.ndarray, proba: Optional[np.ndarray]) -> float:
        p_b = proba[idx] if proba is not None else None
        m = compute_all_metrics(y_true[idx], pred[idx], p_b, positive_class)
        return float(m.get(metric, np.nan))

    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ma = _eval(idx, y_pred_a, y_proba_a)
        mb = _eval(idx, y_pred_b, y_proba_b)
        deltas[b] = mb - ma

    obs_a = compute_all_metrics(y_true, y_pred_a, y_proba_a, positive_class).get(metric, np.nan)
    obs_b = compute_all_metrics(y_true, y_pred_b, y_proba_b, positive_class).get(metric, np.nan)
    obs_delta = float(obs_b - obs_a)
    p_two = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "metric": metric,
        "obs_a": float(obs_a),
        "obs_b": float(obs_b),
        "delta": obs_delta,
        "delta_mean": float(deltas.mean()),
        "delta_std": float(deltas.std(ddof=1)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_value": float(p_two),
        "n_bootstrap": int(n_bootstrap),
    }


def aggregate_seeds(
    per_seed_metrics: List[Dict[str, Any]],
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-seed metric dicts into mean / std / 95% normal CI.

    ``per_seed_metrics`` is a list of dicts returned by ``compute_all_metrics``.
    """
    if not per_seed_metrics:
        return {}
    keys = metric_keys or [k for k, v in per_seed_metrics[0].items() if isinstance(v, (int, float))]
    out: Dict[str, Dict[str, float]] = {}
    n = len(per_seed_metrics)
    for k in keys:
        vals = np.asarray([float(m.get(k, np.nan)) for m in per_seed_metrics], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        sem = std / np.sqrt(vals.size) if vals.size > 1 else 0.0
        ci = 1.96 * sem
        out[k] = {
            "mean": mean,
            "std": std,
            "ci95_low": mean - ci,
            "ci95_high": mean + ci,
            "n_seeds": int(vals.size),
        }
    return out
