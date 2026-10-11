"""Error analysis: misclassified examples grouped by length / confidence / class."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def error_analysis_report(
    texts,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    *,
    id2label: Optional[Dict[int, str]] = None,
    length_bins: tuple = (0, 20, 50, 100, 200, 500, 10000),
) -> Dict[str, Any]:
    """Return a DataFrame of misclassified rows + per-bucket error stats."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    texts = list(texts)
    n_tokens = np.asarray([len(str(t).split()) for t in texts])
    confidence = (
        y_proba.max(axis=1) if y_proba is not None and y_proba.ndim == 2 else np.full(len(texts), np.nan)
    )

    df = pd.DataFrame({
        "text": texts,
        "y_true": y_true,
        "y_pred": y_pred,
        "n_tokens": n_tokens,
        "confidence": confidence,
        "correct": y_true == y_pred,
    })
    if id2label is not None:
        df["y_true_label"] = df["y_true"].map(id2label)
        df["y_pred_label"] = df["y_pred"].map(id2label)


    df["length_bucket"] = pd.cut(df["n_tokens"], bins=list(length_bins), include_lowest=True)
    bucket_stats = (
        df.groupby("length_bucket", observed=True)
        .agg(n=("correct", "size"), error_rate=("correct", lambda s: 1 - s.mean()))
        .reset_index()
    )


    per_class = (
        df.groupby("y_true", observed=True)
        .agg(n=("correct", "size"), error_rate=("correct", lambda s: 1 - s.mean()))
        .reset_index()
    )
    if id2label is not None:
        per_class["label"] = per_class["y_true"].map(id2label)


    errors = df[~df["correct"]].copy()
    if y_proba is not None:
        errors = errors.sort_values("confidence", ascending=False)

    return {
        "errors": errors.reset_index(drop=True),
        "per_class": per_class,
        "per_length_bucket": bucket_stats,
    }
