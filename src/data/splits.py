"""Stratified train/val/test splits + Stratified K-Fold for cross-validation."""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def make_splits(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) DataFrames with stratification on ``label_col``.

    ``val_size`` is expressed as a fraction of the **original** dataset.
    """
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=seed,
        shuffle=True,
    )
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=rel_val,
        stratify=train_val[label_col],
        random_state=seed,
        shuffle=True,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def make_kfold_splits(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    n_folds: int = 5,
    seed: int = 42,
) -> Iterator[Tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield ``(fold_idx, train_df, val_df)`` using StratifiedKFold."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = df[label_col].to_numpy()
    indices = np.arange(len(df))
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(indices, y)):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)
        yield fold_idx, train_df, val_df


def class_distribution(df: pd.DataFrame, label_col: str = "label") -> pd.Series:
    """Return label counts sorted by label index."""
    return df[label_col].value_counts().sort_index()
