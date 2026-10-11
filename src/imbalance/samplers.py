"""Sampler-based and resampling strategies for class imbalance."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler

log = logging.getLogger(__name__)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Inverse-frequency class weights ('balanced' from scikit-learn)."""
    classes = np.arange(num_classes)
    present = np.unique(labels)
    base = compute_class_weight(class_weight="balanced", classes=present, y=labels)
    out = np.ones(num_classes, dtype=np.float32)
    for c, w in zip(present, base):
        out[int(c)] = float(w)
    return out


def make_weighted_sampler(labels: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    """``WeightedRandomSampler`` with per-sample weights = 1 / class freq."""
    cw = compute_class_weights(labels, num_classes)
    sample_weights = cw[labels.astype(int)]
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def random_oversample(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    seed: int = 42,
) -> pd.DataFrame:
    """Duplicate minority rows until every class matches the majority count."""
    counts = df[label_col].value_counts()
    target = int(counts.max())
    rng = np.random.default_rng(seed)
    out = []
    for label, sub in df.groupby(label_col):
        n_extra = target - len(sub)
        out.append(sub)
        if n_extra > 0:
            idx = rng.integers(0, len(sub), size=n_extra)
            out.append(sub.iloc[idx])
    new = pd.concat(out, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    log.info("Oversample: %s -> %s", counts.to_dict(), new[label_col].value_counts().to_dict())
    return new


def random_undersample(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    seed: int = 42,
) -> pd.DataFrame:
    """Sample majority rows down to the minority count."""
    counts = df[label_col].value_counts()
    target = int(counts.min())
    out = []
    for label, sub in df.groupby(label_col):
        out.append(sub.sample(n=target, random_state=seed))
    new = pd.concat(out, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    log.info("Undersample: %s -> %s", counts.to_dict(), new[label_col].value_counts().to_dict())
    return new


def smote_on_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    k_neighbors: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE in embedding space. Returns ``(X_resampled, y_resampled)``."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as e:
        raise ImportError("imbalanced-learn is required for SMOTE") from e

    counts = pd.Series(labels).value_counts()
    if counts.min() <= k_neighbors:
        k_neighbors = max(1, int(counts.min()) - 1)
        log.warning("Reduced SMOTE k_neighbors to %d due to tiny minority", k_neighbors)
    sm = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    X_res, y_res = sm.fit_resample(embeddings, labels)
    log.info("SMOTE: %s -> %s", counts.to_dict(), pd.Series(y_res).value_counts().to_dict())
    return X_res, y_res
