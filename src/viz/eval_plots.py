"""Confusion matrix, ROC, Precision-Recall curve."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.viz.style import apply_style, savefig

apply_style()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    out_path: str,
    id2label: Optional[Dict[int, str]] = None,
    normalize: bool = True,
    title: str = "Confusion matrix",
) -> str:
    classes = sorted(set(map(int, np.concatenate([y_true, y_pred]).tolist())))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    display_cm = cm_norm if normalize else cm
    labels = [id2label.get(c, str(c)) if id2label else str(c) for c in classes]

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(display_cm, cmap="Blues", vmin=0, vmax=display_cm.max())
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if normalize:
                text = f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]})"
            else:
                text = f"{cm[i, j]}"
            color = "white" if display_cm[i, j] > display_cm.max() * 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return savefig(fig, out_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    out_path: str,
    positive_class: int = 1,
    title: str = "ROC curve",
) -> str:
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, positive_class])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    ax.plot(fpr, tpr, color="#4C72B0", label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return savefig(fig, out_path)


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    out_path: str,
    positive_class: int = 1,
    title: str = "Precision-Recall curve",
) -> str:
    p, r, _ = precision_recall_curve(y_true, y_proba[:, positive_class])
    pr_auc = auc(r, p)
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    ax.plot(r, p, color="#DD8452", label=f"PR-AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    return savefig(fig, out_path)
