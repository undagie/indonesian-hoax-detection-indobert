"""Bar plots for ablation studies (augmentation methods, imbalance strategies)."""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz.style import apply_style, savefig

apply_style()


def plot_ablation_bars(
    df: pd.DataFrame,
    *,
    group_col: str,
    metric_col: str,
    err_low_col: Optional[str] = None,
    err_high_col: Optional[str] = None,
    out_path: str,
    title: str = "Ablation study",
    ylabel: Optional[str] = None,
    sort_desc: bool = True,
    annotate: bool = True,
) -> str:
    """Bar plot with optional asymmetric error bars (CI low/high)."""
    df = df.copy()
    if sort_desc:
        df = df.sort_values(metric_col, ascending=False)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(df) + 2), 4.0))
    x = np.arange(len(df))
    bars = ax.bar(x, df[metric_col], color="#4C72B0")
    if err_low_col and err_high_col:
        lo = df[metric_col].values - df[err_low_col].values
        hi = df[err_high_col].values - df[metric_col].values
        ax.errorbar(x, df[metric_col].values, yerr=[lo, hi],
                    fmt="none", ecolor="#333", capsize=4, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(df[group_col], rotation=30, ha="right")
    ax.set_ylabel(ylabel or metric_col)
    ax.set_title(title)
    if annotate:
        for b, v in zip(bars, df[metric_col]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(df[metric_col].max() * 1.15, 0.05))
    return savefig(fig, out_path)


def plot_seed_variance(
    per_seed_df: pd.DataFrame,
    *,
    group_col: str,
    metric_col: str,
    out_path: str,
    title: str = "Per-seed variance",
    ylabel: Optional[str] = None,
) -> str:
    """Box / strip plot of per-seed metric values for each model variant."""
    groups = per_seed_df[group_col].unique().tolist()
    data = [per_seed_df.loc[per_seed_df[group_col] == g, metric_col].values for g in groups]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.8 * len(groups) + 2), 4.2))
    bp = ax.boxplot(data, labels=groups, patch_artist=True, widths=0.55)
    for patch in bp["boxes"]:
        patch.set(facecolor="#A9C5E3", edgecolor="#345")

    for i, vals in enumerate(data, start=1):
        x = np.full_like(vals, i, dtype=float) + np.random.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(x, vals, s=22, color="#C44E52", zorder=3, alpha=0.85)
    ax.set_ylabel(ylabel or metric_col)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return savefig(fig, out_path)
