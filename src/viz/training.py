"""Training curves: train loss and validation metric per epoch."""

from __future__ import annotations

from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.viz.style import apply_style, savefig

apply_style()


def plot_learning_curves(
    history: List[dict],
    *,
    out_path: str,
    metric_key: str = "val_f1_macro",
    title: str = "Learning curves",
) -> str:
    df = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(6.5, 4.0))
    color1, color2 = "#C44E52", "#4C72B0"
    ax1.plot(df["epoch"], df["train_loss"], color=color1, marker="o", label="train_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2 = ax1.twinx()
    ax2.grid(False)
    if metric_key in df.columns:
        ax2.plot(df["epoch"], df[metric_key], color=color2, marker="s", label=metric_key)
        ax2.set_ylabel(metric_key, color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
    ax1.set_title(title)
    return savefig(fig, out_path)
