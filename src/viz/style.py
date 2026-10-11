"""Shared matplotlib style + save helper."""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGURE_DPI, FIGURE_FORMAT


def apply_style() -> None:
    """Apply a clean publication style to matplotlib."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
    })


def savefig(fig, path: str, *, fmt: Optional[str] = None) -> str:
    """Save ``fig`` to ``path``. Creates parent directories. Returns the final path."""
    fmt = fmt or FIGURE_FORMAT
    if not path.lower().endswith(f".{fmt}"):
        path = f"{path}.{fmt}"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, format=fmt)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path
