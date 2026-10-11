"""t-SNE / UMAP scatter of contextual embeddings, coloured by label."""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.viz.style import apply_style, savefig

apply_style()


def plot_embeddings_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    method: str = "tsne",
    out_path: str,
    id2label: Optional[Dict[int, str]] = None,
    seed: int = 42,
    title: Optional[str] = None,
) -> str:
    """Project ``embeddings`` to 2D and plot a class-coloured scatter."""
    method = method.lower()
    if method == "tsne":
        from sklearn.manifold import TSNE

        n_samples = embeddings.shape[0]
        perplexity = max(5, min(30, n_samples // 3 - 1))
        reducer = TSNE(n_components=2, random_state=seed, init="pca",
                       perplexity=perplexity, learning_rate="auto")
    elif method == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError("Install `umap-learn` for UMAP embeddings.") from e
        reducer = umap.UMAP(n_components=2, random_state=seed)
    else:
        raise ValueError(f"unknown method {method!r}, expected 'tsne' or 'umap'")
    xy = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    classes = sorted(np.unique(labels).tolist())
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    for i, cls in enumerate(classes):
        mask = labels == cls
        name = id2label.get(int(cls), str(cls)) if id2label else str(cls)
        ax.scatter(xy[mask, 0], xy[mask, 1], s=14, alpha=0.65,
                   color=palette[i % len(palette)], label=name, edgecolors="none")
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.set_title(title or f"{method.upper()} of contextual embeddings")
    ax.legend(loc="best", frameon=True)
    return savefig(fig, out_path)
