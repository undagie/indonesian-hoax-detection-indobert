"""Publication-quality plotting helpers. All functions save 300 DPI figures by default."""

from src.viz.style import apply_style, savefig
from src.viz.eda import (
    plot_class_distribution,
    plot_text_length_distribution,
    plot_top_tokens,
    plot_wordcloud,
)
from src.viz.training import plot_learning_curves
from src.viz.eval_plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
)
from src.viz.embeddings import plot_embeddings_2d
from src.viz.ablation import plot_ablation_bars, plot_seed_variance

__all__ = [
    "apply_style",
    "savefig",
    "plot_class_distribution",
    "plot_text_length_distribution",
    "plot_top_tokens",
    "plot_wordcloud",
    "plot_learning_curves",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_embeddings_2d",
    "plot_ablation_bars",
    "plot_seed_variance",
]
