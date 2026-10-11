"""EDA plots: class distribution, length distribution, top tokens, word cloud."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz.style import apply_style, savefig

apply_style()


def plot_class_distribution(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    id2label: Optional[Dict[int, str]] = None,
    out_path: str,
    title: str = "Class distribution",
) -> str:
    counts = df[label_col].value_counts().sort_index()
    labels = [id2label.get(int(i), str(i)) if id2label else str(i) for i in counts.index]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.bar(labels, counts.values, color=["#4C72B0", "#DD8452"][: len(counts)])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v):,}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of samples")
    ax.set_title(title)
    ax.set_ylim(0, max(counts.values) * 1.15)
    return savefig(fig, out_path)


def plot_text_length_distribution(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    label_col: str = "label",
    id2label: Optional[Dict[int, str]] = None,
    out_path: str,
    max_tokens: Optional[int] = 400,
    title: str = "Text length distribution (tokens)",
) -> str:
    df = df.copy()
    df["_n_tokens"] = df[text_col].astype(str).str.split().str.len()
    if max_tokens is not None:
        df["_n_tokens"] = df["_n_tokens"].clip(upper=max_tokens)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for cls in sorted(df[label_col].unique()):
        sub = df[df[label_col] == cls]["_n_tokens"]
        label = id2label.get(int(cls), str(cls)) if id2label else str(cls)
        ax.hist(sub, bins=40, alpha=0.55, label=label, edgecolor="white")
    ax.set_xlabel("Tokens per document (whitespace-split)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    return savefig(fig, out_path)


def _top_tokens(texts: Iterable[str], top_k: int = 25, stopwords: Optional[set] = None) -> List[tuple]:
    stop = stopwords or set()
    counter: Counter = Counter()
    for t in texts:
        for w in str(t).lower().split():
            w = w.strip(".,!?;:()[]\"'-")
            if not w or w in stop or w.isdigit():
                continue
            counter[w] += 1
    return counter.most_common(top_k)


DEFAULT_ID_STOPWORDS = {
    "yang", "dan", "di", "itu", "untuk", "dari", "dengan", "pada", "ke",
    "tidak", "akan", "ini", "kita", "adalah", "kami", "atau", "dalam",
    "juga", "saya", "sudah", "telah", "para", "oleh", "saja", "agar",
    "namun", "tetapi", "tapi", "karena", "bahwa", "ada", "lagi", "saat",
    "lebih", "bisa", "kalau", "harus", "jadi", "yaitu", "sebagai", "ya",
    "ke", "se", "kah", "lah", "pun", "nya", "an", "hanya", "masih",
}


def plot_top_tokens(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    label_col: str = "label",
    id2label: Optional[Dict[int, str]] = None,
    out_path: str,
    top_k: int = 20,
    stopwords: Optional[set] = None,
    title: str = "Top tokens per class",
) -> str:
    stopwords = stopwords or DEFAULT_ID_STOPWORDS
    classes = sorted(df[label_col].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4.8), sharey=False)
    if len(classes) == 1:
        axes = [axes]
    for ax, cls in zip(axes, classes):
        sub = df[df[label_col] == cls][text_col]
        items = _top_tokens(sub, top_k=top_k, stopwords=stopwords)
        if not items:
            ax.set_title(f"(no tokens) cls={cls}")
            continue
        words, counts = zip(*items)
        ax.barh(range(len(words)), counts, color="#4C72B0")
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        label = id2label.get(int(cls), str(cls)) if id2label else str(cls)
        ax.set_title(f"{label}")
        ax.set_xlabel("Count")
    fig.suptitle(title)
    fig.tight_layout()
    return savefig(fig, out_path)


def plot_wordcloud(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    label_col: str = "label",
    id2label: Optional[Dict[int, str]] = None,
    out_path: str,
    stopwords: Optional[set] = None,
    title: str = "Word cloud per class",
) -> Optional[str]:
    try:
        from wordcloud import WordCloud
    except ImportError:
        return None
    stopwords = stopwords or DEFAULT_ID_STOPWORDS
    classes = sorted(df[label_col].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(6 * len(classes), 4))
    if len(classes) == 1:
        axes = [axes]
    for ax, cls in zip(axes, classes):
        text = " ".join(df[df[label_col] == cls][text_col].astype(str).tolist())
        wc = WordCloud(
            width=800, height=500, background_color="white",
            stopwords=stopwords, collocations=False,
        ).generate(text or "empty")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        label = id2label.get(int(cls), str(cls)) if id2label else str(cls)
        ax.set_title(label)
    fig.suptitle(title)
    fig.tight_layout()
    return savefig(fig, out_path)
