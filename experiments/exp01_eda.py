"""Experiment 01: exploratory data analysis."""

from __future__ import annotations

import experiments._bootstrap
import os

import pandas as pd

from config import FIGURES_DIR, ID2LABEL, TABLES_DIR
from src.data import load_dataset, preprocess_dataframe
from src.data.preprocess import basic_token_stats
from src.utils.io import setup_logging, write_table
from src.viz.eda import (
    plot_class_distribution,
    plot_text_length_distribution,
    plot_top_tokens,
    plot_wordcloud,
)


def main(synthetic_n: int = 400) -> None:
    log = setup_logging("exp01_eda")
    eda_dir = os.path.join(FIGURES_DIR, "eda")
    os.makedirs(eda_dir, exist_ok=True)

    df = load_dataset(synthetic_n=synthetic_n)
    df = preprocess_dataframe(df, text_col="text", out_col="text")
    log.info("Loaded %d rows: %s", len(df), df["label"].value_counts().to_dict())


    plot_class_distribution(
        df, label_col="label", id2label=ID2LABEL,
        out_path=os.path.join(eda_dir, "class_distribution"),
        title="Class distribution (hoax vs fakta)",
    )
    plot_text_length_distribution(
        df, text_col="text", label_col="label", id2label=ID2LABEL,
        out_path=os.path.join(eda_dir, "length_distribution"),
        title="Document length distribution (tokens)",
    )
    plot_top_tokens(
        df, text_col="text", label_col="label", id2label=ID2LABEL,
        out_path=os.path.join(eda_dir, "top_tokens"),
        top_k=20,
    )
    plot_wordcloud(
        df, text_col="text", label_col="label", id2label=ID2LABEL,
        out_path=os.path.join(eda_dir, "wordclouds"),
    )


    stats_rows = []
    for cls, sub in df.groupby("label"):
        s = basic_token_stats(sub["text"].tolist())
        s["label_id"] = int(cls)
        s["label"] = ID2LABEL.get(int(cls), str(cls))
        stats_rows.append(s)
    overall = basic_token_stats(df["text"].tolist())
    overall.update({"label_id": -1, "label": "overall"})
    stats_rows.append(overall)
    cols = ["label_id", "label", "n", "min", "max", "mean", "median", "p95", "p99"]
    stats_df = pd.DataFrame(stats_rows)[cols]
    paths = write_table(
        stats_df, stem="eda_summary",
        caption="Dataset descriptive statistics per class",
        label="tab:eda_summary",
    )
    log.info("EDA tables written: %s", paths)
    log.info("EDA figures written to: %s", eda_dir)


if __name__ == "__main__":
    main()
