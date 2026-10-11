"""Experiment 06: error analysis and embedding projection for the proposed run."""

from __future__ import annotations

import experiments._bootstrap
import glob
import os
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    Config,
    FIGURES_DIR,
    ID2LABEL,
    NUM_CLASSES,
    RUNS_DIR,
    SEEDS,
    resolve_model_name,
    seed_everything,
)
from src.data import load_dataset, make_splits, preprocess_dataframe
from src.eval.error_analysis import error_analysis_report
from src.models import TextDataset, TransformerClassifier, build_tokenizer, extract_embeddings
from src.utils.io import setup_logging, write_table
from src.viz.embeddings import plot_embeddings_2d


def _find_best_run_dir(tag_substr: Optional[str] = None) -> Optional[str]:
    """Find a completed proposed-run directory under ``results/runs/``."""
    search_tags = (
        [tag_substr]
        if tag_substr
        else [
            "cmp_transformer_imb-combined",
            "imb_combined",
            "cmp_transformer_proposed",
        ]
    )
    for tag in search_tags:
        candidates = sorted(glob.glob(os.path.join(RUNS_DIR, f"{tag}*", "seed*")))
        if candidates:
            return candidates[0]
    return None


def main(run_dir: Optional[str] = None, synthetic_n: int = 400) -> None:
    log = setup_logging("exp06_error_analysis")
    run_dir = run_dir or _find_best_run_dir()
    if run_dir is None or not os.path.isdir(run_dir):
        log.error("No proposed-pipeline run found. Run exp05 first or pass run_dir.")
        return
    log.info("Analysing run: %s", run_dir)

    y_true = np.load(os.path.join(run_dir, "test_y_true.npy"))
    y_pred = np.load(os.path.join(run_dir, "test_y_pred.npy"))
    proba_path = os.path.join(run_dir, "test_y_proba.npy")
    y_proba = np.load(proba_path) if os.path.exists(proba_path) else None


    cfg = Config()
    seed = SEEDS[0]
    seed_everything(seed)
    df = load_dataset(synthetic_n=synthetic_n)
    df = preprocess_dataframe(df, text_col="text", out_col="text")
    _, _, test_df = make_splits(
        df, label_col="label",
        test_size=cfg.TEST_RATIO, val_size=cfg.VAL_RATIO, seed=seed,
    )

    if len(test_df) != len(y_true):
        log.warning(
            "Reconstructed test size (%d) differs from saved predictions (%d). "
            "Re-creating from saved preds only.",
            len(test_df), len(y_true),
        )
        test_df = pd.DataFrame({
            "text": [f"<sample {i}>" for i in range(len(y_true))],
            "label": y_true,
        })


    report = error_analysis_report(
        test_df["text"].tolist(), y_true, y_pred, y_proba, id2label=ID2LABEL,
    )
    err_dir = os.path.join(FIGURES_DIR, "errors")
    os.makedirs(err_dir, exist_ok=True)
    write_table(report["per_class"], stem="errors_per_class")
    write_table(report["per_length_bucket"], stem="errors_per_length_bucket")
    err_df = report["errors"]
    err_df.to_csv(os.path.join(err_dir, "misclassified.csv"), index=False)
    log.info("Wrote %d misclassified rows.", len(err_df))


    best_model_path = os.path.join(run_dir, "best_model.pt")
    if not os.path.exists(best_model_path):
        log.warning("best_model.pt not found in %s - skipping embedding plot.", run_dir)
        return
    import torch

    model_name = resolve_model_name()
    log.info("Loading best model from %s for embedding extraction.", best_model_path)
    model = TransformerClassifier(model_name, num_classes=NUM_CLASSES, dropout=cfg.DROPOUT).to(cfg.DEVICE)
    state = torch.load(best_model_path, map_location=cfg.DEVICE)
    model.load_state_dict(state)
    tokenizer = build_tokenizer(model_name)
    test_ds = TextDataset(test_df["text"].tolist(), y_true.tolist(),
                          tokenizer, max_len=cfg.MAX_LEN)
    emb = extract_embeddings(model, test_ds, batch_size=cfg.EVAL_BATCH_SIZE, device=cfg.DEVICE)

    emb_dir = os.path.join(FIGURES_DIR, "embeddings")
    plot_embeddings_2d(
        emb, y_true, method="tsne",
        out_path=os.path.join(emb_dir, "tsne_proposed_test"),
        id2label=ID2LABEL,
        title="t-SNE of test embeddings (proposed pipeline)",
    )
    try:
        plot_embeddings_2d(
            emb, y_true, method="umap",
            out_path=os.path.join(emb_dir, "umap_proposed_test"),
            id2label=ID2LABEL,
            title="UMAP of test embeddings (proposed pipeline)",
        )
    except ImportError:
        log.warning("UMAP not installed; skipping UMAP plot.")
    log.info("Error analysis complete.")


if __name__ == "__main__":
    main()
