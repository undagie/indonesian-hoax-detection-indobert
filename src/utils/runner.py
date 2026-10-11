"""Orchestrate load, augment, imbalance handling, train, and evaluate for one seed."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    BOOTSTRAP_CI,
    BOOTSTRAP_N,
    Config,
    FIGURES_DIR,
    ID2LABEL,
    LABEL_MAP,
    NUM_CLASSES,
    get_run_config_snapshot,
    get_runs_dir,
    seed_everything,
)
from src.augment import augment_dataframe
from src.data import load_dataset, make_splits, preprocess_dataframe
from src.eval.metrics import bootstrap_metric_ci, compute_all_metrics
from src.imbalance import build_loss, compute_class_weights, make_weighted_sampler
from src.imbalance.samplers import random_oversample, random_undersample
from src.models import (
    TextDataset,
    TransformerClassifier,
    build_tokenizer,
    train_tfidf_baseline,
)
from src.training import Trainer
from src.utils.io import load_json, save_json
from src.viz.eval_plots import plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from src.viz.training import plot_learning_curves

log = logging.getLogger(__name__)


@dataclass
class RunOutput:
    """Container for a single experiment run (one seed)."""

    run_dir: str
    tag: str
    seed: int
    model_name: str
    aug_method: str
    imbalance: str
    test_metrics: Dict[str, Any]
    test_metric_ci: Dict[str, Any]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]




def _load_and_split(
    *,
    cfg: Config,
    seed: int,
    synthetic_n: int = 200,
) -> tuple:
    df = load_dataset(synthetic_n=synthetic_n)
    df = preprocess_dataframe(df, text_col="text", out_col="text")
    train_df, val_df, test_df = make_splits(
        df,
        label_col="label",
        test_size=cfg.TEST_RATIO,
        val_size=cfg.VAL_RATIO,
        seed=seed,
    )
    return train_df, val_df, test_df


def _apply_resample(
    df: pd.DataFrame,
    strategy: str,
    *,
    seed: int,
) -> pd.DataFrame:
    if strategy == "random_oversample":
        return random_oversample(df, seed=seed)
    if strategy == "random_undersample":
        return random_undersample(df, seed=seed)
    return df


def _try_load_cached_run(
    run_dir: str,
    *,
    tag: str,
    seed: int,
    model_name: str,
    aug_method: str,
    imbalance: str,
) -> Optional["RunOutput"]:
    """Return a RunOutput reconstructed from disk if the run already finished."""
    metrics_path = os.path.join(run_dir, "test_metrics.json")
    if not os.path.exists(metrics_path):
        return None
    try:
        test_metrics = load_json(metrics_path)
        ci_path = os.path.join(run_dir, "test_metrics_ci.json")
        test_ci = load_json(ci_path) if os.path.exists(ci_path) else {}
        y_true = np.load(os.path.join(run_dir, "test_y_true.npy"))
        y_pred = np.load(os.path.join(run_dir, "test_y_pred.npy"))
        proba_path = os.path.join(run_dir, "test_y_proba.npy")
        y_proba = np.load(proba_path) if os.path.exists(proba_path) else None
    except Exception as e:
        log.warning("Cached run at %s exists but failed to load (%s) - re-running.", run_dir, e)
        return None
    log.info("Reusing cached run: %s", run_dir)
    return RunOutput(
        run_dir=run_dir,
        tag=tag,
        seed=seed,
        model_name=model_name,
        aug_method=aug_method,
        imbalance=imbalance,
        test_metrics=test_metrics,
        test_metric_ci=test_ci,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )




def train_transformer_run(
    *,
    tag: str,
    seed: int,
    aug_method: str = "none",
    aug_ratio: float = 1.0,
    aug_prob: float = 0.15,
    imbalance: str = "none",
    model_alias_or_name: Optional[str] = None,
    epochs: Optional[int] = None,
    synthetic_n: int = 200,
    save_plots: bool = True,
    save_best_model: bool = True,
    force: bool = False,
) -> RunOutput:
    """Run one seed end-to-end. Skips work if ``test_metrics.json`` already exists."""
    cfg = Config()
    if epochs is not None:
        cfg.EPOCHS = epochs

    from config import resolve_model_name

    model_name = resolve_model_name(model_alias_or_name)
    run_dir = os.path.join(get_runs_dir(tag=tag), f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    if not force:
        cached = _try_load_cached_run(
            run_dir, tag=tag, seed=seed, model_name=model_name,
            aug_method=aug_method, imbalance=imbalance,
        )
        if cached is not None:
            return cached

    seed_everything(seed)

    snapshot = get_run_config_snapshot(
        experiment=tag,
        model_name=model_name,
        seed=seed,
        aug_method=aug_method,
        aug_ratio=aug_ratio,
        aug_prob=aug_prob,
        imbalance_strategy=imbalance,
        epochs=cfg.EPOCHS,
    )
    save_json(snapshot, os.path.join(run_dir, "run_config.json"))


    train_df, val_df, test_df = _load_and_split(cfg=cfg, seed=seed, synthetic_n=synthetic_n)
    save_json(
        {
            "train": train_df["label"].value_counts().to_dict(),
            "val": val_df["label"].value_counts().to_dict(),
            "test": test_df["label"].value_counts().to_dict(),
        },
        os.path.join(run_dir, "split_counts.json"),
    )

    if aug_method != "none":
        train_df = augment_dataframe(
            train_df, aug_method,
            text_col="text", label_col="label",
            ratio=aug_ratio, prob=aug_prob,
            minority_only=True, seed=seed,
        )

    if imbalance in {"random_oversample", "random_undersample"}:
        train_df = _apply_resample(train_df, imbalance, seed=seed)

    tokenizer = build_tokenizer(model_name)
    train_ds = TextDataset(train_df["text"].tolist(), train_df["label"].tolist(),
                           tokenizer, max_len=cfg.MAX_LEN)
    val_ds = TextDataset(val_df["text"].tolist(), val_df["label"].tolist(),
                         tokenizer, max_len=cfg.MAX_LEN)
    test_ds = TextDataset(test_df["text"].tolist(), test_df["label"].tolist(),
                          tokenizer, max_len=cfg.MAX_LEN)

    sampler = None
    class_weights = None
    samples_per_class = (
        train_df["label"].value_counts().sort_index().reindex(range(NUM_CLASSES), fill_value=0).tolist()
    )
    if imbalance == "weighted_sampler":
        sampler = make_weighted_sampler(
            train_df["label"].to_numpy(), num_classes=NUM_CLASSES
        )
    if imbalance in {"class_weight", "ldam_loss", "combined"}:
        class_weights = compute_class_weights(
            train_df["label"].to_numpy(), num_classes=NUM_CLASSES
        ).tolist()

    model = TransformerClassifier(model_name, num_classes=NUM_CLASSES, dropout=cfg.DROPOUT)
    loss_fn = build_loss(
        imbalance,
        num_classes=NUM_CLASSES,
        samples_per_class=samples_per_class,
        class_weights=class_weights,
    )
    trainer = Trainer(
        model, train_ds, val_ds,
        loss_fn=loss_fn,
        device=cfg.DEVICE,
        run_dir=run_dir,
        batch_size=cfg.BATCH_SIZE,
        eval_batch_size=cfg.EVAL_BATCH_SIZE,
        epochs=cfg.EPOCHS,
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
        warmup_ratio=cfg.WARMUP_RATIO,
        max_grad_norm=cfg.MAX_GRAD_NORM,
        grad_accum_steps=cfg.GRAD_ACCUM_STEPS,
        early_stopping_patience=cfg.EARLY_STOPPING_PATIENCE,
        early_stopping_metric=cfg.EARLY_STOPPING_METRIC,
        use_amp=cfg.USE_AMP,
        num_workers=cfg.NUM_WORKERS,
        sampler=sampler,
        save_best_model=save_best_model,
    )
    train_result = trainer.fit()


    pred_out = trainer.predict(test_ds, batch_size=cfg.EVAL_BATCH_SIZE)
    y_true = pred_out["labels"]
    y_pred = pred_out["preds"]
    y_proba = pred_out["proba"]

    test_metrics = compute_all_metrics(y_true, y_pred, y_proba)
    test_ci = {
        m: bootstrap_metric_ci(
            y_true, y_pred, y_proba,
            metric=m, n_bootstrap=BOOTSTRAP_N, ci=BOOTSTRAP_CI, seed=seed,
        )
        for m in ("accuracy", "f1_macro", "f1_hoax", "roc_auc")
    }
    save_json(test_metrics, os.path.join(run_dir, "test_metrics.json"))
    save_json(test_ci, os.path.join(run_dir, "test_metrics_ci.json"))
    np.save(os.path.join(run_dir, "test_y_true.npy"), y_true)
    np.save(os.path.join(run_dir, "test_y_pred.npy"), y_pred)
    if y_proba is not None:
        np.save(os.path.join(run_dir, "test_y_proba.npy"), y_proba)

    if save_plots:
        plot_learning_curves(
            train_result.history,
            out_path=os.path.join(run_dir, "learning_curves"),
            metric_key=f"val_{cfg.EARLY_STOPPING_METRIC}",
            title=f"{tag} seed={seed}",
        )
        plot_confusion_matrix(
            y_true, y_pred,
            out_path=os.path.join(run_dir, "confusion_matrix"),
            id2label=ID2LABEL,
            title=f"Confusion - {tag} seed={seed}",
        )
        if y_proba is not None:
            plot_roc_curve(y_true, y_proba,
                           out_path=os.path.join(run_dir, "roc"),
                           title=f"ROC - {tag} seed={seed}")
            plot_pr_curve(y_true, y_proba,
                          out_path=os.path.join(run_dir, "pr"),
                          title=f"PR - {tag} seed={seed}")

    return RunOutput(
        run_dir=run_dir,
        tag=tag,
        seed=seed,
        model_name=model_name,
        aug_method=aug_method,
        imbalance=imbalance,
        test_metrics=test_metrics,
        test_metric_ci=test_ci,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )




def train_baseline_run(
    *,
    tag: str,
    seed: int,
    classifier: str,
    synthetic_n: int = 200,
    force: bool = False,
) -> RunOutput:
    """Run one TF-IDF baseline seed. Skips work if metrics already exist."""
    cfg = Config()
    run_dir = os.path.join(get_runs_dir(tag=tag), f"{classifier}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    if not force:
        cached = _try_load_cached_run(
            run_dir, tag=tag, seed=seed, model_name=classifier,
            aug_method="none", imbalance="class_weight",
        )
        if cached is not None:
            return cached

    seed_everything(seed)
    train_df, val_df, test_df = _load_and_split(cfg=cfg, seed=seed, synthetic_n=synthetic_n)

    train_combined = pd.concat([train_df, val_df], ignore_index=True)
    res = train_tfidf_baseline(
        classifier,
        train_texts=train_combined["text"].tolist(),
        train_labels=train_combined["label"].tolist(),
        test_texts=test_df["text"].tolist(),
        seed=seed,
    )

    y_true = test_df["label"].to_numpy()
    y_pred = res.y_pred
    y_proba = res.y_proba
    test_metrics = compute_all_metrics(y_true, y_pred, y_proba)
    test_ci = {
        m: bootstrap_metric_ci(
            y_true, y_pred, y_proba,
            metric=m, n_bootstrap=BOOTSTRAP_N, ci=BOOTSTRAP_CI, seed=seed,
        )
        for m in ("accuracy", "f1_macro", "f1_hoax", "roc_auc")
    }

    save_json(test_metrics, os.path.join(run_dir, "test_metrics.json"))
    save_json(test_ci, os.path.join(run_dir, "test_metrics_ci.json"))
    save_json(
        {"classifier": classifier, "seed": seed, "tag": tag,
         "experiment": "baseline_tfidf", "model_name": classifier,
         "aug_method": "none", "imbalance_strategy": "class_weight"},
        os.path.join(run_dir, "run_config.json"),
    )
    np.save(os.path.join(run_dir, "test_y_true.npy"), y_true)
    np.save(os.path.join(run_dir, "test_y_pred.npy"), y_pred)
    if y_proba is not None:
        np.save(os.path.join(run_dir, "test_y_proba.npy"), y_proba)

    plot_confusion_matrix(
        y_true, y_pred,
        out_path=os.path.join(run_dir, "confusion_matrix"),
        id2label=ID2LABEL,
        title=f"Confusion - {classifier} seed={seed}",
    )
    if y_proba is not None:
        plot_roc_curve(y_true, y_proba, out_path=os.path.join(run_dir, "roc"))
        plot_pr_curve(y_true, y_proba, out_path=os.path.join(run_dir, "pr"))

    return RunOutput(
        run_dir=run_dir,
        tag=tag,
        seed=seed,
        model_name=classifier,
        aug_method="none",
        imbalance="class_weight",
        test_metrics=test_metrics,
        test_metric_ci=test_ci,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )
