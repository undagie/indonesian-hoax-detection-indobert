"""Experiment 05: proposed pipeline vs baselines with significance tests.

Proposed config: best imbalance strategy from exp04, no text augmentation.
Writes comparison tables and McNemar / paired-bootstrap results under results/.
"""

from __future__ import annotations

import experiments._bootstrap
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import FIGURES_DIR, SEEDS, TABLES_DIR
from src.eval.statistical import (
    aggregate_seeds,
    mcnemar_test,
    paired_bootstrap_test,
)
from src.models.baselines import AVAILABLE_BASELINES
from src.utils.io import load_json, setup_logging, write_table
from src.utils.runner import RunOutput, train_baseline_run, train_transformer_run
from src.viz.ablation import plot_ablation_bars




def _read_best(stem: str, group_col: str) -> Optional[str]:
    path = os.path.join(TABLES_DIR, f"{stem}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df = df.sort_values("f1_macro_mean", ascending=False)
    return str(df.iloc[0][group_col])


def main(
    seeds: List[int] = None,
    epochs: int = None,
    best_aug: Optional[str] = None,
    best_imb: Optional[str] = None,
    baselines: List[str] = None,
    synthetic_n: int = 400,
) -> None:
    log = setup_logging("exp05_full_comparison")
    seeds = seeds or SEEDS
    baselines = baselines or AVAILABLE_BASELINES

    best_aug = best_aug or _read_best("aug_ablation_aggregated", "augmentation") or "synonym"
    best_imb = best_imb or _read_best("imb_ablation_aggregated", "imbalance") or "focal_loss"
    log.info("Best augmentation = %s | Best imbalance = %s", best_aug, best_imb)

    runs_per_model: Dict[str, List[RunOutput]] = {}


    for clf in baselines:
        runs_per_model[f"tfidf_{clf}"] = []
        for seed in seeds:
            log.info("[baseline %s] seed=%d", clf, seed)
            runs_per_model[f"tfidf_{clf}"].append(
                train_baseline_run(tag="cmp_baseline_tfidf", seed=seed,
                                   classifier=clf, synthetic_n=synthetic_n)
            )


    variants: List[Tuple[str, str, str]] = [
        ("transformer_vanilla", "none", "none"),
        (f"transformer_aug-{best_aug}", best_aug, "none"),
        (f"transformer_proposed", "none", best_imb),
        (f"transformer_aug-imb", best_aug, best_imb),
    ]
    for tag, aug, imb in variants:
        runs_per_model[tag] = []
        for seed in seeds:
            log.info("[%s] seed=%d (aug=%s, imb=%s)", tag, seed, aug, imb)
            runs_per_model[tag].append(
                train_transformer_run(
                    tag=f"cmp_{tag}",
                    seed=seed,
                    aug_method=aug,
                    imbalance=imb,
                    epochs=epochs,
                    synthetic_n=synthetic_n,
                )
            )


    per_seed_rows = []
    agg_rows = []
    for model_name, runs in runs_per_model.items():
        per_seed_metrics = [r.test_metrics for r in runs]
        for r in runs:
            per_seed_rows.append({
                "model": model_name, "seed": r.seed,
                **{k: v for k, v in r.test_metrics.items() if isinstance(v, (int, float))},
            })
        agg = aggregate_seeds(per_seed_metrics)
        flat = {"model": model_name}
        for metric, stats in agg.items():
            flat[f"{metric}_mean"] = stats["mean"]
            flat[f"{metric}_std"] = stats["std"]
            flat[f"{metric}_ci95_low"] = stats["ci95_low"]
            flat[f"{metric}_ci95_high"] = stats["ci95_high"]
        agg_rows.append(flat)

    per_seed_df = pd.DataFrame(per_seed_rows)
    if not agg_rows:
        log.error("Full comparison produced 0 successful runs -- check the "
                  "stack traces above. Skipping table/figure writes.")
        return
    agg_df = pd.DataFrame(agg_rows)
    if "f1_macro_mean" in agg_df.columns:
        agg_df = agg_df.sort_values("f1_macro_mean", ascending=False)
    write_table(per_seed_df, stem="full_comparison_per_seed",
                caption="Per-seed metrics: baselines vs Transformer variants",
                label="tab:full_per_seed")
    write_table(agg_df, stem="full_comparison_aggregated",
                caption="Aggregated metrics: baselines vs Transformer variants (mean/std/95% CI)",
                label="tab:full_agg")


    proposed_runs = runs_per_model["transformer_proposed"]


    proposed_first = proposed_runs[0]
    sig_rows = []
    for model_name, runs in runs_per_model.items():
        if model_name == "transformer_proposed":
            continue

        first = runs[0]
        if len(first.y_true) != len(proposed_first.y_true):
            log.warning("Skipping significance for %s - test set size mismatch.", model_name)
            continue
        mc = mcnemar_test(
            y_true=proposed_first.y_true,
            y_pred_a=first.y_pred,
            y_pred_b=proposed_first.y_pred,
        )
        pb = paired_bootstrap_test(
            y_true=proposed_first.y_true,
            y_pred_a=first.y_pred,
            y_pred_b=proposed_first.y_pred,
            y_proba_a=first.y_proba,
            y_proba_b=proposed_first.y_proba,
            metric="f1_macro",
            n_bootstrap=1000,
            seed=proposed_first.seed,
        )
        sig_rows.append({
            "baseline": model_name,
            "mcnemar_stat": mc["statistic"],
            "mcnemar_p": mc["p_value"],
            "mcnemar_n01": mc["n01"],
            "mcnemar_n10": mc["n10"],
            "delta_f1_macro": pb["delta"],
            "paired_boot_p": pb["p_value"],
            "paired_boot_ci95_low": pb["ci95_low"],
            "paired_boot_ci95_high": pb["ci95_high"],
        })
    sig_df = pd.DataFrame(sig_rows).sort_values("paired_boot_p")
    write_table(sig_df, stem="significance_vs_proposed",
                caption="Statistical tests of the proposed pipeline vs each baseline (seed=first).",
                label="tab:sig_vs_proposed")


    ablation_dir = os.path.join(FIGURES_DIR, "ablation")
    plot_ablation_bars(
        agg_df.rename(columns={"f1_macro_mean": "f1_macro"}),
        group_col="model", metric_col="f1_macro",
        err_low_col="f1_macro_ci95_low", err_high_col="f1_macro_ci95_high",
        out_path=os.path.join(ablation_dir, "full_comparison_f1_macro"),
        title="All models: F1-macro (mean +/- 95% CI)",
        ylabel="F1-macro",
    )
    log.info("Full comparison done.")


if __name__ == "__main__":
    main()
