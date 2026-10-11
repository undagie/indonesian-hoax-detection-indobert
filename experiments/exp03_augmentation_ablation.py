"""Experiment 03: text-augmentation ablation on IndoBERT."""

from __future__ import annotations

import experiments._bootstrap
import os
from typing import List

import pandas as pd

from config import AUG_METHODS, FIGURES_DIR, SEEDS, TABLES_DIR
from src.eval.statistical import aggregate_seeds
from src.utils.io import setup_logging, write_table
from src.utils.runner import train_transformer_run
from src.viz.ablation import plot_ablation_bars, plot_seed_variance


def main(
    methods: List[str] = None,
    seeds: List[int] = None,
    epochs: int = None,
    imbalance: str = "none",
    synthetic_n: int = 400,
) -> None:
    log = setup_logging("exp03_aug_ablation")
    methods = methods or AUG_METHODS
    seeds = seeds or SEEDS
    per_seed_rows = []
    agg_rows = []

    for method in methods:
        per_seed = []
        for seed in seeds:
            log.info("[aug=%s] seed=%d", method, seed)
            try:
                out = train_transformer_run(
                    tag=f"aug_{method}",
                    seed=seed,
                    aug_method=method,
                    imbalance=imbalance,
                    epochs=epochs,
                    synthetic_n=synthetic_n,
                    save_best_model=False,
                )
            except Exception as e:
                log.exception("Augmentation '%s' failed on seed %d: %s", method, seed, e)
                continue
            per_seed.append(out.test_metrics)
            per_seed_rows.append({
                "augmentation": method, "seed": seed,
                **{k: v for k, v in out.test_metrics.items() if isinstance(v, (int, float))},
            })
        if not per_seed:
            continue
        agg = aggregate_seeds(per_seed)
        flat = {"augmentation": method}
        for metric, stats in agg.items():
            flat[f"{metric}_mean"] = stats["mean"]
            flat[f"{metric}_std"] = stats["std"]
            flat[f"{metric}_ci95_low"] = stats["ci95_low"]
            flat[f"{metric}_ci95_high"] = stats["ci95_high"]
        agg_rows.append(flat)

    per_seed_df = pd.DataFrame(per_seed_rows)
    if not agg_rows:
        log.error("Augmentation ablation produced 0 successful runs -- check the "
                  "stack traces above. Skipping table/figure writes.")
        return
    agg_df = pd.DataFrame(agg_rows)
    if "f1_macro_mean" in agg_df.columns:
        agg_df = agg_df.sort_values("f1_macro_mean", ascending=False)

    write_table(per_seed_df, stem="aug_ablation_per_seed",
                caption="Per-seed augmentation ablation metrics",
                label="tab:aug_ablation_per_seed")
    write_table(agg_df, stem="aug_ablation_aggregated",
                caption="Augmentation ablation (mean / std / 95% CI)",
                label="tab:aug_ablation_agg")

    ablation_dir = os.path.join(FIGURES_DIR, "ablation")
    plot_ablation_bars(
        agg_df.rename(columns={"f1_macro_mean": "f1_macro"}),
        group_col="augmentation", metric_col="f1_macro",
        err_low_col="f1_macro_ci95_low", err_high_col="f1_macro_ci95_high",
        out_path=os.path.join(ablation_dir, "aug_ablation_f1_macro"),
        title="Augmentation ablation: F1-macro (mean +/- 95% CI)",
        ylabel="F1-macro",
    )
    plot_seed_variance(
        per_seed_df, group_col="augmentation", metric_col="f1_macro",
        out_path=os.path.join(ablation_dir, "aug_ablation_f1_macro_variance"),
        title="Per-seed F1-macro by augmentation method",
    )
    log.info("Augmentation ablation done.")


if __name__ == "__main__":
    main()
