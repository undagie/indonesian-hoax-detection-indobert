"""Experiment 04: class-imbalance strategy ablation on IndoBERT."""

from __future__ import annotations

import experiments._bootstrap
import os
from typing import List

import pandas as pd

from config import FIGURES_DIR, IMBALANCE_STRATEGIES, SEEDS
from src.eval.statistical import aggregate_seeds
from src.utils.io import setup_logging, write_table
from src.utils.runner import train_transformer_run
from src.viz.ablation import plot_ablation_bars, plot_seed_variance


SKIP_FOR_NOW = {"smote_embed"}


def main(
    strategies: List[str] = None,
    seeds: List[int] = None,
    epochs: int = None,
    aug_method: str = "none",
    synthetic_n: int = 400,
) -> None:
    log = setup_logging("exp04_imb_ablation")
    strategies = strategies or [s for s in IMBALANCE_STRATEGIES if s not in SKIP_FOR_NOW]
    seeds = seeds or SEEDS

    per_seed_rows = []
    agg_rows = []
    for strat in strategies:
        per_seed = []
        for seed in seeds:
            log.info("[imbalance=%s] seed=%d", strat, seed)
            try:
                out = train_transformer_run(
                    tag=f"imb_{strat}",
                    seed=seed,
                    aug_method=aug_method,
                    imbalance=strat,
                    epochs=epochs,
                    synthetic_n=synthetic_n,
                    save_best_model=False,
                )
            except Exception as e:
                log.exception("Imbalance '%s' failed on seed %d: %s", strat, seed, e)
                continue
            per_seed.append(out.test_metrics)
            per_seed_rows.append({
                "imbalance": strat, "seed": seed,
                **{k: v for k, v in out.test_metrics.items() if isinstance(v, (int, float))},
            })
        if not per_seed:
            continue
        agg = aggregate_seeds(per_seed)
        flat = {"imbalance": strat}
        for metric, stats in agg.items():
            flat[f"{metric}_mean"] = stats["mean"]
            flat[f"{metric}_std"] = stats["std"]
            flat[f"{metric}_ci95_low"] = stats["ci95_low"]
            flat[f"{metric}_ci95_high"] = stats["ci95_high"]
        agg_rows.append(flat)

    per_seed_df = pd.DataFrame(per_seed_rows)
    if not agg_rows:
        log.error("Imbalance ablation produced 0 successful runs -- check the "
                  "stack traces above. Skipping table/figure writes.")
        return
    agg_df = pd.DataFrame(agg_rows)
    if "f1_macro_mean" in agg_df.columns:
        agg_df = agg_df.sort_values("f1_macro_mean", ascending=False)

    write_table(per_seed_df, stem="imb_ablation_per_seed",
                caption="Per-seed imbalance ablation metrics",
                label="tab:imb_ablation_per_seed")
    write_table(agg_df, stem="imb_ablation_aggregated",
                caption="Imbalance ablation (mean / std / 95% CI)",
                label="tab:imb_ablation_agg")

    ablation_dir = os.path.join(FIGURES_DIR, "ablation")
    plot_ablation_bars(
        agg_df.rename(columns={"f1_macro_mean": "f1_macro"}),
        group_col="imbalance", metric_col="f1_macro",
        err_low_col="f1_macro_ci95_low", err_high_col="f1_macro_ci95_high",
        out_path=os.path.join(ablation_dir, "imb_ablation_f1_macro"),
        title="Imbalance ablation: F1-macro (mean +/- 95% CI)",
        ylabel="F1-macro",
    )
    plot_seed_variance(
        per_seed_df, group_col="imbalance", metric_col="f1_macro",
        out_path=os.path.join(ablation_dir, "imb_ablation_f1_macro_variance"),
        title="Per-seed F1-macro by imbalance strategy",
    )
    log.info("Imbalance ablation done.")


if __name__ == "__main__":
    main()
