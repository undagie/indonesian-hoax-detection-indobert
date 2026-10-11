"""Experiment 02: TF-IDF baselines across seeds."""

from __future__ import annotations

import experiments._bootstrap
import os
from typing import List

import pandas as pd

from config import FIGURES_DIR, SEEDS, TABLES_DIR
from src.eval.statistical import aggregate_seeds
from src.models.baselines import AVAILABLE_BASELINES
from src.utils.io import setup_logging, write_table
from src.utils.runner import train_baseline_run
from src.viz.ablation import plot_seed_variance


def main(baselines: List[str] = None, synthetic_n: int = 400) -> None:
    log = setup_logging("exp02_baselines")
    baselines = baselines or AVAILABLE_BASELINES
    per_seed_rows = []
    aggregated_rows = []
    for clf in baselines:
        per_seed = []
        for seed in SEEDS:
            log.info("[%s] seed=%d", clf, seed)
            out = train_baseline_run(
                tag="baseline_tfidf",
                seed=seed,
                classifier=clf,
                synthetic_n=synthetic_n,
            )
            per_seed.append(out.test_metrics)
            per_seed_rows.append({
                "model": clf, "seed": seed,
                **{k: v for k, v in out.test_metrics.items() if isinstance(v, (int, float))},
            })
        agg = aggregate_seeds(per_seed)
        flat = {"model": clf}
        for metric, stats in agg.items():
            flat[f"{metric}_mean"] = stats["mean"]
            flat[f"{metric}_std"] = stats["std"]
        aggregated_rows.append(flat)

    per_seed_df = pd.DataFrame(per_seed_rows)
    agg_df = pd.DataFrame(aggregated_rows).sort_values("f1_macro_mean", ascending=False)

    write_table(per_seed_df, stem="baselines_per_seed",
                caption="Per-seed baseline metrics",
                label="tab:baselines_per_seed")
    write_table(agg_df, stem="baselines_aggregated",
                caption="Baseline metrics aggregated over seeds (mean / std)",
                label="tab:baselines_agg")

    ablation_dir = os.path.join(FIGURES_DIR, "ablation")
    plot_seed_variance(
        per_seed_df, group_col="model", metric_col="f1_macro",
        out_path=os.path.join(ablation_dir, "baselines_f1_macro_variance"),
        title="Baselines: F1-macro variance across seeds",
    )
    log.info("Baseline experiment done. See results/tables and %s.", ablation_dir)


if __name__ == "__main__":
    main()
