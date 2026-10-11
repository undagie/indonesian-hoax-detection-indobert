"""Run every experiment step in order. Finished runs are skipped automatically."""

from __future__ import annotations

import experiments._bootstrap
import argparse
import logging
import os
import sys
import time
from typing import List

from src.utils.io import setup_logging

STEPS = ["eda", "baselines", "aug", "imb", "compare", "err", "report"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run every hoax-detection experiment.")
    p.add_argument("--quick", action="store_true",
                   help="Quick smoke: 2 seeds, 3 epochs, fewer augmentation variants.")
    p.add_argument("--skip", default="",
                   help=f"Comma-separated steps to skip. Available: {','.join(STEPS)}")
    p.add_argument("--only", default="",
                   help="Run ONLY the comma-separated steps.")
    p.add_argument("--synthetic-n", type=int, default=400,
                   help="Synthetic data size when no real dataset is provided.")
    return p.parse_args()


def _should_run(step: str, skip: List[str], only: List[str]) -> bool:
    if only:
        return step in only
    return step not in skip


def main() -> None:
    args = _parse_args()
    log = setup_logging("run_all")
    skip = [s.strip() for s in args.skip.split(",") if s.strip()]
    only = [s.strip() for s in args.only.split(",") if s.strip()]

    if args.quick:
        seeds = [42, 101]
        epochs = 3
        aug_subset = ["none", "synonym", "swap", "delete", "back_translation"]
        imb_subset = [
            "none", "class_weight", "weighted_sampler",
            "random_oversample", "focal_loss", "combined",
        ]
    else:
        from config import AUG_METHODS, IMBALANCE_STRATEGIES, SEEDS

        seeds = list(SEEDS)
        epochs = None
        aug_subset = list(AUG_METHODS)
        imb_subset = [s for s in IMBALANCE_STRATEGIES if s != "smote_embed"]

    log.info("=== run_all | seeds=%s | epochs=%s | quick=%s ===",
             seeds, epochs, args.quick)
    t0 = time.time()

    if _should_run("eda", skip, only):
        log.info("Step: EDA")
        from experiments.exp01_eda import main as eda_main

        eda_main(synthetic_n=max(args.synthetic_n, 200))

    if _should_run("baselines", skip, only):
        log.info("Step: classical baselines")
        from experiments.exp02_baselines import main as base_main

        base_main(synthetic_n=args.synthetic_n)

    if _should_run("aug", skip, only):
        log.info("Step: augmentation ablation")
        from experiments.exp03_augmentation_ablation import main as aug_main

        aug_main(
            methods=aug_subset, seeds=seeds, epochs=epochs,
            imbalance="none", synthetic_n=args.synthetic_n,
        )

    if _should_run("imb", skip, only):
        log.info("Step: imbalance ablation")
        from experiments.exp04_imbalance_ablation import main as imb_main

        imb_main(
            strategies=imb_subset, seeds=seeds, epochs=epochs,
            aug_method="none", synthetic_n=args.synthetic_n,
        )

    if _should_run("compare", skip, only):
        log.info("Step: full comparison")
        from experiments.exp05_full_comparison import main as cmp_main

        cmp_main(seeds=seeds, epochs=epochs, synthetic_n=args.synthetic_n)

    if _should_run("err", skip, only):
        log.info("Step: error analysis")
        from experiments.exp06_error_analysis import main as err_main

        err_main(synthetic_n=args.synthetic_n)

    if _should_run("report", skip, only):
        log.info("Step: build master report")
        from experiments.build_report import main as report_main

        report_main()

    log.info("=== DONE in %.1f min ===", (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
