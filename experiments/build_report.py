"""Aggregate ``results/`` into master tables and ``report.md`` (no torch required)."""

from __future__ import annotations

import experiments._bootstrap
import glob
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_report")


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
REPORT_FIG_DIR = os.path.join(RESULTS_DIR, "figures", "report")

METRIC_COLS = (
    "accuracy", "f1_macro", "f1_weighted", "mcc",
    "precision_hoax", "recall_hoax", "f1_hoax",
    "roc_auc", "pr_auc",
)




def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("Failed to read %s: %s", path, e)
        return None


def _seed_from_dir(name: str) -> Optional[int]:
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else None


def _model_prefix_from_leaf(leaf: str) -> Optional[str]:
    """Extract a model prefix from a leaf like 'logreg_seed42' -> 'logreg'."""
    m = re.match(r"(.+?)_seed\d+$", leaf)
    return m.group(1) if m else None


def _walk_runs() -> List[Dict[str, Any]]:
    """Return one dict per run directory containing test_metrics.json."""
    rows: List[Dict[str, Any]] = []
    if not os.path.isdir(RUNS_DIR):
        log.warning("No runs directory at %s", RUNS_DIR)
        return rows

    for metrics_path in sorted(glob.glob(os.path.join(RUNS_DIR, "*", "*", "test_metrics.json"))):
        run_dir = os.path.dirname(metrics_path)
        tag = os.path.basename(os.path.dirname(run_dir))
        leaf = os.path.basename(run_dir)
        metrics = _load_json(metrics_path) or {}
        cfg = _load_json(os.path.join(run_dir, "run_config.json")) or {}
        ci = _load_json(os.path.join(run_dir, "test_metrics_ci.json")) or {}
        seed = _seed_from_dir(leaf) or cfg.get("seed")



        prefix = _model_prefix_from_leaf(leaf)
        effective_tag = f"{tag}__{prefix}" if (prefix and prefix.lower() != "seed") else tag
        row = {
            "tag": effective_tag,
            "tag_raw": tag,
            "run_leaf": leaf,
            "seed": seed,
            "model_name": cfg.get("model_name", prefix or "unknown"),
            "aug_method": cfg.get("aug_method", "n/a"),
            "imbalance": cfg.get("imbalance_strategy", "n/a"),
            "experiment": cfg.get("experiment", "n/a"),
            "epochs": cfg.get("epochs"),
        }
        for k in METRIC_COLS:
            v = metrics.get(k)
            row[k] = float(v) if isinstance(v, (int, float)) else None
        for k in ("f1_macro", "accuracy", "roc_auc"):
            ci_entry = ci.get(k, {}) if isinstance(ci, dict) else {}
            if isinstance(ci_entry, dict):
                row[f"{k}_ci_low"] = ci_entry.get("ci_low")
                row[f"{k}_ci_high"] = ci_entry.get("ci_high")
        rows.append(row)
    return rows


def _aggregate_by_tag(per_run: pd.DataFrame) -> pd.DataFrame:
    """Compute mean/std/95% normal-CI per tag across seeds."""
    if per_run.empty:
        return pd.DataFrame()
    metric_cols = [c for c in METRIC_COLS if c in per_run.columns]
    grouped = per_run.groupby("tag")
    rows = []
    for tag, sub in grouped:
        row = {"tag": tag, "n_seeds": len(sub)}
        for m in metric_cols:
            vals = pd.to_numeric(sub[m], errors="coerce").dropna().to_numpy()
            if vals.size == 0:
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            sem = std / np.sqrt(vals.size) if vals.size > 1 else 0.0
            ci = 1.96 * sem
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_ci_low"] = mean - ci
            row[f"{m}_ci_high"] = mean + ci

        for col in ("model_name", "aug_method", "imbalance", "experiment"):
            row[col] = sub[col].iloc[0] if col in sub.columns else "n/a"
        rows.append(row)
    df = pd.DataFrame(rows)
    if "f1_macro_mean" in df.columns:
        df = df.sort_values("f1_macro_mean", ascending=False)
    return df


def _write_table_no_torch(df: pd.DataFrame, stem: str, *, caption: str = "", label: str = "") -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    df.to_csv(os.path.join(TABLES_DIR, f"{stem}.csv"), index=False)


    cols = [str(c) for c in df.columns]
    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return "" if v is None else str(v)
    rows = [[_fmt(v) for v in r] for r in df.itertuples(index=False, name=None)]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c) for i, c in enumerate(cols)]
    md_lines = [
        "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |",
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    for r in rows:
        md_lines.append("| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |")
    with open(os.path.join(TABLES_DIR, f"{stem}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


    try:
        df.to_latex(
            os.path.join(TABLES_DIR, f"{stem}.tex"),
            index=False, float_format="%.4f",
            caption=caption or stem, label=label or f"tab:{stem}",
            escape=True,
        )
    except Exception as e:
        log.warning("LaTeX export failed for %s: %s", stem, e)


def _plot_comparison(agg: pd.DataFrame, out_path: str) -> None:
    if agg.empty or "f1_macro_mean" not in agg.columns:
        return
    agg = agg.dropna(subset=["f1_macro_mean"]).sort_values("f1_macro_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(agg) + 2), 4.5))
    x = np.arange(len(agg))
    ax.bar(x, agg["f1_macro_mean"], color="#4C72B0")
    if "f1_macro_ci_low" in agg.columns and "f1_macro_ci_high" in agg.columns:
        lo = agg["f1_macro_mean"].values - agg["f1_macro_ci_low"].values
        hi = agg["f1_macro_ci_high"].values - agg["f1_macro_mean"].values
        lo = np.where(np.isfinite(lo), lo, 0.0)
        hi = np.where(np.isfinite(hi), hi, 0.0)
        ax.errorbar(x, agg["f1_macro_mean"].values, yerr=[lo, hi],
                    fmt="none", ecolor="#333", capsize=4, linewidth=1.2)
    for i, v in enumerate(agg["f1_macro_mean"].values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["tag"], rotation=40, ha="right")
    ax.set_ylabel("F1-macro (mean across seeds)")
    ax.set_title("All experiments: F1-macro comparison")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _inventory() -> pd.DataFrame:
    rows = []
    if not os.path.isdir(RESULTS_DIR):
        return pd.DataFrame()
    for dirpath, _, filenames in os.walk(RESULTS_DIR):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, REPO_ROOT)
            ext = os.path.splitext(fn)[1].lower().lstrip(".")
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            rows.append({"path": rel.replace("\\", "/"), "ext": ext, "size_bytes": size})
    df = pd.DataFrame(rows).sort_values("path")
    return df


def _write_markdown_report(agg: pd.DataFrame, per_run: pd.DataFrame, inventory: pd.DataFrame) -> str:
    lines = ["# Master experiment report", ""]
    lines.append(f"- Total runs found: **{len(per_run)}**")
    lines.append(f"- Unique experiment tags: **{per_run['tag'].nunique() if not per_run.empty else 0}**")
    lines.append(f"- Total result artifacts: **{len(inventory)}**")
    lines.append("")

    if not agg.empty and "f1_macro_mean" in agg.columns:
        top = agg.head(5)[["tag", "n_seeds", "f1_macro_mean", "f1_macro_std",
                            "accuracy_mean", "roc_auc_mean"]]
        lines.append("## Top-5 by F1-macro (mean across seeds)")
        lines.append("")
        for _, r in top.iterrows():
            lines.append(
                f"- **{r['tag']}** F1={r['f1_macro_mean']:.4f} "
                f"(+/-{r['f1_macro_std']:.4f}, n_seeds={int(r['n_seeds'])}) "
                f"| Acc={r['accuracy_mean']:.4f} "
                f"| ROC-AUC={r.get('roc_auc_mean', float('nan')):.4f}"
            )
        lines.append("")

    lines.append("## Where to find things")
    lines.append("")
    lines.append("- Per-run raw: `results/runs/<tag>/<leaf>/`")
    lines.append("- Aggregated tables: `results/tables/master_aggregated.{csv,md,tex}`")
    lines.append("- Per-run table: `results/tables/master_per_run.csv`")
    lines.append("- Headline JSON: `results/tables/headline.json`")
    lines.append("- All-models comparison plot: `results/figures/report/compare_all_models.png`")
    lines.append("- Inventory of every artifact: `results/tables/inventory.csv`")
    lines.append("")

    if not agg.empty:
        lines.append("## Full aggregated table")
        lines.append("")
        cols = ["tag", "n_seeds", "f1_macro_mean", "f1_macro_std",
                "accuracy_mean", "roc_auc_mean", "model_name",
                "aug_method", "imbalance"]
        cols = [c for c in cols if c in agg.columns]
        sub = agg[cols].copy()
        sub.columns = [c.replace("_mean", "").replace("_std", "_std") for c in cols]
        lines.append(sub.to_csv(index=False, sep="|").replace("\n", "\n").strip())
        lines.append("")

    out = os.path.join(RESULTS_DIR, "report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out




def main() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(REPORT_FIG_DIR, exist_ok=True)

    rows = _walk_runs()
    per_run = pd.DataFrame(rows)
    log.info("Found %d completed runs across %d tags",
             len(per_run), per_run["tag"].nunique() if not per_run.empty else 0)

    if not per_run.empty:
        per_run.to_csv(os.path.join(TABLES_DIR, "master_per_run.csv"), index=False)

    agg = _aggregate_by_tag(per_run)
    if not agg.empty:
        _write_table_no_torch(
            agg, "master_aggregated",
            caption="Master results: every experiment tag aggregated across seeds",
            label="tab:master_agg",
        )


    headline: Dict[str, Any] = {}
    if not agg.empty and "f1_macro_mean" in agg.columns:
        best = agg.iloc[0]
        headline = {
            "best_tag": str(best["tag"]),
            "best_f1_macro": float(best["f1_macro_mean"]),
            "best_f1_macro_std": float(best.get("f1_macro_std", 0.0) or 0.0),
            "best_accuracy": float(best.get("accuracy_mean", float("nan")) or float("nan")),
            "best_roc_auc": float(best.get("roc_auc_mean", float("nan")) or float("nan")),
            "best_n_seeds": int(best.get("n_seeds", 0) or 0),
            "best_model": str(best.get("model_name", "n/a")),
            "best_aug": str(best.get("aug_method", "n/a")),
            "best_imbalance": str(best.get("imbalance", "n/a")),
        }
    with open(os.path.join(TABLES_DIR, "headline.json"), "w", encoding="utf-8") as f:
        json.dump(headline, f, indent=2)


    _plot_comparison(agg, os.path.join(REPORT_FIG_DIR, "compare_all_models.png"))


    inv = _inventory()
    inv.to_csv(os.path.join(TABLES_DIR, "inventory.csv"), index=False)


    report_path = _write_markdown_report(agg, per_run, inv)
    log.info("Report written: %s", report_path)
    log.info("Done. Open results/report.md to see the summary.")


if __name__ == "__main__":
    main()
