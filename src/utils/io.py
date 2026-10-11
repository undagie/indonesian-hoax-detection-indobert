"""I/O helpers: JSON, multi-format table writer, logging setup."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterable, Optional

import pandas as pd

from config import LOGS_DIR, TABLES_DIR


def setup_logging(name: str = "hoaxid", level: int = logging.INFO) -> logging.Logger:
    """Console + file logging. File goes to ``results/logs/<name>.log``."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = logging.getLogger()
    if getattr(setup_logging, "_done", False):
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s | %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(LOGS_DIR, f"{name}.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    setattr(setup_logging, "_done", True)
    return logger


def save_json(obj: Any, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    return path


def _json_default(o: Any):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_table(
    df: pd.DataFrame,
    stem: str,
    *,
    out_dir: Optional[str] = None,
    formats: Iterable[str] = ("csv", "md", "tex"),
    float_format: str = "%.4f",
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> dict:
    """Write a DataFrame to multiple formats (CSV, Markdown, LaTeX).

    Returns a dict of ``{format: path}``.
    """
    out_dir = out_dir or TABLES_DIR
    os.makedirs(out_dir, exist_ok=True)
    paths: dict = {}
    if "csv" in formats:
        p = os.path.join(out_dir, f"{stem}.csv")
        df.to_csv(p, index=False)
        paths["csv"] = p
    if "md" in formats:
        p = os.path.join(out_dir, f"{stem}.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write(_to_markdown(df, float_format=float_format))
        paths["md"] = p
    if "tex" in formats:
        p = os.path.join(out_dir, f"{stem}.tex")
        df.to_latex(
            p,
            index=False,
            float_format=float_format,
            caption=caption,
            label=label,
            escape=True,
        )
        paths["tex"] = p
    return paths


def _to_markdown(df: pd.DataFrame, *, float_format: str = "%.4f") -> str:
    """Render a DataFrame as GFM markdown without requiring the ``tabulate`` package."""
    try:
        return df.to_markdown(index=False, floatfmt=float_format.replace("%", ""))
    except ImportError:
        pass
    cols = [str(c) for c in df.columns]
    def _fmt(v):
        if isinstance(v, float):
            try:
                return float_format % v
            except (TypeError, ValueError):
                return str(v)
        return str(v)
    rows = [[_fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c) for i, c in enumerate(cols)]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = "\n".join("| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |" for r in rows)
    return f"{header}\n{sep}\n{body}\n"
