"""Conservative text cleaning for Transformer and TF-IDF pipelines."""

from __future__ import annotations

import html
import re
from typing import Iterable, Optional

import pandas as pd

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_WS_RE = re.compile(r"\s+")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")
_NON_PRINT_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(
    text: str,
    *,
    keep_hashtag_text: bool = True,
    lowercase: bool = False,
) -> str:
    """Strip URLs, HTML, mentions; optionally keep hashtag text; optional lowercase."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = _NON_PRINT_RE.sub(" ", t)
    t = _HTML_TAG_RE.sub(" ", t)
    t = _URL_RE.sub(" ", t)
    t = _MENTION_RE.sub(" ", t)
    if keep_hashtag_text:
        t = _HASHTAG_RE.sub(r"\1", t)
    else:
        t = _HASHTAG_RE.sub(" ", t)
    t = _REPEAT_CHAR_RE.sub(r"\1\1", t)
    t = _MULTI_WS_RE.sub(" ", t).strip()
    if lowercase:
        t = t.lower()
    return t


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    out_col: str = "text_clean",
    min_chars: int = 10,
    drop_duplicates: bool = True,
    lowercase: bool = False,
) -> pd.DataFrame:
    """Apply ``clean_text`` over a DataFrame and drop empty/duplicate rows.

    Returns a copy with ``out_col`` added (or replacing ``text_col`` if equal).
    """
    df = df.copy()
    df[out_col] = df[text_col].apply(lambda x: clean_text(x, lowercase=lowercase))
    n0 = len(df)
    df = df[df[out_col].str.len() >= min_chars]
    if drop_duplicates:
        df = df.drop_duplicates(subset=[out_col])
    df = df.reset_index(drop=True)
    return df


def basic_token_stats(texts: Iterable[str]) -> dict:
    """Return basic length statistics in tokens (whitespace-split)."""
    lens = [len(str(t).split()) for t in texts]
    if not lens:
        return {"n": 0}
    import numpy as np

    arr = np.asarray(lens)
    return {
        "n": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
