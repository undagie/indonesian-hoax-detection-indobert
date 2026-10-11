"""Apply a named augmenter to a DataFrame; optionally target the minority class."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.augment.registry import get_augmenter

log = logging.getLogger(__name__)


def augment_dataframe(
    df: pd.DataFrame,
    method: str,
    *,
    text_col: str = "text",
    label_col: str = "label",
    ratio: float = 1.0,
    prob: float = 0.15,
    minority_only: bool = True,
    seed: Optional[int] = 42,
    **augmenter_kwargs,
) -> pd.DataFrame:
    """Append augmented copies. ``ratio`` is copies per selected row; ``prob`` is token noise."""
    if method == "none" or ratio <= 0:
        out = df.copy()
        out["is_augmented"] = False
        return out

    augmenter_kwargs = dict(augmenter_kwargs)
    if method in {"synonym", "swap", "delete", "insert"}:
        augmenter_kwargs.setdefault("alpha", prob)
    elif method == "contextual":
        augmenter_kwargs.setdefault("prob", prob)
    augmenter = get_augmenter(method, seed=seed, **augmenter_kwargs)

    counts = df[label_col].value_counts()
    target_labels = (
        [int(counts.idxmin())] if minority_only else sorted(df[label_col].unique())
    )
    log.info(
        "Augmenting with '%s' (ratio=%.2f, prob=%.2f, target_labels=%s)",
        method, ratio, prob, target_labels,
    )

    augmented_rows = []
    for label in target_labels:
        subset = df[df[label_col] == label]
        n_per_row = max(1, int(round(ratio)))
        for _, row in subset.iterrows():
            for _ in range(n_per_row):
                new_text = augmenter.augment(str(row[text_col]))
                if not new_text or new_text == row[text_col]:
                    continue
                new_row = row.to_dict()
                new_row[text_col] = new_text
                new_row["is_augmented"] = True
                augmented_rows.append(new_row)

    orig = df.copy()
    orig["is_augmented"] = False
    if not augmented_rows:
        log.warning("Augmenter produced 0 new rows.")
        return orig
    aug_df = pd.DataFrame(augmented_rows)
    out = pd.concat([orig, aug_df], ignore_index=True)
    log.info("Augmented dataset size: %d (orig=%d, added=%d)",
             len(out), len(orig), len(aug_df))
    return out
