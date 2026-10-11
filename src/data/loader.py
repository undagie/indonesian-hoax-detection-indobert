"""Load the Indonesian hoax / fake-news corpus as ``text`` / ``label`` rows."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from config import (
    HOAX_ALLOW_SYNTHETIC,
    HOAX_DATASET,
    HOAX_LABEL_COL,
    HOAX_TEXT_COL,
    LABEL_MAP,
    RAW_DATA_DIR,
)

log = logging.getLogger(__name__)


CANONICAL_LABEL_ALIASES = {
    "hoax": "hoax",
    "fake": "hoax",
    "false": "hoax",
    "1": "hoax",
    1: "hoax",
    True: "hoax",
    "fakta": "fakta",
    "valid": "fakta",
    "true": "fakta",
    "real": "fakta",
    "0": "fakta",
    0: "fakta",
    False: "fakta",
}


def _canonicalize_label(value) -> Optional[str]:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        v = value.strip().lower()
    else:
        v = value
    return CANONICAL_LABEL_ALIASES.get(v, None)


def list_available_datasets() -> List[str]:
    """List subfolders in ``dataset/raw/`` that look like datasets (contain a CSV)."""
    root = Path(RAW_DATA_DIR)
    if not root.exists():
        return []
    out = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and any(child.glob("*.csv")):
            out.append(child.name)
    return out


def _read_csv_any(path: Path) -> pd.DataFrame:
    """Read CSV trying common separators / encodings used by Indonesian datasets."""
    for sep in (",", ";", "\t"):
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
                if df.shape[1] >= 2:
                    return df
            except Exception:
                continue
    raise IOError(f"Could not parse CSV at {path}")


def _make_synthetic(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a small Indonesian-flavoured synthetic dataset for smoke tests."""
    rng = np.random.default_rng(seed)
    hoax_templates = [
        "BREAKING!!! Pemerintah diam-diam mengesahkan UU rahasia yang merugikan rakyat, sebarkan!",
        "Ilmuwan menemukan ramuan herbal ajaib yang menyembuhkan {penyakit} dalam 3 hari tanpa obat.",
        "Awas! Vaksin {nama} terbukti mengandung microchip pelacak, hentikan vaksinasi sekarang.",
        "Foto viral menunjukkan {tokoh} bertemu alien di gedung putih kemarin malam.",
        "Bumi sebenarnya datar dan NASA menyembunyikan kebenaran selama puluhan tahun.",
    ]
    fakta_templates = [
        "Menurut data BPS, inflasi tahunan tercatat sebesar {n} persen pada bulan ini.",
        "Penelitian terbaru yang diterbitkan di jurnal Nature menjelaskan mekanisme {topik} secara rinci.",
        "Bank Indonesia mengumumkan suku bunga acuan ditahan di level {n} persen.",
        "Tim peneliti dari ITB merilis prototipe sistem deteksi {topik} dengan akurasi tinggi.",
        "Kementerian Kesehatan melaporkan penurunan kasus {penyakit} sebesar {n} persen tahun ini.",
    ]
    fillers = {
        "penyakit": ["diabetes", "kanker", "hipertensi", "demam berdarah"],
        "nama": ["A", "B", "C"],
        "tokoh": ["presiden", "menteri", "selebriti"],
        "topik": ["machine learning", "perubahan iklim", "energi terbarukan"],
        "n": ["2", "3.5", "5.1", "7"],
    }

    def _fill(template: str) -> str:
        out = template
        for key, vals in fillers.items():
            placeholder = "{" + key + "}"
            if placeholder in out:
                out = out.replace(placeholder, rng.choice(vals))
        return out

    rows = []
    for _ in range(n // 2):
        rows.append({"text": _fill(rng.choice(hoax_templates)), "label": "hoax"})
        rows.append({"text": _fill(rng.choice(fakta_templates)), "label": "fakta"})
    rng.shuffle(rows)
    return pd.DataFrame(rows)


KAGGLE_DATASET_NAME = "kaggle_indonesiafalsenews"
KAGGLE_LABELED_FILE = "Data_latih.csv"


def _load_kaggle_indonesiafalsenews(dataset_dir: Path) -> pd.DataFrame:
    """Load ``Data_latih.csv`` as ``text = judul + narasi``. Ignores unlabelled ``Data_uji.csv``."""
    fp = dataset_dir / KAGGLE_LABELED_FILE
    if not fp.exists():
        raise FileNotFoundError(
            f"Expected {fp} for the Kaggle indonesiafalsenews dataset. "
            "Re-download it via `python scripts/download_data.py`."
        )
    log.info("Loading %s", fp)
    raw = _read_csv_any(fp)
    needed = {"label", "judul", "narasi"}
    missing = needed - set(raw.columns)
    if missing:
        raise KeyError(
            f"{fp} is missing expected columns {missing}; found {list(raw.columns)}."
        )
    judul = raw["judul"].fillna("").astype(str).str.strip()
    narasi = raw["narasi"].fillna("").astype(str).str.strip()
    text = (judul + " " + narasi).str.strip()
    return pd.DataFrame({"text": text, "label_raw": raw["label"]})


def load_dataset(
    name: Optional[str] = None,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
    synthetic_if_missing: Optional[bool] = None,
    synthetic_n: int = 200,
) -> pd.DataFrame:
    """Load the configured dataset. Falls back to a small synthetic set if allowed."""
    name = name or HOAX_DATASET
    text_col = text_col or HOAX_TEXT_COL
    label_col = label_col or HOAX_LABEL_COL

    if synthetic_if_missing is None:
        synthetic_if_missing = HOAX_ALLOW_SYNTHETIC

    dataset_dir = Path(RAW_DATA_DIR) / name

    if name == KAGGLE_DATASET_NAME:
        if not (dataset_dir / KAGGLE_LABELED_FILE).exists():
            if synthetic_if_missing:
                log.warning(
                    "Kaggle dataset not found at %s. Returning synthetic data (n=%d).",
                    dataset_dir,
                    synthetic_n,
                )
                df = _make_synthetic(n=synthetic_n)
            else:
                raise FileNotFoundError(
                    f"{dataset_dir / KAGGLE_LABELED_FILE} not found. "
                    "Run `python scripts/download_data.py` first."
                )
        else:
            df = _load_kaggle_indonesiafalsenews(dataset_dir)
    else:
        csv_files = sorted(dataset_dir.glob("*.csv")) if dataset_dir.exists() else []
        if not csv_files:
            if synthetic_if_missing:
                log.warning(
                    "Dataset '%s' not found at %s. Returning synthetic data (n=%d).",
                    name,
                    dataset_dir,
                    synthetic_n,
                )
                df = _make_synthetic(n=synthetic_n)
            else:
                raise FileNotFoundError(
                    f"No CSV files in {dataset_dir}. Place a CSV with columns "
                    f"'{text_col}' and '{label_col}' there."
                )
        else:
            frames = []
            for fp in csv_files:
                log.info("Loading %s", fp)
                frames.append(_read_csv_any(fp))
            df = pd.concat(frames, ignore_index=True)
            if text_col not in df.columns or label_col not in df.columns:
                raise KeyError(
                    f"Expected columns '{text_col}' and '{label_col}' in dataset; "
                    f"found {list(df.columns)}. Set HOAX_TEXT_COL / HOAX_LABEL_COL env vars."
                )
            df = df[[text_col, label_col]].copy()
            df.columns = ["text", "label_raw"]

    df["label_str"] = df["label_raw"].apply(_canonicalize_label)
    n_before = len(df)
    df = df.dropna(subset=["text", "label_str"])
    df = df[df["text"].astype(str).str.strip().str.len() > 0]
    df = df[df["label_str"].isin(LABEL_MAP.keys())]
    n_after = len(df)
    if n_after < n_before:
        log.info(
            "Dropped %d rows with empty text or missing/unmappable labels (kept %d).",
            n_before - n_after,
            n_after,
        )

    df["label"] = df["label_str"].map(LABEL_MAP).astype(int)
    df = df[["text", "label", "label_str"]].reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    return df
