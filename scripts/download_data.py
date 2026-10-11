"""Download Kaggle ``muhammadghazimuharam/indonesiafalsenews`` into ``dataset/raw/``."""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("download_data")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RAW_DIR = os.path.join(REPO_ROOT, "dataset", "raw")
DATASET_NAME = "kaggle_indonesiafalsenews"
KAGGLE_SLUG = "muhammadghazimuharam/indonesiafalsenews"
LABELLED_FILE = "Data_latih.csv"


def already_present(name: str = DATASET_NAME) -> bool:
    return os.path.exists(os.path.join(RAW_DIR, name, LABELLED_FILE))


def _resolve_kaggle_credentials() -> str:
    """Locate ``kaggle.json`` via KAGGLE_CONFIG_DIR, ~/.kaggle, or repo root."""
    env_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if env_dir and os.path.exists(os.path.join(env_dir, "kaggle.json")):
        return env_dir

    home_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(home_path):
        return os.path.dirname(home_path)

    repo_path = os.path.join(REPO_ROOT, "kaggle.json")
    if os.path.exists(repo_path):
        log.warning(
            "Using kaggle.json from repo root (%s). For better security, move it to "
            "~/.kaggle/kaggle.json (chmod 600) and ensure it is git-ignored.",
            repo_path,
        )
        os.environ["KAGGLE_CONFIG_DIR"] = REPO_ROOT
        return REPO_ROOT

    raise RuntimeError(
        "Kaggle credentials not found. Place your kaggle.json at "
        "~/.kaggle/kaggle.json (chmod 600), or set KAGGLE_CONFIG_DIR. "
        "See https://www.kaggle.com/docs/api for details."
    )


def fetch_kaggle_indonesiafalsenews(force: bool = False) -> int:
    """Download and unzip the dataset. Returns labelled-split row count."""
    out_dir = os.path.join(RAW_DIR, DATASET_NAME)
    os.makedirs(out_dir, exist_ok=True)

    if already_present() and not force:
        log.info("Dataset already present at %s (use --force to re-download)", out_dir)
    else:
        _resolve_kaggle_credentials()
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as e:
            raise RuntimeError(
                "The `kaggle` package is required. Install with `pip install kaggle`."
            ) from e

        api = KaggleApi()
        api.authenticate()
        log.info("Downloading %s -> %s", KAGGLE_SLUG, out_dir)
        api.dataset_download_files(KAGGLE_SLUG, path=out_dir, unzip=True, quiet=False)

    fp = os.path.join(out_dir, LABELLED_FILE)
    if not os.path.exists(fp):
        raise RuntimeError(
            f"Download succeeded but {LABELLED_FILE} not found under {out_dir}. "
            f"Contents: {sorted(os.listdir(out_dir))}"
        )

    import pandas as pd

    df = pd.read_csv(fp)
    n = len(df)
    log.info("OK | %s rows=%d cols=%s", LABELLED_FILE, n, list(df.columns))
    return n


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true",
                   help="Re-download even when dataset is already present.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(RAW_DIR, exist_ok=True)
    try:
        n = fetch_kaggle_indonesiafalsenews(force=args.force)
    except RuntimeError as e:
        log.error("%s", e)
        sys.exit(1)
    rel = os.path.relpath(
        os.path.join(RAW_DIR, DATASET_NAME, LABELLED_FILE), REPO_ROOT
    ).replace("\\", "/")
    print(f"\n=== OK | {n} rows at {rel} ===")
    print("Run experiments with:")
    print("  python -m experiments.run_all")


if __name__ == "__main__":
    main()
