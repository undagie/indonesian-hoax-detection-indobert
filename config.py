"""Experiment configuration for Indonesian hoax detection.

Single source of hyperparameters. Override via ``HOAX_*`` environment variables.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Pin one GPU before importing torch (avoids CUDA assert on broken multi-GPU nodes).
# Override with HOAX_GPU_ID, e.g. HOAX_GPU_ID=2.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("HOAX_GPU_ID", "0")

# Silence tokenizer fork warnings when DataLoader uses num_workers > 0.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError(
        "PyTorch is required. Install via `pip install torch --index-url "
        "https://download.pytorch.org/whl/cu121` for CUDA 12.1 wheels."
    ) from e

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = _SCRIPT_DIR

try:
    from google.colab import drive

    if not os.path.exists("/content/drive"):
        drive.mount("/content/drive")
    BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/HoaxID/dataset"
except Exception:
    BASE_PATH = os.path.join(REPO_ROOT, "dataset")

DATA_DIR = BASE_PATH
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

for _d in (
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CACHE_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    LOGS_DIR,
):
    os.makedirs(_d, exist_ok=True)


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_str(key: str, default: Optional[str]) -> Optional[str]:
    v = os.environ.get(key)
    return v if v else default


HOAX_DATASET = _env_str("HOAX_DATASET", "kaggle_indonesiafalsenews") or "kaggle_indonesiafalsenews"
HOAX_TEXT_COL = _env_str("HOAX_TEXT_COL", "text") or "text"
HOAX_LABEL_COL = _env_str("HOAX_LABEL_COL", "label") or "label"
HOAX_ALLOW_SYNTHETIC = _env_bool("HOAX_ALLOW_SYNTHETIC", False)

LABEL_MAP = {"fakta": 0, "hoax": 1}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = 2

HOAX_RUNS_SUBDIR = _env_str("HOAX_RUNS_SUBDIR", None)
HOAX_MODEL_NAME = _env_str("HOAX_MODEL_NAME", None)

DEFAULT_MODEL = "indolem/indobert-base-uncased"
SUPPORTED_MODELS = {
    "indobert_base": "indolem/indobert-base-uncased",
    "indobertweet": "indolem/indobertweet-base-uncased",
    "indobert_large": "indobenchmark/indobert-large-p2",
    "xlmr_base": "xlm-roberta-base",
    "xlmr_large": "xlm-roberta-large",
    "mbert": "bert-base-multilingual-cased",
}


def resolve_model_name(alias_or_name: Optional[str] = None) -> str:
    """Resolve a short alias or pass through a Hugging Face model id."""
    name = alias_or_name or HOAX_MODEL_NAME or DEFAULT_MODEL
    return SUPPORTED_MODELS.get(name, name)


@dataclass
class Config:
    """Hyperparameters for a 12-16 GB GPU (e.g. RTX 3060 / T4)."""

    SEED: int = 42
    MAX_LEN: int = _env_int("HOAX_MAX_LEN", 384)
    BATCH_SIZE: int = _env_int("HOAX_BATCH_SIZE", 16)
    EVAL_BATCH_SIZE: int = _env_int("HOAX_EVAL_BATCH_SIZE", 32)
    GRAD_ACCUM_STEPS: int = _env_int("HOAX_GRAD_ACCUM", 1)
    NUM_WORKERS: int = _env_int("HOAX_NUM_WORKERS", 2)
    EPOCHS: int = _env_int("HOAX_EPOCHS", 10)
    LR: float = _env_float("HOAX_LR", 2e-5)
    LR_LARGE: float = _env_float("HOAX_LR_LARGE", 1e-5)
    WEIGHT_DECAY: float = _env_float("HOAX_WD", 0.01)
    WARMUP_RATIO: float = _env_float("HOAX_WARMUP", 0.1)
    MAX_GRAD_NORM: float = _env_float("HOAX_GRAD_NORM", 1.0)
    DROPOUT: float = _env_float("HOAX_DROPOUT", 0.1)
    USE_AMP: bool = _env_bool("HOAX_AMP", True)
    USE_TORCH_COMPILE: bool = _env_bool("HOAX_COMPILE", False)
    EARLY_STOPPING_PATIENCE: int = _env_int("HOAX_PATIENCE", 3)
    EARLY_STOPPING_METRIC: str = _env_str("HOAX_ES_METRIC", "f1_macro") or "f1_macro"
    MODEL_NAME: str = field(default_factory=lambda: resolve_model_name())
    NUM_CLASSES: int = NUM_CLASSES
    DEVICE: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    VAL_RATIO: float = 0.1
    TEST_RATIO: float = 0.2
    USE_KFOLD: bool = _env_bool("HOAX_KFOLD", False)
    N_FOLDS: int = _env_int("HOAX_NFOLDS", 5)


AUG_METHODS = [
    "none",
    "synonym",
    "swap",
    "delete",
    "insert",
    "back_translation",
    "contextual",
]
AUG_DEFAULT = _env_str("HOAX_AUG", "none") or "none"
AUG_RATIO = _env_float("HOAX_AUG_RATIO", 1.0)
AUG_PROB = _env_float("HOAX_AUG_PROB", 0.15)
AUG_TARGET_MINORITY_ONLY = _env_bool("HOAX_AUG_MINORITY_ONLY", True)
BACK_TRANSLATION_PIVOT = _env_str("HOAX_BT_PIVOT", "en") or "en"
BACK_TRANSLATION_MODEL_ID2X = "Helsinki-NLP/opus-mt-id-en"
BACK_TRANSLATION_MODEL_X2ID = "Helsinki-NLP/opus-mt-en-id"
CONTEXTUAL_AUG_MODEL = "indolem/indobert-base-uncased"

IMBALANCE_STRATEGIES = [
    "none",
    "class_weight",
    "weighted_sampler",
    "random_oversample",
    "random_undersample",
    "smote_embed",
    "focal_loss",
    "cb_loss",
    "ldam_loss",
    "combined",
]
IMBALANCE_DEFAULT = _env_str("HOAX_IMBALANCE", "none") or "none"
FOCAL_GAMMA = _env_float("HOAX_FOCAL_GAMMA", 2.0)
FOCAL_ALPHA = _env_float("HOAX_FOCAL_ALPHA", 0.25)
CB_BETA = _env_float("HOAX_CB_BETA", 0.999)
LDAM_MAX_M = _env_float("HOAX_LDAM_MAX_M", 0.5)
LDAM_S = _env_float("HOAX_LDAM_S", 30.0)

SEEDS: List[int] = [7, 42, 101, 1234, 1137]
BOOTSTRAP_N: int = _env_int("HOAX_BOOTSTRAP_N", 1000)
BOOTSTRAP_CI: float = _env_float("HOAX_BOOTSTRAP_CI", 0.95)
FIGURE_DPI: int = _env_int("HOAX_FIG_DPI", 300)
FIGURE_FORMAT: str = _env_str("HOAX_FIG_FORMAT", "png") or "png"


def get_runs_dir(
    tag: Optional[str] = None,
    model: Optional[str] = None,
    runs_subdir: Optional[str] = None,
) -> str:
    """Return (and create) the directory for run outputs."""
    subdir = runs_subdir or HOAX_RUNS_SUBDIR
    if subdir:
        path = os.path.join(RUNS_DIR, subdir)
    elif tag:
        path = os.path.join(RUNS_DIR, tag)
    elif model:
        path = os.path.join(RUNS_DIR, model.replace("/", "_"))
    else:
        path = RUNS_DIR
    os.makedirs(path, exist_ok=True)
    return path


def get_run_config_snapshot(
    experiment: str,
    model_name: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Snapshot hyperparameters for a run. Extra kwargs override defaults."""
    cfg = Config()
    snap: Dict[str, Any] = {
        "experiment": experiment,
        "model_name": model_name or cfg.MODEL_NAME,
        "max_len": cfg.MAX_LEN,
        "batch_size": cfg.BATCH_SIZE,
        "grad_accum_steps": cfg.GRAD_ACCUM_STEPS,
        "epochs": cfg.EPOCHS,
        "lr": cfg.LR,
        "weight_decay": cfg.WEIGHT_DECAY,
        "warmup_ratio": cfg.WARMUP_RATIO,
        "max_grad_norm": cfg.MAX_GRAD_NORM,
        "dropout": cfg.DROPOUT,
        "use_amp": cfg.USE_AMP,
        "early_stopping_patience": cfg.EARLY_STOPPING_PATIENCE,
        "early_stopping_metric": cfg.EARLY_STOPPING_METRIC,
        "val_ratio": cfg.VAL_RATIO,
        "test_ratio": cfg.TEST_RATIO,
        "use_kfold": cfg.USE_KFOLD,
        "n_folds": cfg.N_FOLDS,
        "dataset": HOAX_DATASET,
        "text_col": HOAX_TEXT_COL,
        "label_col": HOAX_LABEL_COL,
        "label_map": dict(LABEL_MAP),
        "num_classes": NUM_CLASSES,
        "aug_method": AUG_DEFAULT,
        "aug_ratio": AUG_RATIO,
        "aug_prob": AUG_PROB,
        "aug_minority_only": AUG_TARGET_MINORITY_ONLY,
        "imbalance_strategy": IMBALANCE_DEFAULT,
        "focal_gamma": FOCAL_GAMMA,
        "focal_alpha": FOCAL_ALPHA,
        "cb_beta": CB_BETA,
        "ldam_max_m": LDAM_MAX_M,
        "ldam_s": LDAM_S,
        "seeds": list(SEEDS),
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_ci": BOOTSTRAP_CI,
    }
    snap.update(overrides)
    return snap


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


print(
    f"[HoaxID config] device={Config().DEVICE} | model={resolve_model_name()} | "
    f"dataset={HOAX_DATASET} | aug={AUG_DEFAULT} | imbalance={IMBALANCE_DEFAULT}"
)
