#!/usr/bin/env bash
# Bootstrap a Linux server environment with uv (no sudo required).
# Usage: bash scripts/setup_server.sh [--python 3.11] [--no-data] [--no-model] [--reuse-venv]

set -euo pipefail

PYTHON_VERSION="3.10"
DOWNLOAD_DATA=1
PREDOWNLOAD_MODEL=1
REUSE_VENV=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-data)   DOWNLOAD_DATA=0; shift ;;
        --no-model)  PREDOWNLOAD_MODEL=0; shift ;;
        --reuse-venv) REUSE_VENV=1; shift ;;
        --python)    PYTHON_VERSION="$2"; shift 2 ;;
        -h|--help)
            grep -E "^# " "$0" | sed 's/^# //'
            exit 0
            ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

log()  { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
ok()   { printf "\033[1;32m[ ok ]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }




if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv to ~/.local/bin (no sudo required)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    if [[ -f "${HOME}/.local/bin/env" ]]; then
        # shellcheck disable=SC1091
        source "${HOME}/.local/bin/env"
    fi
else
    ok "uv already installed: $(uv --version)"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv install failed. Add \$HOME/.local/bin to your PATH and retry." >&2
    exit 1
fi




if [[ -d .venv ]]; then
    if [[ "${REUSE_VENV}" == "1" && -f .venv/bin/activate ]]; then
        ok ".venv already exists -- reusing (--reuse-venv set)"
    else
        if [[ "${REUSE_VENV}" == "1" ]]; then
            warn "--reuse-venv requested but .venv is incomplete (no bin/activate). Recreating."
        else
            log "Removing existing .venv for a clean install (pass --reuse-venv to keep it)"
        fi
        rm -rf .venv
    fi
fi

if [[ ! -d .venv ]]; then
    log "Creating .venv with Python ${PYTHON_VERSION}"
    uv venv .venv --python "${PYTHON_VERSION}"
fi

if [[ ! -f .venv/bin/activate ]]; then
    echo "uv venv failed to produce .venv/bin/activate. Try a different --python version." >&2
    exit 1
fi


# shellcheck disable=SC1091
source .venv/bin/activate




log "Installing dependencies from requirements.txt"

# Install torch from the CUDA wheel index (override with TORCH_INDEX).
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
log "Installing torch from ${TORCH_INDEX}"
uv pip install torch --index-url "${TORCH_INDEX}"

# Verify the torch wheel exposes CUDA when expected.
python - <<'PY'
import torch, sys
v = torch.__version__
print(f"installed torch={v}")
if "+cu" not in v:
    print("WARNING: CPU-only torch was installed; GPU experiments will be very slow.")
elif not torch.cuda.is_available():
    print("ERROR: torch built with CUDA but cuda.is_available()==False.")
    print("Likely cause: torch CUDA version is newer than the NVIDIA driver on this server.")
    print("Fix: re-run setup with an older index, e.g.:")
    print("  TORCH_INDEX=https://download.pytorch.org/whl/cu118 bash scripts/setup_server.sh --reuse-venv")
    sys.exit(2)
PY

uv pip install -r requirements.txt
ok "Dependencies installed"




log "Sanity check: torch + GPU"

# Prefer torch-bundled NCCL over an older system copy on LD_LIBRARY_PATH.
if python -c "import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__))" >/dev/null 2>&1; then
    NCCL_DIR="$(python -c 'import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__))')"
    if [[ -d "${NCCL_DIR}/lib" ]]; then
        export LD_LIBRARY_PATH="${NCCL_DIR}/lib:${LD_LIBRARY_PATH:-}"
        ok "Prepended bundled NCCL to LD_LIBRARY_PATH: ${NCCL_DIR}/lib"
    fi
fi

if ! python - <<'PY' 2>/tmp/_torch_check.err; then
import sys, torch
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name} ({p.total_memory/1024**3:.1f} GB)")
PY
    cat /tmp/_torch_check.err >&2
    warn "torch import failed. If you see 'undefined symbol: ncclCommResume', the"
    warn "system NCCL is older than the one torch was built with. Try:"
    warn "  LD_LIBRARY_PATH= python -c 'import torch; print(torch.__version__)'"
    warn "If that works, add to your shell: unset LD_LIBRARY_PATH (or prepend the"
    warn "bundled NCCL dir as we just attempted)."
    exit 1
fi




if [[ "${PREDOWNLOAD_MODEL}" == "1" ]]; then
    log "Pre-downloading IndoBERT-base (one-time, ~500 MB)"
    python - <<'PY'
from transformers import AutoModel, AutoTokenizer
name = "indolem/indobert-base-uncased"
AutoTokenizer.from_pretrained(name)
AutoModel.from_pretrained(name)
print(f"OK -- {name} cached")
PY
fi




if [[ "${DOWNLOAD_DATA}" == "1" ]]; then
    if [[ -s dataset/raw/kaggle_indonesiafalsenews/Data_latih.csv ]]; then
        ok "Dataset kaggle_indonesiafalsenews already present"
    else
        log "Downloading Kaggle indonesiafalsenews dataset (needs ~/.kaggle/kaggle.json)"
        python scripts/download_data.py
    fi
fi


log "Verifying pipeline can load the dataset"
python - <<'PY'
from src.data import load_dataset
df = load_dataset()
print(f"loaded rows={len(df)} | label counts={df['label'].value_counts().to_dict()}")
assert len(df) > 100, "dataset too small -- did the download fail?"
PY




cat <<EOF

\033[1;32m=== Server setup complete ===\033[0m

Next steps:
  source .venv/bin/activate
  nohup python -m experiments.run_all > results/logs/run_all.log 2>&1 &
  tail -f results/logs/run_all.log

When done:
  bash scripts/pack_results.sh    # produces results_<host>_<ts>.tar.gz

EOF
