#!/usr/bin/env bash
# Pack results/ into results_<host>_<timestamp>.tar.gz
# Usage: bash scripts/pack_results.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${ROOT}/results"

if [[ ! -d "${RESULTS_DIR}" ]]; then
    echo "No results/ directory at ${RESULTS_DIR}" >&2
    exit 1
fi

TS="$(date +%Y%m%d_%H%M)"
HOST="$(hostname -s | tr '[:upper:]' '[:lower:]')"
OUT="${ROOT}/results_${HOST}_${TS}.tar.gz"
tar -czf "${OUT}" -C "${ROOT}" results
SIZE_MB=$(du -m "${OUT}" | cut -f1)
echo "Packed -> ${OUT} (${SIZE_MB} MB)"
