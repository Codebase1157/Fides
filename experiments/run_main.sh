#!/usr/bin/env bash
# Main experiments: Tables III (10-fold), IV (LOSO), V (cross-dataset), VI (audits).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

export FIDES_SEED="${FIDES_SEED:-20260101}"

bash run_all.sh --stage pretrain
bash run_all.sh --stage loso
bash run_all.sh --stage tenfold
bash run_all.sh --stage cross
bash run_all.sh --stage audits
bash run_all.sh --stage tables
