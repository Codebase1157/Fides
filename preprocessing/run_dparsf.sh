#!/usr/bin/env bash
# run_dparsf.sh - run DPARSF preprocessing on raw rs-fMRI data
#
# Usage: bash run_dparsf.sh <raw_root> <out_root>
#
# Expects DPARSF/SPM12 on MATLAB path. Reproduces the pipeline of BrainCL [13].

set -euo pipefail

RAW_ROOT="${1:-data/raw}"
OUT_ROOT="${2:-data/processed}"

mkdir -p "${OUT_ROOT}/dparsf"

for dataset in ABIDE_I ABIDE_II REST-meta-MDD SRPBS_MD SRPBS_TS HCP_S1200; do
    echo "[DPARSF] ${dataset}"
    matlab -nodisplay -nosplash -r "\
        addpath(genpath('/opt/DPARSF/DPARSF_V5.4'));\
        addpath(genpath('/opt/spm12'));\
        run_dparsf_pipeline('${RAW_ROOT}/${dataset}', '${OUT_ROOT}/dparsf/${dataset}');\
        exit;"
done

echo "[DPARSF] Done. Outputs in ${OUT_ROOT}/dparsf/"
