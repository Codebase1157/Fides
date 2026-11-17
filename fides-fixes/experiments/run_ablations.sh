#!/usr/bin/env bash
# Ablation study behind Table VII. Each row trains Stage 2 with one component
# disabled or swapped, and reports LOSO ACC / AUC / Site BAcc on ABIDE and
# REST-meta-MDD.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

VARIANTS=(
    w_o_v1_halves
    w_o_v2_sessions
    w_o_v3_traveling_subjects
    w_o_v4_cross_site
    w_o_v5_combat_pairs
    natural_views_to_salience_masking
    w_o_hsic
    w_o_decoder
    w_o_stage1_pretraining
    single_prototype
    motformer_to_brainnettf
    cc200_to_aal3
    fides_full
)

for v in "${VARIANTS[@]}"; do
    for ds in abide rest_meta_mdd; do
        python -m experiments.ablation_runner \
            --variant "$v" \
            --dataset "$ds" \
            --out "results/ablations/${v}_${ds}" \
            --log-dir "logs/ablations/${v}_${ds}" \
            --seed "${FIDES_SEED:-20260101}"
    done
done

python -m results.assemble_ablation_table \
    --root results/ablations \
    --out results/tables/table7_ablations.csv
