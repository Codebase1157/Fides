#!/usr/bin/env bash
# run_all.sh - Top-level reproduction pipeline for FIDES
#
# Usage:
#   bash run_all.sh                    # full pipeline
#   bash run_all.sh --stage pretrain   # only Stage 1
#   bash run_all.sh --stage loso       # only LOSO Stage 2
#   bash run_all.sh --stage tenfold    # only 10-fold Stage 2
#   bash run_all.sh --stage cross      # only cross-dataset evaluation
#   bash run_all.sh --stage ablations  # only ablation study (Table VII)
#   bash run_all.sh --stage audits     # only shortcut audits (Table VI)
#   bash run_all.sh --stage severity   # only severity-subset analysis (Tables XII, XIII)
#   bash run_all.sh --stage scanner    # only scanner-overlap analysis (Tables X, XI)
#
# Environment variables:
#   FIDES_DATA_ROOT  default: ./data/processed
#   FIDES_OUT_ROOT   default: ./results
#   FIDES_LOG_ROOT   default: ./logs
#   FIDES_SEED       default: 20260101

set -euo pipefail

STAGE="all"
if [[ "${1:-}" == "--stage" ]]; then
    STAGE="${2:-all}"
fi

export FIDES_DATA_ROOT="${FIDES_DATA_ROOT:-./data/processed}"
export FIDES_OUT_ROOT="${FIDES_OUT_ROOT:-./results}"
export FIDES_LOG_ROOT="${FIDES_LOG_ROOT:-./logs}"
export FIDES_SEED="${FIDES_SEED:-20260101}"

mkdir -p "${FIDES_OUT_ROOT}" "${FIDES_LOG_ROOT}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "${FIDES_LOG_ROOT}/run_all.log"; }

run_pretrain() {
    log "Stage 1 pretraining on SRPBS traveling subjects + HCP"
    python -m src.training.train_stage1 \
        --config configs/stage1_pretrain.yaml \
        --data-root "${FIDES_DATA_ROOT}" \
        --out "${FIDES_OUT_ROOT}/stage1" \
        --log-dir "${FIDES_LOG_ROOT}/stage1" \
        --seed "${FIDES_SEED}"
}

run_loso() {
    log "Stage 2 LOSO training on ABIDE and REST-meta-MDD"
    for dataset in abide rest_meta_mdd; do
        python -m src.training.train_stage2 \
            --config "configs/stage2_loso_${dataset}.yaml" \
            --data-root "${FIDES_DATA_ROOT}" \
            --pretrained "${FIDES_OUT_ROOT}/stage1/best.pt" \
            --split-dir "splits/loso/${dataset}" \
            --out "${FIDES_OUT_ROOT}/loso/${dataset}" \
            --log-dir "${FIDES_LOG_ROOT}/stage2/${dataset}_loso" \
            --seed "${FIDES_SEED}"
    done
}

run_tenfold() {
    log "Stage 2 10-fold cross-validation"
    for dataset in abide rest_meta_mdd; do
        python -m src.training.train_stage2 \
            --config "configs/stage2_10fold_${dataset}.yaml" \
            --data-root "${FIDES_DATA_ROOT}" \
            --pretrained "${FIDES_OUT_ROOT}/stage1/best.pt" \
            --split-dir "splits/10fold/${dataset}" \
            --out "${FIDES_OUT_ROOT}/10fold/${dataset}" \
            --log-dir "${FIDES_LOG_ROOT}/stage2/${dataset}_10fold" \
            --seed "${FIDES_SEED}"
    done
}

run_cross() {
    log "Cross-dataset evaluation on SRPBS multi-disorder"
    python -m evaluation.cross_dataset \
        --loso-dir "${FIDES_OUT_ROOT}/loso" \
        --srpbs-dir "${FIDES_DATA_ROOT}/srpbs_md" \
        --out "${FIDES_OUT_ROOT}/cross"
}

run_ablations() {
    log "Ablation study (Table VII)"
    bash experiments/run_ablations.sh
}

run_audits() {
    log "Shortcut audits (Table VI): site probe, identity fingerprint, scanner ICC, nuisance probes"
    python -m evaluation.site_probe --loso-dir "${FIDES_OUT_ROOT}/loso" --out "${FIDES_OUT_ROOT}/audits/site_probe.json"
    python -m evaluation.identity_fingerprint --pretrained "${FIDES_OUT_ROOT}/stage1/best.pt" --out "${FIDES_OUT_ROOT}/audits/fingerprint.json"
    python -m evaluation.scanner_icc --pretrained "${FIDES_OUT_ROOT}/stage1/best.pt" --out "${FIDES_OUT_ROOT}/audits/scanner_icc.json"
    python -m evaluation.nuisance_probe --loso-dir "${FIDES_OUT_ROOT}/loso" --out "${FIDES_OUT_ROOT}/audits/nuisance.json"
}

run_severity() {
    log "Severity-subset analysis (Tables XII, XIII)"
    python -m experiments.severity_subset \
        --loso-dir "${FIDES_OUT_ROOT}/loso" \
        --out "${FIDES_OUT_ROOT}/severity"
}

run_scanner() {
    log "Scanner-overlap and unseen-scanner evaluation (Tables X, XI)"
    python -m experiments.scanner_overlap \
        --config configs/scanner_disjoint.yaml \
        --out "${FIDES_OUT_ROOT}/scanner"
}

run_interpret() {
    log "Interpretability: top-20 edges and BrainNet plots (Fig. 5)"
    python -m interpretability.gradient_attribution \
        --loso-dir "${FIDES_OUT_ROOT}/loso" \
        --out "${FIDES_OUT_ROOT}/interpretability"
}

run_tables() {
    log "Assembling result tables"
    python -m results.assemble_tables \
        --loso-dir "${FIDES_OUT_ROOT}/loso" \
        --tenfold-dir "${FIDES_OUT_ROOT}/10fold" \
        --cross-dir "${FIDES_OUT_ROOT}/cross" \
        --audits-dir "${FIDES_OUT_ROOT}/audits" \
        --out "${FIDES_OUT_ROOT}/tables"
}

case "${STAGE}" in
    all)
        run_pretrain
        run_loso
        run_tenfold
        run_cross
        run_audits
        run_ablations
        run_severity
        run_scanner
        run_interpret
        run_tables
        ;;
    pretrain)   run_pretrain ;;
    loso)       run_loso ;;
    tenfold)    run_tenfold ;;
    cross)      run_cross ;;
    ablations)  run_ablations ;;
    audits)     run_audits ;;
    severity)   run_severity ;;
    scanner)    run_scanner ;;
    interpret)  run_interpret ;;
    tables)     run_tables ;;
    *)
        echo "Unknown stage: ${STAGE}"
        exit 2
        ;;
esac

log "Done."
