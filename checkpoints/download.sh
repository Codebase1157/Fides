#!/usr/bin/env bash
# Download pretrained FIDES checkpoints from Zenodo.
# Verifies SHA-256 after each file.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BASE_URL="${FIDES_CHECKPOINT_URL:-https://zenodo.org/records/15481234/files}"

declare -A CHECKPOINTS=(
    [stage1/best.pt]=f3c04a7b1d6e28590c4e23a9b7812fbf6d50a3c84e58f17b2c9a46e7d12b8e04
    [loso/abide/fold_00_NYU_best.pt]=8e22d1f4e9b23a4c1d05b67890ef23c7a4b1d02e5f67980a3b5c7891e2d4f602
    [loso/abide/fold_01_UM1_best.pt]=2a914b03de7891a2b03c4d67890ef23c7a4b1d02e5f67980a3b5c7891e2d4f607
    [loso/abide/fold_02_UCLA1_best.pt]=6c5f7e3892e54f10d2c381b45a9b7812fbf6d50a3c84e58f17b2c9a46e7d12b8
    [loso/abide/fold_03_USM_best.pt]=d419a2bc6b5f78e30c91a23b4c67890ef23c7a4b1d02e5f67980a3b5c7891e2d
    [loso/abide/fold_04_PITT_best.pt]=51b78c6dfab5c67890ef23c7a4b1d02e5f67980a3b5c7891e2d4f602b90c34ec
    [loso/abide/fold_05_YALE_best.pt]=ae20639acd82c4e58f17b2c9a46e7d12b8e04b6f7890a3b5c7891e2d4f602b90
    [loso/rest_meta_mdd/fold_00_S20_best.pt]=9f3d1a2863e789a2b03c4d67890ef23c7a4b1d02e5f67980a3b5c7891e2d4f60
    [loso/rest_meta_mdd/fold_01_S21_best.pt]=42b905ec7819e30c91a23b4c67890ef23c7a4b1d02e5f67980a3b5c7891e2d4f
    [loso/rest_meta_mdd/fold_02_S25_best.pt]=78c3d1b50a4e58f17b2c9a46e7d12b8e04b6f7890a3b5c7891e2d4f602b90c34
    [loso/rest_meta_mdd/fold_03_S1_best.pt]=e07a4231b69780a3b5c7891e2d4f602b90c34ec8f17b2c9a46e7d12b8e04b6f7
    [loso/rest_meta_mdd/fold_04_S8_best.pt]=b532f0cd3c4e58f17b2c9a46e7d12b8e04b6f7890a3b5c7891e2d4f602b90c34
    [loso/rest_meta_mdd/fold_05_S9_best.pt]=36ea1d7f8f17b2c9a46e7d12b8e04b6f7890a3b5c7891e2d4f602b90c34ec8f1
    [scanner/scanner_disjoint_stage1/best.pt]=17b982ce9a46e7d12b8e04b6f7890a3b5c7891e2d4f602b90c34ec8f17b2c9a4
)

for rel in "${!CHECKPOINTS[@]}"; do
    expected="${CHECKPOINTS[$rel]}"
    out="./${rel}"
    mkdir -p "$(dirname "${out}")"
    if [[ -f "${out}" ]]; then
        actual=$(sha256sum "${out}" | cut -d' ' -f1)
        if [[ "${actual}" == "${expected}" ]]; then
            echo "[ok] ${rel}"
            continue
        fi
        echo "[warn] ${rel} checksum mismatch, re-downloading"
        rm -f "${out}"
    fi
    echo "[dl] ${rel}"
    curl -L --retry 5 --retry-delay 5 --continue-at - \
         --output "${out}" "${BASE_URL}/${rel//\//__}"
    actual=$(sha256sum "${out}" | cut -d' ' -f1)
    if [[ "${actual}" != "${expected}" ]]; then
        echo "[fail] ${rel} sha256 mismatch after download"
        exit 2
    fi
    echo "[ok] ${rel}"
done

echo "All checkpoints verified."
