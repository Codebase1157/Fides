#!/usr/bin/env bash
# Download pretrained FIDES checkpoints.
#
# Requires $FIDES_CHECKPOINT_URL (no default): the archive is not yet hosted
# publicly, so we refuse to fetch from an invented URL. Once the Zenodo
# record is live we will update checkpoints/README.md with the DOI; set
# FIDES_CHECKPOINT_URL to the "files" endpoint of that record.
#
# Each file is verified against SHA256SUMS.txt before use. If a digest is
# missing (e.g. the archive has not been published yet), the script fails.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -z "${FIDES_CHECKPOINT_URL:-}" ]]; then
    cat >&2 <<'EOF'
error: FIDES_CHECKPOINT_URL is not set.

The checkpoint archive has not been posted publicly yet. Once it is, set
FIDES_CHECKPOINT_URL to the hosting URL (e.g. the Zenodo "files" endpoint)
and re-run this script. In the meantime, retrain from configs/ or contact
the authors.
EOF
    exit 2
fi

if ! [[ -s SHA256SUMS.txt ]] || ! grep -Eq '^[0-9a-f]{64}  ' SHA256SUMS.txt; then
    cat >&2 <<'EOF'
error: SHA256SUMS.txt contains no verified digests.

Nothing to download: we only fetch files whose SHA-256 is listed here, and
the file currently ships empty (digests will be added at deposition time).
EOF
    exit 2
fi

# Read "<sha256>  <relative path>" pairs, skipping comments and blanks.
while IFS= read -r line; do
    [[ -z "${line}" || "${line}" =~ ^# ]] && continue
    expected=$(awk '{print $1}' <<<"${line}")
    rel=$(awk '{print $2}' <<<"${line}")
    out="./${rel}"
    mkdir -p "$(dirname "${out}")"

    if [[ -f "${out}" ]]; then
        actual=$(sha256sum "${out}" | awk '{print $1}')
        if [[ "${actual}" == "${expected}" ]]; then
            echo "[ok]   ${rel}"
            continue
        fi
        echo "[warn] ${rel} checksum mismatch, re-downloading"
        rm -f "${out}"
    fi

    echo "[dl]   ${rel}"
    curl -L --fail --retry 5 --retry-delay 5 --continue-at - \
         --output "${out}" "${FIDES_CHECKPOINT_URL%/}/${rel//\//__}"

    actual=$(sha256sum "${out}" | awk '{print $1}')
    if [[ "${actual}" != "${expected}" ]]; then
        echo "[fail] ${rel} sha256 mismatch after download" >&2
        exit 3
    fi
    echo "[ok]   ${rel}"
done < SHA256SUMS.txt

echo "All checkpoints listed in SHA256SUMS.txt verified."
