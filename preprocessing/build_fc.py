"""Build functional connectivity matrices from ROI time series.

For every subject or session, computes a Pearson correlation matrix across
atlas ROIs, applies the Fisher z-transform, zeros out self-connections, and
sparsifies by absolute-value threshold.

Usage:
    python preprocessing/build_fc.py --atlas cc200
    python preprocessing/build_fc.py --atlas aal3 --threshold 0.1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("build_fc")

ATLAS_SIZES = {"cc200": 200, "aal3": 166}


def pearson_fc(ts: np.ndarray) -> np.ndarray:
    """Return a Pearson correlation matrix across ROIs."""
    ts = ts - ts.mean(axis=0, keepdims=True)
    std = ts.std(axis=0, keepdims=True) + 1e-8
    ts = ts / std
    fc = (ts.T @ ts) / (ts.shape[0] - 1)
    np.fill_diagonal(fc, 0.0)
    return fc


def fisher_z(fc: np.ndarray) -> np.ndarray:
    fc = np.clip(fc, -0.999, 0.999)
    return np.arctanh(fc)


def sparsify(fc: np.ndarray, threshold: float) -> np.ndarray:
    mask = np.abs(fc) >= threshold
    return fc * mask


def build_subject(ts_path: Path, atlas: str, threshold: float, out_dir: Path) -> None:
    ts = np.load(ts_path)
    expected = ATLAS_SIZES[atlas]
    if ts.shape[1] != expected:
        raise ValueError(f"{ts_path}: expected {expected} ROIs, got {ts.shape[1]}")
    fc = pearson_fc(ts)
    fc = fisher_z(fc)
    fc = sparsify(fc, threshold)
    out_path = out_dir / ts_path.stem / f"{ts_path.stem}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, fc.astype(np.float32))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", choices=list(ATLAS_SIZES), default="cc200")
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--ts-root", default="data/processed/dparsf")
    p.add_argument("--out-root", default="data/processed/fc")
    args = p.parse_args()

    ts_root = Path(args.ts_root)
    out_root = Path(args.out_root) / args.atlas
    out_root.mkdir(parents=True, exist_ok=True)

    paths = sorted(ts_root.rglob("*.npy"))
    LOG.info("Found %d time series under %s", len(paths), ts_root)

    for i, ts_path in enumerate(paths, 1):
        try:
            build_subject(ts_path, args.atlas, args.threshold, out_root)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Skipping %s (%s)", ts_path, exc)
        if i % 50 == 0:
            LOG.info("Processed %d / %d", i, len(paths))

    LOG.info("Done. FC matrices under %s", out_root)


if __name__ == "__main__":
    main()
