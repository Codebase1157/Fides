"""Scanner overlap and unseen-scanner evaluation (Tables X and XI).

Stratifies the SRPBS test set by whether its scanner was seen during
Stage 1 pretraining, and separately evaluates on three unseen-site pools:
ABIDE I held-out sites, ABIDE II institutions absent from training, and
REST-meta-MDD sites that share neither hospital nor scanner with the
training sites.

This script consumes files written by earlier training / prediction
runs; it does not retrain. If the scanner-disjoint retrain is requested
with ``--include-disjoint``, the script expects its outputs to be
present under ``--disjoint-dir`` already; retraining must be launched
via ``configs/scanner_disjoint.yaml`` and is not run from here.

Inputs
------
``--loso-dir``          LOSO output root with per-fold ``srpbs_predictions.npz``.
``--srpbs-csv``         SRPBS phenotypic CSV with a ``scanner`` column.
``--overlapping``       scanner codes seen in Stage 1 (comma-separated).
``--unseen-dir``        directory with predictions on each unseen-site pool.
``--disjoint-dir``      (optional) outputs of the scanner-disjoint retrain.
``--out``               output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("scanner")


def ensemble_preds(loso_dir: Path, diagnosis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref_ids = None
    probs = []
    labels = None
    scanners = None
    for fold_dir in sorted(loso_dir.iterdir()):
        preds_path = fold_dir / "srpbs_predictions.npz"
        if not preds_path.is_file():
            raise FileNotFoundError(preds_path)
        data = np.load(preds_path, allow_pickle=True)
        if str(data["diagnosis"]) != diagnosis:
            continue
        ids = np.asarray(data["subject_id"]).astype(str)
        if ref_ids is None:
            ref_ids = ids
            labels = np.asarray(data["label"]).astype(int)
            scanners = np.asarray(data["scanner"]).astype(str)
        elif not np.array_equal(ids, ref_ids):
            raise RuntimeError("fold predictions disagree on subject ordering")
        probs.append(np.asarray(data["prob"]).astype(float))
    if not probs:
        raise RuntimeError(f"no SRPBS predictions for diagnosis={diagnosis!r}")
    return ref_ids, np.mean(np.stack(probs, 0), axis=0), labels, scanners


def metric_for(mask: np.ndarray, prob: np.ndarray, label: np.ndarray) -> dict:
    if mask.sum() == 0:
        return {"n": 0}
    pred = (prob[mask] >= 0.5).astype(int)
    out = {
        "n": int(mask.sum()),
        "acc": float(balanced_accuracy_score(label[mask], pred) * 100),
    }
    if len(np.unique(label[mask])) > 1:
        out["auc"] = float(roc_auc_score(label[mask], prob[mask]) * 100)
    return out


def load_unseen_pool(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return (
        np.asarray(data["prob"], dtype=float),
        np.asarray(data["label"], dtype=int),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loso-dir", type=Path, required=True)
    p.add_argument("--srpbs-csv", type=Path, required=True)
    p.add_argument(
        "--overlapping",
        required=True,
        help="Comma-separated scanner codes seen during Stage 1 pretraining.",
    )
    p.add_argument("--unseen-dir", type=Path, required=True)
    p.add_argument("--disjoint-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("results/scanner"))
    args = p.parse_args()

    overlapping = {s.strip() for s in args.overlapping.split(",") if s.strip()}
    srpbs_meta = pd.read_csv(args.srpbs_csv).set_index("subject_id")

    args.out.mkdir(parents=True, exist_ok=True)

    stratified = {}
    for diag in ("asd", "mdd"):
        ids, prob, label, scanners = ensemble_preds(args.loso_dir, diag)
        seen_mask = np.array([s in overlapping for s in scanners])
        stratified[diag] = {
            "all": metric_for(np.ones_like(seen_mask), prob, label),
            "overlapping_scanners": metric_for(seen_mask, prob, label),
            "unseen_scanners": metric_for(~seen_mask, prob, label),
        }
    (args.out / "stratified_srpbs.json").write_text(json.dumps(stratified, indent=2))
    LOG.info("wrote %s", args.out / "stratified_srpbs.json")

    unseen = {}
    for name in ("abide1_unseen_sites", "abide2_unseen_institutions", "rest_meta_mdd_unseen_sites"):
        pool_path = args.unseen_dir / f"{name}.npz"
        prob, label = load_unseen_pool(pool_path)
        pred = (prob >= 0.5).astype(int)
        unseen[name] = {
            "n": int(len(label)),
            "acc": float(balanced_accuracy_score(label, pred) * 100),
            "auc": float(roc_auc_score(label, prob) * 100),
        }
    (args.out / "unseen_site_pools.json").write_text(json.dumps(unseen, indent=2))
    LOG.info("wrote %s", args.out / "unseen_site_pools.json")

    if args.disjoint_dir is not None:
        disjoint = {}
        for diag in ("asd", "mdd"):
            ids, prob, label, _ = ensemble_preds(args.disjoint_dir, diag)
            pred = (prob >= 0.5).astype(int)
            disjoint[f"cross_dataset_srpbs_{diag}"] = {
                "acc": float(balanced_accuracy_score(label, pred) * 100),
                "auc": float(roc_auc_score(label, prob) * 100),
                "n": int(len(label)),
            }
        (args.out / "scanner_disjoint.json").write_text(json.dumps(disjoint, indent=2))
        LOG.info("wrote %s", args.out / "scanner_disjoint.json")


if __name__ == "__main__":
    main()
