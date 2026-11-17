"""Cross-dataset evaluation on SRPBS multi-disorder (Table V).

For each diagnosis (ASD and MDD), loads the predictions written by
``src.training.stage2`` for every LOSO fold, averages the sigmoid
probabilities to form an ensemble prediction per SRPBS participant, and
computes balanced accuracy and AUC against the SRPBS labels.

Inputs
------
``--loso-dir``   directory with one subdirectory per fold, each containing
                 ``srpbs_predictions.npz`` with keys:

                     subject_id : (n,) str
                     diagnosis  : {'asd', 'mdd'}
                     prob       : (n,) float, P(diseased)
                     label      : (n,) int,   0 or 1

``--srpbs-csv``  phenotypic CSV with ``subject_id`` and ``diagnosis_group``
                 columns, used only as a sanity check that the fold
                 predictions cover the same participant set.

``--out``        output directory for ``cross_dataset_summary.json``.

If a fold's predictions file is missing, the script fails rather than
substituting anything in its place.
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
LOG = logging.getLogger("cross")


def load_fold_preds(loso_dir: Path, diagnosis: str) -> list[dict]:
    folds = []
    for fold_dir in sorted(loso_dir.iterdir()):
        preds_path = fold_dir / "srpbs_predictions.npz"
        if not preds_path.is_file():
            raise FileNotFoundError(
                f"missing {preds_path}; run stage2 with --write-srpbs-preds first"
            )
        data = np.load(preds_path, allow_pickle=True)
        if str(data["diagnosis"]) != diagnosis:
            continue
        folds.append(
            {
                "subject_id": np.asarray(data["subject_id"]).astype(str),
                "prob": np.asarray(data["prob"]).astype(float),
                "label": np.asarray(data["label"]).astype(int),
            }
        )
    if not folds:
        raise RuntimeError(
            f"no SRPBS predictions for diagnosis={diagnosis!r} under {loso_dir}"
        )
    return folds


def ensemble(folds: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = folds[0]["subject_id"]
    probs = []
    for f in folds:
        if not np.array_equal(f["subject_id"], ref):
            raise RuntimeError("fold predictions disagree on subject ordering")
        probs.append(f["prob"])
    mean_prob = np.mean(np.stack(probs, axis=0), axis=0)
    return ref, mean_prob, folds[0]["label"]


def metrics(prob: np.ndarray, label: np.ndarray) -> dict:
    pred = (prob >= 0.5).astype(int)
    return {
        "acc": float(balanced_accuracy_score(label, pred) * 100),
        "auc": float(roc_auc_score(label, prob) * 100),
        "n": int(len(label)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loso-dir", type=Path, required=True)
    p.add_argument("--srpbs-csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    srpbs = pd.read_csv(args.srpbs_csv)
    summary = {}
    for diag in ("asd", "mdd"):
        folds = load_fold_preds(args.loso_dir, diag)
        ids, prob, label = ensemble(folds)

        expected = set(
            srpbs.loc[srpbs["diagnosis_group"].isin([diag, "nc"]), "subject_id"]
            .astype(str)
            .tolist()
        )
        got = set(ids.tolist())
        if got != expected:
            raise RuntimeError(
                f"predictions for {diag} cover a different participant set than "
                f"{args.srpbs_csv}: {len(got)} vs {len(expected)}"
            )

        summary[f"{diag}_srpbs"] = metrics(prob, label)
        LOG.info("%s: acc=%.2f auc=%.2f n=%d", diag, *summary[f"{diag}_srpbs"].values())

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cross_dataset_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    LOG.info("Wrote %s", args.out / "cross_dataset_summary.json")


if __name__ == "__main__":
    main()
