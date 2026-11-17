"""Site-probe audit (Table VI).

Within each diagnostic group of each LOSO fold, train a multinomial
logistic-regression probe on the five training sites and evaluate balanced
accuracy on the held-out site. Average across groups, then across folds.

Inputs
------
``--loso-dir``   directory with one subdirectory per fold, each containing
                 ``representations.npz`` with keys:

                     z_dis   : (n, d)
                     site    : (n,) str
                     group   : (n,) int, diagnostic group
                     is_test : (n,) bool, held-out-site mask

``--n-sites``    number of training sites (6 for both ABIDE and REST-meta-MDD),
                 used only to compute chance (1 / n_sites).

``--out``        JSON file to write.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("site_probe")


def site_bacc(
    z: np.ndarray,
    sites: np.ndarray,
    groups: np.ndarray,
    is_test: np.ndarray,
) -> float:
    scores = []
    for g in np.unique(groups):
        sel_train = (~is_test) & (groups == g)
        sel_test = is_test & (groups == g)
        if sel_train.sum() < 10 or sel_test.sum() < 2:
            continue
        sc = StandardScaler().fit(z[sel_train])
        probe = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            C=1.0,
            max_iter=2000,
        ).fit(sc.transform(z[sel_train]), sites[sel_train])
        pred = probe.predict(sc.transform(z[sel_test]))
        scores.append(balanced_accuracy_score(sites[sel_test], pred))
    return float(np.mean(scores) * 100) if scores else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loso-dir", type=Path, required=True)
    p.add_argument("--n-sites", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    fold_scores = []
    for fold_dir in sorted(args.loso_dir.iterdir()):
        reps_path = fold_dir / "representations.npz"
        if not reps_path.is_file():
            raise FileNotFoundError(reps_path)
        data = np.load(reps_path, allow_pickle=True)
        score = site_bacc(
            np.asarray(data["z_dis"], dtype=float),
            np.asarray(data["site"]).astype(str),
            np.asarray(data["group"]).astype(int),
            np.asarray(data["is_test"]).astype(bool),
        )
        if np.isnan(score):
            LOG.warning("fold %s skipped (no eligible groups)", fold_dir.name)
            continue
        fold_scores.append(score)
        LOG.info("%s site_bacc=%.2f", fold_dir.name, score)

    if not fold_scores:
        raise RuntimeError("no fold yielded a valid probe score")

    summary = {
        "site_bacc_mean": float(np.mean(fold_scores)),
        "site_bacc_std": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
        "n_folds": len(fold_scores),
        "chance": float(100.0 / args.n_sites),
        "interpretation": "closer to chance = site is less recoverable from z_dis",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    LOG.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
