"""Nuisance probes: framewise displacement, age, sex (Table VI).

For each LOSO fold, a ridge probe predicts a nuisance variable from the
disease representation of the held-out site. FD and age are regression
(R^2); sex is classification (balanced accuracy). We report the across-
fold mean.

Inputs
------
``--loso-dir``   directory with one subdirectory per fold, each containing
                 ``representations.npz`` with keys:

                     z_dis      : (n, d)  disease representation
                     fd         : (n,)    framewise displacement
                     age        : (n,)    age in years
                     sex        : (n,)    0/1

``--out``        JSON file to write.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("nuisance")


def cv_r2(z: np.ndarray, y: np.ndarray, seed: int) -> float:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in kf.split(z):
        sc = StandardScaler().fit(z[train_idx])
        probe = Ridge(alpha=1.0).fit(sc.transform(z[train_idx]), y[train_idx])
        preds[test_idx] = probe.predict(sc.transform(z[test_idx]))
    return float(r2_score(y, preds))


def cv_bacc(z: np.ndarray, y: np.ndarray, seed: int) -> float:
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.zeros_like(y, dtype=int)
    for train_idx, test_idx in kf.split(z, y):
        sc = StandardScaler().fit(z[train_idx])
        probe = LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=2000
        ).fit(sc.transform(z[train_idx]), y[train_idx])
        preds[test_idx] = probe.predict(sc.transform(z[test_idx]))
    return float(balanced_accuracy_score(y, preds) * 100)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loso-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260101)
    args = p.parse_args()

    fd_r2, age_r2, sex_bacc = [], [], []
    seen = 0
    for fold_dir in sorted(args.loso_dir.iterdir()):
        reps_path = fold_dir / "representations.npz"
        if not reps_path.is_file():
            raise FileNotFoundError(reps_path)
        data = np.load(reps_path)
        z = np.asarray(data["z_dis"], dtype=float)
        fd_r2.append(cv_r2(z, np.asarray(data["fd"], dtype=float), args.seed))
        age_r2.append(cv_r2(z, np.asarray(data["age"], dtype=float), args.seed))
        sex_bacc.append(cv_bacc(z, np.asarray(data["sex"], dtype=int), args.seed))
        seen += 1
        LOG.info(
            "%s fd_r2=%.3f age_r2=%.3f sex_bacc=%.2f",
            fold_dir.name, fd_r2[-1], age_r2[-1], sex_bacc[-1],
        )

    if seen == 0:
        raise RuntimeError(f"no fold directories found under {args.loso_dir}")

    summary = {
        "fd_r2_mean": float(np.mean(fd_r2)),
        "age_r2_mean": float(np.mean(age_r2)),
        "sex_bacc_mean": float(np.mean(sex_bacc)),
        "chance_sex_bacc": 50.0,
        "n_folds": seen,
        "interpretation": "lower r^2 / bacc = less nuisance leakage",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    LOG.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
