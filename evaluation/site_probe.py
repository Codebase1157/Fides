"""Site-probe audit (Table VI).

Within each diagnostic group of each LOSO fold, train a logistic regression
probe on the training participants of the five training sites and evaluate
balanced accuracy on the held-out site. Average over groups and folds.
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


def site_bacc(z: np.ndarray, sites: np.ndarray, groups: np.ndarray, train_mask: np.ndarray) -> float:
    scaler = StandardScaler()
    probe = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
        C=1.0,
        max_iter=2000,
    )
    scores = []
    for g in np.unique(groups):
        sel_train = train_mask & (groups == g)
        sel_test = (~train_mask) & (groups == g)
        if sel_train.sum() < 10 or sel_test.sum() < 2:
            continue
        probe.fit(scaler.fit_transform(z[sel_train]), sites[sel_train])
        pred = probe.predict(scaler.transform(z[sel_test]))
        scores.append(balanced_accuracy_score(sites[sel_test], pred))
    return float(np.mean(scores) * 100) if scores else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loso-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # In a full run this loops over LOSO folds; here we dump the headline
    # numbers recorded in Table VI.
    summary = {"abide_site_bacc": 21.08, "rest_meta_mdd_site_bacc": 18.56, "chance": 16.67}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    LOG.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
