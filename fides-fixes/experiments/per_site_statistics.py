"""Per-site LOSO comparison (Tables XIV, XV; Fig. 8).

For FIDES and each strong baseline listed in ``BASELINES``, loads the
per-fold ``metrics.json`` written by the LOSO training runs, computes
paired differences and corrected statistics, and writes a single JSON
summary.

Inputs
------
``--fides-dir``     LOSO output root for FIDES (one subdir per fold).
``--baseline-dir``  LOSO output root for the baselines; expected layout:

        <baseline-dir>/<baseline_name>/<fold_dir>/metrics.json

``--out``           output directory for ``per_site_comparison.json``.

Each per-fold ``metrics.json`` is expected to contain at least:
``acc``, ``auc``, ``held_out_site``.

The script fails if any expected file is missing rather than writing a
placeholder.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from evaluation.statistical_tests import corrected_resampled_t, holm_correct

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("per_site")

BASELINES = ["BrainGFM", "Wang_et_al_10", "BrainCL", "BRIO", "CORE"]


def load_fold_metrics(root: Path) -> dict[str, dict]:
    out = {}
    for fold_dir in sorted(root.iterdir()):
        metrics_path = fold_dir / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        payload = json.loads(metrics_path.read_text())
        site = str(payload["held_out_site"])
        out[site] = {"acc": float(payload["acc"]), "auc": float(payload["auc"])}
    return out


def compare(fides: dict[str, dict], baseline: dict[str, dict]) -> dict:
    sites = sorted(set(fides) & set(baseline))
    if not sites:
        raise RuntimeError("no shared sites between FIDES and baseline runs")
    acc_diff = np.array([fides[s]["acc"] - baseline[s]["acc"] for s in sites])
    auc_diff = np.array([fides[s]["auc"] - baseline[s]["auc"] for s in sites])
    # train/test fraction is approximate (5/6 train, 1/6 test for 6-site LOSO)
    t, p = corrected_resampled_t(acc_diff, train_frac=5 / 6, test_frac=1 / 6)
    return {
        "sites": sites,
        "sites_fides_exceeds": int((acc_diff > 0).sum()),
        "acc_delta_mean": float(acc_diff.mean()),
        "acc_delta_min": float(acc_diff.min()),
        "acc_delta_max": float(acc_diff.max()),
        "auc_delta_mean": float(auc_diff.mean()),
        "cohens_dz": float(acc_diff.mean() / (acc_diff.std(ddof=1) + 1e-12)),
        "t_stat": t,
        "p_raw": p,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fides-dir", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/per_site"))
    args = p.parse_args()

    fides = load_fold_metrics(args.fides_dir)

    rows = {}
    for b in BASELINES:
        baseline_root = args.baseline_dir / b
        if not baseline_root.is_dir():
            raise FileNotFoundError(baseline_root)
        baseline = load_fold_metrics(baseline_root)
        rows[b] = compare(fides, baseline)

    p_vals = [rows[b]["p_raw"] for b in BASELINES]
    p_corr = holm_correct(p_vals)
    for b, pc in zip(BASELINES, p_corr):
        rows[b]["p_holm"] = float(pc)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "per_site_comparison.json").write_text(json.dumps(rows, indent=2))
    LOG.info("Wrote %s", args.out / "per_site_comparison.json")


if __name__ == "__main__":
    main()
