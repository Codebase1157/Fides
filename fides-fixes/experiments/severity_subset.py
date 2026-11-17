"""Severity-subset analysis (Tables XII, XIII; Fig. 7).

For each training site, loads the per-fold representations and severity
scores and computes:

    - Classification ACC and AUC on the severity-labeled subset (for
      FIDES, FIDES-sev, FIDES mu=0, and any baseline that writes
      ``representations.npz`` with the same schema).
    - Out-of-sample ridge-regression R^2 predicting severity from z_dis,
      and from z_id after age/sex residualization.
    - Stratified HSIC between z_dis and z_id, using the same Gaussian
      kernel used in Stage 2.

The script fails if the expected per-fold files are missing.

Inputs
------
``--loso-dir``   LOSO output root; each variant has one subdirectory per
                 fold containing ``representations.npz`` with keys
                 ``z_dis``, ``z_id``, ``diagnosis``, ``severity``, ``age``,
                 ``sex``, and ``prob`` (per-participant disease probability).

``--variants``   subdirectories under ``--loso-dir`` to process.
``--out``        output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("severity")


def residualize(y: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Return residuals of y on an OLS fit of covariates (column-augmented with 1)."""
    x = np.hstack([covariates, np.ones((len(y), 1))])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def cv_r2(z: np.ndarray, y: np.ndarray, seed: int) -> float:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(z):
        sc = StandardScaler().fit(z[tr])
        pred[te] = Ridge(alpha=1.0).fit(sc.transform(z[tr]), y[tr]).predict(sc.transform(z[te]))
    return float(r2_score(y, pred))


def gaussian_kernel(x: np.ndarray, sigma: float) -> np.ndarray:
    d = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    return np.exp(-d / (2 * sigma ** 2))


def hsic(x: np.ndarray, y: np.ndarray, sigma_x: float | None = None, sigma_y: float | None = None) -> float:
    n = len(x)
    if sigma_x is None:
        sigma_x = float(np.median(np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1)))) + 1e-8
    if sigma_y is None:
        sigma_y = float(np.median(np.sqrt(((y[:, None] - y[None]) ** 2).sum(-1)))) + 1e-8
    K = gaussian_kernel(x, sigma_x)
    L = gaussian_kernel(y, sigma_y)
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(K @ H @ L @ H) / (n - 1) ** 2)


def variant_summary(variant_dir: Path, seed: int) -> dict:
    z_dis, z_id, severity, diagnosis, age, sex, prob = [], [], [], [], [], [], []
    for fold_dir in sorted(variant_dir.iterdir()):
        reps_path = fold_dir / "representations.npz"
        if not reps_path.is_file():
            raise FileNotFoundError(reps_path)
        d = np.load(reps_path, allow_pickle=True)
        z_dis.append(np.asarray(d["z_dis"], dtype=float))
        z_id.append(np.asarray(d["z_id"], dtype=float)) if "z_id" in d.files else None
        severity.append(np.asarray(d["severity"], dtype=float))
        diagnosis.append(np.asarray(d["diagnosis"]).astype(int))
        age.append(np.asarray(d["age"], dtype=float))
        sex.append(np.asarray(d["sex"]).astype(int))
        prob.append(np.asarray(d["prob"], dtype=float))
    z_dis = np.concatenate(z_dis)
    severity_arr = np.concatenate(severity)
    diagnosis_arr = np.concatenate(diagnosis)
    age_arr = np.concatenate(age)
    sex_arr = np.concatenate(sex)
    prob_arr = np.concatenate(prob)

    has_sev = ~np.isnan(severity_arr)
    pred = (prob_arr[has_sev] >= 0.5).astype(int)
    out = {
        "n_subset": int(has_sev.sum()),
        "acc": float(balanced_accuracy_score(diagnosis_arr[has_sev], pred) * 100),
        "auc": float(roc_auc_score(diagnosis_arr[has_sev], prob_arr[has_sev]) * 100),
        "r2_zdis": cv_r2(z_dis[has_sev], severity_arr[has_sev], seed),
    }

    if z_id:
        z_id_arr = np.concatenate(z_id)
        covars = np.stack([age_arr, sex_arr.astype(float)], axis=1)
        sev_resid = residualize(severity_arr, covars)
        out["r2_zid_resid"] = cv_r2(z_id_arr[has_sev], sev_resid[has_sev], seed)
        out["hsic_zdis_zid"] = hsic(z_dis[has_sev], z_id_arr[has_sev])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loso-dir", type=Path, required=True)
    p.add_argument(
        "--variants",
        nargs="+",
        default=["fides_full", "fides_sev", "fides_mu0", "braincl"],
    )
    p.add_argument("--out", type=Path, default=Path("results/severity"))
    p.add_argument("--seed", type=int, default=20260101)
    args = p.parse_args()

    summary = {}
    for variant in args.variants:
        variant_dir = args.loso_dir / variant
        if not variant_dir.is_dir():
            raise FileNotFoundError(variant_dir)
        summary[variant] = variant_summary(variant_dir, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "severity_subset.json").write_text(json.dumps(summary, indent=2))
    LOG.info("Wrote %s", args.out / "severity_subset.json")


if __name__ == "__main__":
    main()
