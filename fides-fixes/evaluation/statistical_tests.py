"""Statistical tests used in Section VI-D.

(i)   corrected resampled t-test over 10-fold repeats [69]
(ii)  paired t-test over LOSO held-out sites with participant-level bootstrap
(iii) McNemar for ACC and DeLong for AUC on the cross-dataset test set

Holm correction over the twelve baselines or the five strongest baselines,
reported in Table XV.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("stats")


def corrected_resampled_t(diffs: np.ndarray, train_frac: float, test_frac: float) -> tuple[float, float]:
    """Nadeau and Bengio corrected resampled t statistic."""
    mean = diffs.mean()
    var = diffs.var(ddof=1)
    corr = 1.0 / len(diffs) + test_frac / (1 - test_frac)
    t = mean / np.sqrt(var * corr) if var > 0 else 0.0
    p = 2 * (1 - stats.t.cdf(abs(t), df=len(diffs) - 1))
    return float(t), float(p)


def paired_t(diffs: np.ndarray) -> tuple[float, float]:
    t, p = stats.ttest_rel(diffs, np.zeros_like(diffs))
    return float(t), float(p)


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> float:
    """Return exact p-value of McNemar's test."""
    b = int(((~a_correct) & b_correct).sum())
    c = int((a_correct & (~b_correct)).sum())
    n = b + c
    if n == 0:
        return 1.0
    p = 2 * stats.binom.cdf(min(b, c), n, 0.5)
    return float(min(p, 1.0))


def delong_variance(scores_pos: np.ndarray, scores_neg: np.ndarray) -> tuple[float, float]:
    """Return empirical AUC and its DeLong variance."""
    m, n = len(scores_pos), len(scores_neg)
    ranked = stats.rankdata(np.concatenate([scores_pos, scores_neg]))
    r_pos = ranked[:m]
    auc = (r_pos.sum() - m * (m + 1) / 2) / (m * n)
    sig = auc * (1 - auc) / min(m, n)
    return float(auc), float(sig)


def holm_correct(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(pvals))
    out = [min(1.0, (len(pvals) - r) * p) for r, p in zip(ranks, pvals)]
    return out
