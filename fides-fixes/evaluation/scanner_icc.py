"""Scanner-ICC audit using SRPBS traveling subjects (Table VI).

For each dimension of a representation, fits a one-way random-effects model
across scanners with participant as the random factor and computes

    ICC(1,1) = (MS_between - MS_within) / (MS_between + (k - 1) * MS_within)

with ``k`` the effective number of sessions per participant. We report the
mean ICC across the representation dimensions, which is the per-dimension
average used by the prior work we compare against.

Inputs
------
``--representations``  .npz file with keys:

        z           : (n_sessions, d) float, embeddings
        participant : (n_sessions,) str or int
        scanner     : (n_sessions,) str

``--out``              JSON file to write.

The script fails if the file is missing, rather than substituting a number.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("icc")


def icc_1_1(x: np.ndarray, groups: np.ndarray) -> float:
    """ICC(1,1) across ``groups`` on a 1D measurement ``x``.

    ``groups`` is a 1D array of subject labels.
    """
    unique = np.unique(groups)
    n_subjects = len(unique)
    if n_subjects < 2:
        return float("nan")

    overall = x.mean()
    ss_between, ss_within = 0.0, 0.0
    n_sessions = 0
    for g in unique:
        mask = groups == g
        xg = x[mask]
        ng = len(xg)
        n_sessions += ng
        ss_between += ng * (xg.mean() - overall) ** 2
        ss_within += ((xg - xg.mean()) ** 2).sum()

    df_b = n_subjects - 1
    df_w = n_sessions - n_subjects
    if df_w <= 0 or df_b <= 0:
        return float("nan")
    ms_b = ss_between / df_b
    ms_w = ss_within / df_w
    k = n_sessions / n_subjects
    denom = ms_b + (k - 1) * ms_w
    if denom <= 0:
        return 0.0
    return float((ms_b - ms_w) / denom)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--representations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if not args.representations.is_file():
        raise FileNotFoundError(args.representations)

    data = np.load(args.representations, allow_pickle=True)
    z = np.asarray(data["z"], dtype=float)
    participant = np.asarray(data["participant"]).astype(str)
    scanner = np.asarray(data["scanner"]).astype(str)

    LOG.info(
        "loaded z=%s, %d participants, %d scanners",
        z.shape,
        len(np.unique(participant)),
        len(np.unique(scanner)),
    )

    per_dim = np.array([icc_1_1(z[:, j], participant) for j in range(z.shape[1])])
    per_dim = per_dim[~np.isnan(per_dim)]
    if per_dim.size == 0:
        raise RuntimeError("no finite ICC values; check inputs")

    summary = {
        "n_dims_used": int(per_dim.size),
        "icc_mean": float(per_dim.mean()),
        "icc_median": float(np.median(per_dim)),
        "icc_q25": float(np.quantile(per_dim, 0.25)),
        "icc_q75": float(np.quantile(per_dim, 0.75)),
        "interpretation": "higher is better; z should be stable across scanners",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    LOG.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
