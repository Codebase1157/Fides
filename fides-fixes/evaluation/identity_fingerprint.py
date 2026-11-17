"""Identity fingerprinting audit (Table VI).

Each HCP test participant's second-session representation is matched to
the most correlated first-session representation. Chance is 1 / N_test.
A disease representation ought not to identify healthy individuals.

Inputs
------
``--representations``  .npz with keys:

        z1          : (n, d)  first-session representations
        z2          : (n, d)  second-session representations, paired with z1
        subject_id  : (n,)    optional, for logging only

``--out``              JSON file to write.

The script uses cosine similarity for matching; this matches the setup of
the original audit paper.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("fingerprint")


def top1_accuracy(z1: np.ndarray, z2: np.ndarray) -> float:
    z1n = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-8)
    z2n = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-8)
    sim = z2n @ z1n.T
    pred = sim.argmax(axis=1)
    return float((pred == np.arange(len(z2))).mean() * 100)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--representations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if not args.representations.is_file():
        raise FileNotFoundError(args.representations)

    data = np.load(args.representations, allow_pickle=True)
    z1 = np.asarray(data["z1"], dtype=float)
    z2 = np.asarray(data["z2"], dtype=float)
    if z1.shape != z2.shape:
        raise RuntimeError(f"z1 and z2 must share shape, got {z1.shape} vs {z2.shape}")

    acc = top1_accuracy(z1, z2)
    n = z1.shape[0]
    summary = {
        "n_test": int(n),
        "fingerprint_acc": acc,
        "chance": float(100.0 / n),
        "interpretation": "closer to chance is better for a disease representation",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    LOG.info("Wrote %s (acc=%.2f, chance=%.2f)", args.out, acc, 100.0 / n)


if __name__ == "__main__":
    main()
