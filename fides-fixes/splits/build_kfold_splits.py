"""Build stratified k-fold splits from a phenotypic CSV.

Stratifies by the (site, diagnosis) pair so every fold sees a comparable
site and diagnosis composition. Writes one JSON file per fold under
``--out`` with the schema used elsewhere in the repo.

Like ``build_loso_splits.py``, no subject IDs are hard-coded: the output
is a deterministic function of the phenotypic CSV and ``--seed``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

REQUIRED_COLUMNS = ("subject_id", "site", "diagnosis")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_participants(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path}: missing required columns {missing}. "
            f"Expected at least {REQUIRED_COLUMNS}."
        )
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["site"] = df["site"].astype(str)
    df["diagnosis"] = df["diagnosis"].astype(int)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--participants", type=Path, required=True)
    p.add_argument("--dataset", required=True, help="e.g. 'abide'")
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260101)
    args = p.parse_args()

    df = load_participants(args.participants)
    csv_digest = file_sha256(args.participants)
    args.out.mkdir(parents=True, exist_ok=True)

    ids = df["subject_id"].to_numpy()
    strat = (df["site"] + "__" + df["diagnosis"].astype(str)).to_numpy()
    kf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.seed
    )

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(ids, strat)):
        rng = np.random.default_rng(args.seed + fold_idx)
        inner = np.array(train_val_idx)
        rng.shuffle(inner)
        n_val = max(1, int(round(len(inner) * args.val_fraction)))
        val_idx = inner[:n_val]
        train_idx = inner[n_val:]

        payload = {
            "dataset": args.dataset,
            "scheme": f"{args.n_folds}fold",
            "fold": fold_idx,
            "seed": args.seed + fold_idx,
            "source_csv": str(args.participants),
            "source_csv_sha256": csv_digest,
            "train": sorted(ids[train_idx].tolist()),
            "val":   sorted(ids[val_idx].tolist()),
            "test":  sorted(ids[test_idx].tolist()),
        }
        out_path = args.out / f"fold_{fold_idx:02d}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(
            f"wrote {out_path}  "
            f"train={len(payload['train'])}  val={len(payload['val'])}  "
            f"test={len(payload['test'])}"
        )


if __name__ == "__main__":
    main()
