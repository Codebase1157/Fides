"""Build LOSO (leave-one-site-out) splits from a phenotypic CSV.

Reads the dataset's participant table (one row per subject, with columns
``subject_id``, ``site``, ``diagnosis``, ``age``, ``sex``), and writes one
JSON file per site in ``--sites``.

For each held-out site, participants from that site form the test set,
and participants from every other site are split into a training and a
validation fold (stratified by diagnosis) using ``--val-fraction``.

No subject IDs are hard-coded; the output is a function of the input CSV
and ``--seed`` alone. The CSV's SHA-256 is written into every fold file
so that reviewers can verify a given fold came from the same phenotypic
release they are looking at.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


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


def one_fold(
    df: pd.DataFrame,
    held_out: str,
    val_fraction: float,
    seed: int,
) -> dict:
    test_df = df[df["site"] == held_out]
    pool_df = df[df["site"] != held_out]
    if len(test_df) == 0:
        raise ValueError(f"No participants for site {held_out!r} in the CSV")
    if len(pool_df) == 0:
        raise ValueError("Pool is empty; need more than one site")

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    pool_ids = pool_df["subject_id"].to_numpy()
    pool_y = pool_df["diagnosis"].to_numpy()
    train_idx, val_idx = next(splitter.split(pool_ids, pool_y))

    return {
        "train": sorted(pool_ids[train_idx].tolist()),
        "val": sorted(pool_ids[val_idx].tolist()),
        "test": sorted(test_df["subject_id"].tolist()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--participants", type=Path, required=True)
    p.add_argument("--dataset", required=True, help="e.g. 'abide'")
    p.add_argument(
        "--sites",
        nargs="+",
        required=True,
        help="Held-out site codes, in the order they should be numbered.",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260101)
    args = p.parse_args()

    df = load_participants(args.participants)
    csv_digest = file_sha256(args.participants)
    args.out.mkdir(parents=True, exist_ok=True)

    for idx, site in enumerate(args.sites):
        fold_seed = args.seed + idx
        parts = one_fold(df, site, args.val_fraction, fold_seed)
        payload = {
            "dataset": args.dataset,
            "scheme": "loso",
            "fold": idx,
            "held_out_site": site,
            "seed": fold_seed,
            "source_csv": str(args.participants),
            "source_csv_sha256": csv_digest,
            **parts,
        }
        out_path = args.out / f"fold_{idx:02d}_{site}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(
            f"wrote {out_path}  "
            f"train={len(parts['train'])}  val={len(parts['val'])}  "
            f"test={len(parts['test'])}"
        )


if __name__ == "__main__":
    main()
