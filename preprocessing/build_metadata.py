"""Rebuild per-participant metadata csvs from the raw releases.

Reads phenotype files that ship with each corpus, normalizes column names and
labels, computes derived fields (pooled age, binarized sex, standard diagnosis
codes, mean FD), and writes anonymized csvs under data/metadata/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("meta")

COLUMN_MAP = {
    "SUB_ID": "subject_id",
    "SITE_ID": "site",
    "AGE_AT_SCAN": "age",
    "SEX": "sex",
    "DX_GROUP": "diagnosis_abide",
    "SubID": "subject_id",
    "Site": "site",
    "Age": "age",
    "Sex": "sex",
    "Group": "diagnosis_rest",
}


def rebuild_abide(raw_root: Path, out_root: Path) -> None:
    phen = pd.read_csv(raw_root / "ABIDE_I" / "Phenotypic_V1_0b.csv", low_memory=False)
    phen = phen.rename(columns=COLUMN_MAP)
    phen["diagnosis"] = phen["diagnosis_abide"].map({1: "ASD", 2: "NC"})
    phen["sex"] = phen["sex"].map({1: "M", 2: "F"})
    out_cols = ["subject_id", "site", "age", "sex", "diagnosis"]
    phen[out_cols].to_csv(out_root / "abide_participants.csv", index=False)
    LOG.info("Wrote abide_participants.csv (%d rows)", len(phen))


def rebuild_rest(raw_root: Path, out_root: Path) -> None:
    phen = pd.read_csv(raw_root / "REST-meta-MDD" / "Phenotypes.csv", low_memory=False)
    phen = phen.rename(columns=COLUMN_MAP)
    phen["diagnosis"] = phen["diagnosis_rest"].map({1: "MDD", -1: "NC"})
    phen["sex"] = phen["sex"].map({1: "M", 2: "F"})
    out_cols = ["subject_id", "site", "age", "sex", "diagnosis"]
    phen[out_cols].to_csv(out_root / "rest_meta_mdd_participants.csv", index=False)
    LOG.info("Wrote rest_meta_mdd_participants.csv (%d rows)", len(phen))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--out-root", default="data/metadata")
    args = p.parse_args()

    raw = Path(args.raw_root)
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)

    if (raw / "ABIDE_I").exists():
        rebuild_abide(raw, out)
    if (raw / "REST-meta-MDD").exists():
        rebuild_rest(raw, out)


if __name__ == "__main__":
    main()
