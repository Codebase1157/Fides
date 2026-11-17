"""Assemble headline tables from the per-experiment JSON dumps."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("assemble")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loso-dir", default="results/loso")
    p.add_argument("--tenfold-dir", default="results/10fold")
    p.add_argument("--cross-dir", default="results/cross")
    p.add_argument("--audits-dir", default="results/audits")
    p.add_argument("--out", default="results/tables")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Table IV (LOSO)
    rows = []
    for ds in ("abide", "rest_meta_mdd"):
        folds_file = Path(args.loso_dir) / ds / "folds_summary.json"
        if not folds_file.exists():
            continue
        folds = json.loads(folds_file.read_text())
        for f in folds:
            rows.append({"dataset": ds, "fold": f["fold"], "acc": f["best"]["acc"], "auc": f["best"]["auc"]})
    write_csv(out / "table4_loso.csv", rows)
    LOG.info("Wrote %d rows into table4_loso.csv", len(rows))


if __name__ == "__main__":
    main()
