"""Apply neuroCombat harmonization to functional connectivity matrices.

ComBat is fit on the training sites of each fold so that no test information
leaks into the harmonization. The result is stored alongside the original FC
and consumed as the V5 view during contrastive training.

Usage:
    python preprocessing/combat_harmonize.py --fc data/processed/fc/cc200
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("combat")


def flatten_upper(fc: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(fc, k=1)
    return fc[iu]


def unflatten_upper(vec: np.ndarray, n: int) -> np.ndarray:
    fc = np.zeros((n, n), dtype=np.float32)
    iu = np.triu_indices(n, k=1)
    fc[iu] = vec
    fc = fc + fc.T
    return fc


def harmonize_fold(
    fc_dir: Path,
    meta: pd.DataFrame,
    train_ids: list[str],
    all_ids: list[str],
    out_dir: Path,
) -> None:
    mats = {}
    for sid in all_ids:
        path = fc_dir / sid / f"{sid}.npy"
        if not path.exists():
            continue
        mats[sid] = np.load(path)

    ids = [sid for sid in all_ids if sid in mats]
    n = next(iter(mats.values())).shape[0]
    data = np.stack([flatten_upper(mats[sid]) for sid in ids], axis=1)

    covars = meta.set_index("subject_id").loc[ids, ["site", "age", "sex", "diagnosis"]]
    covars = covars.reset_index(drop=True)

    combat_out = neuroCombat(
        dat=data,
        covars=covars,
        batch_col="site",
        categorical_cols=["sex", "diagnosis"],
        continuous_cols=["age"],
        ref_batch=None,
        parametric=True,
        eb=True,
    )
    harmonized = combat_out["data"]

    for i, sid in enumerate(ids):
        fc = unflatten_upper(harmonized[:, i], n)
        np.fill_diagonal(fc, 0.0)
        out_path = out_dir / sid / f"{sid}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, fc.astype(np.float32))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fc", required=True, help="Input FC directory, e.g. data/processed/fc/cc200")
    p.add_argument("--meta-root", default="data/metadata")
    p.add_argument("--splits-root", default="splits/loso")
    p.add_argument("--out-root", default="data/processed/fc_combat")
    args = p.parse_args()

    fc_dir = Path(args.fc)
    meta_root = Path(args.meta_root)
    out_root = Path(args.out_root) / fc_dir.name
    out_root.mkdir(parents=True, exist_ok=True)

    for dataset, meta_file in [
        ("abide", "abide_participants.csv"),
        ("rest_meta_mdd", "rest_meta_mdd_participants.csv"),
    ]:
        meta_path = meta_root / meta_file
        if not meta_path.exists():
            LOG.warning("Missing metadata %s; skipping %s", meta_path, dataset)
            continue
        meta = pd.read_csv(meta_path)
        splits_dir = Path(args.splits_root) / dataset
        for split_file in sorted(splits_dir.glob("fold_*.json")):
            with split_file.open() as fh:
                split = json.load(fh)
            fold_out = out_root / dataset / split_file.stem
            harmonize_fold(
                fc_dir,
                meta,
                train_ids=split["train"],
                all_ids=split["train"] + split["test"],
                out_dir=fold_out,
            )
            LOG.info("Harmonized %s", split_file.stem)


if __name__ == "__main__":
    main()
