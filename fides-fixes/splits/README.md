# Splits

LOSO (leave-one-site-out) and stratified 10-fold splits for ABIDE and
REST-meta-MDD. Splits are **not** checked in with hand-written subject
lists. They are regenerated from the phenotypic CSVs in
`data/metadata/` by the two scripts in this directory, so the IDs stay
tied to whichever release of the raw data the reader actually downloaded.

## Why splits are generated, not committed

The participant IDs under ABIDE, ABIDE II, and REST-meta-MDD change
between releases (new scans added, participants withdrawn, QC updates).
Committing a snapshot of IDs under those names would misrepresent which
participants were in each fold for anyone who downloaded the data at a
different date. The generator scripts take the dated phenotypic CSV
distributed with each corpus and produce a deterministic split, so the
same CSV + seed always yields the same JSON files.

## Expected layout after generation

```
splits/
├── loso/
│   ├── abide/
│   │   └── fold_<idx>_<site>.json       # 6 files, one per ABIDE site
│   └── rest_meta_mdd/
│       └── fold_<idx>_<site>.json       # 6 files, one per REST-meta-MDD site
└── 10fold/
    ├── abide/
    │   └── fold_<idx>.json              # 10 files
    └── rest_meta_mdd/
        └── fold_<idx>.json              # 10 files
```

Each JSON file has the schema:

```json
{
  "dataset": "abide",
  "scheme": "loso",
  "held_out_site": "NYU",
  "fold": 0,
  "train": ["<SUB_ID>", ...],
  "val":   ["<SUB_ID>", ...],
  "test":  ["<SUB_ID>", ...],
  "seed": 20260101,
  "source_csv": "data/metadata/abide_participants.csv",
  "source_csv_sha256": "<digest of the CSV at generation time>"
}
```

## Generating

```bash
# LOSO (one fold per site)
python splits/build_loso_splits.py \
    --participants data/metadata/abide_participants.csv \
    --dataset abide \
    --sites NYU UM1 UCLA1 USM PITT YALE \
    --out splits/loso/abide \
    --seed 20260101

python splits/build_loso_splits.py \
    --participants data/metadata/rest_meta_mdd_participants.csv \
    --dataset rest_meta_mdd \
    --sites S20 S21 S25 S1 S8 S9 \
    --out splits/loso/rest_meta_mdd \
    --seed 20260101

# Stratified 10-fold (stratified by site x diagnosis)
python splits/build_kfold_splits.py \
    --participants data/metadata/abide_participants.csv \
    --dataset abide \
    --n-folds 10 \
    --out splits/10fold/abide \
    --seed 20260101

python splits/build_kfold_splits.py \
    --participants data/metadata/rest_meta_mdd_participants.csv \
    --dataset rest_meta_mdd \
    --n-folds 10 \
    --out splits/10fold/rest_meta_mdd \
    --seed 20260101
```

## Site codes

For ABIDE the six training sites are NYU, UM1, UCLA1, USM, PITT, YALE.
For REST-meta-MDD they are S20, S21, S25, S1, S8, S9. These match the
`site` column of the phenotypic CSVs distributed by the two consortia.
Any additional sites are kept out of the training pool and reserved for
unseen-site evaluation (see `experiments/scanner_overlap.py`).
