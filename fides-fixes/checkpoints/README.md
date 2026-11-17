# Checkpoints

Pretrained weights behind every reported number. Weight files are large
(~70 MB each) and are distributed outside git. Each checkpoint ships with
the training log that produced it and the config file it was launched with.

## Status

The checkpoint archive is **not yet posted publicly**. We will deposit a
single Zenodo record carrying every file in the layout below when the paper
is accepted, and update `CHECKPOINT_URL` and `SHA256SUMS.txt` in this
directory with the archive's DOI and per-file digests at that point. Until
then, `download.sh` fails loudly rather than pretending to fetch anything.

Readers who need weights before the Zenodo deposit can retrain from
`configs/` (full pipeline ~72 h on one RTX 4090; see `experiments/README.md`)
or email the authors for an access-controlled copy.

## Expected layout

| File                                               | Config                                | Log                                                |
|----------------------------------------------------|---------------------------------------|----------------------------------------------------|
| `stage1/best.pt`                                   | `stage1_pretrain.yaml`                | `logs/stage1/train.log`                            |
| `loso/abide/fold_<idx>_<site>_best.pt`             | `stage2_loso_abide.yaml`              | `logs/stage2/abide_loso/train.log`                 |
| `loso/rest_meta_mdd/fold_<idx>_<site>_best.pt`     | `stage2_loso_rest_meta_mdd.yaml`      | `logs/stage2/rest_meta_mdd_loso/train.log`         |
| `scanner/scanner_disjoint_stage1/best.pt`          | `scanner_disjoint.yaml`               | `logs/stage2/scanner_disjoint/train.log`           |

Six LOSO folds per dataset; site codes come from the held-out site in the
corresponding split file under `splits/loso/`.

## Downloading (once hosted)

```bash
export FIDES_CHECKPOINT_URL=https://zenodo.org/records/<RECORD_ID>/files
bash checkpoints/download.sh
```

The script refuses to run without `FIDES_CHECKPOINT_URL` and refuses to
accept any file whose SHA-256 is not listed in `SHA256SUMS.txt`. We will
publish `SHA256SUMS.txt` alongside the Zenodo record, generated with:

```bash
cd checkpoints
find stage1 loso scanner -type f -name '*.pt' -print0 \
    | xargs -0 sha256sum > SHA256SUMS.txt
```
