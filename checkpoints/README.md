# Checkpoints

Pretrained model weights behind every reported number. Weight files are large
(~70 MB each) and are distributed outside git. Each checkpoint ships with its
training log in `logs/` and its config in `configs/`.

## Available checkpoints

| File                                          | Config                                | Log                                   | Size | SHA-256 (first 8) |
|-----------------------------------------------|---------------------------------------|---------------------------------------|------|-------------------|
| `stage1/best.pt`                              | `stage1_pretrain.yaml`                | `logs/stage1/train.log`               | 68M  | f3c04a7b          |
| `loso/abide/fold_00_NYU_best.pt`              | `stage2_loso_abide.yaml`              | `logs/stage2/abide_loso/train.log`    | 71M  | 8e22d1f4          |
| `loso/abide/fold_01_UM1_best.pt`              | `stage2_loso_abide.yaml`              | same                                  | 71M  | 2a914b03          |
| `loso/abide/fold_02_UCLA1_best.pt`            | `stage2_loso_abide.yaml`              | same                                  | 71M  | 6c5f7e38          |
| `loso/abide/fold_03_USM_best.pt`              | `stage2_loso_abide.yaml`              | same                                  | 71M  | d419a2bc          |
| `loso/abide/fold_04_PITT_best.pt`             | `stage2_loso_abide.yaml`              | same                                  | 71M  | 51b78c6d          |
| `loso/abide/fold_05_YALE_best.pt`             | `stage2_loso_abide.yaml`              | same                                  | 71M  | ae20639a          |
| `loso/rest_meta_mdd/fold_00_S20_best.pt`      | `stage2_loso_rest_meta_mdd.yaml`      | `logs/stage2/rest_meta_mdd_loso/train.log` | 71M | 9f3d1a28     |
| `loso/rest_meta_mdd/fold_01_S21_best.pt`      | same                                  | same                                  | 71M  | 42b905ec          |
| `loso/rest_meta_mdd/fold_02_S25_best.pt`      | same                                  | same                                  | 71M  | 78c3d1b5          |
| `loso/rest_meta_mdd/fold_03_S1_best.pt`       | same                                  | same                                  | 71M  | e07a4231          |
| `loso/rest_meta_mdd/fold_04_S8_best.pt`       | same                                  | same                                  | 71M  | b532f0cd          |
| `loso/rest_meta_mdd/fold_05_S9_best.pt`       | same                                  | same                                  | 71M  | 36ea1d7f          |
| `scanner/scanner_disjoint_stage1/best.pt`     | `scanner_disjoint.yaml`               | `logs/stage2/scanner_disjoint/train.log` | 68M | 17b982ce       |

## Downloading

Checkpoints are hosted on Zenodo and mirrored on OSF. Download with:

```bash
bash checkpoints/download.sh
```

Each file is verified against its SHA-256 before use. If a download fails,
the script resumes from the last complete block.
