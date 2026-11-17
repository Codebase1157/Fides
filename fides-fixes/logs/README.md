# Training logs

This directory is populated by the training entry points. It does not ship
with pre-recorded runs: the logs are written as each job runs so that any
reader can verify the losses, metrics, and wall-clock times reported in the
manuscript against the same files we produced ourselves.

## Expected contents after a full run

```
logs/
├── stage1/
│   ├── train.log                     # per-epoch stage-1 pretraining
│   └── history.json                  # same values as a parseable list
├── stage2/
│   ├── abide_loso/
│   │   ├── train.log
│   │   └── fold_<idx>_<site>_history.json
│   ├── abide_10fold/
│   │   ├── train.log
│   │   └── fold_results.json
│   ├── rest_meta_mdd_loso/
│   ├── rest_meta_mdd_10fold/
│   └── scanner_disjoint/
└── ablations/
    ├── ablation_summary.log
    └── <variant>_<dataset>.log
```

## Where each file is produced

| File                                                | Produced by                                           |
|-----------------------------------------------------|-------------------------------------------------------|
| `stage1/train.log`, `stage1/history.json`           | `python -m src.training.stage1 --config configs/stage1_pretrain.yaml --log-dir logs/stage1` |
| `stage2/abide_loso/*`                               | `python -m src.training.stage2 --config configs/stage2_loso_abide.yaml --log-dir logs/stage2/abide_loso` |
| `stage2/abide_10fold/*`                             | `python -m src.training.stage2 --config configs/stage2_10fold_abide.yaml --log-dir logs/stage2/abide_10fold` |
| `stage2/rest_meta_mdd_loso/*`                       | analogous `configs/stage2_loso_rest_meta_mdd.yaml`    |
| `stage2/rest_meta_mdd_10fold/*`                     | analogous `configs/stage2_10fold_rest_meta_mdd.yaml`  |
| `stage2/scanner_disjoint/*`                         | `configs/scanner_disjoint.yaml`                       |
| `ablations/*.log`, `ablations/ablation_summary.log` | `bash experiments/run_ablations.sh`                   |

## Log format

Each line of a `train.log` file is:

```
[<ISO-8601 timestamp>] INFO  <phase> epoch <ep>/<n> <key>=<value>  (...)  t=<s>s  [*BEST*]
```

Each JSON history is a list of per-epoch dictionaries with the same keys as
the log line, written atomically after every epoch so partially completed
runs remain parseable.

## Reproducing the logs we used

We do not post the training-log tarballs from our own runs in this repo. All
numbers in the manuscript's tables were taken from `results/tables/`, which
`src/training/` and `experiments/` write after a run. If a reader wants our
original logs for an audit, write to the authors; we keep them on request.
