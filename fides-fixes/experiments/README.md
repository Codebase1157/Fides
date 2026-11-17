# Experiments

Each experiment script runs a slice of the manuscript and writes its
results under `results/<experiment>/`. The names map 1:1 to tables and
figures:

| Script                        | Table / Figure                   |
|-------------------------------|----------------------------------|
| `run_main.sh`                 | Tables III, IV, V, VI            |
| `run_ablations.sh`            | Table VII                        |
| `scanner_overlap.py`          | Tables X, XI                     |
| `severity_subset.py`          | Tables XII, XIII, Fig. 7         |
| `per_site_statistics.py`      | Tables XIV, XV, Fig. 8           |

All experiment scripts expect that the LOSO and 10-fold runs have already
written their per-fold result files (`metrics.json`, `predictions.npz`,
`representations.npz`) under `results/loso/` and `results/10fold/`. If a
needed input is missing, the script raises a `FileNotFoundError` rather
than filling in a placeholder number.

## Reproducing from scratch

```bash
# Produces logs/stage1/, logs/stage2/, results/loso/, results/10fold/
bash experiments/run_main.sh       # ~34 h on one RTX 4090

# Produces results/ablations/ and logs/ablations/
bash experiments/run_ablations.sh  # ~22 h

# Summaries and plots
python -m experiments.scanner_overlap --loso-dir results/loso --srpbs-dir data/processed/srpbs_md --out results/scanner
python -m experiments.severity_subset --loso-dir results/loso --out results/severity
python -m experiments.per_site_statistics --loso-dir results/loso --out results/per_site
```

All scripts read `configs/` and respect `FIDES_SEED`.
