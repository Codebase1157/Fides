# Experiments

Each experiment script runs a slice of the manuscript and writes its results
under `results/<experiment>/`. The names map 1:1 to tables and figures:

| Script                        | Table / Figure                   |
|-------------------------------|----------------------------------|
| `run_main.sh`                 | Tables III, IV, V, VI            |
| `run_ablations.sh`            | Table VII                        |
| `run_sensitivity.py`          | Fig. 4                           |
| `interpretability_topk.py`    | Fig. 5                           |
| `scanner_overlap.py`          | Tables X, XI                     |
| `severity_subset.py`          | Tables XII, XIII, Fig. 7         |
| `per_site_statistics.py`      | Tables XIV, XV, Fig. 8           |
| `transfer_to_patients.py`     | Fig. 6                           |

## Reproducing from scratch

```bash
bash experiments/run_main.sh       # ~34 h
bash experiments/run_ablations.sh  # ~22 h
python experiments/run_sensitivity.py
python experiments/scanner_overlap.py
python experiments/severity_subset.py
python experiments/per_site_statistics.py
python experiments/transfer_to_patients.py
python experiments/interpretability_topk.py
```

All scripts read `configs/` and respect `FIDES_SEED`.
