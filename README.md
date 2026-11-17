# FIDES: Identifiable Disentanglement of Disease, Identity, Site, and State in Functional Brain Networks via Natural-View Contrastive Learning

Official reproduction repository for the FIDES paper. This repository contains
source code, configurations, dataset preparation scripts, training pipelines,
evaluation code, and the exact splits used to produce every number reported in
the manuscript.

> **Note on data distribution.** We cannot redistribute the raw datasets.
> ABIDE, REST-meta-MDD, SRPBS (multi-disorder and traveling-subject releases),
> and HCP S1200 all require separate data-use agreements. `data/README.md`
> gives the official request URLs and the preprocessing scripts needed to
> reproduce our processed inputs from the raw files.

## Table of Contents
1. [Installation](#installation)
2. [Data preparation](#data-preparation)
3. [Reproducing the main results](#reproducing-the-main-results)
4. [Repository layout](#repository-layout)
5. [Pretrained checkpoints](#pretrained-checkpoints)
6. [Baselines](#baselines)
7. [Hardware and runtime](#hardware-and-runtime)
8. [Citing FIDES](#citing-fides)
9. [License](#license)

## Installation

```bash
git clone https://github.com/FIDES-brain/fides-code.git
cd fides-code

# Option A: conda
conda env create -f environment/environment.yml
conda activate fides

# Option B: pip (Python >= 3.10)
python -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements.txt
```

Verified on Ubuntu 22.04 with CUDA 12.1 and a single NVIDIA GeForce RTX 4090
(24 GB). A CPU-only run is supported for small smoke tests but is impractical
for the full pipeline.

## Data preparation

1. Request access to each dataset and place the raw archives in `data/raw/`
   (see `data/README.md` for folder names).
2. Run DPARSF preprocessing with the AAL3/CC200 atlases:
   ```bash
   bash preprocessing/run_dparsf.sh data/raw data/processed
   python preprocessing/build_fc.py --atlas cc200 --out data/processed/fc
   python preprocessing/combat_harmonize.py --fc data/processed/fc
   ```
3. Build the five natural-view pair indices:
   ```bash
   python src/views/build_views.py --config configs/views.yaml
   ```

## Reproducing the main results

`run_all.sh` chains Stage 1 pretraining, LOSO training, 10-fold training,
cross-dataset evaluation, ablations, and the shortcut audits.

```bash
bash run_all.sh                 # full pipeline (~72 h on one RTX 4090)
bash run_all.sh --stage pretrain
bash run_all.sh --stage loso
bash run_all.sh --stage audits
```

Per-table entry points are listed in `experiments/README.md`. The exact random
seeds, split files, and baseline hyperparameters that produced the published
numbers are checked into `splits/` and `configs/`.

## Repository layout

```
.
├── data/              # Dataset preparation: raw/, processed/, metadata/
├── preprocessing/     # fMRI preprocessing, FC construction, ComBat
├── src/               # Core FIDES implementation
│   ├── models/        # MoTFormer, four latent blocks, prototypes
│   ├── losses/        # Block-wise contrastive, HSIC, reconstruction
│   ├── training/      # Stage 1 and Stage 2 training loops
│   ├── views/         # V1-V5 natural-view generation
│   └── utils/         # Logging, metrics, config, seed helpers
├── configs/           # YAML configs for every experiment
├── splits/            # 10-fold, LOSO, and cross-dataset splits
├── baselines/         # 28 baseline methods
├── experiments/       # Experiment scripts and definitions
├── checkpoints/       # Pretrained weights (see section below)
├── logs/              # Training logs (stage1/, stage2/, ablations/)
├── evaluation/        # Accuracy, AUC, probes, statistical tests
├── interpretability/  # Gradient attributions and BrainNet plots
├── results/           # Numerical outputs behind every table
├── figures/           # Scripts and PDF/PNG for every figure
├── supplementary/     # Additional analyses and extended tables
├── environment/       # Dependency pins and hardware notes
├── run_all.sh         # Top-level pipeline script
├── requirements.txt   # Pip mirror of environment.yml
├── LICENSE            # MIT license
└── README.md          # This file
```

## Pretrained checkpoints

Stage 1 and the six LOSO Stage 2 checkpoints for ABIDE and REST-meta-MDD are
released as a separate archive (see `checkpoints/README.md` for the download
URL and SHA-256 sums). Each checkpoint is accompanied by its training log in
`logs/` and its config in `configs/`.

## Baselines

All 28 baseline methods are re-trained on our splits and preprocessing. For
every baseline we provide a thin wrapper that calls the authors' original code
or our reimplementation, together with the hyperparameter grid and the
selected configuration. See `baselines/README.md`.

## Hardware and runtime

| Stage                       | GPU hours (RTX 4090) |
|-----------------------------|----------------------|
| Stage 1 pretraining         | ~14                  |
| LOSO Stage 2, both datasets | ~34                  |
| 10-fold Stage 2             | ~18                  |
| Ablations and audits        | ~22                  |
| Baselines (28 methods)      | ~110                 |

## Citing FIDES

```bibtex
@article{fides2026,
  title   = {FIDES: Identifiable Disentanglement of Disease, Identity, Site,
             and State in Functional Brain Networks via Natural-View
             Contrastive Learning},
  author  = {First Author and Second Author and Third Author and Corresponding Author},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2026}
}
```

## License

Code is released under the MIT License (see `LICENSE`). Datasets remain under
their original terms.
