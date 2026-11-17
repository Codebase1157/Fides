# Data preparation

We cannot redistribute the raw imaging data for any of the five corpora used
in this paper. All of them are freely available for research after a
data-use agreement with the host consortium. Follow the steps below to obtain
and preprocess the data.

## Required corpora

| Corpus                          | Role in FIDES                         | Access URL                                          |
|---------------------------------|---------------------------------------|-----------------------------------------------------|
| ABIDE I                         | ASD diagnosis, V1/V4/V5               | https://fcon_1000.projects.nitrc.org/indi/abide/    |
| ABIDE II                        | Unseen-site evaluation (Table XI)     | https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html |
| REST-meta-MDD                   | MDD diagnosis, V1/V4/V5               | http://rfmri.org/REST-meta-MDD                      |
| SRPBS multi-disorder            | External test                         | https://bicr-resource.atr.jp/srpbsopen/             |
| SRPBS traveling-subject         | V3 pairs, invariance audit            | https://bicr-resource.atr.jp/srpbs_tscohort/        |
| HCP S1200                       | V2 pairs                              | https://www.humanconnectome.org/study/hcp-young-adult |

## Expected layout under `data/raw/`

```
data/raw/
├── ABIDE_I/
│   └── Outputs/dparsf/nofilt_noglobal/rois_cc200/
├── ABIDE_II/
│   └── Outputs/dparsf/nofilt_noglobal/rois_cc200/
├── REST-meta-MDD/
│   └── Phase1-Sharing/Results/ROISignals_FunImgARCWF/
├── SRPBS_MD/
│   └── Nifti/
├── SRPBS_TS/
│   └── Nifti/
└── HCP_S1200/
    └── {Subject}/MNINonLinear/Results/
```

## Preprocessing

Run the full pipeline with:

```bash
bash preprocessing/run_dparsf.sh data/raw data/processed
python preprocessing/build_fc.py --atlas cc200
python preprocessing/build_fc.py --atlas aal3
python preprocessing/combat_harmonize.py --fc data/processed/fc
python preprocessing/build_metadata.py
```

The DPARSF steps follow the BrainCL configuration [13]: head-motion
correction, nuisance regression (24 motion parameters, WM, CSF), MNI
normalization, 0.01-0.10 Hz band-pass filter, CC200 (default) and AAL3
(robustness check) atlases, Pearson correlation into a 200x200 or 166x166
functional connectivity matrix, Fisher-z transform, and absolute-value
threshold of 0.1 to sparsify.

## Metadata

`data/metadata/` contains csv tables with one row per participant:

- `abide_participants.csv`
- `abide2_participants.csv`
- `rest_meta_mdd_participants.csv`
- `srpbs_md_participants.csv`
- `srpbs_ts_sessions.csv`
- `hcp_participants.csv`
- `severity_scores.csv` (ADOS totals for ABIDE, HAMD-17 for REST-meta-MDD)
- `scanners.csv` (manufacturer, model, site, role)

Each file is an anonymized index: it carries subject IDs, site, scanner, age,
sex, diagnosis, and any derived labels used by the splits in `splits/`. No
imaging data is stored here.
