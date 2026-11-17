# data/processed/

Preprocessing writes functional connectivity matrices and derived inputs here:

```
data/processed/
├── fc/
│   ├── cc200/
│   │   ├── abide/<subject_id>.npy
│   │   ├── rest_meta_mdd/<subject_id>.npy
│   │   ├── srpbs_md/<subject_id>.npy
│   │   ├── srpbs_ts/<subject_id>_<session>.npy
│   │   └── hcp/<subject_id>_<session>.npy
│   └── aal3/...
├── fc_combat/
│   └── cc200/...
├── views/
│   ├── v1_halves.json
│   ├── v2_sessions.json
│   ├── v3_ts_pairs.json
│   ├── v4_cross_site.json
│   └── v5_combat.json
└── nuisance/
    ├── framewise_displacement.csv
    └── motion_params.csv
```

Contents are not tracked by git.
