# Splits

Deterministic subject-level splits used in the manuscript. These files carry
only subject ids; the actual imaging data stays under `data/processed/`.

Layout:

```
splits/
├── 10fold/
│   ├── abide/fold_{00..09}.json
│   └── rest_meta_mdd/fold_{00..09}.json
├── loso/
│   ├── abide/fold_{00..05}_{SITE}.json
│   └── rest_meta_mdd/fold_{00..05}_{SITE}.json
└── cross_dataset/
    ├── abide_to_srpbs.json
    └── rest_meta_mdd_to_srpbs.json
```

Each file contains:
- 10-fold: `{"fold": N, "train": [...], "test": [...]}`
- LOSO:    `{"held_out_site": "NYU", "train": [...], "test": [...]}`
- Cross:   `{"train_source": "abide", "train": [...], "test_source": "srpbs_md", "test": [...]}`

Regenerate deterministically with:

```bash
python -c "import importlib.util, pathlib; spec=importlib.util.spec_from_file_location('gen', 'splits/_gen.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"
```

Seed: 20260101 (matches `FIDES_SEED`).
