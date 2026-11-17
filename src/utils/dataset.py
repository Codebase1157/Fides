"""Torch Dataset for functional connectivity matrices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FCDataset(Dataset):
    """Functional connectivity dataset with site, diagnosis and severity."""

    def __init__(self, fc_root: Path, subject_ids: list[str], meta_csv: Path,
                 severity_csv: Path | None = None):
        self.fc_root = Path(fc_root)
        self.ids = subject_ids
        meta = pd.read_csv(meta_csv).set_index("subject_id")
        self.meta = meta.loc[[sid for sid in subject_ids if sid in meta.index]]
        if severity_csv and Path(severity_csv).exists():
            sev = pd.read_csv(severity_csv).set_index("subject_id")
            self.severity = sev.loc[self.meta.index.intersection(sev.index)]
        else:
            self.severity = pd.DataFrame(columns=["score"])

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        sid = self.meta.index[idx]
        fc = np.load(self.fc_root / sid / f"{sid}.npy")
        row = self.meta.loc[sid]
        return {
            "subject_id": sid,
            "fc": torch.from_numpy(fc).float(),
            "site": str(row["site"]),
            "diagnosis": str(row["diagnosis"]),
            "age": float(row.get("age", np.nan)),
            "sex": str(row.get("sex", "U")),
            "severity": float(self.severity.loc[sid, "score"]) if sid in self.severity.index else np.nan,
        }


def load_split(split_path: Path) -> dict:
    with Path(split_path).open() as fh:
        return json.load(fh)
