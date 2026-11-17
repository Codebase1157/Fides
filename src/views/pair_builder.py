"""Construct the five natural-view positive-pair indices used by FIDES.

V1 halves       : two halves of one scan (within-scan).
V2 sessions     : two sessions of the same participant.
V3 traveling    : one participant imaged on two scanners (SRPBS TS).
V4 cross-site   : two participants with the same diagnosis, different sites.
V5 ComBat       : a network paired with its ComBat-harmonized version.

Each class produces a json manifest of pair ids consumed by the training loop.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class _PairBuilder:
    name: str = ""

    def __init__(self, seed: int = 20260101):
        self.rng = random.Random(seed)

    def build(self, *args, **kwargs) -> list[dict]:
        raise NotImplementedError

    def save(self, pairs: list[dict], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            json.dump({"name": self.name, "n_pairs": len(pairs), "pairs": pairs}, fh, indent=2)


class V1Halves(_PairBuilder):
    name = "v1_halves"

    def build(self, ts_root: Path, subjects: Iterable[str]) -> list[dict]:
        pairs = []
        for sid in subjects:
            ts_path = ts_root / f"{sid}.npy"
            if not ts_path.exists():
                continue
            ts = np.load(ts_path, mmap_mode="r")
            T = ts.shape[0]
            mid = T // 2
            pairs.append({"id": sid, "a": [0, mid], "b": [mid, T]})
        return pairs


class V2Sessions(_PairBuilder):
    name = "v2_sessions"

    def build(self, sessions_csv: Path) -> list[dict]:
        df = pd.read_csv(sessions_csv)
        pairs = []
        for sid, grp in df.groupby("subject_id"):
            ses = grp["session_id"].tolist()
            if len(ses) < 2:
                continue
            for i in range(len(ses)):
                for j in range(i + 1, len(ses)):
                    pairs.append({"subject_id": sid, "a": ses[i], "b": ses[j]})
        return pairs


class V3Traveling(_PairBuilder):
    name = "v3_ts_pairs"

    def build(self, ts_sessions_csv: Path) -> list[dict]:
        df = pd.read_csv(ts_sessions_csv)
        pairs = []
        for sid, grp in df.groupby("subject_id"):
            scanners = grp.groupby("scanner")["session_id"].first().to_dict()
            keys = list(scanners.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    pairs.append({
                        "subject_id": sid,
                        "a": {"scanner": keys[i], "session_id": scanners[keys[i]]},
                        "b": {"scanner": keys[j], "session_id": scanners[keys[j]]},
                    })
        return pairs


class V4CrossSite(_PairBuilder):
    name = "v4_cross_site"

    def build(self, participants_csv: Path, k_per_subject: int = 2) -> list[dict]:
        df = pd.read_csv(participants_csv)
        by_group: dict[tuple[str, str], list[str]] = defaultdict(list)
        for _, row in df.iterrows():
            by_group[(row["diagnosis"], row["site"])].append(row["subject_id"])
        pairs = []
        for (dx, site_a), ids_a in by_group.items():
            other_sites = [s for (d, s), _ in by_group.items() if d == dx and s != site_a]
            if not other_sites:
                continue
            for sid_a in ids_a:
                for _ in range(k_per_subject):
                    site_b = self.rng.choice(other_sites)
                    sid_b = self.rng.choice(by_group[(dx, site_b)])
                    pairs.append({"a": sid_a, "b": sid_b, "diagnosis": dx})
        return pairs


class V5Combat(_PairBuilder):
    name = "v5_combat"

    def build(self, subjects: Iterable[str]) -> list[dict]:
        return [{"id": sid, "a": sid, "b": sid, "harmonized_b": True} for sid in subjects]
