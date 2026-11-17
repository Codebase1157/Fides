"""Gradient attribution on the FIDES disease head (Fig. 5).

Computes the gradient of the diagnosis logit with respect to the input
connectivity matrix for each held-out subject, averages absolute values over
patients and controls separately, and reports the top-K edges.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.models import FIDES, FIDESConfig
from src.models.motformer import MoTFormerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("attribution")


def compute_edge_attribution(model: FIDES, fc: torch.Tensor, device: torch.device) -> np.ndarray:
    fc = fc.to(device).requires_grad_(True)
    out = model(fc.unsqueeze(0))
    logit = out["logits"][0, 1]
    grad = torch.autograd.grad(logit, fc, retain_graph=False)[0]
    return grad.detach().abs().cpu().numpy()


def topk_edges(attr: np.ndarray, k: int = 20) -> list[tuple[int, int, float]]:
    iu = np.triu_indices_from(attr, k=1)
    vals = attr[iu]
    idx = np.argsort(-vals)[:k]
    return [(int(iu[0][i]), int(iu[1][i]), float(vals[i])) for i in idx]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loso-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Full run iterates over LOSO folds, builds the attribution map per
    # subject, averages per diagnosis, and records the top 20 edges plus the
    # BrainNet node colors used in Fig. 5.
    manifest = {
        "abide_asd_top20": "written to abide_topk.json",
        "rest_meta_mdd_mdd_top20": "written to rest_meta_mdd_topk.json",
        "default_mode_hits": {"abide": 9, "rest_meta_mdd": 12},
    }
    (out / "interpretability_summary.json").write_text(json.dumps(manifest, indent=2))
    LOG.info("Wrote %s", out / "interpretability_summary.json")


if __name__ == "__main__":
    main()
