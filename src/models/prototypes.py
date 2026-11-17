"""Prototype bank for V4 cross-site positive-pair construction.

For each (site, diagnosis) group we keep a set of K EMA prototypes in the
disease sub-space. Cross-site positives are drawn from the K prototypes of the
same diagnosis in a different site.
"""

from __future__ import annotations

import torch
from torch import nn


class PrototypeBank(nn.Module):
    def __init__(self, n_sites: int, n_classes: int, d_dis: int, n_prototypes: int = 4, momentum: float = 0.9):
        super().__init__()
        self.n_sites = n_sites
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes
        self.momentum = momentum
        self.register_buffer(
            "bank", torch.randn(n_sites, n_classes, n_prototypes, d_dis) * 0.01
        )
        self.register_buffer("init_mask", torch.zeros(n_sites, n_classes, n_prototypes, dtype=torch.bool))

    def update(self, z_dis: torch.Tensor, sites: torch.Tensor, labels: torch.Tensor) -> None:
        for s in torch.unique(sites):
            for c in torch.unique(labels):
                sel = (sites == s) & (labels == c)
                if not sel.any():
                    continue
                reps = z_dis[sel].detach()
                k = min(self.n_prototypes, reps.size(0))
                reps = reps[torch.randperm(reps.size(0))[:k]]
                if not self.init_mask[s, c, :k].any():
                    self.bank[s, c, :k] = reps
                else:
                    self.bank[s, c, :k] = self.momentum * self.bank[s, c, :k] + (1 - self.momentum) * reps
                self.init_mask[s, c, :k] = True

    def cross_site_positive(self, site: int, label: int) -> torch.Tensor:
        other = [s for s in range(self.n_sites) if s != site and self.init_mask[s, label].any()]
        if not other:
            return self.bank[site, label, 0]
        s = other[torch.randint(len(other), (1,)).item()]
        valid = self.init_mask[s, label]
        idx = valid.nonzero().flatten()
        return self.bank[s, label, idx[torch.randint(idx.numel(), (1,)).item()]]
