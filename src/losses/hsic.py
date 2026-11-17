"""Stratified Hilbert-Schmidt Independence Criterion.

Suppresses residual dependence between latent blocks within each
(site, diagnosis) stratum, as described in Section IV-E of the manuscript.
"""

from __future__ import annotations

import torch


def rbf_kernel(x: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    if sigma is None:
        with torch.no_grad():
            dists = torch.cdist(x, x)
            sigma = float(dists.median().clamp_min(1e-6))
    pair = torch.cdist(x, x) ** 2
    return torch.exp(-pair / (2 * sigma ** 2))


def _hsic(k: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    n = k.size(0)
    h = torch.eye(n, device=k.device) - torch.ones_like(k) / n
    kh = h @ k @ h
    return torch.trace(kh @ l) / (n - 1) ** 2


def stratified_hsic(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    strata: torch.Tensor,
    min_group: int = 4,
) -> torch.Tensor:
    losses = []
    for g in torch.unique(strata):
        sel = strata == g
        if sel.sum() < min_group:
            continue
        ka = rbf_kernel(z_a[sel])
        kb = rbf_kernel(z_b[sel])
        losses.append(_hsic(ka, kb))
    if not losses:
        return z_a.new_zeros(())
    return torch.stack(losses).mean()
