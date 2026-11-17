"""Smoke tests that run on CPU in under a minute."""

from __future__ import annotations

import torch

from src.losses import block_nt_xent, stratified_hsic
from src.models import FIDES, FIDESConfig


def test_fides_forward_shapes():
    cfg = FIDESConfig()
    cfg.n_rois = 32
    cfg.encoder.n_rois = 32
    cfg.encoder.d_model = 16
    cfg.encoder.readout_dim = 32
    model = FIDES(cfg)
    fc = torch.randn(4, 32, 32)
    fc = 0.5 * (fc + fc.transpose(-1, -2))
    out = model(fc)
    assert out["z_dis"].shape == (4, cfg.d_dis)
    assert out["z_id"].shape == (4, cfg.d_id)
    assert out["logits"].shape == (4, 2)
    assert out["fc_hat"].shape == (4, 32, 32)


def test_block_nt_xent_symmetric():
    z = torch.randn(4, 8)
    assert block_nt_xent(z, z) >= 0.0
    assert torch.isclose(block_nt_xent(z, z), block_nt_xent(z, z))


def test_stratified_hsic_nonnegative():
    z1 = torch.randn(12, 8)
    z2 = torch.randn(12, 8)
    strata = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    assert stratified_hsic(z1, z2, strata) >= 0.0
