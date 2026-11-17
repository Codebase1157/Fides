"""Symmetric MSE reconstruction loss on functional connectivity."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def fc_reconstruction_loss(fc_hat: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
    diag = torch.arange(fc.size(-1), device=fc.device)
    target = fc.clone()
    target[:, diag, diag] = 0.0
    return F.mse_loss(fc_hat, target)
