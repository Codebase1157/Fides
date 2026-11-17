"""Block-wise NT-Xent contrastive loss over a chosen latent block."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def block_nt_xent(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Symmetric NT-Xent on L2-normalized embeddings.

    Args:
        z1, z2: (B, D) matching pairs across a view type
        temperature: softmax temperature
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_12 + loss_21)
