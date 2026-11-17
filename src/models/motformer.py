"""Multi-order Topology-aware Transformer (MoTFormer) encoder.

Follows the BrainCL architecture [13]: preference-random-walk injection of
multi-hop topology, hop-wise gated attention, and a learnable readout. The
output is a per-graph embedding that FIDES projects into four latent blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MoTFormerConfig:
    n_rois: int = 200
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    n_hops: int = 3
    dropout: float = 0.2
    readout_dim: int = 128


class HopGatedAttention(nn.Module):
    """Multi-head attention with per-hop gates."""

    def __init__(self, cfg: MoTFormerConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_k = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.hop_gate = nn.Parameter(torch.ones(cfg.n_hops))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, hops: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D); hops: (B, H, N, N) multi-hop topology bias
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        att = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        gates = F.softplus(self.hop_gate)
        bias = torch.einsum("bhij,h->bij", hops, gates)
        att = att + bias.unsqueeze(1)
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        out = (att @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class MoTFormerBlock(nn.Module):
    def __init__(self, cfg: MoTFormerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.att = HopGatedAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, hops: torch.Tensor) -> torch.Tensor:
        x = x + self.att(self.ln1(x), hops)
        x = x + self.mlp(self.ln2(x))
        return x


def preference_random_walk(adj: torch.Tensor, n_hops: int, alpha: float = 0.15) -> torch.Tensor:
    """Compute n_hops powers of a normalized adjacency (preference walk)."""
    deg = adj.sum(-1, keepdim=True) + 1e-8
    p = adj / deg
    hops = [torch.eye(adj.size(-1), device=adj.device).expand_as(adj)]
    cur = p
    for _ in range(n_hops - 1):
        hops.append(cur)
        cur = alpha * p + (1 - alpha) * (cur @ p)
    return torch.stack(hops, dim=1)  # (B, H, N, N)


class MoTFormer(nn.Module):
    def __init__(self, cfg: MoTFormerConfig | None = None):
        super().__init__()
        cfg = cfg or MoTFormerConfig()
        self.cfg = cfg
        self.node_embed = nn.Linear(cfg.n_rois, cfg.d_model)
        self.blocks = nn.ModuleList(MoTFormerBlock(cfg) for _ in range(cfg.n_layers))
        self.ln = nn.LayerNorm(cfg.d_model)
        self.readout = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.readout_dim),
            nn.GELU(),
            nn.Linear(cfg.readout_dim, cfg.readout_dim),
        )

    def forward(self, fc: torch.Tensor) -> torch.Tensor:
        """fc: (B, N, N) functional connectivity matrix."""
        adj = (fc.abs() > 0).float()
        hops = preference_random_walk(adj, self.cfg.n_hops)
        x = self.node_embed(fc)
        for blk in self.blocks:
            x = blk(x, hops)
        x = self.ln(x).mean(dim=1)  # mean pooling across nodes
        return self.readout(x)
