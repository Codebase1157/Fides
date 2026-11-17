"""FIDES encoder: MoTFormer trunk + four latent blocks + decoder + classifier.

Four blocks:
    z_dis  : disease
    z_id   : identity
    z_site : site
    z_stt  : momentary state
    z_full : concatenation, used by the reconstruction decoder
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .motformer import MoTFormer, MoTFormerConfig


@dataclass
class FIDESConfig:
    encoder: MoTFormerConfig = field(default_factory=MoTFormerConfig)
    d_dis: int = 32
    d_id: int = 32
    d_site: int = 16
    d_state: int = 16
    n_classes: int = 2
    n_rois: int = 200
    sev_head: bool = False


class BlockProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FCDecoder(nn.Module):
    def __init__(self, z_dim: int, n_rois: int):
        super().__init__()
        self.n_rois = n_rois
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, 4 * z_dim),
            nn.GELU(),
            nn.Linear(4 * z_dim, n_rois * n_rois),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        fc = self.mlp(z).view(z.size(0), self.n_rois, self.n_rois)
        fc = 0.5 * (fc + fc.transpose(-1, -2))
        diag = torch.arange(self.n_rois, device=fc.device)
        fc[:, diag, diag] = 0.0
        return fc


class FIDES(nn.Module):
    def __init__(self, cfg: FIDESConfig | None = None):
        super().__init__()
        cfg = cfg or FIDESConfig()
        cfg.encoder.n_rois = cfg.n_rois
        self.cfg = cfg
        self.encoder = MoTFormer(cfg.encoder)
        d = cfg.encoder.readout_dim
        self.proj_dis = BlockProjector(d, cfg.d_dis)
        self.proj_id = BlockProjector(d, cfg.d_id)
        self.proj_site = BlockProjector(d, cfg.d_site)
        self.proj_state = BlockProjector(d, cfg.d_state)
        self.classifier = nn.Linear(cfg.d_dis, cfg.n_classes)
        self.decoder = FCDecoder(cfg.d_dis + cfg.d_id + cfg.d_site + cfg.d_state, cfg.n_rois)
        if cfg.sev_head:
            self.sev_head = nn.Sequential(
                nn.Linear(cfg.d_dis, cfg.d_dis),
                nn.GELU(),
                nn.Linear(cfg.d_dis, 1),
            )
        else:
            self.sev_head = None

    def encode(self, fc: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(fc)
        return {
            "z_dis": self.proj_dis(h),
            "z_id": self.proj_id(h),
            "z_site": self.proj_site(h),
            "z_state": self.proj_state(h),
            "h": h,
        }

    def forward(self, fc: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.encode(fc)
        z_full = torch.cat([out["z_dis"], out["z_id"], out["z_site"], out["z_state"]], dim=-1)
        out["z_full"] = z_full
        out["logits"] = self.classifier(out["z_dis"])
        out["fc_hat"] = self.decoder(z_full)
        if self.sev_head is not None:
            out["sev_hat"] = self.sev_head(out["z_dis"]).squeeze(-1)
        return out
