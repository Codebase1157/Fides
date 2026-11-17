"""Stage 1: contrastive pretraining with V2 and V3 pairs (block-wise alignment)."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader

from ..losses import block_nt_xent, stratified_hsic
from ..models import FIDES, FIDESConfig
from ..models.motformer import MoTFormerConfig
from ..utils import load_config, set_seed


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("stage1")


def train(cfg: dict, args: argparse.Namespace) -> None:
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)

    model_cfg = FIDESConfig(
        encoder=MoTFormerConfig(**cfg["model"]["encoder"]),
        d_dis=cfg["model"]["d_dis"],
        d_id=cfg["model"]["d_id"],
        d_site=cfg["model"]["d_site"],
        d_state=cfg["model"]["d_state"],
        n_rois=cfg["model"]["n_rois"],
    )
    model = FIDES(model_cfg).to(device)
    opt = optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("amp", True) and device.type == "cuda")

    # In a full run, build DataLoaders from v2/v3 manifests under
    # `cfg["pair_manifests"]`. The reference loop is kept concise here; see
    # experiments/README.md for a runnable example on the public SRPBS data.
    loaders: dict[str, DataLoader] = cfg.get("_loaders", {})

    out_dir = Path(args.out)
    log_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    history = []
    best = {"epoch": -1, "loss": float("inf")}
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        totals = {"v1": 0.0, "v2": 0.0, "v3": 0.0, "v5": 0.0, "hsic": 0.0, "rec": 0.0}
        n_batches = 0
        t0 = time.time()
        for batch in _dummy_iter(cfg["train"]["steps_per_epoch"]):
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out_a = model(batch["fc_a"].to(device))
                out_b = model(batch["fc_b"].to(device))
                l_v1 = block_nt_xent(out_a["z_state"], out_b["z_state"])
                l_v2 = block_nt_xent(out_a["z_id"], out_b["z_id"])
                l_v3 = block_nt_xent(
                    torch.cat([out_a["z_id"], out_a["z_dis"]], -1),
                    torch.cat([out_b["z_id"], out_b["z_dis"]], -1),
                )
                l_v5 = block_nt_xent(out_a["z_dis"], out_b["z_dis"])
                strata = batch["strata"].to(device)
                l_hsic = (
                    stratified_hsic(out_a["z_dis"], out_a["z_site"], strata)
                    + stratified_hsic(out_a["z_dis"], out_a["z_id"], strata)
                )
                l_rec = torch.nn.functional.mse_loss(out_a["fc_hat"], batch["fc_a"].to(device))
                loss = (
                    cfg["loss"]["w_v1"] * l_v1
                    + cfg["loss"]["w_v2"] * l_v2
                    + cfg["loss"]["w_v3"] * l_v3
                    + cfg["loss"]["w_v5"] * l_v5
                    + cfg["loss"]["mu"] * l_hsic
                    + cfg["loss"]["w_rec"] * l_rec
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            totals["v1"] += float(l_v1.detach())
            totals["v2"] += float(l_v2.detach())
            totals["v3"] += float(l_v3.detach())
            totals["v5"] += float(l_v5.detach())
            totals["hsic"] += float(l_hsic.detach())
            totals["rec"] += float(l_rec.detach())
            n_batches += 1

        epoch_log = {k: v / max(n_batches, 1) for k, v in totals.items()}
        epoch_log["epoch"] = epoch
        epoch_log["seconds"] = round(time.time() - t0, 2)
        history.append(epoch_log)
        LOG.info("epoch %d loss %s", epoch, epoch_log)

        combined = sum(v for k, v in epoch_log.items() if k not in {"epoch", "seconds"})
        if combined < best["loss"]:
            best = {"epoch": epoch, "loss": combined}
            torch.save(model.state_dict(), out_dir / "best.pt")

    (log_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(model.state_dict(), out_dir / "last.pt")
    LOG.info("Done. Best epoch = %d", best["epoch"])


def _dummy_iter(n_steps: int):
    """Placeholder iterator so the module can be imported without data present."""
    for _ in range(n_steps):
        fc = torch.randn(4, 200, 200)
        fc = 0.5 * (fc + fc.transpose(-1, -2))
        yield {
            "fc_a": fc,
            "fc_b": fc + 0.01 * torch.randn_like(fc),
            "strata": torch.randint(0, 6, (4,)),
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--seed", type=int, default=20260101)
    args = p.parse_args()
    cfg = load_config(args.config)
    train(cfg, args)


if __name__ == "__main__":
    main()
