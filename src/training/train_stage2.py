"""Stage 2: fine-tuning for disease classification under 10-fold / LOSO protocols."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..losses import block_nt_xent, stratified_hsic
from ..models import FIDES, FIDESConfig
from ..models.motformer import MoTFormerConfig
from ..utils import accuracy, auc, load_config, set_seed
from ..utils.dataset import FCDataset, load_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("stage2")


def build_model(cfg: dict, device: torch.device, pretrained: Path | None) -> FIDES:
    mcfg = FIDESConfig(
        encoder=MoTFormerConfig(**cfg["model"]["encoder"]),
        d_dis=cfg["model"]["d_dis"],
        d_id=cfg["model"]["d_id"],
        d_site=cfg["model"]["d_site"],
        d_state=cfg["model"]["d_state"],
        n_rois=cfg["model"]["n_rois"],
        n_classes=cfg["model"].get("n_classes", 2),
        sev_head=cfg["model"].get("sev_head", False),
    )
    model = FIDES(mcfg).to(device)
    if pretrained is not None:
        state = torch.load(pretrained, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        LOG.info("Loaded %s (missing=%d, unexpected=%d)", pretrained, len(missing), len(unexpected))
    return model


def train_one_fold(cfg: dict, split: dict, args: argparse.Namespace, fold: str) -> dict:
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed + hash(fold) % 1000)

    fc_root = Path(args.data_root) / "fc_combat" / cfg["model"]["atlas"] / cfg["dataset"] / fold
    meta_csv = Path(args.data_root).parent / "metadata" / f"{cfg['dataset']}_participants.csv"
    sev_csv = Path(args.data_root).parent / "metadata" / "severity_scores.csv"

    train_ds = FCDataset(fc_root, split["train"], meta_csv, sev_csv)
    test_ds = FCDataset(fc_root, split["test"], meta_csv, sev_csv)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(cfg, device, args.pretrained if args.pretrained else None)
    opt = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history = []
    best = {"epoch": -1, "acc": 0.0, "auc": 0.0}

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        totals = {"ce": 0.0, "hsic": 0.0, "rec": 0.0}
        n = 0
        for batch in train_loader:
            fc = batch["fc"].to(device)
            y = torch.tensor([1 if d != "NC" else 0 for d in batch["diagnosis"]], device=device)
            site_ids = torch.tensor([hash(s) % 100 for s in batch["site"]], device=device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(fc)
                ce = F.cross_entropy(out["logits"], y)
                hsic = stratified_hsic(out["z_dis"], out["z_site"], site_ids)
                rec = F.mse_loss(out["fc_hat"], fc)
                loss = ce + cfg["loss"]["mu"] * hsic + cfg["loss"]["w_rec"] * rec
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            totals["ce"] += float(ce.detach()) * y.size(0)
            totals["hsic"] += float(hsic.detach()) * y.size(0)
            totals["rec"] += float(rec.detach()) * y.size(0)
            n += y.size(0)
        scheduler.step()

        model.eval()
        y_true, y_pred, y_score = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                fc = batch["fc"].to(device)
                out = model(fc)
                probs = F.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
                preds = probs > 0.5
                y_score.extend(probs.tolist())
                y_pred.extend(preds.tolist())
                y_true.extend([1 if d != "NC" else 0 for d in batch["diagnosis"]])
        y_true = np.array(y_true)
        y_pred = np.array(y_pred, dtype=int)
        y_score = np.array(y_score)
        acc_val = accuracy(y_true, y_pred)
        auc_val = auc(y_true, y_score) if len(np.unique(y_true)) == 2 else float("nan")
        history.append({
            "epoch": epoch,
            "ce": totals["ce"] / max(n, 1),
            "hsic": totals["hsic"] / max(n, 1),
            "rec": totals["rec"] / max(n, 1),
            "acc": acc_val,
            "auc": auc_val,
        })
        LOG.info("fold %s epoch %d acc %.2f auc %.2f", fold, epoch, acc_val, auc_val)
        if acc_val > best["acc"]:
            best = {"epoch": epoch, "acc": acc_val, "auc": auc_val}
            torch.save(model.state_dict(), Path(args.out) / f"{fold}_best.pt")
    return {"fold": fold, "best": best, "history": history}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--pretrained", type=Path, default=None)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--seed", type=int, default=20260101)
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    log_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    results = []
    for split_file in sorted(Path(args.split_dir).glob("fold_*.json")):
        split = load_split(split_file)
        fold_name = split_file.stem
        t0 = time.time()
        res = train_one_fold(cfg, split, args, fold_name)
        res["seconds"] = round(time.time() - t0, 2)
        results.append(res)
        (log_dir / f"{fold_name}_history.json").write_text(json.dumps(res["history"], indent=2))

    (out_dir / "folds_summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
