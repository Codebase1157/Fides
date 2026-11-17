"""Entry point that builds V1-V5 pair indices from configured sources."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from .pair_builder import V1Halves, V2Sessions, V3Traveling, V4CrossSite, V5Combat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("build_views")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/views.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_root = Path(cfg["out_root"])

    V1Halves(seed=cfg["seed"]).save(
        V1Halves(seed=cfg["seed"]).build(Path(cfg["v1"]["ts_root"]), cfg["v1"]["subjects"]),
        out_root / "v1_halves.json",
    )
    V2Sessions(seed=cfg["seed"]).save(
        V2Sessions(seed=cfg["seed"]).build(Path(cfg["v2"]["sessions_csv"])),
        out_root / "v2_sessions.json",
    )
    V3Traveling(seed=cfg["seed"]).save(
        V3Traveling(seed=cfg["seed"]).build(Path(cfg["v3"]["ts_sessions_csv"])),
        out_root / "v3_ts_pairs.json",
    )
    V4CrossSite(seed=cfg["seed"]).save(
        V4CrossSite(seed=cfg["seed"]).build(
            Path(cfg["v4"]["participants_csv"]),
            k_per_subject=cfg["v4"].get("k_per_subject", 2),
        ),
        out_root / "v4_cross_site.json",
    )
    V5Combat(seed=cfg["seed"]).save(
        V5Combat(seed=cfg["seed"]).build(cfg["v5"]["subjects"]),
        out_root / "v5_combat.json",
    )
    LOG.info("Views written to %s", out_root)


if __name__ == "__main__":
    main()
