"""Entry point for training any baseline on our splits.

Usage:
    python baselines/run_baseline.py --name BrainCL --dataset abide --protocol loso
    python baselines/run_baseline.py --name BrainGFM --dataset rest_meta_mdd --protocol 10fold
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("baseline")

NAME_TO_MODULE = {
    "SVM":            "baselines.svm.runner",
    "RF":             "baselines.rf.runner",
    "GAT":            "baselines.gat.runner",
    "BrainGNN":       "baselines.braingnn.runner",
    "IBGNN":          "baselines.ibgnn.runner",
    "BrainIB":        "baselines.brainib.runner",
    "Xu_et_al":       "baselines.xu_et_al.runner",
    "VanillaTF":      "baselines.vanillatf.runner",
    "BrainNetTF":     "baselines.brainnettf.runner",
    "Yu_et_al":       "baselines.yu_et_al.runner",
    "MSSTAN":         "baselines.msstan.runner",
    "Miniformer":     "baselines.miniformer.runner",
    "BrainCL":        "baselines.braincl.runner",
    "GCA":            "baselines.gca.runner",
    "A-GCL":          "baselines.a_gcl.runner",
    "Wang_et_al_22":  "baselines.wang_et_al_22.runner",
    "CIA-GCL":        "baselines.cia_gcl.runner",
    "Zhang_et_al_24": "baselines.zhang_et_al_24.runner",
    "Zhou_et_al":     "baselines.zhou_et_al.runner",
    "Song_et_al":     "baselines.song_et_al.runner",
    "Zhang_et_al_28": "baselines.zhang_et_al_28.runner",
    "MADCLE":         "baselines.madcle.runner",
    "CORE":           "baselines.core.runner",
    "BRIO":           "baselines.brio.runner",
    "Ji_et_al":       "baselines.ji_et_al.runner",
    "SF-DisenNet":    "baselines.sf_disennet.runner",
    "Wang_et_al_10":  "baselines.wang_et_al_10.runner",
    "BrainGFM":       "baselines.braingfm.runner",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, choices=list(NAME_TO_MODULE))
    p.add_argument("--dataset", required=True, choices=["abide", "rest_meta_mdd"])
    p.add_argument("--protocol", required=True, choices=["10fold", "loso", "cross"])
    p.add_argument("--config", required=False)
    p.add_argument("--out", default="results/baselines")
    args = p.parse_args()

    mod_path = NAME_TO_MODULE[args.name]
    try:
        module = importlib.import_module(mod_path)
    except ImportError:
        LOG.error("Module not installed: %s", mod_path)
        raise

    cfg_path = args.config or f"baselines/configs/{args.name.lower()}_{args.dataset}_{args.protocol}.yaml"
    LOG.info("Running %s / %s / %s with %s", args.name, args.dataset, args.protocol, cfg_path)
    result = module.run(dataset=args.dataset, protocol=args.protocol, config_path=cfg_path)

    out_dir = Path(args.out) / args.name / args.dataset / args.protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    LOG.info("Wrote %s", out_dir / "result.json")


if __name__ == "__main__":
    main()
