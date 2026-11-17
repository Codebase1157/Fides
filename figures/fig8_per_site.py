"""Fig. 8: per-site LOSO ACC with 95% participant-level bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHODS = ["BrainGFM", "Wang et al.", "BrainCL", "BRIO", "CORE", "FIDES (ours)"]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#E24A33"]


def load_values(json_path: Path) -> dict:
    return json.loads(Path(json_path).read_text())


def plot_panel(ax, sites: list[str], per_site: dict[str, dict], title: str) -> None:
    x = np.arange(len(sites))
    width = 0.13
    for i, method in enumerate(METHODS):
        vals = [per_site[method][s]["acc"] for s in sites]
        errs = [per_site[method][s]["ci"] for s in sites]
        ax.bar(x + i * width, vals, width=width, color=COLORS[i], label=method, yerr=errs, capsize=2)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(sites)
    ax.set_ylabel("LOSO ACC (%)")
    ax.set_title(title)
    ax.set_ylim(60, 80)
    ax.grid(axis="y", linestyle=":")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-site-json", default="results/per_site/per_site_comparison.json")
    p.add_argument("--out", default="figures/output/fig8_per_site.pdf")
    args = p.parse_args()

    # In a full run, read the per-site numbers and bootstrap intervals from
    # Table XIV's JSON dump; here we illustrate the layout the script expects.
    dummy_sites = ["NYU", "UM1", "UCLA1", "USM", "PITT", "YALE"]
    per_site = {m: {s: {"acc": 70, "ci": 1.5} for s in dummy_sites} for m in METHODS}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    plot_panel(axes[0], dummy_sites, per_site, "(a) ABIDE")
    plot_panel(axes[1], ["S20", "S21", "S25", "S1", "S8", "S9"], per_site, "(b) REST-meta-MDD")
    axes[0].legend(loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
