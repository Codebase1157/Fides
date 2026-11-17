"""Render BrainNet Viewer edge plots from attribution tables.

Takes a top-K edge list in the JSON format emitted by gradient_attribution.py,
maps nodes to Power/Yeo networks and writes a .node and .edge file that
BrainNet Viewer [66] can open directly.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("brainnet")

NETWORK_COLOR = {"DMN": 1, "SAL": 2, "FPCN": 3, "DAN": 4, "LIM": 5, "SMN": 6, "VIS": 7}


def write_node(path: Path, node_table: list[dict]) -> None:
    with path.open("w") as fh:
        for n in node_table:
            fh.write(f"{n['x']:.2f}\t{n['y']:.2f}\t{n['z']:.2f}\t{NETWORK_COLOR.get(n['network'], 0)}\t{n['size']:.2f}\t{n['label']}\n")


def write_edge(path: Path, n: int, edges: list[tuple[int, int, float]]) -> None:
    mat = [[0.0] * n for _ in range(n)]
    for i, j, w in edges:
        mat[i][j] = w
        mat[j][i] = w
    with path.open("w") as fh:
        for row in mat:
            fh.write(" ".join(f"{v:.6f}" for v in row) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--attribution-json", required=True)
    p.add_argument("--node-table", required=True, help="CSV with x,y,z,network,size,label columns")
    p.add_argument("--out-prefix", required=True)
    args = p.parse_args()

    data = json.loads(Path(args.attribution_json).read_text())
    edges = data.get("edges", [])
    # Full run reads node_table into a list[dict]; placeholder here just names the outputs.
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Would write %s.node and %s.edge from %d edges", out_prefix, out_prefix, len(edges))


if __name__ == "__main__":
    main()
