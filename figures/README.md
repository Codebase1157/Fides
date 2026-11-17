# figures/

Scripts and outputs for every manuscript figure.

| Figure | Content                                           | Script                     |
|--------|---------------------------------------------------|----------------------------|
| 1      | View types and block alignment                    | `fig1_views.py`            |
| 2      | Four-block FIDES architecture                     | `fig2_architecture.py`     |
| 3      | t-SNE of BrainCL vs FIDES embeddings              | `fig3_tsne.py`             |
| 4      | Sensitivity of LOSO ACC to hyperparameters        | `fig4_sensitivity.py`      |
| 5      | Top-20 edges by attribution                       | `fig5_edges.py`            |
| 6      | Scanner-invariance transfer to patients           | `fig6_transfer.py`         |
| 7      | ADOS and HAMD-17 severity distributions           | `fig7_severity.py`         |
| 8      | Per-site LOSO ACC with 95% bootstrap intervals    | `fig8_per_site.py`         |

Outputs land in `figures/output/`. Each script reads the JSON tables produced
by `experiments/` so the figures regenerate from the same source numbers as
the tables.
