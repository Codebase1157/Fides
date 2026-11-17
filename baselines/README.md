# Baselines

28 baseline methods, grouped into six families. For each baseline we provide
either (a) a thin wrapper that calls the authors' original code, or (b) a
reimplementation when the authors have not released code. Every baseline is
re-trained on our splits and preprocessing.

| # | Method        | Family                 | Code source                      |
|---|---------------|------------------------|----------------------------------|
| 1 | SVM [62]      | Traditional            | `sklearn.svm.SVC` wrapper        |
| 2 | RF [63]       | Traditional            | `sklearn.ensemble` wrapper       |
| 3 | GAT [64]      | GNN                    | https://github.com/PetarV-/GAT   |
| 4 | BrainGNN [4]  | GNN                    | https://github.com/xxlya/BrainGNN_Pytorch |
| 5 | IBGNN [42]    | GNN                    | https://github.com/HennyJie/IBGNN.git |
| 6 | BrainIB [43]  | GNN                    | https://github.com/SJYuCNEL/brain-and-Information-Bottleneck |
| 7 | Xu et al.[44] | GNN                    | reimpl. under `baselines/xu_et_al/` |
| 8 | VanillaTF[65] | Transformer            | reimpl. under `baselines/vanillatf/` |
| 9 | BrainNetTF[6] | Transformer            | https://github.com/Wayfear/BrainNetworkTransformer |
|10 | Yu et al. [7] | Transformer            | reimpl. under `baselines/yu_et_al/` |
|11 | MSSTAN [8]    | Transformer            | reimpl. under `baselines/msstan/` |
|12 | Miniformer[9] | Transformer            | reimpl. under `baselines/miniformer/` |
|13 | BrainCL [13]  | Transformer            | https://github.com/author/braincl |
|14 | GCA [20]      | Graph contrastive      | https://github.com/CRIPAC-DIG/GCA |
|15 | A-GCL [21]    | Graph contrastive      | reimpl. under `baselines/a_gcl/` |
|16 | Wang et al.[22]| Graph contrastive     | reimpl. under `baselines/wang_et_al/` |
|17 | CIA-GCL [23]  | Graph contrastive      | reimpl. under `baselines/cia_gcl/` |
|18 | Zhang et al.[24]| Graph contrastive    | reimpl. under `baselines/zhang_et_al_24/` |
|19 | Zhou et al.[25]| Graph contrastive     | reimpl. under `baselines/zhou_et_al/` |
|20 | Song et al.[32]| Site-robust           | reimpl. under `baselines/song_et_al/` |
|21 | Zhang et al.[28]| Disentangled         | reimpl. under `baselines/zhang_et_al_28/` |
|22 | MADCLE [29]   | Disentangled           | reimpl. under `baselines/madcle/` |
|23 | CORE [34]     | Site-robust            | reimpl. under `baselines/core/` |
|24 | BRIO [35]     | Site-robust            | reimpl. under `baselines/brio/` |
|25 | Ji et al.[33] | Disentangled           | reimpl. under `baselines/ji_et_al/` |
|26 | SF-DisenNet[31]| Disentangled          | reimpl. under `baselines/sf_disennet/` |
|27 | Wang et al.[10]| Foundation            | https://github.com/author/brain-foundation |
|28 | BrainGFM [11] | Foundation             | https://github.com/author/braingfm |

See `baselines/configs/` for the selected hyperparameter grid per baseline.
All runs share our splits and preprocessing.
