# V005 Final K60 Results Mapping

This page is the single public mapping between the final manuscript V005 and
the reproducibility archive. It supersedes earlier result descriptions.

## Frozen Protocol

| Item | Final setting |
|---|---|
| Datasets | PV-Fruit (`F_new`), PV-Vegetable (`V_new`), complete MCLD-11 (`M_new`), DFLD-BR-4 (`G_new`) |
| Exact duplicate handling | SHA256 byte-level duplicate removal |
| pHash | Near-duplicate risk audit only; not used to remove images or evaluate results |
| Feature backbone | Frozen timm ConvNeXt-XLarge ImageNet-22K checkpoint |
| UMAP | 100 dimensions, `n_neighbors=15`, `min_dist=0.1`, Euclidean metric |
| Clustering | KMeans, Birch, and Agglomerative; `K=60` |
| Seeds | 11, 22, 33, 44, 55 |
| Evaluation | Transductive selective unsupervised clustering; labels only for post-hoc many-to-one alignment and external evaluation |

## Main Result Mapping

All values below are the five-seed mean +/- sample standard deviation for
`method=all3`, `umap_dim=100`, and `n_clusters=60`, read from
`results/final_k60/main_results.csv`.

| Manuscript dataset | Dataset code | Acc_kept | Coverage | Conservative overall accuracy |
|---|---|---:|---:|---:|
| PV-Fruit | `F_new` | 99.95% +/- 0.02% | 61.37% +/- 2.06% | 61.34% +/- 2.05% |
| PV-Vegetable | `V_new` | 99.07% +/- 0.24% | 82.62% +/- 5.67% | 81.84% +/- 5.48% |
| MCLD-11 | `M_new` | 90.28% +/- 0.64% | 73.36% +/- 1.51% | 66.23% +/- 1.28% |
| DFLD-BR-4 | `G_new` | 99.87% +/- 0.02% | 64.41% +/- 1.50% | 64.32% +/- 1.51% |

`Acc_kept` is the post-hoc aligned clustering accuracy of retained samples.
The conservative overall accuracy equals `acc_kept * coverage`; rejected
samples remain in its denominator and are not assumed to be intrinsically
misclassified.

## Archive-To-Manuscript Map

| Manuscript material | Authoritative archived source |
|---|---|
| Main C3 results and full ablation | `results/final_k60/main_results.csv`, `results/final_k60/ablation_results.csv` |
| All-versus-retained ARI/NMI/AMI | `results/final_k60/all_vs_retained_metrics.csv` |
| Seed-level paired Bootstrap intervals | `results/final_k60/bootstrap_results.csv` |
| Class-level accuracy and rejection | `results/final_k60/classwise_results.csv` |
| Runtime/environment | `results/final_k60/runtime_results.csv` |
| Retained and all-plus-Rejected matrices | `results/final_k60/confusion_matrices/` |
| Main and supplementary figure files | `figures/v13_figure_report/reproduced_figures/` |
| Figure source-data map | `docs/protocolA_figure_manifest.md` |
| DINOv2 external baseline | `external_baselines/dinov2/external_dinov2_results.csv` |

The figure report at `figures/v13_figure_report/index.html` links the same
final source files and figures. The matrix images originate from real
server-generated, per-sample prediction records rather than reconstructed
summary statistics.
